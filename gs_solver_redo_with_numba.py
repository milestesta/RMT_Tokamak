import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.special
import scipy.constants
from numba import int32,float32, boolean
from numba import prange
from numba.experimental import jitclass
from timeit import default_timer as timer
#Establishes constants
pi = np.pi
mu = scipy.constants.mu_0

#Creates global arrays which store elliptical integral values
#scipy.special does not work with numba, so this is my best solution
elliptic_int_1 = np.zeros(10000)# #will store values for complete elliptic integral of the first kind.  
elliptic_int_2 = np.zeros(10000)# #will store values for complete elliptic integral of the second kind. 
for ind,x in enumerate(np.linspace(0,1,10000)):
    if ind == 10000-1: #occurs for input=1.  We do it this way to have a spacing of about 0.001 between x points
        break #avoids asymptote at ellipk(x=1)
    elliptic_int_1[ind] = scipy.special.ellipk(x)
    elliptic_int_2[ind] = scipy.special.ellipe(x)
    
spec = [
    ('minR',float32),
    ('majR',float32),
    ('triangularity',float32),
    ('elongation',float32),
    ('Rdim',int32),
    ('Zdim',int32),
    ('N_poles',int32),
    ('sim_width',float32),
    ('sim_height',float32),
    ('a',float32),
    ('b',float32),
    ('c',float32),
    ('is_solovev',boolean)
]

@jitclass(spec)
class RRT_Tokamak(object):
    """
    A class for our plasma modeling within a tokamak

    Parameters:
        minR (float): minor radius in meters
        majR (float): major radius in meters
        tri (float): the triangularity of plasma shape
        elo (float): the elongation of the plasma shape
        Rdim (int): number of points to sample in radial direction
        Zdim (int): number of points to sample in z direction
        N_poles (int): Order to which you expand the multipole 
            aproximation to the actual contribution from the coils. 
        sim_width (float): (simulation width) range of R values to sample
        sim_height (float): (simulation height) range of Z values to sample
        a (float): solovev constant for pressure.
        b (float): solovev constant for toroidal field.
        c (float): solovev constant for shaping. 
    """

    def __init__(self,minR=0.1,majR=1,tri=0.8,elo=1,
            Rdim=10,Zdim=10,N_poles=10,
            sim_width=0.3,sim_height=0.3, a=1.2, b=-1.0, c=1.1, is_solovev = True):
        #Sets user-input constraints
        self.minR = minR
        self.majR = majR
        self.triangularity = tri
        self.elongation = elo
        self.Rdim = Rdim
        self.Zdim = Zdim
        self.N_poles = N_poles
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.a = a
        self.b = b
        self.c = c
        self.is_solovev = is_solovev
        
    def _create_Anm(self, N_poles):
        """
            Create Anm matrix: Eqn 34 in Fitzpatrick paper
            This will use notation from Eqn 17 in: https://doi.org/10.1016/0021-9991(86)90041-0
            Only for the symmetric case!

            returns:
                output (array): Anm matrix without the n=1 multipole
            """

            #+1 below to ensure we have the correct number of multipoles when
            #we remove the n=1 multipole
        A_NM_array = np.zeros((N_poles+1, N_poles+1), float)

        for n in range(0, N_poles+1):
            if (n%2 == 0):
                A_NM_array[n][0] = 1
            else:
                A_NM_array[n][1] = 1

        A_NM_array[1][1] = 0
        A_NM_array[4][2] = -4 #F4 from Eqn 15 in linked paper above

        for n in range(4, N_poles+1):
            for m in range(2, N_poles+1):
                A_NM_array[n][m] = -1*A_NM_array[n][m-2]*(((n+2-m)*(n-m))/(m*(m-1))) #Eqn 34

        #Removes n=1 multipole
        output = np.zeros((N_poles, N_poles+1), float)
        for row in range(N_poles):
            if row == 0:
                shift = 0
            else:
                shift = 1 #removes n=1 multipole 
            for col in range(N_poles+1):
                output[row][col] = A_NM_array[row+shift][col]
        return output

    def _field_due_to_pole_N(self, R_o, Z_o, N, A_NM_array, N_poles):
        """Calculates the field due to the Nth symmetric multipole, a' la' Eqn 17 in: https://doi.org/10.1016/0021-9991(86)90041-0

        Args:
            R_o (float): Radial location at which the field should be calculated. 
            Z_o (float): Vertical location at which the field should be calculated. 
            N (int): Pole number
            A_NM_array (float): Array holding the coefficients needed to calculate the multipoles. 
            N_poles (float): The total number of poles we expand up to. 

        Returns:
            float: The flux at R_o, Z_o due to the Nth multipole. 
        """
        if N == 0:
            return 1
        else:
            psi = 0
            for m in range(0, N_poles):
                psi += A_NM_array[N][m]*(R_o**((N+1)-m))*(Z_o**m) #N+1 as we are actually a pole higher due to the missing n = 1 pole. 
            return psi

    def _create_computational_grid(self):
        """This function creates the full computational grid in the (R, Z) plane, indexed as grid[Z, R]. 
        It returns an array that is of the shape [Zdim, Rdim], as well as arrays [Rdim, 1] and [Zdim, 1] which 
        contain all of the points in the axis. 
        """
        dR = self.sim_width/self.Rdim
        dZ = self.sim_height/self.Zdim
        
        computational_grid = np.array((self.Zdim, self.Rdim), float)
        R_array = np.array([(self.majR - (self.sim_width/2)) + i*dR for i in range(0, self.Rdim)])
        Z_array = np.array([k*dZ - (self.sim_height/2) for k in range(0, self.Zdim)])
        return(R_array, Z_array)
    
    def _psi_val_on_LCFS(self):
        """This function calculates/defines the value that we want the flux to be on the LCFS.
        """
        a = self.a
        b = self.b
        c = self.c
        return(((a-c)*((b+c)**2.0))/(8*(c**2.0))) #solovev lcfs. 
    
    def _shape_of_LCFS(self, R_array, Z_array, psi_x):
        """This function contains the shape of the LCFS in terms of R, Z, and psi_X, outputted as a list of [Z, R] points. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            psi_x (float): The intended value of the flux at the LCFS. 
        
        Returns: 
            lcfs_array (float): The spatial [Z, R] locations of the points on the LCFS. 
        """
        if self.is_solovev:
            a = self.a
            b = self.b
            c = self.c
            #If the shape is the symmetric solovev:
            zmin = -((((b+c)*(a-c))/(2*(c**2.0)))**0.5)*self.majR
            zmax = ((((b+c)*(a-c))/(2*(c**2.0)))**0.5)*self.majR
            dR = R_array[1] - R_array[0]
            lcfs_list = [] #to be converted to a np array after. 
            for k in range(0, len(Z_array)):
                Z = Z_array[k]
                if (Z >= zmin) and (Z <= zmax):
                    alpha = (2*c*self.majR*Z*Z)/(a-(c*self.majR*self.majR))
                    beta = (((b+c)*Z*Z)/(a-c)) - ((2*psi_x)/((a-c)*self.majR*self.majR))
                    r_plus = (((self.majR*self.majR)-(alpha*self.majR))+(self.majR*(((alpha*alpha)-(4*beta))**0.5)))**0.5
                    r_minus = (((self.majR*self.majR)-(alpha*self.majR))-(self.majR*(((alpha*alpha)-(4*beta))**0.5)))**0.5
                    for i in range(0, len(R_array)):
                        R = R_array[i]
                        if np.abs(R-r_plus) <= (dR/2):
                            lcfs_list.append([Z, r_plus])
                        if np.abs(R-r_minus) <= (dR/2):
                            lcfs_list.append([Z, r_minus])
            lcfs_array = np.array(lcfs_list)
        else:
            dR = R_array[1]-R_array[0]
            dZ = Z_array[1]-Z_array[0]
            aspect = self.minR/self.majR
            R_0 = self.majR
            R_max = R_0 + R_0*aspect
            R_min = R_0 - R_0*aspect
            lcfs_list = [] #to be converted to a np array after.
            #R_lcfs = R_0*(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(self.triangularity))) #add more shaping terms as desired
            for i in range(0, len(R_array)):
                R = R_array[i]
                if abs(R-R_min) <= dR/2:
                    R_min_index = i
                if abs(R-R_max) <= dR/2:
                    R_max_index = i
            print(R_min_index)
            print(R_max_index)

            # for k in range(0, len(Z_array)):
            #     Z = Z_array[k]
            #     theta = np.arcsin(Z/(aspect*self.elongation))
                
            #     for i in range(R_min_index, R_max_index+1):
            for i in range(R_min_index, R_max_index+1):
                R = R_array[i]
                for k in range(0, int(len(Z_array)/2)):
                    Z = Z_array[k]
                    theta = np.arctan(Z/R)
                    Z_prospect = aspect*self.elongation*np.sin(theta)
                    print(Z_prospect)
                    if np.abs(Z-Z_prospect) <= dZ:
                        print("bingo!")
                        lcfs_list.append([Z, R])
                        lcfs_list.append([-Z, R])
            lcfs_array = np.array(lcfs_list)
        print(lcfs_array)
        return(lcfs_array)
    
    def _plasma_list(self, R_array, Z_array, psi_x):
        """This function returns two lists of the points on the computational grid that contain plasma. The first list of 
        points is in the form[[Z1, R1], [Z2, R2], .... ], and the second is the same but with the index of each point. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            psi_x (float): The intended value of the flux at the LCFS. 
        """
        if self.is_solovev:
            a = self.a
            b = self.b
            c = self.c
            #If the shape is the symmetric solovev:
            zmin = -((((b+c)*(a-c))/(2*(c**2.0)))**0.5)*self.majR
            zmax = ((((b+c)*(a-c))/(2*(c**2.0)))**0.5)*self.majR
            plasma_position_points = []
            plasma_index_points = []
            
            for k in range(0, len(Z_array)):
                Z = Z_array[k]
                if (Z >= zmin) and (Z <= zmax):
                    alpha = (2*c*self.majR*Z*Z)/(a-(c*self.majR*self.majR))
                    beta = (((b+c)*Z*Z)/(a-c)) - ((2*psi_x)/((a-c)*self.majR*self.majR))
                    rmax = (((self.majR*self.majR)-(alpha*self.majR))+(self.majR*(((alpha*alpha)-(4*beta))**0.5)))**0.5
                    rmin = (((self.majR*self.majR)-(alpha*self.majR))-(self.majR*(((alpha*alpha)-(4*beta))**0.5)))**0.5
                    for i in range(0, len(R_array)):
                        R = R_array[i]
                        if (R >= rmin) and (R <= rmax):
                            plasma_position_points.append([Z, R])
                            plasma_index_points.append([k, i])
            plasma_positions = np.array(plasma_position_points)
            plasma_indices = np.array(plasma_index_points)
        return(plasma_positions, plasma_indices)
            
    def _plasma_grid(self, plasma_index_list):
        """This function returns an array of the same shape as compuational grid, with a one where the 
        grid contains a plasma point, and a zero where it doesn't.
        
        Args:
        plasma_index_list (float): numpy array [#of plasma points by 2] containing the indices of where plasma points are.
        
        Returns:
        plasma_grid (float): numpy array [Zdim by Rdim] with zeros where there is no plasma, and ones where there is. 

        
        """
        plasma_grid = np.zeros((self.Zdim, self.Rdim), float)
        for index in plasma_index_list:
            plasma_grid[index[0], index[1]] = 1
        return(plasma_grid)
    
    def show_plasma(self):
        
        """This function just plots the location of the plasma on the computational grid. 
        """
        R_array, Z_array = self._create_computational_grid()
        psi_x = self._psi_val_on_LCFS()
        plasma_points, plasma_indices = self._plasma_list(R_array, Z_array, psi_x)
        print("number of plasma points is: " + str(len(plasma_points)))
        print(plasma_indices[3])
        plt.scatter(plasma_points[:,1], plasma_points[:, 0])
        plt.figure()
        plt.imshow(self._plasma_grid(plasma_indices))
        plt.show()
        
    def _greens_function(self, R_p, Z_p, R, Z):
        """calculates the value of the green's function based on the plasma at (R_p, Z_p), as viewed at (R, Z)

        Args:
            R_p (float): radial location of the plasma point. 
            Z_p (float): vertical location of the plasma point. 
            R (float): radial location at which we want the flux evaluated. 
            Z (float): vertical location at which we want the flux evaluated. 
        """
        k = ((4.0*R_p*R)/(((R + R_p)*(R + R_p)) + ((Z-Z_p)*(Z - Z_p))))**(0.5)

        index = int(np.round(k,4)*10000)
        if index in [9999,10000]: #avoids asymptote (can occur because of our rounding)
            index = 9998

        K = elliptic_int_1[index]
        E = elliptic_int_2[index]
        return((1/(2*pi))*(((R*R_p)**(0.5))/k)*(((2-(k*k))*K) - (2*E)))
    
    def _plasma_current(self, R_array, Z_array, plasma_points, plasma_indices):
        """calculates the current at all points in the computational grid, giving 0 if the point
        is outside of the LCFS. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            plasma_points (float): spatial location [Z, R] of all plasma points. 
            plasma_indices (int): index location [Z, R] of all plasma points. 
            
        Returns:
            current_grid (float): The value of the plasma current at each point [Z, R] in the computational grid. 
        """
        a = self.a
        b = self.b
        c = self.c
        R_dim = len(R_array)
        Z_dim = len(Z_array)
        current_grid = np.zeros((Z_dim, R_dim), float)
        for n in range(0, len(plasma_points)):
            R = plasma_points[n, 1]
            k = plasma_indices[n, 0]
            i = plasma_indices[n, 1]
            current_grid[k, i] = (-a*R) - ((b*(self.majR**2.0))/R)

        return(current_grid)            
    
    def show_current_grid(self):
        """This function just plots the plasma current grid. 
        """
        R_array, Z_array = self._create_computational_grid()
        psi_x = self._psi_val_on_LCFS()
        plasma_points, plasma_indices = self._plasma_list(R_array, Z_array, psi_x)
        current_grid = self._plasma_current(R_array, Z_array, plasma_points, plasma_indices)
        plt.imshow(current_grid)
        plt.show()
    
    def _plasma_flux(self, R_array, Z_array, plasma_points, plasma_indices, current_grid):
        """This function calculates the flux at all positions in the grid due to the plasma alone. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            plasma_points (float): spatial location [Z, R] of all plasma points. 
            plasma_indices (int): index location [Z, R] of all plasma points. 
            current_grid (float): The value of the plasma current at each point [Z, R] in the computational grid. 
            
        Returns:
            plasma_flux (float): The flux at each grid point [Z, R] due to just the plasma. 
        """
        R_dim = len(R_array)
        Z_dim = len(Z_array)
        dR = R_array[1] - R_array[0]
        dZ = Z_array[1] - Z_array[0]
        plasma_flux = np.zeros((Z_dim, R_dim), float)
        for k in range(0, len(Z_array)):
            for i in range(0, len(R_array)):
                R = R_array[i]
                Z = Z_array[k]
                for n in range(0, len(plasma_points)):
                    R_p = plasma_points[n, 1]
                    Z_p = plasma_points[n, 0]
                    i_p = plasma_indices[n, 1]
                    k_p = plasma_indices[n, 0]
                    if (R_p != R) and (Z_p != Z):
                        plasma_flux[k, i] += dR*dZ*current_grid[k_p, i_p]*self._greens_function(R_p, Z_p, R, Z)
        return(plasma_flux)
                
    def show_plasma_flux(self):
        """This function just plots the plasma flux. 
        """
        R_array, Z_array = self._create_computational_grid()
        psi_x = self._psi_val_on_LCFS()
        plasma_points, plasma_indices = self._plasma_list(R_array, Z_array, psi_x)
        current_grid = self._plasma_current(R_array, Z_array, plasma_points, plasma_indices)
        plasma_flux = self._plasma_flux(R_array, Z_array, plasma_points, plasma_indices, current_grid)
        
        psi_fig, psi_ax = plt.subplots()
        psi_CS = psi_ax.contour(R_array, Z_array, plasma_flux, levels = np.linspace(np.min(plasma_flux), np.max(plasma_flux), 40))
        plt.show()
    
    
    def _multipole_matching(self, R_array, Z_array, LCFS_points, psi_x, plasma_flux, A_NM_ARRAY):
        """This function performs the matching routine, figuring out how much of each multipole needs to be included to set the 
        value of the flux at the LCFS to the desired psi_x. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            LCFS_points (float): The spatial [Z, R] locations of the points on the LCFS. 
            psi_x (float): The desired value of the flux on the LCFS. 
            plasma_flux (float): The flux at each grid point [Z, R] due to just the plasma. 
            A_NM_ARRAY (float): Array needed to compute the multipoles. 
        Returns:
            multipole_contributions (float): The coefficient that each multipole is multiplied by to get psi_x on the LCFS. 
        """
        #finding the points at which we'll perform the matching. Evenly spaced in Z, and need N_poles
        #matching points for N_poles.
        initial_R = LCFS_points[:, 1]
        initial_Z = LCFS_points[:, 0]
        theta = np.arctan(initial_Z/initial_R)
        sorting_indices = np.argsort(theta)
        sorted_R = initial_R[sorting_indices]
        sorted_Z = initial_Z[sorting_indices]
        index_spacing = np.linspace(0, len(sorted_R)-1, self.N_poles, dtype=int)
        matching_R = sorted_R[index_spacing]
        matching_Z = sorted_Z[index_spacing]
        
        
        N_poles = self.N_poles
        dR = R_array[1]-R_array[0]
        dZ = Z_array[1]-Z_array[0]

        print(matching_R)
        print(matching_Z)

        plt.scatter(matching_R, matching_Z)
        plt.show()
        delta_psi = np.zeros((N_poles, 1), float)
        psi_multipole_matrix = np.zeros((N_poles, N_poles), float)
        for i in range(0, N_poles):
            R = matching_R[i]
            Z = matching_Z[i]
            R_index = int((R-self.majR+(self.sim_width/2))/dR)
            Z_index = int((Z+(self.sim_height/2))/dZ)
            delta_psi[i] = psi_x - plasma_flux[Z_index, R_index]
            print("delta psi: " + str(delta_psi[i]))
            print("psi lcfs: " + str(psi_x))
            matching_R[i] = R
            matching_Z[i] = Z
            for n in range(0, N_poles):
                psi_multipole_matrix[i, n] = self._field_due_to_pole_N(R, Z, n, A_NM_ARRAY, N_poles)
                print("field due to pole " + str(n)+ " at this point: " + str(psi_multipole_matrix[i, n]))
            print("_-_-_")
        multipole_contributions = np.linalg.solve(psi_multipole_matrix, delta_psi)
        print("multipole contributions: ")
        print(multipole_contributions)
        return(multipole_contributions, matching_R, matching_Z)
    
    def _multipole_flux(self, R_array, Z_array, LCFS_points, psi_x, plasma_flux, A_NM_ARRAY):
        """The flux at each grid point due to the multipoles alone. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid. 
            LCFS_points (float): The spatial [Z, R] locations of the points on the LCFS. 
            psi_x (float): The desired value of the flux on the LCFS. 
            plasma_flux (float): The flux at each grid point [Z, R] due to just the plasma. 
            A_NM_ARRAY (float): Array needed to compute the multipoles. 
        Returns:
            multipole_flux (float): An array holding the flux at each [Z, R] grid point, due to the multipoles alone. 
        """
        R_dim = len(R_array)
        Z_dim = len(Z_array)
        multipole_flux = np.zeros((Z_dim, R_dim), float)
        N_poles = self.N_poles
        multipole_contributions, matching_R, matching_Z = self._multipole_matching(R_array, Z_array, LCFS_points, psi_x, plasma_flux, A_NM_ARRAY)
        for k in range(0, Z_dim):
            Z = Z_array[k]
            for i in range(0, R_dim):
                R = R_array[i]
                for n in range(0, N_poles):
                    multipole_flux[k, i] += (multipole_contributions[n]*self._field_due_to_pole_N(R, Z, n, A_NM_ARRAY, N_poles))
        return(multipole_flux)
    
    def show_lcfs(self, R_array, Z_array, plasma_flux):
        A_NM_ARRAY = self._create_Anm(self.N_poles)
        R_array, Z_array = self._create_computational_grid()
        psi_x = self._psi_val_on_LCFS()
        lcfs = self._shape_of_LCFS(R_array, Z_array, psi_x)
        trash, matching_R, matching_Z = self._multipole_matching(R_array, Z_array, lcfs, psi_x, plasma_flux, A_NM_ARRAY)
        plt.scatter(lcfs[:, 1], lcfs[:, 0])
        plt.scatter(matching_R, matching_Z)
        plt.show()
    def show_multipole_flux(self, R_array, Z_array, multipole_flux):
        """Plots a contour plot of the multipole flux. 

        Args:
            R_array (float): numpy array of the radial points in the computational grid. 
            Z_array (float): numpy array of the vertical points in the computational grid.
            multipole_flux (float): An array holding the flux at each [Z, R] grid point, due to the multipoles alone. 
        """
        psi_fig, psi_ax = plt.subplots()
        psi_CS = psi_ax.contour(R_array, Z_array, multipole_flux, levels = np.linspace(np.min(multipole_flux), np.max(multipole_flux), 40))
        plt.show()
    
    def get_psi(self):
        """This function calls all of the previously defined functions in order to calculate the overal flux on the computational grid. 
        Returns:
            total_flux (float): The flux at each [Z, R] point due to all possible sources. 
            plasma_flux (float): The flux at each [Z, R] point due to just the plasma. 
            multipole_flux (float): The flux at each [Z, R] point due to just the multipoles. 
        """
        N_poles = self.N_poles
        R_array, Z_array = self._create_computational_grid()
        psi_x = self._psi_val_on_LCFS()
        plasma_points, plasma_indices = self._plasma_list(R_array, Z_array, psi_x)
        plasma_grid = self._plasma_grid(plasma_indices)
        LCFS_points = self._shape_of_LCFS(R_array, Z_array, psi_x)
        current_grid = self._plasma_current(R_array, Z_array, plasma_points, plasma_indices)
        plasma_flux = self._plasma_flux(R_array, Z_array, plasma_points, plasma_indices, current_grid)
        A_NM_ARRAY = self._create_Anm(N_poles)
        multipole_flux = self._multipole_flux(R_array, Z_array, LCFS_points, psi_x, plasma_flux, A_NM_ARRAY)
        total_flux = multipole_flux+plasma_flux
        return(total_flux, plasma_flux, multipole_flux)

    def plot_psi(self, psi_grid, nlevs):
        """This function plots a contour plot of any supplied flux on the computational grid. 

        Args:
            psi_grid (float): Any flux function of dimension [Zdim, Rdim]
            nlevs (int): The number of contours desired. 
        """
        R_array, Z_array = self._create_computational_grid()
        psi_fig, psi_ax = plt.subplots()
        psi_CS = psi_ax.contour(R_array, Z_array, psi_grid, levels = np.linspace(np.min(psi_grid), np.max(psi_grid), nlevs))
        psi_ax.clabel(psi_CS, inline=True, fontsize=4)
        plt.show()
    
    def psi_output(grid,fname='psi_grid.dat'):
        """
        This outputs the raw psi data on the r,z grid.

        Parameters:
            grid (array,float): this is the standard output of RRT_Tokamak.compute_psi()
            fname (string): this denotes the name of the output file to write to
        """

        if grid.ndim != 2:
            raise ValueError('Error: input for psi_output() is not 2 dimensional')

        with open(fname, "w") as output:
            output.write('r = R + (r_ind - rdim/2)*(sim_width/rdim)\n')
            output.write('z = (z_ind - zdim/2)*(sim_height/zdim)\n')
            output.write('\n')
            output.write(f'Number of r points: {grid.shape[1]}\n')
            output.write(f'Number of z points: {grid.shape[0]}\n')
            output.write('\n')
            output.write(' r_ind  z_ind         psi\n')
            for zind in range(grid.shape[0]):
                for rind in range(grid.shape[1]):
                    if grid[zind][rind] < 1e-8:
                        continue #skips printing grid value if it is zero
                    else:
                        output.write('{:^6} {:^6}    {:8.10f}\n'.format(rind,zind,grid[zind][rind]))
        
             
            
