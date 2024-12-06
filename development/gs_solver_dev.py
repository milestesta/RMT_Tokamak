import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.constants
from numba import int64,float64,boolean
from numba import prange
from numba.experimental import jitclass

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
    ('minR',float64),
    ('majR',float64),
    ('triangularity',float64),
    ('elongation',float64),
    ('Rdim',int64),
    ('Zdim',int64),
    ('N_poles',int64),
    ('sim_width',float64),
    ('sim_height',float64),
    ('a',float64),
    ('b',float64),
    ('c',float64),
    ('just_plasma',boolean),
    ('is_solovev', boolean)
]

@jitclass(spec)
class RMT_Tokamak(object):
    """
    A class for our plasma modeling within a tokamak. Based on the work in "Tao Xu and Richard Fitzpatrick 2019 Nucl. Fusion 59 064002" (DOI 10.1088/1741-4646/ab1ce3) and 
    "Toroidally symmetric polynomial multipole solutions of the vector laplace equation" by M.F Reusch and G.H Neilson. https://doi.org/10.1016/0021-9991(86)90041-0

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
        just_plasma (boolean): True only analyzes within LCFS, False analyses entire space 
        is_solovev (boolean): Flag for using the Solovev solver. 
    """

    def __init__(self,minR=0.1,majR=1,tri=0.8,elo=1,
                    Rdim=10,Zdim=10,N_poles=10,
                    sim_width=0.3,sim_height=0.3,a=1.2, b=-1.0, c=1.1,just_plasma=False, is_solovev=False):
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
        self.just_plasma = just_plasma
        self.a = a
        self.b = b
        self.c = c
        self.is_solovev = is_solovev

        #Error check for R,Z partitions
        if Rdim%2 != 0:
            raise ValueError("Error: Rdim must be even")
        if Zdim%2 != 0:
            raise ValueError("Error: Zdim must be even")

    def _pts_in_lcfs(self):
        """
        Determines which points are within plasma
        Only works for Miller surface now

        Parameters:
            aspect (float): tells us the dimensions of our Miller lcfs

        Returns:
            plasma_grid_arr (array,int): stores (R,Z) points within plasma
            plasma_ind (array,int): the indices of plasma points in grid
        """
        #Defines local variables to avoid calling object so much
        minR = self.minR
        majR = self.majR
        Rdim = self.Rdim
        Zdim = self.Zdim
        sim_width = self.sim_width 
        sim_height = self.sim_height
        elo = self.elongation
        is_solovev = self.is_solovev

        aspect = minR/majR

        R_min = majR - (aspect*majR)
        R_max = majR + (aspect*majR)

        dR = sim_width/Rdim
        dZ = sim_height/Zdim

        #will store points [r][z], but have extra padded zeros
        #we will exclude the extra zeros when outputting all points
        plasma_grid_full = np.zeros((Rdim*Zdim,2)) 
        plasma_grid_full_inds = np.zeros((Rdim*Zdim,2))
        count = 0 #will count how many points are within the lcfs
        if is_solovev == False:
            for i in range(0,Rdim):
                R = majR + (i - Rdim/2)*dR
                if (R < R_min) or (R_max < R):
                    continue #if R outside bounds, skip to next R value
                for j in range(0,Zdim):
                    Z = (j - Zdim/2)*dZ
                    theta_point = np.arctan(Z/(R-majR + 0.001))
                    Z_max = aspect*elo*np.sin(theta_point)
                    if abs(Z) <= abs(Z_max):
                        plasma_grid_full[count][0] = R
                        plasma_grid_full[count][1] = Z
                        
                        plasma_grid_full_inds[count][0] = i
                        plasma_grid_full_inds[count][1] = j

                        count += 1
        
        ## Solovev:
        elif is_solovev == True:
            a = self.a
            b = self.b
            c = self.c
            psi_x = self._psi_on_LCFS()
            Z_min = -majR*np.sqrt(((b+c)*(a-c))/(2*c*c))
            Z_max = majR*np.sqrt(((b+c)*(a-c))/(2*c*c))
            for k in range(0, Zdim):
                Z = (k - Zdim/2)*dZ
                if abs(Z) <= Z_max: #only works for up down symmetric. 
                    alpha = (2*c*(Z**2.0))/((a-c)*majR)
                    beta = (((b+c)*(Z**2.0)) - (2*psi_x/(majR**2.0)))/(a-c)
                    r_less = np.sqrt((majR**2.0) - (alpha*majR) - (majR*np.sqrt((alpha**2.0) - (4.0*beta)))) ## not the issue!
                    r_more = np.sqrt((majR**2.0) - (alpha*majR) + (majR*np.sqrt((alpha**2.0) - (4.0*beta)))) ## not the issue! 
                    for i in range(0, Rdim):
                        R = majR + (i - Rdim/2)*dR
                        if (R <= r_more) and (R >= r_less):

                            plasma_grid_full[count][0] = R
                            plasma_grid_full[count][1] = Z

                                
                            plasma_grid_full_inds[count][0] = i
                            plasma_grid_full_inds[count][1] = k


                            count += 1 

                 
                 
                 
        #Only reports coordinates within plasma, no extra cells
        plasma_grid_arr = np.zeros((count,2))
        for i in range(count):
            plasma_grid_arr[i][0] = plasma_grid_full[i][0]
            plasma_grid_arr[i][1] = plasma_grid_full[i][1]
        
        #Stores indices for plasma points
        plasma_grid_arr_inds = np.zeros((count,2))
        for i in range(count):
            plasma_grid_arr_inds[i][0] = plasma_grid_full_inds[i][0]
            plasma_grid_arr_inds[i][1] = plasma_grid_full_inds[i][1]

        return plasma_grid_arr,plasma_grid_arr_inds

    def _create_Anm(self):
        """
        Create Anm matrix: Eqn 34 in Fitzpatrick paper
        This will use notation from Eqn 17 in: https://doi.org/10.1016/0021-9991(86)90041-0
        Only for the symmetric case!

        returns:
            output (array): Anm matrix without the n=1 multipole
        """
        #Defines local variables to avoid calling object so much
        N_poles = self.N_poles

        #+1 below to ensure we have the correct number of multipoles when
        #we remove the n=1 multipole
        A_NM_array = np.zeros((N_poles+1, N_poles+1))

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
        output = np.zeros((N_poles, N_poles+1))
        for row in range(N_poles):
            if row == 0:
                shift = 0
            else:
                shift = 1 #removes n=1 multipole 
            for col in range(N_poles+1):
                output[row][col] = A_NM_array[row+shift][col]

        return output

    def _psi_on_LCFS(self,a=1.2,b=-1,c=1.1):
        """
        This function specifies the desired value of the flux on the LCFS

        Returns:
            value of flux on LCFS. 
        """
        #Using notation from the following paper:
        #Tao Xu and Richard Fitzpatrick 2019 Nucl. Fusion 59 064002
        #Defines surface
        return (a-c)*(b+c)*(b+c)*(self.majR**4.0)/(8.0*c*c) #just using the solovev lcfs value for now (Eqn 11)


    def _LCFS(self,r,theta):
        """This function takes in a radius length and a poloidal angle, and returns a the R, Z coordinates
        of the corresponding point on the LCFS. 

        Args:
            r (float): radial distance from the magnetic axis (R = majR, Z = 0) to the lcfs point. Not generally used. 
            theta (float): Poloidal coordinate from the outboard mid-plane (R = majR + r, Z = 0, counterclockwise)
        Returns:
            R_lcfs (float): R coordinate of the lcfs point.
            Z_lcfs (float): Z coordinate of the lcfs point. 
        """
        #Defines local variables to avoid calling object so much
        minR = self.minR
        majR = self.majR
        tri = self.triangularity
        elo = self.elongation
        psi_x = self._psi_on_LCFS()
        is_solovev = self.is_solovev

        #GJW Shape: TODO: Implement this!!!!
        #R_lcfs = R_0 - r*np.cos(omega(theta))#add more shaping terms as desired
        #Z_lfcs = r*np.sin(theta)
        
        #Miller Shape:
        if is_solovev == False:
            aspect = minR/majR
            R_lcfs = majR*(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(tri))) #add more shaping terms as desired
            Z_lfcs = aspect*elo*np.sin(theta)
            
        #Solovev Shape:
        elif is_solovev == True:
            a = self.a
            b = self.b
            c = self.c
            Z_min = -majR*np.sqrt(((b+c)*(a-c))/(2.0*c*c))
            Z_max = majR*np.sqrt(((b+c)*(a-c))/(2.0*c*c))
            Z = Z_min + (((Z_max-Z_min)/(2.0*pi))*theta)
            # which r we return should alternate. 
            d_theta = (2.0*pi)/self.N_poles
            sign = ((np.floor(theta/d_theta))%2.0)
            alpha = (2.0*c*(Z**2.0))/((a-c)*majR)
            beta = (((b+c)*(Z**2.0)) - (2*psi_x/(majR**2.0)))/(a-c)
            R = np.sqrt((majR**2.0) - (alpha*majR) + (((-1.0)**sign)*(majR*np.sqrt((alpha**2.0) - (4.0*beta)))))

            R_lcfs = R
            Z_lfcs = Z                                                      

        return(R_lcfs, Z_lfcs)

    def _greens_function(self,R_p, Z_p, R, Z): #calculates the value of the green's function based on the plasma at (R_p, Z_p), as viewed at (R, Z)
        """This function calculates the green's function response (see equation 23 in Xu, Fitzpatrick) at R, Z based on the plasma current at 
        R_p, Z_p

        Args:
            R_p (float): Real space R location of the plasma current. 
            Z_p (float): Real space Z location of the plasma current. 
            R (float): Real space R location of the observer. 
            Z (float): Real space Z location of the observer.
        
        Returns:
            (float): Value of the Green's response.  
        """
        k = ((4.0*R_p*R)/(((R + R_p)*(R + R_p)) + ((Z-Z_p)*(Z - Z_p))))**(0.5)

        index = int(np.round(k,4)*10000)
        if index in [9999,10000]: #avoids asymptote (can occur because of our rounding)
            index = 9998

        K = elliptic_int_1[index]
        E = elliptic_int_2[index]
        return((1/(2*pi))*(((R*R_p)**(0.5))/k)*(((2-(k*k))*K) - (2*E)))

    def _pressure_term(self,R_p, Z_p): #the pressure term in the GS equaion
        #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
        """Calculates the first term in equation 3 of Xu Fitzpatrick, essentially the diamagnetic pressure current. 

        Args:
            R_p (float): Real space R location of the plasma current. 
            Z_p (float): Real space Z location of the plasma current.
        """
        a = 1.2 #EDIT THIS LATER
        return(-a*R_p)#up down symmetric solovev tororidal current. 

    def _toroidal_field_term(self,R_p, Z_p): #the toroidal field term in the GS. 
        #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
        """Calculates the toroidal field component of the source term in the Grad-Shafranov equation. (second term in equation
        3 of Xu Fitzpatrick)

        Args:
            R_p (float): Real space R location of the plasma current. 
            Z_p (float): Real space Z location of the plasma current.
        Returns:
            (float): second term in equation 3 of Xu Fitzpatrick 
        """
        b = -1 #EDIT THIS LATER
        return(-b*(self.majR**2.0)/R_p) #up down symmetric solovev  toroidal field 

    def _plasma_current(self,R_p, Z_p):
        """Combines the pressure and toroidal contributions to the plasma current to produce the total plasma current at R_p, Z_p. Depending
        on the is_solovev flag it will return either the solovev current (equation 3 of Xu Fitzpatrick) or a typical current for a miller surface.

        Args:
            R_p (float): Real space R location of the plasma current. 
            Z_p (float): Real space Z location of the plasma current.
        
        Returns:
            (float): Total plasma current at R_p, Z_p
        """
        if self.is_solovev == True:
            current = ((1.0)*(self._pressure_term(R_p, Z_p) + self._toroidal_field_term(R_p, Z_p)))
        elif self.is_solovev == False:
            minR = self.minR
            majR = self.majR
            plat = minR*0.1
            peak = 0.5
            # elo = self.elongation
            # tri = self.triangularity
            # theta = np.arctan(Z_p/((majR-R_p)+0.001))#+0.001 to avoid division by 0. 
            # aspect = minR/majR
            # r = R_p/(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(tri)))
            # plat = minR*0.5/(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(tri)))
            r = np.sqrt(((R_p-majR)**2.0) + ((Z_p)**2.0))
            if r < plat:
                current = -peak
            else:
                current = -peak*(1-(0.5*(1+np.tanh(10*(r - (minR/2))/((minR-plat))))))
            # elo = self.elongation
            # tri = self.triangularity
            # theta = np.arctan(Z_p/((majR-R_p)+0.001))#+0.001 to avoid division by 0. 
            # aspect = minR/majR
            # psuedo_psi = R_p/(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(tri)))
            # current = -np.exp(-np.abs(psuedo_psi/majR))
        return(current)

    def _plasma_flux(self,R_o, Z_o, plasma_grid_list): #calculates the flux at (R, Z) due to the plasma current at each location listed in plasma_grid
        """Calculates the poloidal flux function at the point R_o, Z_o due to all plasma currents (stored in plasma_grid_list). 

        Args:
            R_o (float): Observer R location.
            Z_o (float): Observer Z location. 
            plasma_grid_list (int array): The indices of all elements of the computational grid that contain plasma. 
        Returns:
            (float): poloidal flux functino at (R_o, Z_o)
        """
        #Defines local variables to avoid calling object so much
        sim_width = self.sim_width
        sim_height = self.sim_height
        Rdim = self.Rdim
        Zdim = self.Zdim

        dR = sim_width/Rdim
        dZ = sim_height/Zdim
        n_points = plasma_grid_list.shape[0]
        flux = 0.0
        for n in range(0, (n_points)):
            R_p = plasma_grid_list[n][0]
            Z_p = plasma_grid_list[n][1]
            if ((R_o!=R_p) and (Z_o!=Z_p)):#to avoid self action. 
                flux += dR*dZ*self._greens_function(R_p, Z_p, R_o, Z_o)*self._plasma_current(R_p, Z_p)
        return(flux)

    def _multipole_contributions(self,plasma_grid_arr,A_NM_array):
        """This function solves a matrix equation to figure out the contribution from each pole in the multipole expansion
        of the flux due to the coils required to fix the Dirichlet boundary condition that psi = psi_x on the LCFS. 

        Args:
            plasma_grid_arr (int array): Contains the indices of all grid points containing plasma. 
            A_NM_array (float array): contains the coefficients needed to calculate the multipole expansion to order N. 

        Returns:
            (float array): a one dimensional vector containing the weight coefficient of each pole. 
        """
        #Defines local variables to avoid calling object so much
        N_poles = self.N_poles
        minR = self.minR
        majR = self.majR
        psi_x = self._psi_on_LCFS()
        a = self.a
        b = self.b
        c = self.c
        
        #Matching on the LCFS
        matching_thetas = np.zeros(N_poles)
        matching_R = np.zeros(N_poles) #R values of the matching points.  
        matching_Z = np.zeros(N_poles) #Z values of the matching points. 

        #From original definitions
        r_lcfs = minR

        aspect = minR/majR
        # R_min = majR - (aspect*majR)
        # R_max = majR + (aspect*majR)
        R_min = np.sqrt((majR**2.0) + (2*(np.sqrt((2*psi_x)/(a-c)))))
        R_max = np.sqrt((majR**2.0) - (2*(np.sqrt((2*psi_x)/(a-c)))))
        dr = (R_max - R_min)/N_poles
        for theta in range(0, N_poles):
            #Begin solovev matching. Uncomment the next two lines to use symmetric solovev. 
            # matching_R[theta] = R_min + (dr*theta)
            # matching_Z[theta] = ((-1.0)**(theta))*(((psi_lcfs - ((1.0/8.0)*(a-c)*(((matching_R[theta])*(matching_R[theta]) - (majR**2.0))**2.0)))/(0.5*((b*(majR**2.0))+(c*matching_R[theta]*matching_R[theta]))))**0.5)
    
            #uncomment the next two lines for angular definition of the LCFS points. 
            matching_thetas[theta] = (2*theta*pi/N_poles) + 0.001
            matching_R[theta], matching_Z[theta] = self._LCFS(r_lcfs, matching_thetas[theta])
        # plt.figure()
        # plt.scatter(matching_R, matching_Z)
        # plt.show()
        #Matching routine
        delta_psi = np.zeros(N_poles)#difference between the desired psi_lcfs and the psi due to the plasma current. 
        psi_multipole_matrix = np.zeros((N_poles, N_poles)) #psi_i_j has contribution of pole i at lcfs site j
        plasma_psi_lcfs = np.zeros(N_poles) #psi values at the matching points due to just the plasma current. 

        psi_lcfs = self._psi_on_LCFS()

        for i in range(0,N_poles):
            plasma_flux_temp = self._plasma_flux(matching_R[i], matching_Z[i], plasma_grid_arr)
            plasma_psi_lcfs[i] = plasma_flux_temp
            delta_psi[i] = psi_lcfs - plasma_psi_lcfs[i]

            for n in range(0,N_poles):
                psi_multipole_matrix[i][n] = self._field_due_to_pole_N(matching_R[i], matching_Z[i], n, A_NM_array) #to skip the non-existent n = 1 pole. 
        # print(np.linalg.solve(psi_multipole_matrix, delta_psi))
        return np.linalg.solve(psi_multipole_matrix, delta_psi) #contains the coefficient that quantifies the amount of that

    def _field_due_to_pole_N(self,R_o, Z_o, N, A_NM_array):
        """This function calculates the flux at a point R_o, Z_o due to the Nth multipole. 

        Args:
            R_o (float): Observer R location.
            Z_o (float): Observer Z location. 
            N (int): The order of pole that we want. Note that N=1 is skipped, so N>2 should be indexed forward. IE requesting N = 2 actually gives the 3rd pole. 
            A_NM_array (float array): contains the coefficients needed to calculate the multipole expansion to order N.

        Returns:
            float: The poloidal flux function fue the the Nth multipole. 
        """
        if N == 0:
            return 1
        else:
            psi = 0
            for m in range(0, self.N_poles):
                psi += A_NM_array[N][m]*(R_o**((N+1)-m))*(Z_o**m) #N+1 as we are actually a pole higher due to the missing n = 1 pole. 
            return psi

    def _field_due_to_all_poles(self,R,Z,A_NM_array,multipole_contributions):
        """This function uses the weighting coefficients to calculate the overall coil contribution at a point R, Z. 

        Args:
            R (float): Observer R location. 
            Z (float): Observer Z location. 
            A_NM_array (float array): contains the coefficients needed to calculate the multipole expansion to order N.
            multipole_contributions (float array): a one dimensional vector containing the weight coefficient of each pole. 
        Returns: 
            (float): The poloidal flux function at R, Z due to all poles. Basically the coil contribution. 
        """
        psi_mp = 0.0
        for n in range(0, self.N_poles):
            psi_mp += self._field_due_to_pole_N(R, Z, n, A_NM_array)*multipole_contributions[n] 
        return(psi_mp)
    
    def plasma_current_grid(self):
        """
        Returns:
            current_grid (float array): plasma current at all points in [Z][R] sampling. Will be 0 outside the LCFS. 
        """
        #Defines local variables to avoid calling object so much
        sim_width = self.sim_width
        sim_height = self.sim_height
        Rdim = self.Rdim
        Zdim = self.Zdim
        majR = self.majR

        #Determines points within plasma
        plasma_grid_arr,plasma_grid_arr_inds = self._pts_in_lcfs()

        #Determines Amn matrix
        Anm = self._create_Anm()

        #Determines multipole contributions
        MPC = self._multipole_contributions(plasma_grid_arr,Anm)

        current_grid = np.zeros((Zdim, Rdim)) #flux at all grid points

        for pt in prange(0, plasma_grid_arr.shape[0]):
            #Defines point
            R,Z = plasma_grid_arr[pt]
            i,j = plasma_grid_arr_inds[pt].astype(int64)
            current_grid[j][i] = self._plasma_current(R, Z)
        return(current_grid)
                    
    def compute_psi(self):
        """
        Returns:
            computational_grid (float array): psi at all points in [Z][R] sampling
        """
        #Defines local variables to avoid calling object so much
        sim_width = self.sim_width
        sim_height = self.sim_height
        Rdim = self.Rdim
        Zdim = self.Zdim
        majR = self.majR

        #Determines points within plasma
        plasma_grid_arr,plasma_grid_arr_inds = self._pts_in_lcfs()

        #Determines Amn matrix
        Anm = self._create_Anm()

        #Determines multipole contributions
        MPC = self._multipole_contributions(plasma_grid_arr,Anm)

        dR = sim_width/Rdim
        dZ = sim_height/Zdim

        computational_grid = np.zeros((Zdim, Rdim)) #flux at all grid points

        if self.just_plasma is False: #If analyzing the entire R-Z space
            for i in prange(0, Rdim): #prange paralellizes for loop
                R = majR + (i - (Rdim/2.0))*dR
                for j in range(0, Zdim): 
                    Z = (j - (Zdim/2.0))*dZ
                    computational_grid[j][i] = self._field_due_to_all_poles(R,Z,Anm,MPC) + self._plasma_flux(R, Z, plasma_grid_arr)
        else: #If analyzing the just plasma space
            for pt in prange(0, plasma_grid_arr.shape[0]):
                #Defines point
                R,Z = plasma_grid_arr[pt]
                i,j = plasma_grid_arr_inds[pt].astype(int64)

                computational_grid[j][i] = self._field_due_to_all_poles(R,Z,Anm,MPC) + self._plasma_flux(R, Z, plasma_grid_arr)

        return computational_grid
    
    @staticmethod
    def _D(grid,dir,dir_delta):
        """
        Computes the first derivative. https://en.wikipedia.org/wiki/Finite_difference_coefficient#

        Parameters:
            grid (array,float): this is what we will be differentiating
            dir (int): this is the direction we will differentiate (which axis). 0=row and 1=col
            dir_delta (float): this is the spacing between points in our desired differentiation direction

        Returns:
            der (array,float): this is the first derivative
        """

        if dir not in [0,1]: 
            raise ValueError(f'Direction for derivative must be 0 (row) or 1 (col)')
        
        der = np.empty(grid.shape) # will store our derivative

        if dir==0: # Derivative along the row
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if col in [0,1]: # first two columns (forward derivative)
                        der[row][col] = (-11/6*grid[row][col] + 3*grid[row][col] - 3/2*grid[row][col] + 1/3*grid[row][col])/dir_delta
                    elif col in [grid.shape[1]-2,grid.shape[1]-1]: # last two columns (backward derivative)
                        der[row][col] = (-1/3*grid[row][col-3] + 3/2*grid[row][col-2] - 3*grid[row][col-1] + 11/6*grid[row][col])/dir_delta
                    else: # middle (full 5-pt-derivative)
                        der[row][col] = (1/12*grid[row][col-2] - 2/3*grid[row][col-1] + 2/3*grid[row][col+1] - 1/12*grid[row][col+2])/dir_delta
        elif dir==1: # Derivative along the column
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if row in [0,1]: # first two columns (forward derivative)
                        der[row][col] = (-11/6*grid[row][col] + 3*grid[row+1][col] - 3/2*grid[row+2][col] + 1/3*grid[row+3][col])/dir_delta
                    elif row in [grid.shape[0]-2,grid.shape[0]-1]: # last two columns (backward derivative)
                        der[row][col] = (-1/3*grid[row-3][col] + 3/2*grid[row-2][col] - 3*grid[row-1][col] + 11/6*grid[row][col])/dir_delta
                    else: # middle (full 5-pt-derivative)
                        der[row][col] = (1/12*grid[row-2][col] - 2/3*grid[row-1][col] + 2/3*grid[row+1][col] - 1/12*grid[row+2][col])/dir_delta
        
        return der

    def compute_B(self,psi_grid):
        """
        Computes the r and z components of our magnetic field (Vector field) determined by
            Bz = +(1/R) * d(psi)/dr
            Br = -(1/R) * d(psi)/dz

        Parameters:
            psi_grid (array,float): this is the standard output of compute_psi

        Returns:
            Bz (array,float): this is the z-component of the magnetic field at psi[z][r]
            Br (array,float): this is the r-component of the magnetic field at psi[z][r]
        """

        #Defines local variables to avoid calling object so much
        R = self.majR
        sim_height = self.sim_height
        sim_width = self.sim_width
        Rdim = self.Rdim
        Zdim = self.Zdim

        # Computes first derivatives
        # psi[z][r]
        dZ = sim_height/Zdim
        Dz_psi = self._D(psi_grid,dir=1,dir_delta=dZ)
        dR = sim_width/Rdim
        Dr_psi = self._D(psi_grid,dir=0,dir_delta=dR)

        Bz =  (1/R)*Dr_psi # z component of magnetic field
        Br = -(1/R)*Dz_psi # r component of magnetic field

        return Bz,Br

    @staticmethod    
    def cyl2xyz(pos_cyl): 
        """
        Converts data point from cylinderical coordinates to rectangular coordinates

        Parameters:
            pos_cyl (array,float): position to change coordinates [phi][z][r]
        
        Returns:
            pos_xyz (array,float): position in rectangular coordinates [x][y][z]
        """
        pos_xyz = np.zeros(3)
        pos_xyz[0] = pos_cyl[2]*np.cos(pos_cyl[0])
        pos_xyz[1] = pos_cyl[2]*np.sin(pos_cyl[0])
        pos_xyz[2] = pos_cyl[1]

        return pos_xyz

def psi_output(grid,just_plasma = False, fname='psi_grid.dat'):
    """
    This outputs the raw psi data on the r,z grid.

    Parameters:
        grid (array,float): this is the standard output of RMT_Tokamak.compute_psi()
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
                if just_plasma == True:
                    if grid[zind][rind] < 1e-8:
                        continue #skips printing grid value if it is zero
                    else:
                        output.write('{:^6} {:^6}    {:8.10f}\n'.format(rind,zind,grid[zind][rind]))
                else:
                    output.write('{:^6} {:^6}    {:8.10f}\n'.format(rind,zind,grid[zind][rind]))
        

def current_output(grid, just_plasma = False, fname='current_grid.dat'):
    """
    This outputs the plasma current data on the r,z grid.

    Parameters:
        grid (array,float): this is the standard output of RMT_Tokamak.compute_psi(), but this time for the plasma current. 
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
        output.write(' r_ind  z_ind         current\n')
        for zind in range(grid.shape[0]):
            for rind in range(grid.shape[1]):
                if just_plasma == True:
                    if grid[zind][rind] < 1e-8:
                        continue #skips printing grid value if it is zero
                    else:
                        output.write('{:^6} {:^6}    {:8.10f}\n'.format(rind,zind,grid[zind][rind]))
                else:
                    output.write('{:^6} {:^6}    {:8.10f}\n'.format(rind,zind,grid[zind][rind]))

