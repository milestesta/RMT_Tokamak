import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.constants
from numba import int32,float32,boolean
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
    ('minR',float32),
    ('majR',float32),
    ('triangularity',float32),
    ('elongation',float32),
    ('Rdim',int32),
    ('Zdim',int32),
    ('N_poles',int32),
    ('sim_width',float32),
    ('sim_height',float32),
    ('just_plasma',boolean)
]

@jitclass(spec)
class RMT_Tokamak(object):
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
        just_plasma (boolean): True only analyzes within LCFS, False analyses entire space 
    """

    def __init__(self,minR=0.1,majR=1,tri=0.8,elo=1,
                    Rdim=10,Zdim=10,N_poles=10,
                    sim_width=0.3,sim_height=0.3,just_plasma=False):
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

    def miller_surface(self,a=1.2,b=-1,c=1.1):
        """
        Establishes surface?

        Returns:
            ?
        """
        #Using notation from the following paper:
        #Tao Xu and Richard Fitzpatrick 2019 Nucl. Fusion 59 064002
        #Defines surface
        return (a-c)*(b+c)*(b+c)*(self.majR**4.0)/(8.0*c*c) #just using the solovev lcfs value for now (Eqn 11)

    def _LCFS(self,r,theta):
        #Defines local variables to avoid calling object so much
        minR = self.minR
        majR = self.majR
        tri = self.triangularity
        elo = self.elongation

        #GJW Shape: TODO: Implement this!!!!
        #R_lcfs = R_0 - r*np.cos(omega(theta))#add more shaping terms as desired
        #Z_lfcs = r*np.sin(theta)
        
        #Miller Shape:
        aspect = minR/majR
        R_lcfs = majR*(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(tri))) #add more shaping terms as desired
        Z_lfcs = aspect*elo*np.sin(theta)
        
        #Solovev Shape:
        return(R_lcfs, Z_lfcs)

    def _greens_function(self,R_p, Z_p, R, Z): #calculates the value of the green's function based on the plasma at (R_p, Z_p), as viewed at (R, Z)
        k = ((4.0*R_p*R)/(((R + R_p)*(R + R_p)) + ((Z-Z_p)*(Z - Z_p))))**(0.5)

        index = int(np.round(k,4)*10000)
        if index in [9999,10000]: #avoids asymptote (can occur because of our rounding)
            index = 9998

        K = elliptic_int_1[index]
        E = elliptic_int_2[index]
        return((1/(2*pi))*(((R*R_p)**(0.5))/k)*(((2-(k*k))*K) - (2*E)))

    def _pressure_term(self,R_p, Z_p): #the pressure term in the GS equaion
        #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
        a = 1.2 #EDIT THIS LATER
        return(-a)#up down symmetric solovev tororidal current. 

    def _toroidal_field_term(self,R_p, Z_p): #the toroidal field term in the GS. 
        #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
        b = -1 #EDIT THIS LATER
        return(-b*R_p*self.majR) #up down symmetric solovev  toroidal field 

    def _plasma_current(self,R_p, Z_p):
        return((1.0)*(self._pressure_term(R_p, Z_p) + self._toroidal_field_term(R_p, Z_p)))

    def _plasma_flux(self,R_o, Z_o, plasma_grid_list): #calculates the flux at (R, Z) due to the plasma current at each location listed in plasma_grid
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
        #Defines local variables to avoid calling object so much
        N_poles = self.N_poles
        minR = self.minR
        majR = self.majR

        #Matching on the LCFS
        matching_thetas = np.zeros(N_poles, float)
        matching_R = np.zeros(N_poles, float) #R values of the matching points.  
        matching_Z = np.zeros(N_poles, float) #Z values of the matching points. 

        #From original definitions
        r_lcfs = minR

        aspect = minR/majR
        R_min = majR - (aspect*majR)
        R_max = majR + (aspect*majR)
        dr = (R_max - R_min)/N_poles
        for theta in range(0, N_poles):
            #Begin solovev matching. Uncomment the next two lines to use symmetric solovev. 
            #matching_R[theta] = R_min + (dr*theta)
            #matching_Z[theta] = ((-1.0)**(theta))*(((psi_lcfs - ((1.0/8.0)*(a-c)*(((matching_R[theta])*(matching_R[theta]) - (R_0*R_0))**2.0)))/(0.5*((b*R_0*R_0)+(c*matching_R[theta]*matching_R[theta]))))**0.5)
    
            #uncomment the next two lines for angular definition of the LCFS points. 
            matching_thetas[theta] = (2*theta*pi/N_poles) + 0.001
            matching_R[theta], matching_Z[theta] = self._LCFS(r_lcfs, matching_thetas[theta])

        #Matching routine
        delta_psi = np.zeros(N_poles, float)#difference between the desired psi_lcfs and the psi due to the plasma current. 
        psi_multipole_matrix = np.zeros((N_poles, N_poles), float) #psi_i_j has contribution of pole i at lcfs site j
        plasma_psi_lcfs = np.zeros(N_poles, float) #psi values at the matching points due to just the plasma current. 

        psi_lcfs = self.miller_surface()

        for i in range(0,N_poles):
            plasma_flux_temp = self._plasma_flux(matching_R[i], matching_Z[i], plasma_grid_arr)
            plasma_psi_lcfs[i] = plasma_flux_temp
            delta_psi[i] = psi_lcfs - plasma_psi_lcfs[i]

            for n in range(0,N_poles):
                psi_multipole_matrix[i][n] = self._field_due_to_pole_N(matching_R[i], matching_Z[i], n, A_NM_array) #to skip the non-existent n = 1 pole. 

        return np.linalg.solve(psi_multipole_matrix, delta_psi) #contains the coefficient that quantifies the amount of that

    def _field_due_to_pole_N(self,R_o, Z_o, N, A_NM_array):
        if N == 0:
            return 1
        else:
            psi = 0
            for m in range(0, self.N_poles):
                psi += A_NM_array[N][m]*(R_o**((N+1)-m))*(Z_o**m) #N+1 as we are actually a pole higher due to the missing n = 1 pole. 
            return psi

    def _field_due_to_all_poles(self,R,Z,A_NM_array,multipole_contributions):
        psi_mp = 0.0
        for n in range(0, self.N_poles):
            psi_mp += self._field_due_to_pole_N(R, Z, n, A_NM_array)*multipole_contributions[n] 
        return(psi_mp)

    def compute_psi(self):
        """
        Returns:
            computational_grid (array): psi at all points in [Z][R] sampling
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

        computational_grid = np.zeros((Zdim, Rdim), float) #flux at all grid points

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
                i,j = plasma_grid_arr_inds[pt].astype(int32)

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
