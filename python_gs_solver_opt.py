import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.constants
from numba import njit
from timeit import default_timer as timer

########## List of edits ##########
# numpy -> np (this is standard)
# matplotlib.pyplot -> plt (this is standard)
# pi = np.pi (more convenient imo)
# used f-strings for outputting raw data
# slightly cleaned up A_MN array creation
# If LCFS is always one piece, we could have a faster way to check with breadth-first-search
# Put plasma_grid as input to plasma_flux function (numba)
# Stored complete elliptic integrals in arrays instead of calling sicpy.special (numba)
# Fully implemented numba (FAST)
# added comments throughout (good to double-check for correctness)
###################################

#Before optimization attempts: ~ 100 seconds
#After optimization attemps:   ~   2 seconds
timing_start = timer()

#Shaping parameters: MKS Units ____-----_____------______-----_______
pi = np.pi
a = 0.1          #minor radius, in meters. 
R_0 = 1.0        #major radius, in meters. 
psi_lcfs = 1.0   #psi on the lcfs. 
r_lcfs = a
aspect = a/R_0
triangularity = 0.8
elongation = 1.0
mu = scipy.constants.mu_0
N_shaping = 3 #number of shaping terms in the GJW formulation of the lcfs. TODO: Implement that! 

#solovev constants: (R_0 is defined above) using notation from the fitzpatric Xu paper. 
#This will recreate Figures 3,5: Tao Xu and Richard Fitzpatrick 2019 Nucl. Fusion 59 064002
c =  1.1
b = -1.0
a =  1.2
psi_lcfs = (a-c)*(b+c)*(b+c)*(R_0**4.0)/(8.0*c*c) #just using the solovev lcfs value for now (Eqn 11)
print(f"Psi at matching surface: {psi_lcfs}")
#solovev R min and max: 
#R_min = ((R_0**2.0) - (((8.0*psi_lcfs)/(a-c))**0.5))**0.5
#R_max = ((R_0**2.0) + (((8.0*psi_lcfs)/(a-c))**0.5))**0.5

#miller R min and max:
R_min = R_0 - (aspect*R_0)
R_max = R_0 + (aspect*R_0)

print(f"leftmost plasma point:  {R_min}")
print(f"rightmost plasma point: {R_max}")
#simulation parameters (N_R,N_Z must be even):
N_R = 100 #number of points sampling R-coordinate
N_Z = 100 #number of points sampling z-coordinate
simulation_width = 0.3
simulation_height = 0.3
dR = simulation_width/N_R
dZ = simulation_height/N_Z 

#first of all I calculate the functional form of the mulitpole expansion in (R, Z) coordinates up to the desired order N_poles, where I will
#later match to N points on the lcfs.

#Order to which you expand the multipole aproximation to the actual contribution from the coils. 
N_poles = 10
A_NM_array = np.zeros((N_poles+1, N_poles+1), float) #plus one to make sure we actually get the number of multipoles that we want once we
#subtract off the n = 1 multipole. 

#This is the symmetric case ()
for n in range(0, N_poles+1): #we don't need to declare things =0 since they are already zero
    if (n%2 == 0):
        A_NM_array[n][0] = 1
    else:
        A_NM_array[n][1] = 1

A_NM_array[1][1] = 0
A_NM_array[4][2] = -4 #F4 from Eqn 15 in following paper

#This uses notation from Eqn 17 in: https://doi.org/10.1016/0021-9991(86)90041-0
for n in range(4, N_poles+1):
    for m in range(2, N_poles+1): #Python doesn't have integer division problems
        A_NM_array[n][m] = -1*A_NM_array[n][m-2]*(((n+2-m)*(n-m))/(m*(m-1))) #Eqn 34 (why index -2)(missing factor of 2 in denominator?)

A_NM_array = np.delete(A_NM_array, 1, axis = 0)

#At this point we have the array of multiple coefficients (sans the n = 1 null row). A_NM_array[n][m] multiplies R^(n-m) X Z^m, summing over m to give the nth multipole

#The simulation box is a vacuum with embedded coils producing a poloidal field (represented by the multipoles) that confines a plasma that is
#entirely contained witin the LCFS. For the Shafranov operator GS, we are solving GS(psi) = -mu_0*(R^2)*(dP/dPsi) - g*(dg/dPsi), for P being
#the plasma pressure as a function of flux label, and g(psi) being the toroidal component of the magnetic field.  

#The plasma contribution to the poloidal field will be calculated using Green's functions on the R,Z grid. We then choose the contribution of
#the coil multipole moments by demanding the Psi = Psi_lcfs on the LCFS. So we are basically choosing the pressure, toroidal field and shape of the LCFS, 
#then seeing if we can use coils to crush the plasma into that shape. 

#When calculating the flux due to the plasma, I want to eventually use Von Hagenow's method, but for now I'll just do the double sum over 
#R_p and Z_p, as I am not smart enough. 

elliptic_int_1 = np.zeros(10000)# #will store values for complete elliptic integral of the first kind.  
elliptic_int_2 = np.zeros(10000)# #will store values for complete elliptic integral of the second kind. 
for ind,x in enumerate(np.linspace(0,1,10000)):
    if ind == 10000-1: #occurs for input=1.  We do it this way to have a spacing of about 0.001 between x points
        break #avoids asymptote at ellipk(x=1)
    elliptic_int_1[ind] = scipy.special.ellipk(x)
    elliptic_int_2[ind] = scipy.special.ellipe(x)

@njit
def greens_function(R_p, Z_p, R, Z): #calculates the value of the green's function based on the plasma at (R_p, Z_p), as viewed at (R, Z)
    k = ((4.0*R_p*R)/(((R + R_p)*(R + R_p)) + ((Z-Z_p)*(Z - Z_p))))**(0.5)

    index = int(np.round(k,4)*10000)
    if index in [9999,10000]: #avoids asymptote (can occur because of our rounding)
        index = 9998

    K = elliptic_int_1[index]
    E = elliptic_int_2[index]
    return((1/(2*pi))*(((R*R_p)**(0.5))/k)*(((2-(k*k))*K) - (2*E)))

#If this is 
def is_plasma(R, Z): #says whether the input point is within the LCFS, and thus whether it contains any plasma.
    #solovev method: 
    """ 
    if (R <= R_max) and (R >= R_min):
        Z_2_max = (((psi_lcfs - ((1.0/8.0)*(a-c)*(((R)*(R) - (R_0*R_0))**2.0)))/(0.5*((b*R_0*R_0)+(c*R*R)))))
        Z_2 = Z*Z
        if Z_2 <= Z_2_max:
            return(True)
        else:
            return(False)
    else:
        return(False)
    """
    #miller shape plasma method: 
    if (R_min <= R) and (R <= R_max):
        theta_point = np.arctan(Z/(R-R_0 + 0.001))
        Z_max = aspect*elongation*np.sin(theta_point)
        if abs(Z) <= abs(Z_max):
            return(True)
        else: 
            return(False)
    else:
        return(False)

@njit
def pressure_term(R_p, Z_p): #the pressure term in the GS equaion
    #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
    return(-a)#up down symmetric solovev tororidal current. 

@njit
def toroidal_field_term(R_p, Z_p): #the toroidal field term in the GS. 
    #return(np.exp((((((R_p-R_0)*(R_p-R_0)) + (Z_p*Z_p))/(1000)))*(-1.0)))
    return(-b*R_p*R_0) #up down symmetric solovev  toroidal field 

@njit
def plasma_current(R_p, Z_p):
    return((1.0)*(pressure_term(R_p, Z_p) + toroidal_field_term(R_p, Z_p)))
    #return(-c_1/mu)

def solovev_psi(R_o, Z_o): #up down symmetric solovev  equilibrium. 
    return((0.5*((b*R_0*R_0) + c*R_o*R_o)*Z_o*Z_o) + ((1/8)*(a - c)*((R_o*R_o) - (R_0*R_0))*((R_o*R_o) - (R_0*R_0))))


def omega(theta): #The conversion between the altered polidal angle omega and the normal poloidal angle theta. 
    return(theta)

@njit
def LCFS(r, theta): #calculates a point on the LFCS.
    
    #GJW Shape: TODO: Implement this!!!!
    #R_lcfs = R_0 - r*np.cos(omega(theta))#add more shaping terms as desired
    #Z_lfcs = r*np.sin(theta)
    
    #Miller Shape:
    R_lcfs = R_0*(1.0 + aspect*np.cos(theta + np.sin(theta)*np.arcsin(triangularity))) #add more shaping terms as desired
    Z_lfcs = aspect*elongation*np.sin(theta)
    
    #Solovev Shape:
    return(R_lcfs, Z_lfcs)

@njit
def plasma_flux(R_o, Z_o, plasma_grid_list): #calculates the flux at (R, Z) due to the plasma current at each location listed in plasma_grid
    #n_points = len(plasma_grid_list)
    n_points = plasma_grid_list.shape[0]
    flux = 0.0
    for n in range(0, (n_points)):
        R_p = plasma_grid_list[n][0]
        Z_p = plasma_grid_list[n][1]
        if ((R_o!=R_p) and (Z_o!=Z_p)):#to avoid self action. 
            flux += dR*dZ*greens_function(R_p, Z_p, R_o, Z_o)*plasma_current(R_p, Z_p)
    return(flux)

@njit
def field_due_to_pole_N(R_o, Z_o, N):
    if N == 0:
        return(1.0)
    else:
        psi = 0
        for m in range(0, N_poles):
            psi += A_NM_array[N][m]*(R_o**((N+1)-m))*(Z_o**m) #N+1 as we are actually a pole higher due to the missing n = 1 pole. 
        return(psi)


#defining the points in the plasma interior.
plasma_grid = [] #spatial location of all points inside the LCFS, will be a list of locations. 
for i in range(0, N_R):
    for j in range(0, N_Z):
        R = R_0 + (i - (N_R/2.0))*dR
        Z = (j - (N_Z/2.0))*dZ
        if is_plasma(R, Z) == True:
            plasma_grid.append([R, Z])
       
#We need to put our plasma points into an array for numba to work
plasma_grid_arr = np.zeros((len(plasma_grid),2))
for ind,pos in enumerate(plasma_grid):
    plasma_grid_arr[ind][0] = pos[0]
    plasma_grid_arr[ind][1] = pos[1]

print(f"Number of plasma points: {len(plasma_grid)}")
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("")
#Matching on the LCFS:
matching_thetas = np.zeros(N_poles, float)
matching_R = np.zeros(N_poles, float) #R values of the matching points.  
matching_Z = np.zeros(N_poles, float) #Z values of the matching points. 
dr = (R_max - R_min)/N_poles
for theta in range(0, N_poles):
    #Begin solovev matching. Uncomment the next two lines to use symmetric solovev. 
    #matching_R[theta] = R_min + (dr*theta)
    #matching_Z[theta] = ((-1.0)**(theta))*(((psi_lcfs - ((1.0/8.0)*(a-c)*(((matching_R[theta])*(matching_R[theta]) - (R_0*R_0))**2.0)))/(0.5*((b*R_0*R_0)+(c*matching_R[theta]*matching_R[theta]))))**0.5)
    
    #uncomment the next two lines for angular definition of the LCFS points. 
    matching_thetas[theta] = (2*theta*pi/N_poles) + 0.001
    matching_R[theta], matching_Z[theta] = LCFS(r_lcfs, matching_thetas[theta])
plasma_psi_lcfs = [] #psi values at the matching points due to just the plasma current. 

#Matching routine
delta_psi = np.zeros(N_poles, float)#difference between the desired psi_lcfs and the psi due to the plasma current. 
psi_multipole_matrix = np.zeros((N_poles, N_poles), float) #psi_i_j has contribution of pole i at lcfs site j
for i in range(0, N_poles):
    print("_-_-_-")
    plasma_flux_temp = plasma_flux(matching_R[i], matching_Z[i], plasma_grid_arr)
    plasma_psi_lcfs.append(plasma_flux_temp)
    print(f"what was calculated: {plasma_flux_temp}")
    print(f"what was stored: {plasma_psi_lcfs}")
    print(f"lcfs plasma psi value: {plasma_psi_lcfs[i]}")
    print(f"coordinates: {matching_R[i]}, {matching_Z[i]}")
    print("_-_-_-")
    print("_-_-_-")
    delta_psi[i] = psi_lcfs - plasma_psi_lcfs[i]
    #psi_multipole_matrix[i][0] = 1/psi_lcfs 
    for n in range(0, N_poles):
        psi_multipole_matrix[i][n] = field_due_to_pole_N(matching_R[i], matching_Z[i], n) #to skip the non-existent n = 1 pole. 

print(plasma_psi_lcfs)
plasma_psi_lcfs_check = np.zeros(N_poles, float)


multipole_contributions = np.linalg.solve(psi_multipole_matrix, delta_psi) #contains the coefficient that quantifies the amount of that
#multipole that is required to crush the plasma into the lcfs. 
print("Contribution from each multipole: ")
print(multipole_contributions)
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("")

#testing to check that that worked. 
lcfs_check = []
temp_point = 0
for i in range (0, N_poles):
    for j in range (0, N_poles):
        temp_point += multipole_contributions[j]*psi_multipole_matrix[i][j]
    lcfs_check.append(temp_point + plasma_psi_lcfs[i] - psi_lcfs)
    temp_point = 0

print("Checking to make sure the lcfs matching was done correctly: ")
print("Should be all close to zero if done right: ")
print(lcfs_check)
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("")

@njit
def field_due_to_all_poles(R, Z):
    psi_mp = 0.0
    for n in range(0, N_poles):
        psi_mp += field_due_to_pole_N(R, Z, n)*multipole_contributions[n] 
    return(psi_mp)


#In theory, now I just need to calculate the field across the grid using the plasma and multipole contributions. 
R_array = np.zeros(N_R)
Z_array = np.zeros(N_Z)
for i in range(0, N_R):
    R_array[i] = R_0 + (i - (N_R/2.0))*dR

for j in range(0, N_Z):
    Z_array[j] = (j - (N_Z/2.0))*dZ    


min_psi = 0.0
max_psi = 0.0
center_cut = np.zeros(N_R)

computational_grid = np.zeros((N_Z, N_R), float) #flux at all grid points
for i in range(0, N_R):
    for j in range(0, N_Z):
        R = R_0 + (i - (N_R/2.0))*dR
        Z = (j - (N_Z/2.0))*dZ
        computational_grid[j][i] = field_due_to_all_poles(R, Z) + plasma_flux(R, Z, plasma_grid_arr)
        if computational_grid[j][i] < min_psi:
            min_psi = computational_grid[j][i]
        elif computational_grid[j][i] > max_psi:
            max_psi = computational_grid[j][i]

########## Makes plots ##########
center_cut_index = int(N_Z/2)
for i in range(0, N_R):
    center_cut[i] = computational_grid[center_cut_index][i]
    #print(center_cut[i])
print(f"min psi: {min_psi}")
print(f"max_psi: {max_psi}")
#print(computational_grid)
print(f"psi lcfs: {psi_lcfs}")
n_contours = 10
dPsi = (max_psi - min_psi)/n_contours
plt.plot(R_array, center_cut)

fig, ax = plt.subplots()
Levels = [psi_lcfs]
for n in range(0, n_contours):
    Levels.append(min_psi+ dPsi*n)
Levels.sort()
CS = ax.contour(R_array, Z_array, computational_grid, levels = np.linspace(min_psi, max_psi, 40))
#ax.clabel(CS, inline=True, fontsize=4
plt.show()

timing_end = timer()
print(f'Time to complete run: {timing_end-timing_start} seconds')