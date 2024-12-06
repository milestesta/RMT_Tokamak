import numpy as np
import matplotlib.pyplot as plt
import gs_solver as gs
from simulate import simulate
from finite_difference import finite_difference


###Specify the Dimensions/Parameters of your Tokamak.

## Machine and Grid Geometry: 

R_0 = 1 # Major Radius (meters)
N_R = 100 # Number of R grid points
N_Z = 100 # Number of Z grid points
simulation_width = 0.2  # Width of the simulation box (meters)
simulation_height = 0.2 # Height of the simulation box (meters)

## LCFS Specifications: 
is_solovev = False #Boolean flag for whether a Solovev solution is desired. If False, uses Miller Geometry. 
minor_radius = 0.1 #The characteristic size of the LCFS. (meters) 
elongation = 0.8 #Vertical elongation of the LCFS. 
triangularity = 0.7 #Triangularity of the LCFS, must be less than 1. 
a = 1.2 #First Solov'ev Parameter
b = -1.0 #Second Solov'ev Parameter
c = 1.1 #Third Solov'ev Parameter
N_poles = 12 #Number of poles in the multipole expansion of the coil contribution to psi. 

## Simulation parameters:

initial_position = [0.0,0.0,1.01] # Initial position of the test particle (meters)
initial_velocity = [5.0,0.01,0.00] # Initial velocity of the test particle (meters/second)
time_step = 0.001 # Simulation time step (seconds)
n_time_steps = 120000 # Number of time steps to run the simulation over. 

### Initialising the model:
print('Starting Psi calculation',flush=True)
model = gs.RRT_Tokamak(majR=R_0,Rdim=N_R,Zdim=N_Z, tri=triangularity, elo=elongation, sim_width=simulation_width,sim_height=simulation_height, N_poles = N_poles, just_plasma=False, is_solovev=False)
psi = model.compute_psi()
print('Psi calculation complete',flush=True)

## Writing the output to standard I/O. 
gs.psi_output(psi) #writes psi to output file
gs.current_output(model.plasma_current_grid()) #writes the current to output file. 

### Running the Finite Differences Verification (STILL IN DEVELOPMENT, PROBABLY INCORRECT)

fd_grid, current_grid, difference_grid = finite_difference("psi_grid.dat", "current_grid.dat", simulation_width/N_R, simulation_height/N_Z, R_0)


### Running the particle simulation. 

print('Starting simulation',flush=True)
xt = simulate(model,x0=initial_position,v0=initial_velocity,dt=time_step,tsteps=n_time_steps)
print('Simulation complete',flush=True)

########## Makes plots ##########

dR = simulation_width/N_R
R_array = np.zeros(N_R)
for i in range(0, N_R):
    R_array[i] = R_0 + (i - (N_R/2.0))*dR

dZ = simulation_height/N_Z      
Z_array = np.zeros(N_Z)
for j in range(0, N_Z):
    Z_array[j] = (j - (N_Z/2.0))*dZ 

# Plotting the Finite Difference Results


fig_fd,ax_fd = plt.subplots(nrows=1,ncols=3,figsize=(12,4),dpi=100)
ax_fd[0].set_title(r'$-Rj_{\phi}$')
ax_fd[0].set_xlabel('R [m]')
ax_fd[0].set_ylabel('Z [m]')
ax_fd[0].imshow(current_grid, extent=[np.min(R_array), np.max(R_array), np.min(Z_array), np.max(Z_array)])

ax_fd[1].set_title(r'$\nabla^*\psi_\text{F.D}$')
ax_fd[1].set_xlabel('R [m]')
ax_fd[1].set_ylabel('Z [m]')
ax_fd[1].imshow(fd_grid, extent=[np.min(R_array), np.max(R_array), np.min(Z_array), np.max(Z_array)])

ax_fd[2].set_title(r'Difference')
ax_fd[2].set_xlabel('R [m]')
ax_fd[2].set_ylabel('Z [m]')
ax_fd[2].imshow(difference_grid, extent=[np.min(R_array), np.max(R_array), np.min(Z_array), np.max(Z_array)])

fig_fd.tight_layout()

# Plots phi component of the simulation

fig,ax = plt.subplots()
phi = []
for angle in range(xt.shape[0]):
    phi.append(np.mod(xt[angle][0],2*np.pi))

ax.set_xlabel('time series')
ax.set_ylabel(r'$\phi$ (radians)')
ax.plot(np.linspace(0,1,xt.shape[0]),phi)

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4),dpi=100)

# Plots central cut of psi

center_cut = np.zeros(N_R)
center_cut_index = int(N_Z/2)
center_cut_lcfs = np.ones(N_R)*model._psi_on_LCFS()

for i in range(0, N_R):
    center_cut[i] = psi[center_cut_index][i]

ax[0].set_title(r'Center cut along $z=0$')
ax[0].set_ylabel(r'$\psi$')
ax[0].set_xlabel('r [m]')
ax[0].plot(R_array, center_cut)
ax[0].plot(R_array, center_cut_lcfs)

# Plots constant psi contours on our grid

psi_lcfs = model._psi_on_LCFS() # determines last closed flux surface (LCFS)
min_psi = np.min(psi) # minimum value of psi for contours
max_psi = np.max(psi) # maximum value of psi for contours

ax[1].set_title(r'Constant $\psi$ contours')
ax[1].set_xlabel('r [m]')
ax[1].set_ylabel('z [m]')
CS = ax[1].contour(R_array, Z_array, psi, levels = np.linspace(min_psi, 2*psi_lcfs, 40))
CS = ax[1].contour(R_array, Z_array, psi, levels=[psi_lcfs])

# Plots magnetic field (vector field)

Bz,Br = model.compute_B(psi) # computes the magentic field

ax[2].set_title(r'$\mathbf{B}$ and trajectory')
ax[2].set_xlabel('r [m]')
ax[2].set_ylabel('z [m]')
ax[2].quiver(R_array,Z_array,Br,Bz)

# Plots plasma trajectory

ax[2].scatter(xt[:,2],xt[:,1],s=0.5,c='r')

fig.tight_layout()
plt.show()
