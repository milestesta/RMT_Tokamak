#Example script to use this repository

import numpy as np
import matplotlib.pyplot as plt
import gs_solver_rebuild as gs
# from simulate import simulate

R_0 = 1 # meters
N_R = 400
N_Z = 400
simulation_width = 0.3  # meters
simulation_height = 0.3 # meters
N_poles = 13

# Computes psi on our grid

print('Starting Psi calculation',flush=True)

model = gs.RRT_Tokamak(majR=R_0,Rdim=N_R,Zdim=N_Z,N_poles=N_poles,sim_width=simulation_width,sim_height=simulation_height,is_solovev=True)
psi,psi_plasma,psi_multipole = model.compute_flux()
gs.psi_output(psi) #writes psi to output file

# Simulates particle in our tokamak

# print('Starting simulation',flush=True)

# xt = simulate(model,x0=[0.0,0.0,1.01],v0=[5.0,0.01,0.00],dt=0.005,tsteps=1200)

########## Makes plots ##########

print('Creating plots',flush=True)

dR = simulation_width/N_R
R_array = np.zeros(N_R)
for i in range(0, N_R):
    R_array[i] = R_0 + (i - (N_R/2.0))*dR

dZ = simulation_height/N_Z      
Z_array = np.zeros(N_Z)
for j in range(0, N_Z):
    Z_array[j] = (j - (N_Z/2.0))*dZ 

# Plots phi component of the simulation
# fig,ax = plt.subplots()
# phi = []
# for angle in range(xt.shape[0]):
#     phi.append(np.mod(xt[angle][0],2*np.pi))

# ax.set_xlabel('time series')
# ax.set_ylabel(r'$\phi$ (radians)')
# ax.plot(np.linspace(0,1,xt.shape[0]),phi)

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4),dpi=100)

# Plots central cut of psi

center_cut = np.zeros(N_R)
center_cut_index = int(N_Z/2)

for i in range(0, N_R):
    center_cut[i] = psi[center_cut_index][i]

ax[0].set_title(r'Center cut along $z=0$')
ax[0].set_ylabel(r'$\psi$')
ax[0].set_xlabel('r [m]')
ax[0].plot(R_array, center_cut)

# Plots constant psi contours on our grid

# psi_lcfs = model.miller_surface() # determines last closed flux surface (LCFS)
min_psi = np.min(psi) # minimum value of psi for contours
max_psi = np.max(psi) # maximum value of psi for contours

ax[1].set_title(r'Constant $\psi$ contours')
ax[1].set_xlabel('r [m]')
ax[1].set_ylabel('z [m]')
CS = ax[1].contour(R_array, Z_array, psi, levels = np.linspace(min_psi, max_psi, 50))

# Plots magnetic field (vector field)

Bz,Br = model.compute_B(psi) # computes the magentic field

skip = 5
ax[2].set_title(r'$\mathbf{B}$ and trajectory')
ax[2].set_xlabel('r [m]')
ax[2].set_ylabel('z [m]')
ax[2].quiver(R_array[0::],Z_array,Br,Bz)

# Plots plasma trajectory

# ax[2].scatter(xt[:,2],xt[:,1],s=0.5,c='r')

fig.tight_layout()
plt.show()
