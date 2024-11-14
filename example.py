#Example script to use this repository

import numpy as np
import matplotlib.pyplot as plt
import gs_solver as gs
from simulate import simulate

R_0 = 1 # meters
N_R = 100
N_Z = 100
simulation_width = 0.3  # meters
simulation_height = 0.3 # meters

# Computes psi on our grid

print('Starting Psi calculation',flush=True)

model = gs.RRT_Tokamak(majR=R_0,Rdim=N_R,Zdim=N_Z,sim_width=simulation_width,sim_height=simulation_height,just_plasma=False)
psi = model.compute_psi()
gs.psi_output(psi) #writes psi to output file

print('Psi calculation complete',flush=True)

# Simulates particle in our tokamak

print('Starting simulation',flush=True)

xt = simulate(model,x0=[0.0,0.0,1.01],v0=[0.0,0.01,0.01],dt=0.001,tsteps=60000)

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

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4),dpi=100)

# Plots central cut of psi

ax[0].set_title(r'Center cut along $z=0$')
ax[0].set_ylabel(r'$\psi$')
ax[0].set_xlabel('r [m]')

center_cut = np.zeros(N_R)
center_cut_index = int(N_Z/2)

for i in range(0, N_R):
    center_cut[i] = psi[center_cut_index][i]

ax[0].plot(R_array, center_cut)

# Plots constant psi contours on our grid

psi_lcfs = model.miller_surface() # determines last closed flux surface (LCFS)
min_psi = np.min(psi) # minimum value of psi for contours
max_psi = np.max(psi) # maximum value of psi for contours

ax[1].set_title(r'Constant $\psi$ contours')
ax[1].set_xlabel('r [m]')
ax[1].set_ylabel('z [m]')
CS = ax[1].contour(R_array, Z_array, psi, levels = np.linspace(min_psi, max_psi, 40))

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
