#Example script to use this repository

import numpy as np
import matplotlib.pyplot as plt
from gs_solver import RRT_Tokamak,psi_output
from timeit import default_timer as timer

R_0 = 1 # meters
N_R = 100
N_Z = 100
simulation_width = 0.3  # meters
simulation_height = 0.3 # meters

timing_start = timer()

model = RRT_Tokamak(majR=R_0,Rdim=N_R,Zdim=N_Z,sim_width=simulation_width,sim_height=simulation_height,just_plasma=True)
psi = model.compute_psi() # computes psi just within the plasma on our grid

timing_end = timer()
print(f'Time to complete run: {np.round(timing_end-timing_start,5)} seconds')

psi_output(psi) #writes psi to output file

########## Makes plots ##########

# First determines a center cut through the plasma
center_cut = np.zeros(N_R)

dR = simulation_width/N_R
R_array = np.zeros(N_R)
for i in range(0, N_R):
    R_array[i] = R_0 + (i - (N_R/2.0))*dR

dZ = simulation_height/N_Z      
Z_array = np.zeros(N_Z)
for j in range(0, N_Z):
    Z_array[j] = (j - (N_Z/2.0))*dZ 

center_cut_index = int(N_Z/2)
for i in range(0, N_R):
    center_cut[i] = psi[center_cut_index][i]

plt.plot(R_array, center_cut)

# Next plots constant psi contours on our grid
fig, ax = plt.subplots()

psi_lcfs = model.miller_surface() # determines last closed flux surface (LCFS)
min_psi = np.min(psi) # minimum value of psi for contours
max_psi = np.max(psi) # maximum value of psi for contours

n_contours = 10
dPsi = (max_psi - min_psi)/n_contours
Levels = [psi_lcfs]

for n in range(0, n_contours):
    Levels.append(min_psi+ dPsi*n)
Levels.sort()

CS = ax.contour(R_array, Z_array, psi, levels = np.linspace(min_psi, max_psi, 40))
plt.show()