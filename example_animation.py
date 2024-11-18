# Creates animation of our plasma particle

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import cv2
import gs_solver as gs
from simulate import simulate

R_0 = 1 # meters
N_R = 500
N_Z = 500
simulation_width = 0.3  # meters
simulation_height = 0.3 # meters

# Computes psi on our grid

print('Starting Psi calculation',flush=True)

model = gs.RRT_Tokamak(majR=R_0,Rdim=N_R,Zdim=N_Z,sim_width=simulation_width,sim_height=simulation_height,just_plasma=False)
psi = model.compute_psi()

# Simulates particle in our tokamak

print('Starting simulation',flush=True)

xt = simulate(model,x0=[0.0,0.0,1.01],v0=[5.0,0.01,0.00],dt=0.005,tsteps=5000)

# Creates frames of time-series data

print('Creating frames',flush=True)

fig = plt.figure(figsize=(12,6))

ax3d = fig.add_subplot(1,2,1, projection='3d')

ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('z [m]')
ax3d.set_xlim(-R_0-simulation_width/2,R_0+simulation_width/2)
ax3d.set_ylim(-R_0-simulation_width/2,R_0+simulation_width/2)
ax3d.set_zlim(-simulation_height/2,simulation_width/2)
ax3d.plot([-R_0-simulation_width/2,-R_0-simulation_width/2],[0,0],[0,0],linewidth=1,linestyle='solid',color='k') # plots x-axis
ax3d.plot([0,0],[-R_0-simulation_width/2,-R_0-simulation_width/2],[0,0],linewidth=1,linestyle='solid',color='k') # plots y-axis
ax3d.plot([0,0],[0,0],[-0.15,0.15],linewidth=1,linestyle='solid',color='k') # plots z-axis

ax2d = fig.add_subplot(1,2,2)

ax2d.set_xlabel('r [m]')
ax2d.set_ylabel('z [m]')
ax2d.set_xlim(R_0-simulation_width/2,R_0+simulation_width/2)
ax2d.set_ylim(-simulation_height/2,simulation_height/2)

dR = simulation_width/N_R
R_array = np.zeros(N_R)
for i in range(0, N_R):
    R_array[i] = R_0 + (i - (N_R/2.0))*dR

dZ = simulation_height/N_Z      
Z_array = np.zeros(N_Z)
for j in range(0, N_Z):
    Z_array[j] = (j - (N_Z/2.0))*dZ 

min_psi = np.min(psi) # minimum value of psi for contours
max_psi = np.max(psi) # maximum value of psi for contours
ax2d.contour(R_array, Z_array, psi, levels = np.linspace(min_psi, max_psi, 40),linewidths=0.5,linestyles='solid',colors='k') # plots constant psi contours

skip = 10 # plots every skip-th time step for video
out_dir = "../animation-frames"
for frame,pos in enumerate(xt[0::skip]):

    pos_xyz = model.cyl2xyz(pos) # converts cylindrical coordinates to rectangular for plotting
    curr_data_3d = ax3d.scatter(pos_xyz[0],pos_xyz[1],pos_xyz[2],s=80,c='r') # plots 3D data

    curr_data_2d = ax2d.scatter(pos[2],pos[1],s=80,c='r') # plots 2D data

    fig.tight_layout(w_pad=4) # ensures the labels do not overlap
    plt.savefig(os.path.join(out_dir,f'{frame}.png'))
    curr_data_3d.remove() #clears 3D data from frame
    curr_data_2d.remove() #clears 2D data from frame

# Joins time-series frames into a video

print('Joining frames',flush=True)

video_dir = './'
video_name = 'plasma_trajectory.avi'

frames = [img for img in os.listdir(out_dir) if img.endswith(".png")] # collects locations of frames
frames.sort(key=lambda x:int(x.split('.')[0])) # sorts frames in chronological order
frame = cv2.imread(os.path.join(out_dir,frames[0]))
height,width,layers = frame.shape # gets dimensions for output video

fps = 24 # frames-per-second
video = cv2.VideoWriter(os.path.join(video_dir,video_name),0,fps,(width,height))
for frame in frames:
    video.write(cv2.imread(os.path.join(out_dir,frame)))

cv2.destroyAllWindows()
video.release()
