import numpy as np
import matplotlib.pyplot as plt
def finite_difference(psi_file_name, current_file_name, dR, dZ, majR, preamble_count = 7):
    raw_data = []
    line_counter = 0
    #initial import. 
    with open(psi_file_name, "r") as f:
        for line in f:
            line = line.split()
            raw_data.append(line)

    Rdim = int(raw_data[3][-1])
    Zdim = int(raw_data[4][-1])
    N_data_points = Rdim*Zdim

    
    unsplit_psi_data = np.asarray(raw_data[preamble_count:(preamble_count + N_data_points)])
    psi_r_index = unsplit_psi_data[:, 0].astype(int)
    psi_z_index = unsplit_psi_data[:, 1].astype(int)
    psi_vals = unsplit_psi_data[:, 2].astype(float)


    Rmin = majR - ((Rdim/2)*dR)
    Rmax = majR + ((Rdim/2)*dR)
    Zmin = -((Zdim/2)*dZ)
    Zmax = ((Zdim/2)*dZ)
    
            
    #Doing the same for the current grid. 
    raw_data = []
    line_counter = 0
    #initial import. 
    with open(current_file_name, "r") as f:
        for line in f:
            line = line.split()
            raw_data.append(line)

    Rdim = int(raw_data[3][-1])
    Zdim = int(raw_data[4][-1])
    N_data_points = Rdim*Zdim

    
    unsplit_current_data = np.asarray(raw_data[preamble_count:(preamble_count + N_data_points)])
    current_r_index = unsplit_current_data[:, 0].astype(int)
    current_z_index = unsplit_current_data[:, 1].astype(int)
    current_vals = unsplit_current_data[:, 2].astype(float)
    
    #All data imported, now doing the finite differences and comparing to the current. 
    R_array = np.linspace(Rmin, Rmax, Rdim)
    Z_array = np.linspace(Zmin, Zmax, Zdim)
    psi_matrix = np.zeros((Zdim, Rdim), float)
    current_matrix = np.zeros((Zdim, Rdim), float)
    for k in range(0, Zdim):
        for i in range(0, Rdim):
            psi_matrix[k][i] = psi_vals[i+(k*(Zdim))]
            current_matrix[k][i] = current_vals[i+(k*(Zdim))]
    
    
    fd_grid = np.zeros((Zdim, Rdim), float)
    for k in range(1, Zdim-1):
        for i in range(1, Rdim-1):
            d2R_psi = (psi_matrix[k, i+1] + psi_matrix[k, i-1] - (2*psi_matrix[k, i]))/(dR*dR)
            d2Z_psi = (psi_matrix[k+1, i] + psi_matrix[k-1, i] - (2*psi_matrix[k, i]))/(dZ*dZ)
            dR_psi= (psi_matrix[k, i+1] - psi_matrix[k, i-1])/(2*R_array[i]*dR)
            fd_grid[k, i] = -dR_psi + d2R_psi + d2Z_psi
    plt.imshow(fd_grid)
    plt.figure()
    plt.imshow(current_matrix)
    plt.show()
    return(fd_grid)
finite_difference("psi_grid.dat", "current_grid.dat", 0.2/50, 0.2/50, 1.0)