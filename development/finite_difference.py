import numpy as np
import matplotlib.pyplot as plt
def finite_difference(psi_file_name, current_file_name, dR, dZ, majR, preamble_count = 7):
    """This code runs a finite difference version of the Grad-Shafranov equation on the psi made by gs_solver. You feed in the 
    name of the current and psi files specified in the gs_solver output functions, as well as the grid spacing and major radius of the 
    tokamak. If the output file has not been altered, leave the preamble_count as 7. 

    Args:
        psi_file_name (string): Name given to the output of the gs_solver class in the function psi_output. 
        current_file_name (string): Name given to the output of the gs_solver class in the function current_output. 
        dR (float): R grid spacing used in gs_solver
        dZ (float): Z grid spacing used in gs_solver
        majR (float): major radius of the tokamak specified in gs_solver. 
        preamble_count (int, optional): number of lines of preamble in the outputs of psi_output and current_output
    
    Returns:
        fd_grid (float, array):
        current_grid (float, array): Returns the RHS of the GS equation, should be equal to fd_grid. 
        difference (float, array): difference between fd_grid and current grid. Should be zero. 
    """
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
    
    #scaling the current matrix by -R to get the RHS of the GS. 
    for k in range(0, Zdim):
        for i in range(0, Rdim):
            current_matrix[k][i] = -(current_matrix[k][i])*(R_array[i])
    # current_matrix = current_matrix[1:-1][1:-1] #cutting off the edges so we can compare to the finite difference grid. 
    
    fd_grid = np.zeros((Zdim, Rdim), float)
    for k in range(1, Zdim-1):
        for i in range(1, Rdim-1):
            d2R_psi = (psi_matrix[k, i+1] + psi_matrix[k, i-1] - (2*psi_matrix[k, i]))/(dR*dR)
            d2Z_psi = (psi_matrix[k+1, i] + psi_matrix[k-1, i] - (2*psi_matrix[k, i]))/(dZ*dZ)
            dR_psi= (psi_matrix[k, i+1] - psi_matrix[k, i-1])/(2*R_array[i]*dR)
            fd_grid[k, i] = -dR_psi + d2R_psi + d2Z_psi

    difference = fd_grid - current_matrix #difference between the solution and the current. 

    
    return(fd_grid, current_matrix, difference)
