import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.constants

def solovev(Rmin, Rmax, Zmin, Zmax, Rdim, Zdim):
    R_array = np.linspace(Rmin, Rmax, Rdim)
    Z_array = np.linspace(Zmin, Zmax, Zdim)
    solovev_grid = np.zeros((Zdim, Rdim), float)
    a = 1.2
    b = -1.0
    c_0 = 1.1
    majR = 1.0
    for k in range(0, Zdim):
        for i in range(0, Rdim):
            R = R_array[i]
            Z = Z_array[k]
            zeta = ((R*R) - (majR*majR))/(2*majR)
            term1 = 0.5*(b+c_0)*majR*majR*Z*Z
            term2 = c_0*majR*zeta*Z*Z
            term3 = 0.5*(a-c_0)*majR*majR*zeta*zeta
            psi_tot = term1+term2+term3
            solovev_grid[k, i] = psi_tot
    return(solovev_grid)

            
def plasma_current(Rmin, Rmax, Zmin, Zmax, Rdim, Zdim):
    R_array = np.linspace(Rmin, Rmax, Rdim)
    Z_array = np.linspace(Zmin, Zmax, Zdim)
    current_grid = np.zeros((Zdim, Rdim), float)
    a = 1.2
    b = -1.0
    majR = 1.0
    for k in range(0, Zdim):
        for i in range(0, Rdim):
            current_grid[k, i] = -R_array[i]*(((-a)*R_array[i]) + ((-b)*majR*majR/R_array[i]))
    
    return(current_grid)
def GS_finite_diff(psi, Rmin, Rmax, Zmin, Zmax, Rdim, Zdim):
    """_summary_

    Args:
        psi (_type_): _description_
        Rmin (_type_): _description_
        Rmax (_type_): _description_
        Zmin (_type_): _description_
        Zmax (_type_): _description_
        Rdim (_type_): _description_
        Zdim (_type_): _description_
    """
    R_array = np.linspace(Rmin, Rmax, Rdim)
    final_grid = np.zeros((Zdim, Rdim), float)
    dR = (Rmax - Rmin)/Rdim
    dZ = (Zmax - Zmin)/Zdim
    print("Dr = " + str(dR))
    for k in range(1, Zdim-1):
        for i in range(1, Rdim-1):
            d2R_psi = (psi[k, i+1] + psi[k, i-1] - (2*psi[k, i]))/(dZ*dZ)
            d2Z_psi = (psi[k+1, i] + psi[k-1, i] - (2*psi[k, i]))/(dZ*dZ)
            dR_psi_1_R = (psi[k, i+1] - psi[k, i-1])/(2*R_array[i]*dR)
            final_grid[k, i] = d2R_psi - dR_psi_1_R + d2Z_psi
    return(final_grid)   

def numerical_derivative(psi, Rmin, Rmax, Zmin, Zmax, Rdim, Zdim):
    R_array = np.linspace(Rmin, Rmax, Rdim)
    final_grid = np.zeros((Zdim, Rdim), float)
    dR = (Rmax - Rmin)/Rdim
    dZ = (Zmax - Zmin)/Zdim
    print("Dr = " + str(dR))
    for k in range(1, Zdim-1):
        for i in range(1, Rdim-1):
            d2R_psi = (psi[k, i+1] + psi[k, i-1] - (2*psi[k, i]))/(dR*dR)
            d2Z_psi = (psi[k+1, i] + psi[k-1, i] - (2*psi[k, i]))/(dZ*dZ)
            dR_psi= (psi[k, i+1] - psi[k, i-1])/(2*R_array[i]*dR)
            final_grid[k, i] = -dR_psi + d2R_psi + d2Z_psi
    return(final_grid)
NR = 200
NZ = 200
psi = np.zeros((NZ, NR), float)
raw_data = []
with open("miller_gs_solution", "r") as f:
    for line in f:
        line = line.split()
        for point in line:
            raw_data.append(float(point))
row = []

for k in range(0, NZ):
    for i in range(0, NR):
        psi[k, i] = raw_data[(k*NZ)+ i]


gs_psi = numerical_derivative(psi, 0.7, 1.3, -0.3, 0.3, 100, 100)
current = plasma_current(0.7, 1.3, -0.3, 0.3, 100, 100)
solovev_psi = solovev(0.7, 1.3, -0.3, 0.3, 100, 100)
solovev_gs = numerical_derivative(solovev_psi, 0.7, 1.3, -0.3, 0.3, 100, 100)
R_array = np.linspace(0.7, 1.3, 100)
Z_array = np.linspace(-0.3, 0.3, 100)
solovev_check = (solovev_gs[1:-1, 1:-1] - current[1:-1, 1:-1])/current[1:-1, 1:-1]
plt.imshow(psi)
plt.figure()
plt.imshow(gs_psi)
# plt.figure()
# plt.imshow(current)
# plt.title("solovev current")
# plt.figure()
# plt.imshow(solovev_psi)
# plt.title("solovev flux")
# plt.figure()
# plt.imshow(solovev_gs)
# plt.title("finite difference of solovev")
# plt.figure()
# plt.imshow(solovev_check)
# plt.title("difference between current and solovev fd")
# fig_check, ax_check = plt.subplots()
# CS_check = ax_check.contour(R_array[1:-1], Z_array[1:-1], solovev_psi[1:-1, 1:-1], 20)
plt.show()



# A = 1
# N_check = 500
# R_array = np.linspace(1, 2, N_check)
# Z_array = np.linspace(-1, 1, N_check)
# dR = 1/N_check
# dZ = 1/N_check
# original_function = np.zeros((N_check, N_check), float)
# analytic_solution = np.zeros((N_check, N_check), float)
# difference = np.zeros((N_check, N_check), float)

# for k in range(0, N_check):
#     for i in range(0, N_check):
#         analytic_solution[k, i] = 2*A*Z_array[k]*(2 + (3*R_array[i]*R_array[i]))
#         original_function[k, i] = A*Z_array[k]*Z_array[k]*Z_array[k]*R_array[i]*R_array[i]

# numerical_solution = numerical_derivative(original_function, 1, 2, -1, 1, N_check, N_check)
# # difference = numerical_solution[1:-1, 1:-1]-analytic_solution[1:-1, 1:-1]
# for k in range(1, N_check-2):
#     for i in range(1, N_check-2):
#         difference[k, i] = np.abs(analytic_solution[k, i] - numerical_solution[k, i])/np.abs(analytic_solution[k, i])
# difference = np.abs(analytic_solution - numerical_solution)
# difference = difference[1:-1, 1:-1]
# plt.imshow(original_function)
# plt.title("original_function")
# plt.figure()
# plt.imshow(analytic_solution)
# plt.title("analytic_solution")
# plt.figure()
# plt.imshow(numerical_solution)
# plt.title("numerical_solution")
# plt.figure()
# plt.imshow(difference)
# plt.title("difference")
# plt.show()