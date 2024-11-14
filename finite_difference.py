import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.constants

def GS_finite_diff(psi, Rmin, Rmax, Zmin, Zmax, Rdim, Zdim):
    """This function uses finite differences to make sure that the psi value that the main code produces
    does actually solve the GS equation for our inputted plasma current. 

    Args:
        psi_grid (float array [Z][R]): contains the GS solution from compute_psi.
        Rmin (float): The leftmost R point of the simulation box. 
        Rmax (float): The rightmost R point of the simulation box. 
        Zmin (float): The lowest Z point of the simulation box. 
        Zmax (float): The highest Z point of the simulation box. 
        Rdim (int): The number of R points.
        Zdim (int): The number of Z points.
    
    Returns:
        An array containing the finite differences version of the GS operator applied to psi_grid. Should be equal to -mu_0 R j_plasma.   
    """
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