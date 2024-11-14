# Simulates plasma particle in a tokamak

import numpy as np
from numba import njit

@njit
def simulate(model,x0,v0,dt,tsteps):
    """
    Simulates particle movement through discretizing Newton's second law

    Parameters:
        model (class): initialized class for RRT_tokamak
        x0 (float,3): initial position [phi][z][r]
        v0 (float): initial velocity [phi][z][r]
        dt (float): time step
        tsteps (int): number of time steps

    Returns
        xt (float,array): array containing position data for particle
    """

    # Quick error check
    if len(x0) != 3:
        raise ValueError('Error: x0 does not have three components')
    if len(v0) != 3:
        raise ValueError('Error: v0 does not have three components')

    # Defines a few constants/arrays to use later
    dR = model.sim_width/model.Rdim
    R_array = np.zeros(model.Rdim)
    for i in range(0, model.Rdim):
        R_array[i] = model.majR + (i - (model.Rdim/2.0))*dR

    dZ = model.sim_height/model.Zdim
    Z_array = np.zeros(model.Zdim)
    for j in range(0, model.Zdim):
        Z_array[j] = (j - (model.Zdim/2.0))*dZ 

    # Calculates the magnetic field at each point
    psi = model.compute_psi()
    Bz,Br = model.compute_B(psi)
    Bphi = 0.0007 # about the average magnitude for each component

    # Initializes output vector
    xt = np.zeros((tsteps+1,3)) # +1 to not count start as a timestep
    xt[0] = x0

    # Iterates over all time steps
    vcurr = np.array(v0)
    for ii in range(tsteps):

        # Converts position at current timstep to indices
        # Finds the closest position, then returns its indices
        rind = np.argmin(np.abs(R_array-xt[ii][2]))
        zind = np.argmin(np.abs(Z_array-xt[ii][1]))

        # Acceleration for current timestep [phi][z][r]
        # Lorentz force law: F = -(-q)* np.cross(v,B)
        B = [Bphi,Bz[zind][rind],Br[zind,rind]]
        a = 500*np.cross(vcurr,B)

        # Calculates next timestep (Newton's second law)
        xt[ii+1] = xt[ii] + vcurr*dt + 0.5*a*dt**2
        vcurr += a*dt

    return xt
