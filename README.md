# RMT_Tokamak
Welcome to the RMT Tokamak Code, by Kieran McDonald, Joe Roll, and Miles Testa. 

The most recent version of the code is in the "development" directory, and contains the newer features (Running Solov'ev Solutions, as well as Finite Differences verification). The Finite Differences code is buggy and still under review. The main directory contains a stable build that can only be used for Miller Geometry. 

## Description
This code implements the numerical Grad-Shafranov (GS) solver for finding the poloidal flux function ($\psi$) in an up-down symmetric tokamak in the (R, Z) plane. The method is a slight extension of that found in Xu and Fitzpatrick's "Vacuum Solution for Solov'ev's equilibrium configuration in tokamaks", hereafter referred to as (Xu, Fitzpatrick). Assuming that all plasma is confined to the region of the tokamak within the so-called "Last Closed Flux Surface" (hereafter referred to as the LCFS), which is the last closed contour of $\psi$, we are solving the system:

$$\nabla^* \psi = R \frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial \psi}{\partial R} \right) + \frac{\partial^2 \psi}{\partial Z^2} = -j_{\phi} R \text{ for points within the LCFS}$$
$$\nabla^* \psi = 0 \text{ for points outside of the LCFS.}$$ 

where $j_{\phi}$ is the toroidal current.  Note that we apply the same normalization for the field, current, and flux as in Xu, Fitzpatrick. 

We enforce the Dirichlet boundary condition that $\psi_{LCFS}$ have a certain value by adding on a homogenous solution to $\nabla^* \psi = 0$ in the form of a multipole expansion that represents the effect of distance poloidal field coils. The order of poles expanded to for this matching is a hyperparameter specified by the user, and requires some trial and error to avoid over-weighting the contribution from the coils over the plasma currents. The form of the multipole expansion comes from the paper "Toroidally Symmetric Polynomial Multipole Solutions of the Vector Laplace Equation" by Reusch and Neilson. As this paper is behind a pay wall, we do not attach it, and merely provide the DOI link: (https://doi.org/10.1016/0021-9991(86)90041-0). 

The user can implement a Solov'ev-type solution using the "is_solovev" flag. If "is_solovev" is False, a LCFS of the following form is implemented (https://doi.org/10.1063/1.872666):

$$R(\theta) = R_o\Big(1 + \epsilon \cos\big(\theta + \arcsin(\delta)\sin(\theta)\big)\Big)$$
$$Z(\theta) = \epsilon \kappa \sin(\theta)$$

Where the shaping parameters are:

$$R_o: \text{ The major radius of the tokamak.}$$
$$\epsilon = \frac{a}{R_o}: \text{ The inverse aspect ratio of the tokamak, for minor radius a.}$$
$$\delta: \text{ triangularity of the LCFS, must be between 0 and 1.}$$
$$\kappa: \text{ The elongation of the LCFS.}$$ 

This parameterization is known as the "Miller Geometry", and was first introduced in the paper "Noncircular, finite aspect ratio, local equilibrium model" by Miller, Chu, Greene, Lin-Liu and Waltz in the lat 90s. As this paper is behind a pay-wall, we do not attach it, and merely provide the DOI link: (https://doi.org/10.1063/1.872666) 

The toroidal current chosen is specified via physical intuition from transport codes/experiments. For the Solov'ev, the current is chosen as it is an exact solution to the GS equation. For the miller, a physically reasonable choice is specified, though the user is welcome to implement their own currents as guided by results from their own transport simulations/experiments. 

For more details on the theory, see the attached papers, as well as the plasma_background.PDF file within the Background Information directory. 

Using the LCFS and $\psi$, we find the magnetic field at each point and use this to simulate the motion of a particle in the tokamak. As we assume no equilibrium toroidal fluid flow, the electric field is assumed to be zero (due to the Ideal MHD Ohm's law), and the motion is merely due to magnetic fields.  

Our implementation utilizes $\texttt{Numba}$ to reduce the computation time by a factor of about 50. 

## Common Usage

A more detailed version of this is shown within [`example.py`](https://github.com/milestesta/RMT_Tokamak/blob/main/example_files/example.py).

The `RMT_Tokamak` class is contained within the [`gs_solver.py`](https://github.com/milestesta/RMT_Tokamak/blob/main/gs_solver.py) file.  The main usage of this code is to compute our poloidal magnetic flux $\psi$.  A quick example of this on a $500\times500$ grid is

```python
import numpy as np
import gs_solver as gs

N_R = 500 # number of points along r
N_Z = 500 # number of points along z

model = gs.RRT_Tokamak(Rdim=N_R,Zdim=N_Z) # initializes class
psi = model.compute_psi() # computes psi
```

## Plot and Animation Examples

<img src=https://github.com/milestesta/RRT_Tokamak/blob/main/Example%20Plots/Constant%20Psi%20contours%20(presentation).png width="500">

<img src=https://github.com/milestesta/RRT_Tokamak/blob/main/Example%20Plots/Triangulation%20and%20Elongation%20(from%20presentation).png width="500">

<img src=https://github.com/milestesta/RRT_Tokamak/blob/main/Example%20Plots/Particle%20Motion%20Simulation%20GIF.gif width="500">

## Installation
Necessary packages

`NumPy`
    
    pip install numpy 

`SciPy`
    
    pip install scipy

`Numba`
    
    pip install numba


Download and install the [`gs_solver.py`](https://github.com/milestesta/RMT_Tokamak/blob/main/gs_solver.py) and/or the [`simulate.py`](https://github.com/milestesta/RMT_Tokamak/blob/main/simulate.py) directly from GitHub

    pip install git+https://github.com/milestesta/RMT_Tokamak/blob/main/gs_solver.py

    pip install git+https://github.com/milestesta/RMT_Tokamak/blob/main/simulate.py

## References
For more information, consider reading the following papers/books.

Base model we built upon (pdf version can be found in Background Information folder): 

> Tao Xu & Richard Fitzpatrick. "Vacuum solution for Solov'ev's equilibrium configuration in tokamaks" Nuclear Fusion 2019 https://iopscience.iop.org/article/10.1088/1741-4326/ab1ce3 

Multipole sum: 

> N. F. Reusch & G. H. Neilson. "Toroidally symmetric polynomial multipole solutions of the vector laplace equation" Journal of Computational Physics 1984 https://www.sciencedirect.com/science/article/abs/pii/0021999186900410

Miller Surface Geometry: 

> R. L. Miller, M. S. Chu, J. M. Greene, Y. R. Lin-Liu, R. E. Waltz; Noncircular, finite aspect ratio, local equilibrium model. Phys. Plasmas 1 April 1998; 5 (4): 973–978. https://doi.org/10.1063/1.872666

Grad Shafranov Equation (Chapter 6):

> Jeffrey P. Friedberg. "Ideal MHD" Cambridge University Press https://doi.org/10.1017/CBO9780511795046
