# RRT_Tokamak
Welcome to the RRT Tokamak Code, by Kieran McDonald, Joe Roll and Miles Testa

## Description
We implement the numerical method for finding the last closed (magnetic) flux surface (LCFS) in a tokamak found in Xu and Fitzpatrick's "Vacuum Solution for Solov'ev's equilibrium configuration in tokamaks". Our implementation utilizes numba to reduce the computation time by a factor of about 50. Using the LCFS and $\psi$, we find the magnetic field at each point and use this to simulate the motion of a particle in the tokamak. 

## References
For more information, consider reading the following papers/books.

Base model we built upon (pdf version can be found in repository): 

> Tao Xu & Richard Fitzpatrick. "Vacuum solution for Solov'ev's equilibrium configuration in tokamaks" Nuclear Fusion 2019 https://iopscience.iop.org/article/10.1088/1741-4326/ab1ce3 

Multipole sum: 

> N. F. Reusch & G. H. Neilson. "Toroidally symmetric polynomial multipole solutions of the vector laplace equation" Journal of Computational Physics 1984 https://www.sciencedirect.com/science/article/abs/pii/0021999186900410

Grad Shafranov Equation (Chapter 6):

> Jeffrey P. Friedberg. "Ideal MHD" Cambridge University Press https://doi.org/10.1017/CBO9780511795046

## Example Usage

<img src=https://github.com/milestesta/RRT_Tokamak/blob/main/Example%20Plots/Screenshot%202024-11-27%20at%2011.13.15.png width="500">

## Installation
Necessary packages

numpy
    
    pip install numpy 

scipy
    
    pip install scipy

numba
    
    pip install numba


Download and install the gs_solver.py and/or the simulate.py directly from GitHub

    pip install git+https://github.com/milestesta/RRT_Tokamak/blob/main/gs_solver.py

    pip install git+https://github.com/milestesta/RRT_Tokamak/blob/main/simulate.py





