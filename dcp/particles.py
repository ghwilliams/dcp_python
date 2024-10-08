##
#
##
import math
import cmath
import numpy as np

import dcp_python.dcp.bogoliubov as bogoliubov
from dcp_python.dcp.base import R as dcp_R

##
# Computes the number of quanta for a pair of modes.
#
# b   boundary condition option (see notes below)
# q   Motion law
#
# Notes on arguments a and b:
#   Very important: a and b must be used with only
#   one of the following two sets of values:
#      b = 0   -> Dirichlet-Dirichlet
#      b = 1/2 -> Neumann-Dirichlet
def NQuanta_per_mode(t, n, m, b, q): 
    alpha = bogoliubov.alpha(t, n, m, b, q)
    beta = bogoliubov.beta(t, n, m, b, q)

    Nm = abs(beta)**2 

    return Nm

##
# Computes the number of particle per field mode.
#
# b   boundary condition option (see notes below)
# q   Motion law
#
# Notes on arguments a and b:
#   Very important: a and b must be used with only
#   one of the following two sets of values:
#      b = 0   -> Dirichlet-Dirichlet
#      b = 1/2 -> Neumann-Dirichlet
def NParticles_per_mode(t, n, b, q, Mmax):
    tol1 = 1E-3  # Convergence tolerance
    
    Mstart = int(1 - 2*b) # Starting summation mode   

    max_convergence_count = 5
    convergence_count = 0
    
    Npart = 0
    Npart_old = 0

    for m in range(Mstart, Mmax + 1):
        Nm = NQuanta_per_mode(t, n, m, b, q)        
        Npart = Npart + Nm

        if abs(Npart - Npart_old) < tol1*Npart_old:
            convergence_count = convergence_count + 1
        else:
            convergence_count = 0

        if convergence_count >= max_convergence_count:
            break            

        Npart_old = Npart

    return Npart

##
# Computes the total number of particles for a given
# total time.
#
# t   Movement duration
# l   Motion law
# b   Boundary condition
#     b = 0    -> Dirichlet-Dirichlet
#     b = 1/2  -> Dirichlet-Neumann/Neumann-Dirichlet
# nmax
# mmax
#
def NParticles_per_time(t, l, b, Nmax, Mmax):
    tol = 1E-3

    L0 = l(0)
    Npart = 0
    Npart_old = Npart
    Nmin = 1    
    
    conv_count = 0   
    max_conv_count = 5
    for n in range(Nmin, Nmax):
        res = NParticles_per_mode(t, n, b, l, Mmax) 
        Npart = Npart + res

        if abs(Npart - Npart_old) < tol*Npart_old:
            conv_count = conv_count + 1
        else:
            conv_count = 0
        if conv_count >= max_conv_count:
            break      

        Npart_old = Npart    

    return Npart



