##
# Procedures for computation of the Bogoliubov coefficients.
#
import math
import cmath
import numpy as np
from scipy.integrate import quad

from dcp_python.dcp.base import R as dcp_R
from dcp_python.dcp.numerics import dcp_quad

##
# Bogoliubov coefficient "alpha"
#
# b   boundary condition option (see notes below)
# Notes on arguments a and b:
#   Very important: a and b must be used with only
#   one of the following two sets of values:
#      b = 0   -> Dirichlet-Dirichlet
#      b = 1/2 -> Neumann-Dirichlet
def alpha(t, m, n, b, q):
    L0 = q(0.)
    tmin = t/L0 - 1
    tmax = t/L0 + 1

    i1_real = dcp_quad(_alpha_real, tmin, tmax, args_list=(t, m, n, b, q))
    i1_real = i1_real - (tmax - tmin)
    i1_imag = dcp_quad(_alpha_imag, tmin, tmax, args_list=(t, m, n, b, q))
    i1_imag = i1_imag - (tmax - tmin)
    alpha = 0.5*math.sqrt((m+b)/(n+b))*complex(i1_real, i1_imag)
    return alpha


def _alpha_real(x, t, m, n, b, q):
    L0 = q(0)
    r = dcp_R(q, L0*x)
    c = math.cos(math.pi * ((n + b) * r - (m + b) * x))
    return 1 + c


def _alpha_imag(x, t, m, n, b, q):
    L0 = q(0)
    r = dcp_R(q, L0*x)
    c = math.sin(math.pi * ((n + b) * r - (m + b) * x))
    return 1 + c


##
# Bogoliubov coefficient "beta"
#
# b   boundary condition option (see notes below)
# Notes on arguments a and b:
#   Very important: a and b must be used with only
#   one of the following two sets of values:
#      b = 0   -> Dirichlet-Dirichlet
#      b = 1/2 -> Neumann-Dirichlet
def beta(t, m, n, b, q):
    L0 = q(0)
    tmin = t/L0 - 1
    tmax = t/L0 + 1

    i1_real = dcp_quad(_beta_real, tmin, tmax, args_list=(t, m, n, b, q))
    i1_real = i1_real - (tmax - tmin)
    i1_imag = dcp_quad(_beta_imag, tmin, tmax, args_list=(t, m, n, b, q))
    i1_imag = i1_imag - (tmax - tmin)
    beta = -0.5*math.sqrt((m+b)/(n+b))*complex(i1_real, i1_imag)
    return beta


def _beta_real(x, t, m, n, b, q):
    L0 = q(0)
    r = dcp_R(q, L0*x) 
    c = math.cos(math.pi * ((n + b) * r + (m + b) * x))
    return 1 + c


def _beta_imag(x, t, m, n, b, q):
    L0 = q(0)
    r = dcp_R(q, L0*x)
    c = math.sin(math.pi * ((n + b) * r + (m + b) * x))
    return 1 + c

