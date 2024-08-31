##
# The "dynamical casimir package" is a Python package for doing
# calculation of the energy density and production of particles
# inside a one-dimensional non-static cavity with an arbitrary 
# initial field state.
#	
# Authors: D. T. Alves^(1,*) and E. R. Granhen^(2,3)
#
# (1)-Faculdade de Física, Universidade Federal do Pará, 
# 66075-110, Belém, PA,  Brazil
#
# (2)-Faculdade de Ciências Exatas e Naturais, Universidade Federal do Pará, 
# 68505-080, Marabá, PA,  Brazil
#
# (3)-Centro Brasileiro de Pesquisas Físicas, Rua Dr. Xavier Sigaud,
# 150, 22290-180, Rio de Janeiro, RJ, Brazil
#
# (*) Corresponding author. Email address: danilo@ufpa.br ; danilo.t.alves@gmail.com 
#
# ABSTRACT:
# We present a Python package for doing calculation of the exact
# value of the energy density and particles spectra for a real
# massless scalar field in a two-dimensional spacetime, inside
# a non-static cavity with an arbitrary initial field state. The
# routines are based on the exact numerical approach proposed
# by Cole and Schieve (Phys. Rev. A 52 (1995) 4405), 
# and also on the extension of this approach to an arbitrary
# initial field state, developed by the present authors, in
# collaboration with Lima and Silva (Phys. Rev. D 81 (2010) 025016).  
#
# Release Date : 
# Requirements : Python3
#
print('Dynamical Casimir package (release date: 2022)(c)')
print('by Danilo T. Alves and Edney R. Granhen.');
print('The authors kindly request that this software be referenced,\n'\
      'if it is used in work resulting in publication, by citing\n'\
      'the corresponding article in  Comp. Phys. Communications.'); 

# Dynamical Casimir Package - abbreviation: "dcp" 

import sys
import math
from scipy import optimize as opt
import sympy

"""
  Imports of local modules.
"""
from dcp.numerics import bracket_zero
from dcp.numerics import dcp_quad

##
# n
# Routine for calculation of the function "n" (number of
# reflections) (see Ref. D. T. Alves, E. R. Granhen,
# H. O. Silva and M. G. Lima, Phys. Rev. D 81 (2010) 025016). 
#
# The "n" command performs the calculation of the number of
# reflections n(z), the value of z_tilde(z) and the list of instants
# [t1,..., tn] related to the map of the null line t + x = z
# into t - z = z_tilde(z) in the static zone.
#
# Calling sequence: n(L, z, s)
#
# Arguments:
#    L  A procedure which defines the law of motion
#       for the right boundary.
#    z  A real number which defines the null line (t + x = z).
#    s  Assumes values 1, 2, 3 or 4, and controls the information
#       returned.
#  return:
#  (s = 1) just the number 'n' of reflections;
#  (s = 2) just the value of z_final(z);
#  (s = 3) the list [n, z_tilde(z)];
#  (s = 4) the list [n, z_final, [t1,...,tn]]
def n(L, z, s):
    def f_opt(t):
        return t + L(t) - z_final

    z_final = z
    l0 = L(0)
    
    # Interval for searching the roots
    t_min = -l0
    t_max = z + 2 * l0

    res = bracket_zero(f_opt, t_min, t_max)
    if not res[0]:
        print('dcp_n:Could not bracket a zero.')
        exit(0)
        
    t_min = res[1]
    t_max = res[2]

    n = 0
    t_list = []
    if z <= l0:
        (tk, r) = opt.brentq(f_opt, t_min, t_max, full_output=True)
        if not r.converged:
            print('dcp_n:Failed to solve equation!')
            exit(1)

        t_list.append(tk)
        z_final = z
    else:
        while z_final > l0:
            (tk, r) = opt.brentq(f_opt, t_min, t_max, full_output=True)
            if not r.converged:
                print('dcp_n:Failed to solve equation!')
                exit(1)

            z_final = tk - L(tk)
            t_list.append(tk)
            n = n + 1

    if s == 1:
        return n
    elif s == 2:
        return z_final
    elif s == 3:
        return n, z_final
    elif s == 4:
        return n, z_final, t_list
    else:
        print('ERROR:dcp_n: Invalid value for argument "s"')
        sys.exit(1)


##
# Routines for calculation of the auxiliary functions "A" and
# "B" defined in Ref. D. T. Alves, E. R. Granhen, H. O. Silva
# and M. G. Lima, Phys. Rev. D 81 (2010) 025016.
#
# The "A" command performs calculation of the function A(t).
# The "B" command performs calculation of the function B(t).
#
# Calling Sequence: A(L, t) and B(L, t)
#    
# Parameters:
#   L   a procedure which defines the law of motion
#       for the right boundary;
#   t   a real number representing the instant t.

def A(L, t):
    return (-1. + L.DL(t)) ** 2 / (1. + L.DL(t)) ** 2


def B(L, t):
    return -1. / (12 * math.pi) * L.D3L(t) / \
        ((1. + L.DL(t)) ** 3 * (1. - L.DL(t))) - \
        1. / (4 * math.pi) * (L.D2L(t) ** 2 * L.DL(t)) / \
        ((1. + L.DL(t)) ** 4 * (1. - L.DL(t)) ** 2)


##
# A_tilde
# Routine for calculation of the functions "A_tilde" defined 
# in Ref. D. T. Alves, E. R. Granhen, H. O. Silva and
# M. G. Lima, Phys. Rev. D 81 (2010) 025016.
#
# The A_tilde command performs calculation of the function
# "A_tilde".
#
# Calling Sequence: A_tilde(L, z)
#    
# Parameters:
#   L   a procedure which defines the law of motion
#       for the right boundary;
#   z   a real number which defines the null line t + x = z.
def A_tilde(L, z):
    res = n(L, z, 4)
    n_ = res[0]
    z_ = res[1]
    Tlist = res[2]
    zz = 1.
    for i in range(n_):
        zz = zz * A(L, Tlist[i])

    return zz


##
# B_tilde
# Routine for calculation of the functions "B_tilde" defined 
# in Ref. D. T. Alves, E. R. Granhen, H. O. Silva and
# M. G. Lima, Phys. Rev. D 81 (2010) 025016.
#
# The B_tilde command performs calculation of the function
# "B_tilde".
#
# Calling Sequence: B_tilde(L, z)
#    
# Parameters:
#   L   a procedure which defines the law of motion for
#       the right boundary;
#   z   a real number which defines the null line t + x = z.
def B_tilde(L, z):
    n_, z_, Tlist = n(L, z, 4)

    zz = 0.
    for j in range(n_):
        prod = 1.
        for i in range(j):
            prod = prod * A(L, Tlist[i])
        zz = zz + B(L, Tlist[j]) * prod

    return zz


##
# h
# Routine for calculation of the function "h" defined 
# in Ref. D. T. Alves, E. R. Granhen, H. O. Silva and
# M. G. Lima, Phys. Rev. D 81 (2010) 025016.
#
# The "h" command performs calculation of the function h(z).
#
# Calling Sequence: h(s, L, z)
#    
# Parameters:
#  s    a procedure which defines the function h^{(s)}(z).
#  L    a procedure which defines the law of motion for the right boundary;
#  z    a real number which defines the null line t + x = z.
def h(hs, L, z):
    L0 = L(0)
    n_, z_final = n(L, z, 3)
    h_static = hs(z_final) 
    if n_ == 0:
        return h_static

    return h_static*A_tilde(L, z) + B_tilde(L, z)


##
# g
# Routine for calculation of the function "g" defined 
# in Ref. D. T. Alves, E. R. Granhen, H. O. Silva and
# M. G. Lima, Phys. Rev. D 81 (2010) 025016.
#
# The "g" command performs calculation of the function g(z).
#
# Calling Sequence: g(s, L, z)
#    
# Parameters:
#  s   a procedure which defines the function h^{(s)};
#  L   a procedure which defines the law of motion for
#      the right boundary;
#  z    a real number which defines the null line t + x = z.
def g(gs, L, DL, z):
    n_, z_final, Tlist = n(L, z, 4)
    g_static = gs(z_final)
    if n_ == 0:
        return g_static

    g_out = 1.
    for i in range(n_):
        g_out = g_out * A(DL, Tlist[i])

    return g_static * g_out


##
# R
# Routine for calculation of the function "R"
# (Moore's function) (see Cole and Schieve,
# Phys. Rev. A 52, 4405 (1995)).
#
# The "R" command performs calculation of the Moore
# function R(z).
#
# Calling Sequence: R(q, z)
#    
# Parameters:
#   q   a procedure which defines the law of motion for
#       the right boundary;
#   z   a real number which defines the null line t + x = z.
#
def R(L, z):
    L0 = L(0)
    n_, z_final = n(L, z, 3)

    return 2 * n_ + z_final / L0

##
# Tcas_DD
def Tcas_DD(L0):
    return -math.pi/(24*L0**2)

##
# Tcas_DN
def Tcas_DN(L0):
    return math.pi/(48*L0**2)


def Ecav_DD(t, L):
    L0 = L(0)
    e_cas = -math.pi/(24*L0)
    h_s = lambda z: math.pi/(48.*L0**2)
    
    e = -dcp_quad(_h, t - L0, t + L0, args_list=(h_s, L))
    
    return e - e_cas

def _h(u, h_s, L):
    return h(h_s, L, u) 
    

