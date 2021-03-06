# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019
@author: Léo Meyer

"""

import numpy as np
import A1
from scipy import integrate
from scipy.optimize import root
from numpy.linalg import norm
import matplotlib.pyplot as plt
#=================================================

# On utilise la méthode de Cranck-Nicolson pour la diffusion dans l'EDP, pour obtenir un scéma implicite. Contrairement au code matlab, on choisit de fixer TG0 à la place de L(0).

r0 = 15   # mum
vl = 1e6  # from mol to mum
a  = 0.5  # mol.mum^-2 h^-1       # need a larger a to have bimodal
b  = 0.27 # mol.mum^-2 h^-1
B  = 125  # mol.h^-1
r_theta = 200  # mum
nr = 3    # hill radius
l_theta = 0.01 #  mol
T_theta = 0.1  # mol

TG0 = 4 # initial total lipid in mol


#--------------------------------------------------
# Pas en r et r_max

nx    = 500
r_max = 300 # mum
dx    = r_max/float(nx-1)

rmin = r0 - dx

r = rmin + (dx * np.arange(1,nx+1) - dx)

#--------------------------------------------------
# Diffusion parameter

D = 50 * 1e3 # needed to be large : units? not sure it makes sens

#--------------------------------------------------
# Initial density of cells : gaussian

mu = 45.0
si = 0.01

u0 = np.exp(-(r - mu)**2 * si)
u = u0 # density vector
#--------------------------------------------------
# Volumes

v  = (4/3.) * np.pi * r**3
v0 = (4/3.) * np.pi * r0**3

#--------------------------------------------------

T = TG0 - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2)) # variable extracellular lipid, in mol

#--------------------------------------------------


def R(T_, r,a,r_theta,nr,T_theta,B,b,l_theta,v0,vl) :
	func_mass = lambda x : np.exp(-(x - mu)**2 * si)

	mass = integrate.quad(func_mass,r0,r_max)[0]
	equi = np.exp(1/float(D) * dx * (np.cumsum(A1.TauR(r,a,r_theta,nr,T_theta,B,b,l_theta,v0,vl,T_)) - r[0]))
	
	int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
	
	const = mass/int_equi
		
	u = const * equi 
   
	R_L = TG0 - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2))
	
	if R_L < 0:
		R_L = 0
	
	return R_L
	
def R_(T_, r,a,r_theta,nr,T_theta,B,b,l_theta,v0,vl) :
	return R(T_,r,a,r_theta,nr,T_theta,B,b,l_theta,v0,vl) - T_

#--------------------------------------------------

func_mass = lambda x : np.exp(-(x - mu)**2 * si)

mass = integrate.quad(func_mass,r0,r_max)[0]

def u_from_T(T_, q) :

	a, r_theta, l_theta, T_theta, D, TG0, v0, B, b = q
	tau= lambda x : A1.TauR(x, a, r_theta, nr, T_theta, B, b, l_theta, v0, vl, T_)
	
	equi = np.exp(1/float(D) * dx * np.cumsum(np.array(list(map(tau,r)))))
	equi = np.array(equi, dtype=np.float128)
	
	int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
#	print(int_equi)
	
	const = mass/int_equi
	u = const*equi
	
	if (int_equi == 0 or np.isnan(int_equi) or np.isinf(int_equi) or len(np.where(np.isinf(u))[0]) != 0 or len(np.where(np.isnan(u))[0]) != 0) :
#		print('Error',equi[-1])
		u = 100 * np.ones(len(r))
	
	return u

#a = 0.50
#T_theta = 
yo = 0
for a in np.arange(0.25, .55, 0.005):
    for D in 1e3 * np.arange(1,100,10):
#for T_theta in 0.00001 * 10 ** np.arange(0,6, 0.5):
#    for l_theta in 0.0001 * 10 ** np.arange(0,4, 0.5):

        T_inf = root(R_,TG0, args=(r,a,r_theta,nr,T_theta,B,b,l_theta,v0,vl), tol=1e-12).x
        u = u_from_T(T_inf[0], (a,r_theta,l_theta,T_theta,D, TG0, v0, B, b))
        #plt.plot(u)
        #plt.savefig("{}.pdf".format(yo))
        #yo += 1
        ndfx = np.diff(u) 
        nx = np.where(ndfx[1:] * ndfx[:-1] < 0)[0]
        print(T_theta,l_theta, len(nx))
