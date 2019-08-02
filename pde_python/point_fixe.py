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

# On utilise la méthode de Cranck-Nicolson pour la diffusion dans l'EDP, pour obtenir un scéma implicite. Contrairement au code matlab, on choisit de fixer Ltotal à la place de L(0).

r0 = 15   # mum
vl = 1e6  # from mol to mum
a  = 0.5  # mol.mum^-2 h^-1       # need a larger a to have bimodal
b  = 0.27 # mol.mum^-2 h^-1
B  = 125  # mol.h^-1
kr = 200  # mum
nr = 3    # hill radius
kt = 0.01 #  mol
kL = 0.1  # mol

Ltotal = 4 # initial total lipid in mol


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
# Boundary conditions
u[0]  = 0
u[-1] = 0

#--------------------------------------------------
# Volumes

v  = (4/3.) * np.pi * r**3
v0 = (4/3.) * np.pi * r0**3

#--------------------------------------------------

L0 = Ltotal - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2)) # variable extracellular lipid, in mol

#--------------------------------------------------


def R(L_) :
	func_mass = lambda x : np.exp(-(x - mu)**2 * si)

	mass = integrate.quad(func_mass,r0,r_max)[0]
	equi = np.exp(1/float(D) * dx * (np.cumsum(A1.A1(r,a,kr,nr,kL,B,b,kt,v0,vl,L_)) - r[0]))
	
	int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
	
	const = mass/int_equi
		
	u = const*equi 

	R_L = Ltotal - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2))
	
	if R_L < 0:
		R_L = 0
	
	return R_L
	
def R_(L_) :
	return R(L_) - L_

#--------------------------------------------------


L_inf = root(R_,Ltotal,tol=1e-12).x

print ('L_inf = ',L_inf)

func_mass = lambda x : np.exp(-(x - mu)**2 * si)

mass = integrate.quad(func_mass,r0,r_max)[0]
equi = np.exp(1/float(D) * dx * (np.cumsum(A1.A1(r,a,kr,nr,kL,B,b,kt,v0,vl,L_inf)) - r[0]))
	
int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
	
const = mass/int_equi
		
u = const*equi 

u_inf = np.load("u_inf.npy")


plt.plot(r,u,label="Point fixe")
plt.plot(r,u_inf,label="Schéma numérique")
plt.legend()
plt.title("Comparaison entre le résultat du point fixe et du schéma numérique")
plt.show()
	
#=================================================
