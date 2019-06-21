import numpy as np

from numpy import linalg

import matplotlib.pyplot as plt

from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

import drr1

import A1

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

L = Ltotal - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2)) # variable extracellular lipid, in mol

#--------------------------------------------------
# Construction des matrices pour le schéma explicites

E  = np.ones(nx-5)
E  = np.concatenate(([1],2*E,[1]),axis = None)
E_ = np.ones(nx-4)

C = sparse.diags([-E_,E,-E_],[-1,0,1])
	
C = sparse.csc_matrix(C)

#--------------------------------------------------
Ls = []

t=0.0

u_old = np.zeros(np.size(u0))
norme = 1
K=0


#--------------------------------------------------
while(norme > 1e-12) :
	K+=1
	u_old = np.array(u)
	L_old = L
	temp = drr1.drr1(r,a,kr,nr,kL,B,b,kt,v0,vl,L)
	lg = temp[0]
	lp = temp[1]
	       
	dp    = vl * (lg - lp)
	mdp   = np.maximum(0,dp)
	idp   = np.minimum(0,dp)
	
	
	Flux = mdp * u #compute right (c>0) flux
    
	Flux[0:-2] = Flux[0:-2] + idp[0:-2] * u[1:-1]
	Flux[0:-2] = Flux[0:-2] - D/(2*dx) * (u[1:-1] - u[0:-2])
	    
	# Boundary condition
	    
	Flux[-1]   = 0
	Flux[0]    = 0
	
	dt = 0.9 * dx/np.amax(np.absolute(dp))
	beta = D*dt/(2*dx**2)
	
	A  = sparse.identity(nx-3) + beta * C
	A  = sparse.csc_matrix(A)
	  
	u[1:-2] = spsolve(A,u[1:-2] - (dt/dx) * (Flux[1:-2] - Flux[0:-3]))
	u[0]  = 0
	u[-1] = 0
	
	L = Ltotal - dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2))
	
	if L<0 :
		L = 0
	    
	t = t + dt
	    
	Ut = dx * np.sum((v - v0) * u * (4 * np.pi * r**2/vl**2))
	    
	Ls.append([t,L,Ut])
	
	norme = linalg.norm(u_old-u)/float(linalg.norm(u_old))


#--------------------------------------------------

func_mass = lambda x : np.exp(-(x - mu)**2 * si)

func_tau  = lambda x : A1.A1(x,a,kr,nr,kL,B,b,kt,v0,vl,Ls[-1][1])
	
mass = integrate.quad(func_mass,r0,r_max)[0]

equi = np.zeros(len(r))
for i in range(len(r)) :
	equi[i] = np.exp(1/float(D) * integrate.quad(func_tau,r0,r[i])[0])

int_equi = dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))

C = mass/int_equi


plt.plot(r,u,label = "u")
plt.plot(r,C*equi,label = "u_inf")
plt.legend(loc = "upper right")
plt.show()

print("Convergence atteinte en",K,"étapes avec norme =",norme)


#=================================================
