# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019

@author: leo
"""

import numpy as np

import scipy.optimize as opt
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import A1

#==========================================================

# Ce code met en place l'algorithme ABC basique.

global nr,B,b,vl,mu,si,mass,dx,rmin,r,v,v0

nr = 3 
B  = 125
b  = 0.27
vl = 1e6

mu = 45.0
si = 0.01



# Fonctions utiles par la suite

def u_from_L(L_,q) :

	a,kr,kl,kL,D,Ltotal = q
	
	equi = np.exp(1/float(D) * dx * (np.cumsum(A1.A1(r,a,kr,nr,kL,B,b,kl,v0,vl,L_))))
	
	int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
	
	const = mass/int_equi

	u = const*equi
	
	return u


#-----------------------------------------------------------


def R(L_,q) :
	
	Ltotal = q[-1]

	u_ = u_from_L(L_,q)

	R_L = Ltotal - dx * np.sum((v - v0) * u_ * (4 * np.pi * r**2/vl**2))	
	
	if R_L<0 :
		R_L = 0
	
	return R_L

	
def R_(L_,q) :
	return R(L_,q) - L_

	
#-----------------------------------------------------------

	
def xhi2_pde(p,u0) :

	L = opt.root(R_,p[-1],args=(p)).x
	
	u = u_from_L(L,p)
	
	#step = int(5/dx)
	
	#norm = np.sum((u[step::]-u0[step::])**2)  + len(np.where(p<0)[0])*1e6
	norm = np.sum((u-u0)**2)  + len(np.where(p<0)[0])*1e6
	
	return norm
	


#=========================================================


r0 = 15

r_max = 300

nx = 500

dx    = (r_max - r0)/(nx-1)
	
rmin = r0 - dx

r = rmin + (dx * np.arange(1,nx+1) - dx)

#---------------------------------------------------------

v  = (4/3.) * np.pi * r**3
v0 = (4/3.) * np.pi * r0**3

#-----------------------------------------------------------


func_mass = lambda x : np.exp(-(x - mu)**2 * si)

mass = integrate.quad(func_mass,r0,r_max)[0]

#---------------------------------------------------------
# Paramètres de départ

var = ['a','kr','kl','kL','D','Ltotal']

a   = 0.5
kr  = 200.
kl  = 0.01
kL  = 0.1
D   = 50. * 1e3
Ltotal = 4.

p0 = np.array([a,kr,kl,kL,D,Ltotal])

L0 = opt.root(R_,p0[-1],args=(p0)).x
	
u0 = u_from_L(L0,p0)

#noise = 0.02 * np.random.normal(size=r.size)
	
#u0+=noise


#---------------------------------------------------------

epsilon = 0.1

N = 10000

P = np.zeros((N,6))

# On remplis P de N jeu de paramètres avec pour chacun une distribution antérieure

P[:,0] = np.random.uniform(0,2,N)
#P[:,0] = a * np.ones(N)

P[:,1] = kr * np.ones(N)

P[:,2] = kl * np.ones(N)

P[:,3] = np.random.uniform(0,1,N)
#P[:,3] = kL * np.ones(N)

P[:,4] = D * np.ones(N)

P[:,5] = np.random.uniform(0,8,N)
#P[:,5] = Ltotal * np.ones(N)


# On construit set_P, l'ensemble des jeu de paramètre qu'on va garder.
set_P  = p0

E=[0]

for index,p in enumerate(P) :
	print(index)
	diss =  xhi2_pde(p,u0)
	#print(diss)
	if diss < epsilon :
		set_P = np.vstack([set_P,p])
		E.append(diss)
		
print (len(set_P))

nb_bins = 500

fig = plt.figure()
ax0_a    = fig.add_subplot(231)
ax1_a    = fig.add_subplot(234)
ax0_KL   = fig.add_subplot(232)
ax1_KL   = fig.add_subplot(235)
ax0_Ltot = fig.add_subplot(233)
ax1_Ltot = fig.add_subplot(236)


ax0_a.hist(set_P[:,0],bins = nb_bins,range = [0,2])
ax0_a.hist(P[:,0],bins = nb_bins,alpha= 0.5,range = [0,2])
ax1_a.hist(set_P[:,0],bins = nb_bins)

ax0_KL.hist(set_P[:,3],bins = nb_bins,range = [0,1])
ax0_KL.hist(P[:,3],bins = nb_bins,alpha= 0.5,range = [0,1])
ax1_KL.hist(set_P[:,3],bins = nb_bins)

ax0_Ltot.hist(set_P[:,5],bins = nb_bins,range = [0,8])
ax0_Ltot.hist(P[:,5],bins = nb_bins,alpha= 0.5,range = [0,8])
ax1_Ltot.hist(set_P[:,5],bins = nb_bins)

ax1_a.axvline(x=p0[0],ls="--",color='black')
ax1_KL.axvline(x=p0[3],ls="--",color='black')
ax1_Ltot.axvline(x=p0[5],ls="--",color='black')

ax1_a.set(xlabel = r'$a$')
ax1_KL.set(xlabel = r'$\kappa_L$')
ax1_Ltot.set(xlabel = r'$L_{total}$')

ax0_a.set(ylabel='Nombre de paramètre')
ax1_a.set(ylabel='Nombre de paramètre')

orange_patch = mpatches.Patch(color='orange', alpha=0.5)
blue_patch = mpatches.Patch(color='blue')

fig.tight_layout()
lgd = ax1_Ltot.legend([orange_patch,blue_patch],['Paramètres générés','Paramètres conservés'],bbox_to_anchor=(1.04,1), loc="upper left")
#fig.savefig('../Rapport/Images/abc_distrib.png',dpi=1200,bbox_extra_artists=[lgd,],bbox_inches='tight')
fig.show()

fig,axs = plt.subplots(3)

axs[0].plot(set_P[:,0],E,"o",markersize=1)
axs[1].plot(set_P[:,3],E,"o",markersize=1)
axs[2].plot(set_P[:,5],E,"o",markersize=1)

axs[0].set(xlabel = r'$a$',ylabel='Erreur')
axs[1].set(xlabel = r'$\kappa_L$',ylabel='Erreur')
axs[2].set(xlabel = r'$L_{total}$',ylabel='Erreur')

axs[0].axvline(x=p0[0],ls="--",color='black')
axs[1].axvline(x=p0[3],ls="--",color='black')
axs[2].axvline(x=p0[5],ls="--",color='black')

fig.tight_layout()
#plt.savefig('../Rapport/Images/abc_error.png',dpi=1200)
plt.show()
