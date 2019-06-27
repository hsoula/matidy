# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019

@author: leo
"""

import numpy as np

import scipy.optimize as opt
from scipy import integrate

import matplotlib.pyplot as plt

import A1

#==========================================================

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


#---------------------------------------------------------

epsilon = 0.05

N = 10000

P = np.zeros((N,6))

# On remplis P de N jeu de paramètres

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

for index,p in enumerate(P) :
	print(index)
	diss =  xhi2_pde(p,u0)
	#print(diss)
	
	if diss < epsilon :
		set_P = np.vstack([set_P,p])
		
print (len(set_P))

#hist_a = np.histogram(set_P[:,0],bins = 100,range = [0.3,0.7])[0]
#
#s_a = np.linspace(0.3,0.7,100)
#
#plt.plot(s_a,hist_a)
#plt.show()

#for i in [0,3,5] :
#	fig, axs = plt.subplots(2)
#	axs[0].hist([set_P[:,i],P[:,i]],bins = 100,stacked = 'bar')
#	axs[1].hist(set_P[:,i],bins = 100)
#	
#	fig.tight_layout()
#	plt.show()

nb_bins = 500

fig, axs = plt.subplots(2,3)

axs[0][0].hist(set_P[:,0],bins = nb_bins,range = [0,2])
axs[0][0].hist(P[:,0],bins = nb_bins,alpha= 0.5,range = [0,2])
axs[1][0].hist(set_P[:,0],bins = nb_bins)

axs[0][1].hist(set_P[:,3],bins = nb_bins,range = [0,1])
axs[0][1].hist(P[:,3],bins = nb_bins,alpha= 0.5,range = [0,1])
axs[1][1].hist(set_P[:,3],bins = nb_bins)

axs[0][2].hist(set_P[:,5],bins = nb_bins,range = [0,8])
axs[0][2].hist(P[:,5],bins = nb_bins,alpha= 0.5,range = [0,8])
axs[1][2].hist(set_P[:,5],bins = nb_bins)


axs[1][0].set(xlabel = "a")
axs[1][1].set(xlabel = "KL")
axs[1][2].set(xlabel = "Ltot")

fig.tight_layout()
plt.show()



"""
plot_set = np.random.randint(len(set_P),size = 100)

for i in plot_set :
	Lp = opt.root(R_,set_P[i,-1],args=(set_P[i])).x
	
	up = u_from_L(Lp,set_P[i])
	
	plt.plot(r,up)

plt.plot(r,u0)
plt.show()
"""