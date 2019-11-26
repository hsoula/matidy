# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:41:05 2019

@author: Léo Meyer
"""

import abcpmc

import numpy as np

import scipy.optimize as opt
from scipy import integrate

import matplotlib.pyplot as plt

import A1

from stat_sol import dx,r0,r_max,r,nx,stationnary_sol

#==========================================================

global nr,B,b,vl,mu,si,mass,dx,rmin,r,v,v0,u0

nr = 3 
B  = 125
b  = 0.27
vl = 1e6

mu = 45.0
si = 0.01




# Fonctions utiles par la suites

def u_from_T(T_,q) :

	a, kr, l_theta, T_theta, D, TG0 = q
	tau= lambda x : A1.TauR(x, a, kr, nr, T_theta, B, b, l_theta, v0, vl, T_)
	
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

#-----------------------------------------------------------

def R(T_,q) :
	
	TG0 = q[-1]

	u_ = u_from_T(T_,q)

	R_L = TG0 - dx * np.sum((v - v0) * u_ * (4 * np.pi * r**2/vl**2))	
	
	if R_L<0 :
		R_L = 0
	
	return R_L

	
def R_(T_,q) :
	return R(T_,q) - T_

	
#-----------------------------------------------------------

	
def dist(x,y) :
	#step = max(int((20-r0)/dx),np.argmax(u0!=0))
	step=0	
	norm = np.sum((x[step::]-y[step::])**2)
	
	return norm
	


#=========================================================

def data(theta) :
	
	T = opt.root(R_,theta[-1],args=(theta)).x
	
	u = u_from_T(T,theta)

	return u


#=========================================================

v  = (4/3.) * np.pi * r**3
v0 = (4/3.) * np.pi * r0**3

#-----------------------------------------------------------

func_mass = lambda x : np.exp(-(x - mu)**2 * si)

mass = integrate.quad(func_mass,r0,r_max)[0]

#---------------------------------------------------------
# Paramètres de départ

var = [r'$a$',r'$r_\theta$',r'$L_\theta$',r'$T_\theta$',r'$D$',r'$TG0$']

a   = 0.5
kr  = 200.
l_theta  = 0.01
T_theta  = 0.1
D   = 50. * 1e3
TG0 = 4.

p0 = np.array([a, kr, l_theta, T_theta, D, TG0])

#Une donné synthétique

"""
sol = stationnary_sol(p0,mass)

sol.add_noise()

u0 = sol.u

plt.plot(r,u0)
plt.show()
"""
#---------------------------------------------------------

#ABC avec des données expérimentales

mass = 1. # On travaille avec des fonctions de densités

histo = np.loadtxt('../matidy/data/Hys-D0-1-1.txt')

u0 = np.histogram(histo,bins = nx, range = (r0,r_max))[0]

lamb = (dx*np.sum(u0))/(1-0.2)

u0=u0/lamb

synth = stationnary_sol(p0,mass)

p_ = synth.fit(u0)
print(p_)

synth.change_param(p_)
#synth.add_noise(0.001)

plt.plot(r,synth.u)
plt.plot(r,u0)
plt.show()

#print(dist(synth.u,u0))

#---------------------------------------------------------

#On choisit une moyenne 

#means = p0
means = p_

#On choisit une distriution antérieure

#prior = abcpmc.TophatPrior(min = [0,150,0,0,4*1e4,0],max=[2,300,0.1,1,6*1e4,8])

prior = abcpmc.TophatPrior(min = [0.5 * param for param in means], max = [1.5 * param for param in means])

#prior = abcpmc.GaussianPrior(p0,np.eye(len(p0)))

#---------------------------------------------------------

#On définit une suite d'epsilon

T = 10

eps_start = 5
eps_end   = 0.1
eps = abcpmc.LinearEps(T, eps_start, eps_end)

#---------------------------------------------------------

sampler = abcpmc.Sampler(N=1000, Y = synth.u, postfn = data, dist = dist)

pools = []


for pool in sampler.sample(prior, eps) :
	print(pool.t)
#	eps.eps = np.percentile(pool.dists, 75)
#	if eps.eps < eps_end:
#		   eps.eps = eps_end
	pools.append(pool)
sampler.close()

#---------------------------------------------------------

#On affiche l'évolution de la valeur moyenne au cour du temps

for i in range(len(means)):
	moments = np.array([abcpmc.weighted_avg_and_std(pool.thetas[:,i], pool.ws, axis=0) for pool in pools])
	plt.errorbar(range(T), moments[:, 0], moments[:, 1],label=var[i])
	plt.hlines(means[i], 0, T, linestyle="dotted", linewidth=0.7)
	plt.legend()
	plt.xlim([-.5, T])
	plt.show()

#---------------------------------------------------------
	
#On affiche les distributions postérieures

import corner
samples = np.vstack([pool.thetas for pool in pools])
fig = corner.corner(samples, labels = var,truths= means)
fig.savefig('../Rapport/Images/abcpmc_data.png')
plt.show()

#---------------------------------------------------------

#On affiche la moyenne, l'écart type et le coefficient de variation.
for i in range(len(means)) :
		x_ = sum(samples[:,i])/len(samples)
		sig = np.sqrt(sum(samples[:,i]**2)/len(samples) -x_**2)
		sig_norm = sig/x_
		
#		sd_p = np.sqrt(sum(samples[:,i]**2)/len(samples) -means[i]**2)
#		sd_p_norm = sd_p/(max(samples[:,i]) - min(samples[:,i]))
		
		print("{:12} : {:.4f} \u00B1 {:.4f} | {:.4f}".format(var[i],x_,sig,sig_norm))
