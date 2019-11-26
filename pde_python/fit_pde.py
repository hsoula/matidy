# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019
@author: Léo Meyer

"""

import numpy as np

import matplotlib.pyplot as plt

from stat_sol import dx,r0,r_max,r,nx,stationnary_sol

from scipy import integrate

import os
#==========================================================

#On cherche à faire une estimation de paramètres pour faire 'fit' le resultat en temps long de notre schéma numérique avec les données expérimentales.
#Au lieu de calculer le schéma à chaque estimation on passe par le calcul du point fixe sur L^infini.

#=========================================================

#Estimations sur des données expérimentales :
#On dispose de 5 données expérimentales, et on fait un 'fi' sur chacune. La liste P représente l'estimation de départ pour chaque donnée.

# Paramètres de départ

var = ['a','r_theta','l_theta','T_theta','D','TG0']
 
a   = [0.5,.5,.5,.4,.6]
r_theta  = [200.,200.,200.,150.,200.]
l_theta  = 0.01
T_theta  = 0.1
D   = 50. * 1e3	
TG0 = [4.,4.,4.,3.,6.]

j = int((25-r0)/dx)
indexes=[j,j,j,j-5,j]

X = [.2,.2,.2,.3,.1]

P = [[a[i],r_theta[i],l_theta,T_theta,D,TG0[i],indexes[i],X[i]] for i in range(5)]

#---------------------------------------------------------

mass = 1. # On travaille avec des fonctions de densités

files = os.listdir("../data/")
files.sort()
for index,file in enumerate(files) :
	
	p = P[index]	
	
	print (file)
	
	histo = np.loadtxt('../data/'+file)
		
	u0 = np.histogram(histo,bins = nx, range = (r0,r_max))[0]
	
	first_d = np.argmax(u0!=0)
	
#	plt.plot(r[first_d::],u0[first_d::])
#	plt.xlim([0,r_max+50])
#	plt.show()
	
	obj = stationnary_sol(p,1.)
	
	p_ = obj.fit(u0)
	obj_f = stationnary_sol(p_,1.)
	
	index_ = int(p_[-2])
	print(p_)
	X_= p_[-1]
	
	plt.plot(r,obj_f.u,label = "u03BB = {:.2f} et {} = {:.2f}".format(X_,r'$r_{data}$',r0 + dx*(index_-1)))
	
	np.save("fit_result",p_)
	
	u_plot = u0/((dx*np.sum(u0[index_:]))/(1-X_))
	
	index_min = np.argmax(u_plot!=0)
	
	
	plt.plot(r[index_:],u_plot[index_:],label = 'Données prises en compte')
	plt.plot(r[index_min:index_:],u_plot[index_min:index_:], label = 'Données non prises en compte')
	plt.axvline(x=r0+dx*(index_-1),ls="--",color = "black")
	plt.legend()
	plt.xlabel('Rayon en u03BCm')
	plt.ylabel('Fréquence des adipocytes')
	#plt.savefig('../Rapport/Images/'+file[:-4]+'.png',dpi=1200)	
	plt.show()
	break
#=========================================================
	
# On fait de l'estimation sur donnée synthétique pour observer l'erreur pour chaque paramètre.
# Dans un premier temps on regarde l'erreur moyenne en fonction de l'estimation de départ puis on regarde l'erreur
# pour une mmême estimation de départ mais différente jeu de paramètre de génération.
	

"""
mu = 45.0
si = 0.01

func_mass = lambda x : np.exp(-(x - mu)**2 * si)

mass = integrate.quad(func_mass,r0,r_max)[0]

var = ['a', 'r_theta', 'l_theta', 'D', 'TG0']

a   = 0.5
r_theta  = 200.
l_theta  = 0.01
T_theta  = 0.1
D   = 50. * 1e3
TG0 = 4.

p0_ = np.array([a,r_theta,l_theta,D,TG0,0,0])

N = 100

I = np.linspace(0.1,2,N)

S = [i*p0_ for i in I]

E = np.zeros((N,7))

s  = np.random.uniform(0.8,1.2,7)

p = s*p0_

sol0=stationnary_sol(p,mass)
sol0.add_noise()

for index,p0 in enumerate(S) :
	obj = stationnary_sol(p0,mass)
	q = obj.fit(sol0.u)
	obj.change_param(q)
	plt.plot(r,obj.u,label="{:.2f}".format(I[index]))
	plt.plot(r,sol0.u,label='data')
	plt.legend()
	plt.show()
	
#	plt.plot(r,sol0.u)
#	plt.plot(r,obj.u)
#	plt.show()
	
	E[index] = np.abs(q-p)/p
	
print(E)
for i in range(5) :
	plt.plot(I,E[:,i],label=var[i])
	plt.ylim((0,1))
	plt.legend()
	plt.xlabel('coefficient multiplicateur de p0')
	plt.ylabel('Erreur relative')
	plt.show()
"""
#--------------------------------------------------------

"""
N = 100

for j in range(N) :

	s  = np.random.uniform(0.8,1.2,8)

	p = s*p0_
	
	sol0.change_param(p)
	sol0.add_noise()
	
	#noise = 0.02 * np.random.normal(size=r.size)
	
	#u0_+=noise

	obj = stationnary_sol(p0,mass)
	q = obj.fit(sol0.u)
	obj.change_param(q)

	E[j] = np.abs(q - p)/p
	P[j] = q

error = np.zeros(6)
for e_i in E :
	error += e_i[:6]

error/=N

for v,e in zip(var,error) :
	print (v,' -> ',e)
"""

#=========================================================
# On affiche la corrélation entre les 3 paramètres d'intérêts : a, T_theta et TG0.
"""
fig, axs = plt.subplots(1,3)

axs[0].plot(P[:,0],P[:,3],"o")
axs[1].plot(P[:,0],P[:,-1],"o")
axs[2].plot(P[:,3],P[:,-1],"o")

axs[0].set(xlabel="a",ylabel="T_theta")
axs[1].set(xlabel="a",ylabel="TG0")
axs[2].set(xlabel="T_theta",ylabel="TG0")

fig.tight_layout()
plt.show()
"""
