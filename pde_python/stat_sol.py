# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:28:40 2019

@author: leo
"""

import scipy.optimize as opt

import numpy as np

import A1

#=========================================================
#Initialisations des paramètres. Ces paramètres peuvent ensuite être utilisés via l'importation.

#Paramètres biologiques :

global nr,B,b,vl,mu,si

nr = 3 
B  = 125
b  = 0.27
vl = 1e6

#---------------------------------------------------------

#Paramètres numétiques :

global dx,r0,r_max,r,v,v0

r0 = 6

r_max = 300

nx = 500

dx    = (r_max - r0)/float(nx-1)

r = r0 + (dx * np.arange(1,nx+1) - dx)

#---------------------------------------------------------

v  = (4/3.) * np.pi * r**3
v0 = (4/3.) * np.pi * r0**3

#=========================================================

# La classe suivante sers à représenter une solution stationnaire du modèle. Elle prend en argument un jeu de paramètre p
# de la forme suivante : p = np.array([a,kr,kl,KL,D,Ltotal,index,X]) et une masse (i.e. un nombre d'adipocytes).
# Comme le point fixe n'a pas besoin de index et X, on peut fournir un jeu de paramètre de la forme np.array([a,kr,kl,KL,D,Ltotal]) mais il faudra modifier la fonction self.__fit_test en conséquence,
# pour fixer index et X.
# A chaque changement de parmètre, il faut appeller la méthode self.change_param qui va changer les champs de la classe
# et recalculer un point fixe puis une solution stationnaire en fonction des nouveaux paramètres.
# La méthode add_noise permet d'ajouter du bruit à la solution stationnaire.
# La méthode self.fit() sert à faire une estimation de paramètre à partir d'une donnée u0 et en partant de self.param.


class stationnary_sol :
	def __init__(self,param,mass) :
		
		self.mass  = mass
		self.change_param(param)
		
	def change_param(self,param) :
		
		self.param = param
		
		self.a,self.kr,self.kl,self.kL,self.D,self.Ltotal = self.param[0:6]
		self.L_root = self.__solve()
		self.u = self.__get_u(self.L_root)
		
	def __solve(self) :
		
		return opt.root(self.__R,self.Ltotal).x 
		
	def __get_u(self,L_) :
		
		equi = np.exp(1/float(self.D) * dx * (np.cumsum(A1.A1(r,self.a,self.kr,nr,self.kL,B,b,self.kl,v0,vl,L_))))
		int_equi =  dx * (np.sum(equi) - 0.5 * (equi[0] + equi[-1]))
		const = self.mass/int_equi
		u = const*equi
		
		return u
		
	def __R(self,L_) :
	
		u_ = self.__get_u(L_)
		R_L = self.Ltotal - dx * np.sum((v - v0) * u_ * (4 * np.pi * r**2/vl**2))
		if R_L<0 :
			R_L = 0
		
		return R_L - L_
		
	def add_noise(self) :
		noise = 0.02 * np.random.normal(size=r.size)
		self.u+=noise
	def __fit_test(self,param,u0) :
		
		self.change_param(param)
		indexmin = np.argmax(u0!=0)
		
		#Ici on peut choisir de modifier index et X qui représente l'index de départ pour le 'fit' et le pourcentage d'adipocytes à gauche de l'index
		
		# Cas 1 : index et X sont dans le jeu de paramètre (typiquement si on 'fit' une donnée expérimentale)
		
		index = int(param[-2])
		X = (dx*np.sum(u0[index:]))/(1-param[-1])
		
		# Cas 2 : on 'fit' sur toute la donnée (typiquement sur une donnée synthétique)
		
#		index=0
#		X = 1.
		
		# Cas 3 : on fixe index et X.
		
#		index = int((25-r0)/dx)
#		X = (dx*np.sum(u0))/(1-0.2)
		
		u_data = u0/X
		
		# Si on utilise les cas 2 ou 3, il faut commenter les derniers tests de limitations dans le calcul de la norme.
		
		norm = np.sum((self.u[index::]-u_data[index::])**2)  + 1e10*(len(np.where(param<0)[0]) +  (r0 + dx *index > 50 or index<indexmin) + (param[-1]>0.40))
		
		return norm
		
	def fit(self,u0) :
		
		y = opt.minimize(lambda x : self.__fit_test(x,u0),self.param,method='Nelder-Mead',options={'maxfev':5000})
#		print(y.message)
		
		return y.x
	
#=========================================================