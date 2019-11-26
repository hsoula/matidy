# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019
@author: Léo Meyer

"""

import numpy as np

#====================================================

# Calcul de la fonction de transport

def TauR(r, a, r_theta, nr, T_theta, B, b, l_theta, v0, vl, T):
    
	v = 4/3. * np.pi * r**3	
	Tau_r = (a /(4*np.pi) * 1./(1 + (r/r_theta)**nr) * T/(T_theta+T)) - (B + b * r**2)/(4*np.pi * r**2) * (v - v0)/(v - v0 + l_theta*vl)
	
	y = vl * Tau_r
	
	return y
	
#====================================================
