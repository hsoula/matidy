import numpy as np

#====================================================

# Calcul de la fonction de transport

def A1(r,a,kr,nr,kL,B,b,kt,v0,vl,L0) :
	v = 4/3. * np.pi * r**3
	
	Tau_r = (a /(4*np.pi) * 1./(1 + (r/kr)**nr) * L0/(kL+L0)) - (B + b * r**2)/(4*np.pi * r**2) * (v - v0)/(v - v0 + kt*vl)
	
	y = vl * Tau_r
	
	return y
	
#====================================================
