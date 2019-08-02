# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:39:36 2019
@author: Léo Meyer

"""

import numpy as np

def drr1(r,a,kr,nr,kL,B,b,kt,v0,vl,L)    :
	
    v = (4/3.) * np.pi * r **3

    dri = (a /(4*np.pi) * 1./(1 + (r/kr)**nr) * L/(kL+L))
    dro = (B + b * r**2)/(4*np.pi * r**2) * (v - v0)/(v - v0 + kt*vl)
       
    return (dri,dro)
	
	
