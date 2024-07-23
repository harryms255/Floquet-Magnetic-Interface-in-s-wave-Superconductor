# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:11:55 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")
kf=0.65
Delta=0.1
B=Delta*0
theta=np.pi/2

km_values=np.linspace(0,2,101)

#theta_values=np.linspace(0,np.pi,101)

Cm_1=continuum_phase_boundaries(kf, km_values, Delta, B, theta, 1)
Cm_2=continuum_phase_boundaries(kf, km_values, Delta, B, theta, -1)

plt.figure()

plt.plot(km_values,Cm_1)
plt.plot(km_values,Cm_2)
plt.xlabel("$k_m/k_F$")
#plt.xlabel(r"$\theta/\pi$")
plt.ylabel(r"$C_m$")