# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:46:14 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

t=1
mu=-3.6
Delta=0.1
B=0.9*Delta
theta=np.pi/3

km_values=np.linspace(0,np.pi,1001)

phase_boundaries_1=TB_phase_boundaries(t, mu, Delta, km_values, B, theta, 1,kx=0)
phase_boundaries_2=TB_phase_boundaries(t, mu, Delta, km_values, B, theta, -1,kx=0)
phase_boundaries_3=TB_phase_boundaries(t, mu, Delta, km_values, B, theta, 1,kx=np.pi)
phase_boundaries_4=TB_phase_boundaries(t, mu, Delta, km_values, B, theta, -1,kx=np.pi)

plt.figure()
plt.plot(km_values/np.pi,phase_boundaries_1)
plt.plot(km_values/np.pi,phase_boundaries_2)
plt.plot(km_values/np.pi,phase_boundaries_3)
plt.plot(km_values/np.pi,phase_boundaries_4)
