# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:00:31 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

kf=0.65
m=1
km=kf
Delta=0.1
B=0
Vm=1
theta=np.pi/2
y=0

omega_values=np.linspace(-5*Delta,5*Delta,101)
kx_values=np.linspace(-2,2,101)

LDOS_values=np.zeros((len(omega_values),len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    for omega_indx,omega in enumerate(omega_values):
        LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, y, kf, m, km, Delta, B, Vm, theta)
        
        
plt.figure()
plt.imshow(LDOS_values,cmap=plt.cm.Blues,vmin=-10,vmax=10)
plt.gca().invert_yaxis()