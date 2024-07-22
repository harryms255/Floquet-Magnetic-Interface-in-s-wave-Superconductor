# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:00:31 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

kf=0.65
km=kf
Delta=0.1
B=0
Cm=Delta**(1/2)
B_values=np.linspace(0,Delta,11)
Cm_values=np.linspace(0,1.5,21)
theta=np.pi/2*0
y=0

for B in tqdm(B_values):
    omega_values=np.linspace(-5*Delta,5*Delta,251)
    kx_values=np.linspace(-2,2,251)
    
    LDOS_values=np.zeros((len(omega_values),len(kx_values)))
    
    for kx_indx,kx in enumerate(kx_values):
        for omega_indx,omega in enumerate(omega_values):
            LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, y, kf, km, Delta, B, Cm, theta)
            
            
    plt.figure()
    plt.imshow(LDOS_values,cmap=plt.cm.Blues,vmax=1)
    plt.gca().invert_yaxis()