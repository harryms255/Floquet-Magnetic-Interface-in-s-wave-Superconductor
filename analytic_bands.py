# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:06:20 2024

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


# kx_values=np.linspace(-5,5,101)
# band_structure=np.zeros((2,len(kx_values)))

# for kx_indx,kx in enumerate(tqdm(kx_values)):
#     band_structure[:,kx_indx]=in_gap_band_structure(kx, kf, m, km, Delta, B, Vm, theta)
    
# plt.figure()
# for i in range(2):
#     plt.plot(kx_values,band_structure[i,:])
    


kx_values=np.linspace(-5,5,101)
omega_values=np.linspace(-0.5*Delta,0.5*Delta,101)
det_values=np.zeros((len(omega_values),len(kx_values)))
for kx_indx,kx in enumerate(tqdm(kx_values)):
    for omega_indx,omega in enumerate(omega_values):
        det_values[omega_indx,kx_indx]=np.linalg.det(np.linalg.inv(T_matrix(omega, kx, kf, m, km, Delta, B, Vm, theta)))
        
plt.figure()
sns.heatmap(det_values,cmap="plasma")
