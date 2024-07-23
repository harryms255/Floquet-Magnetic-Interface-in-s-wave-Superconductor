# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:00:31 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

kf=0.65
km=1
Delta=0.1
B=0.5*Delta
B_values=np.linspace(0,Delta,11)
Cm_values=np.linspace(0,1.5,51)
theta_values=np.linspace(0,np.pi/2,11)
y=0



omega_values=np.linspace(-2*Delta,2*Delta,251)
kx_values=np.linspace(-2,2,251)

LDOS_values=np.zeros((len(omega_values),len(kx_values)))
for theta in tqdm(theta_values):
    Cm=continuum_phase_boundaries(kf, km, Delta, B, theta, 1)
    for kx_indx,kx in enumerate(kx_values):
        for omega_indx,omega in enumerate(omega_values):
            LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, y, kf, km, Delta, B, Cm, theta)
            
            
    plt.figure()
    sns.heatmap(LDOS_values,cmap="plasma",vmax=2)
    plt.gca().invert_yaxis()
    plt.title(r"$C_m={:.2f}$, $\theta={:.2f}\pi$, $B={:.1f}\Delta$".format(np.real(Cm),theta/np.pi,B/Delta))