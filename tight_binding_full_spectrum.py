# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:58:37 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

Ny=101
t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.9*Delta
theta=np.pi*0.2
Vm=0.9*TB_phase_boundaries(t, mu, Delta, km, B, theta, 1)

kx_values=np.linspace(-np.pi,np.pi,1001)

spectrum=np.zeros((4*Ny,len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum[:,kx_indx]=np.linalg.eigvalsh(static_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta))

plt.figure(figsize=[12,8])
for i in range(4*Ny):
    plt.plot(kx_values/np.pi, spectrum[i,:]/Delta,c="k")
plt.ylim(top=2,bottom=-2)
plt.xlabel("$k_x/\pi$")
plt.ylabel("$E/\Delta$")
plt.tight_layout()


