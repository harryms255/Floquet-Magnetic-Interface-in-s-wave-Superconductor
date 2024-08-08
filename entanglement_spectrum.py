# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:04:51 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

Nx=50
Ny=21
t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.9*Delta
theta=np.pi/2
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=0)

Vm_values=np.linspace(-6,6,101)

x_position=np.zeros((4*Nx//2*Ny,len(Vm_values)))
spectrum=np.zeros((4*Nx//2*Ny,len(Vm_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    parameters=[t,mu,Delta,km,B,Vm,theta]
    spectrum[:,Vm_indx],x_position[:,Vm_indx]=entanglement_spectrum(Nx, Ny, static_tight_binding_Hamiltonian, parameters)
    
plt.figure()
for i in range(4*Nx//2*Ny):
    sc=plt.scatter(Vm_values,spectrum[i,:],c=x_position[i,:],vmax=np.max(x_position))
    
plt.colorbar(sc)
plt.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1),linewidth=5,linestyle="dashed")
plt.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1),linewidth=5,linestyle="dashed")
plt.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi),linewidth=5,linestyle="dashed")
plt.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1,kx=np.pi),linewidth=5,linestyle="dashed")
    