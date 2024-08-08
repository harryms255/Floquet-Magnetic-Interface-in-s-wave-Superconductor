# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:03:01 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

Nx=100
Ny=21
t=1
mu=-3.6
km=0.65
Delta=0.1
B=0
theta=np.pi/2
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=0)
Vm=0.8*Vm_crit
driving_frequency=2*B

Vm_values=np.linspace(-6,6,101)
B_values=np.linspace(-Delta,Delta,101)
x_position=np.zeros((4*Nx//2*Ny,len(Vm_values)))
spectrum=np.zeros((4*Nx//2*Ny,len(Vm_values)))

# for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
for B_indx,B in enumerate(tqdm(B_values)):
    parameters=[t,mu,Delta,km,B,Vm,theta]
    spectrum[:,B_indx],x_position[:,B_indx]=real_space_spectrum(Nx, Ny, driving_frequency,static_tight_binding_Hamiltonian, parameters)
    
fig,ax=plt.subplots()
for i in range(4*Nx//2*Ny):
    sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position))
    
cbar=plt.colorbar(sc)
cbar.ax.get_yaxis().labelpad = 40
cbar.ax.set_ylabel("$\sqrt{<x^2>}$", rotation=270)
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1),linewidth=5,linestyle="dashed")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1),linewidth=5,linestyle="dashed")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi),linewidth=5,linestyle="dashed")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1,kx=np.pi),linewidth=5,linestyle="dashed")
# ax.set_xlabel("$V_m/t$")
ax.set_xlabel("$B/t$")
ax.set_ylabel("$E/\Delta$")
ax.set_ylim(bottom=-5,top=5)