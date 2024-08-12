# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:03:01 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

#plt.close("all")

Nx=50
Ny=51
Nev=10
t=1
mu=-3.6
km=0.65
Delta=1
B=0.5*Delta
theta=np.pi/2
sparse=False

Vm_values=np.linspace(0,6,101)
B_values=np.linspace(-Delta,Delta,101)

if sparse==True:
    x_position=np.zeros((Nev,len(Vm_values)))
    spectrum=np.zeros((Nev,len(Vm_values)))
else:
    x_position=np.zeros((4*Nx//2*Ny,len(Vm_values)))
    spectrum=np.zeros((4*Nx//2*Ny,len(Vm_values)))


for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    parameters=[t,mu,Delta,km,B,Vm,theta]
    if sparse==True:
        spectrum[:,Vm_indx],x_position[:,Vm_indx]=sparse_real_space_spectrum(Nx, Ny,Nev,static_tight_binding_Hamiltonian, parameters)
    else:
        spectrum[:,Vm_indx],x_position[:,Vm_indx]=real_space_spectrum(Nx, Ny,static_tight_binding_Hamiltonian, parameters)
    
fig,ax=plt.subplots()


if sparse==True:
    for i in range(Nev):
        sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position),cmap="plasma")
else:
    for i in range(4*Nx//2*Ny):
        sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position),cmap="plasma")
    
cbar=plt.colorbar(sc)
cbar.ax.get_yaxis().labelpad = 40
cbar.ax.set_ylabel("$\sqrt{<x^2>}$", rotation=270)
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1),linewidth=5,linestyle="dashed",color="black")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1),linewidth=5,linestyle="dashed",color="black")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
ax.set_xlabel("$V_m/t$")
ax.set_ylabel("$E/\Delta$")
ax.set_ylim(bottom=-5,top=5)