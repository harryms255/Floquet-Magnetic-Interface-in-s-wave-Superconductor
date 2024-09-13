# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:57:36 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

Nx=10
Ny=11
Nev=20
t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.5*Delta
theta=np.pi/2

Vm_values=np.linspace(0,6,101)
B_values=np.linspace(-Delta,Delta,101)


x_position=np.zeros((Nev,len(Vm_values)))
spectrum=np.zeros((Nev,len(Vm_values)))


for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    parameters=[t,mu,Delta,km,B,Vm,theta]
    # spectrum[:,Vm_indx],x_position[:,Vm_indx]=real_space_floquet_spectrum(Nx, Ny, Nev,t, mu, Delta, km, B, Vm, theta)
    spectrum[:,Vm_indx]=real_space_floquet_spectrum(Nx, Ny, Nev,t, mu, Delta, km, B, Vm, theta)
    
fig,ax=plt.subplots()



for i in range(Nev):
    sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position),cmap="plasma")
    
cbar=plt.colorbar(sc)
cbar.ax.get_yaxis().labelpad = 40
cbar.ax.set_ylabel("$\sqrt{<x^2>}$", rotation=270)
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1),linewidth=5,linestyle="dashed",color="black")
#ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1),linewidth=5,linestyle="dashed",color="black")
ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
# ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
ax.set_xlabel("$V_m/t$")
ax.set_ylabel(r"$\epsilon/Period$")