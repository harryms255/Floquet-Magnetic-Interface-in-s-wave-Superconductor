# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:56:24 2024

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
omega=2*B
theta=0.5*np.pi
period=2*np.pi/omega
Vm=1

parameters=[t,mu,Delta,Vm,km]


#spectrum of floquet hamiltonian
kx_values=np.linspace(-np.pi,np.pi,101)
spectrum=np.zeros((4*Ny,len(kx_values)))
for kx_indx,kx in enumerate(tqdm(kx_values)):
    
    FH=floquet_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta)
    spectrum[:,kx_indx]=np.linalg.eigvalsh(FH)


fig,ax=plt.subplots()
for i in range(4*Ny):
    ax.plot(kx_values/np.pi,spectrum[i,:]*period/np.pi)
ax.set_xlabel("$k_x/\pi$")
ax.set_ylabel("$\epsilon T/\pi$")


#instaneous spectrum of Hamiltonian at time t
# kx=0
# T_values=np.linspace(0,period,1001)
# spectrum=np.zeros((4*Ny,len(T_values)))

# for T_indx,T in enumerate(tqdm(T_values)):
#     spectrum[:,T_indx]=np.linalg.eigvalsh(driven_tight_binding_Hamiltonian(kx, Ny, T, period, t, mu, Delta, Vm, km))
    
# fig,ax=plt.subplots()
# for i in range(4*Ny):
#     ax.plot(T_values/period,spectrum[i,:],"k")
# ax.set_xlabel("$T/period$")
# ax.set_ylabel("$E/t$")
