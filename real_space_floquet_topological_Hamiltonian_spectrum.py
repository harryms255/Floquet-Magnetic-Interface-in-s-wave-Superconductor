# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:31:06 2024

@author: Harry MullineauxSanders
"""

from functions_file import *


plt.close("all")

Nx=150
t=1
mu=-3.6
km=0.65
Delta=0.1
theta=0.5*np.pi
B=0.5*Delta
period=np.pi/B
Vm_values=np.linspace(0,6,51)

spectrum=np.zeros((4*Nx,len(Vm_values)))

for Vm_indx, Vm in enumerate(tqdm(Vm_values)):
    
    spectrum[:,Vm_indx]=np.linalg.eigvalsh(real_space_floquet_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta))

plt.figure()
for i in range(4*Nx):
    plt.plot(Vm_values,spectrum[i,:]*period/np.pi,"k")