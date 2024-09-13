# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:03:48 2024

@author:Harry MullineauxSanders

"""

from functions_file import *


plt.close("all")

Nx=25
t=1
mu=-3.6
km=0.65
Delta=0.1
theta=0.5*np.pi
B=0.5*Delta

disorder_values=[0,0.5*(Delta-B)]

Vm_values=np.linspace(0,6,101)

for disorder in disorder_values:

    # spectrum=np.zeros((4*Nx,len(Vm_values)))
    spectrum=np.zeros((10,len(Vm_values)))
    
    for Vm_indx, Vm in enumerate(tqdm(Vm_values)):
        
        #spectrum[:,Vm_indx]=np.linalg.eigvalsh(real_space_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder))
        spectrum[:,Vm_indx]=spl.eigsh(dok_matrix(real_space_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta)).tocsc(),k=10,return_eigenvectors=False,sigma=0,which="LM")
    
    #floquet_spectrum=np.sort(np.real((1j/period*np.log(np.exp(-1j*period*spectrum)))),axis=0)
    
    
    plt.figure()
    for i in range(10):
        plt.scatter(Vm_values,spectrum[i,:]/Delta,color="k",marker=".")
        
    plt.xlabel("$V_m/t$")
    plt.ylabel("$\epsilon/\Delta$")
    phase_boundaries_1=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1)
    phase_boundaries_2=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi)
    
    plt.axvline(x=phase_boundaries_1,linestyle="dashed",color="black")
    plt.axvline(x=phase_boundaries_2,linestyle="dashed",color="black")