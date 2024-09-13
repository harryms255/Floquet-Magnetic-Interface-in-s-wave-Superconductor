# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:46:55 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

def topological_hamiltonian_spectral_localiser(x,E,k,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0):
    X=one_D_position_operator(Nx)
    H=real_space_floquet_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)
    
    # k=0.001
    
    L=np.zeros((8*Nx,8*Nx),dtype=complex)
    
    L[:4*Nx,4*Nx:]=k*(X-x*np.identity(4*Nx))-1j*(H-E*np.identity(4*Nx))
    L[4*Nx:,:4*Nx]=k*(X-x*np.identity(4*Nx))+1j*(H-E*np.identity(4*Nx))
   
    return L

def topological_hamiltonian_localiser_gap(x,E,k,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0):
    L=dok_matrix(topological_hamiltonian_spectral_localiser(x, E, k,Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)).tocsc()
    
    #eigenvalues=np.linalg.eigvalsh(L.A)
    eigenvalues=spl.eigsh(L,k=1,sigma=0,return_eigenvectors=False,which="LM")
    
    gap=np.min(abs(eigenvalues))
    
    return gap



Nx_values=[50,75,100,125,150]
t=1
mu=-3.6
Delta=0.1
km=0.65
B=0.5*Delta
theta=0.5*np.pi
E=0
x=0
Vm=1.5

plt.figure()
k_values=10**np.linspace(0,-5,11)
spectral_localiser_values=np.zeros(((len(k_values)),len(Nx_values)))

for Nx_indx,Nx in enumerate(Nx_values):
    
    for k_indx,k in enumerate(tqdm(k_values)):
        spectral_localiser_values[k_indx,Nx_indx,]=topological_hamiltonian_localiser_gap(x, E, k, Nx, t, mu, Delta, km, B, Vm, theta)
    plt.plot(np.log10(k_values),spectral_localiser_values[:,Nx_indx],"x-",label="$N_x={}$".format(Nx))
plt.legend()
plt.xlabel(r"$\log_{10}(\kappa)$")
plt.ylabel(r"$\Delta L_{x=0}$")
