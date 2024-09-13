# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:40:46 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

def topological_hamiltonian_spectral_localiser(x,E,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0):
    X=one_D_position_operator(Nx)
    H=real_space_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)
    
    k=10**(-4)
    
    L=np.zeros((8*Nx,8*Nx),dtype=complex)
    
    L[:4*Nx,4*Nx:]=k*(X-x*np.identity(4*Nx))-1j*(H-E*np.identity(4*Nx))
    L[4*Nx:,:4*Nx]=k*(X-x*np.identity(4*Nx))+1j*(H-E*np.identity(4*Nx))
   
    return L
    


def topological_hamiltonian_localiser_gap(x,E,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0):
    L=dok_matrix(topological_hamiltonian_spectral_localiser(x, E, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)).tocsc()
    
    #eigenvalues=np.linalg.eigvalsh(L.A)
    eigenvalues=spl.eigsh(L,k=1,sigma=0,return_eigenvectors=False,which="LM")
    
    gap=np.min(abs(eigenvalues))
    
    return gap

def topological_hamiltonian_class_D_invariant(x,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0):
    X=one_D_position_operator(Nx)
    H=real_space_floquet_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)
    
    k=10**(-4)
    
    C=k*(X-x*np.identity(4*Nx))+1j*H
    
    # eigenvalues=np.linalg.eigvalsh(C)
    
    # # sig=len(eigenvalues[eigenvalues<0])
    # # invariant=(-1)**(sig%2)
    
    # invariant=np.prod(np.sign(eigenvalues))
    
    invariant,det=np.linalg.slogdet(C)
    
    return invariant


Nx=100
t=1
mu=-3.6
km=0.65
Delta=0.1
theta=0.5*np.pi
B=0.1*Delta
Vm_values=[1]
disorder=(Delta-B)*0

x_values=np.linspace(-10,Nx+9,25)
plt.figure()
for Vm in Vm_values:

    localiser_gap_invariant_values=np.zeros(len(x_values))
    
    for x_indx,x in enumerate(tqdm(x_values)):
        localiser_gap_invariant_values[x_indx]=topological_hamiltonian_class_D_invariant(x, Nx, t, mu, Delta, km, B, Vm, theta)
    
    
    plt.plot(x_values,localiser_gap_invariant_values,"-x",label=r"$V_m={:.1f}t$".format(Vm))
    plt.xlabel("$x$")
    plt.ylabel(r"$\nu(x)$")
    plt.xlim(left=min(x_values),right=max(x_values))
plt.axvline(x=0,linestyle="dashed",color="black")
plt.axvline(x=Nx-1,linestyle="dashed",color="black")
plt.legend()