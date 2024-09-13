# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:00:13 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

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
Vm=1.5
k=10**(-4)
x_values=np.linspace(0,1,25)
localiser_gap_values=np.zeros((len(Nx_values),len(x_values)))

plt.figure()
for Nx_indx,Nx in enumerate(Nx_values):
    x_Nx_values=Nx*x_values
    
    for x_indx,x in enumerate(tqdm(x_Nx_values)):
        localiser_gap_values[Nx_indx,x_indx]=topological_hamiltonian_localiser_gap(x, E, k, Nx, t, mu, Delta, km, B, Vm, theta)
        

    plt.plot(x_values,localiser_gap_values[Nx_indx,:],"x-",label="$N_x={}$".format(Nx))
    
plt.legend()
plt.xlabel(r"$x/N_x$")
plt.ylabel(r"$\Delta L_{x,E=0}$")
plt.xlim(left=x_values[0],right=x_values[-1])
plt.ylim(bottom=0)

# plt.axvline(x=0,color="black",linestyle="dashed")
# plt.axvline(x=1,color="black",linestyle="dashed")
