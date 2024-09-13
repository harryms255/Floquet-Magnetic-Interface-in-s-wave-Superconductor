# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:38:31 2024

@author: Harry MullineauxSanders
"""

from functions_file import *


#plt.close("all")

Nx=150
t=1
mu=-3.6
km=0.65
Delta=0.1
theta=0.5*np.pi
B=0.5*Delta
Vm=1
disorder_values=[0,0.5*(Delta-B)]
disorder_config=[[0 for x in range(Nx)],[uniform(-1,1) for x in range(Nx)]]
period=np.pi/B
E=0

E_values=np.linspace(-np.pi,np.pi,25)
x_values=np.linspace(0,1,51)*Nx
localiser_gap_values=np.zeros(len(x_values))

# plt.figure()
# for disorder in disorder_values:
#     for x_indx,x in enumerate(tqdm(x_values)):
#         localiser_gap_values[x_indx]=topological_hamiltonian_localiser_gap(x, E, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config)
#     plt.plot(x_values,localiser_gap_values,"x-",label=r"$W={:.2f}(\Delta-B)$".format(disorder/(Delta-B)))
# plt.xlabel(r"$x$")
# plt.ylabel(r"$\Delta L_{x,E=0}$")
# plt.legend()
# plt.ylim(bottom=0)
# plt.xlim(left=x_values[0],right=x_values[-1])
# plt.axvline(x=0,color="black",linestyle="dashed")
# plt.axvline(x=Nx-1,color="black",linestyle="dashed")


localiser_gap_values=np.zeros((len(E_values),len(x_values),2))
for disorder_indx,disorder in enumerate(disorder_values):
    
    
    for x_indx,x in enumerate(tqdm(x_values)):
        for E_indx,E in enumerate(E_values):
            localiser_gap_values[E_indx,x_indx,disorder_indx]=topological_hamiltonian_localiser_gap(x, E, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config[disorder_indx])
            
    plt.figure()
    sns.heatmap(localiser_gap_values[:,:,disorder_indx],cmap="viridis",vmin=0)
    plt.gca().invert_yaxis()
    plt.xlabel("$x$")
    plt.ylabel("$\epsilon T/\pi$")
    
    
# x=0
# Vm_values=np.linspace(0,6,51)
# localiser_gap_values=np.zeros((len(E_values),len(Vm_values)))
# disorder=0

# for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
#     for E_indx,E in enumerate(E_values):
#         localiser_gap_values[E_indx,Vm_indx]=topological_hamiltonian_localiser_gap(x, E, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder)
        
# plt.figure()
# sns.heatmap(localiser_gap_values,cmap="viridis",vmin=0)
# plt.gca().invert_yaxis()
# plt.xlabel("$V_m$")
# plt.ylabel(r"$E\tau/\pi$")


