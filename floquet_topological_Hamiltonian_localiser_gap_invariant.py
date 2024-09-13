# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:45:04 2024

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
B=0.1*Delta
Vm_values=[0.5,1,1.5]
disorder_values=[0,0.5*(Delta-B)]
disorder_config=[[0 for x in range(Nx)],[uniform(-1,1) for x in range(Nx)]]

# x_values=np.linspace(-10,Nx+9,50)
# fig,ax=plt.subplots(1,1,num="Floquet Spectral Localiser Invariant with disorder")
# for Vm in Vm_values:

#     localiser_gap_invariant_values=np.zeros(len(x_values))
    
#     for x_indx,x in enumerate(tqdm(x_values)):
#         localiser_gap_invariant_values[x_indx]=topological_hamiltonian_class_D_invariant(x, Nx, t, mu, Delta, km, B, Vm, theta)
    
    
#     ax.plot(x_values,localiser_gap_invariant_values,"-x",label=r"$V_m={:.1f}t$".format(Vm))
# ax.set_xlabel("$x$")
# ax.set_ylabel(r"$\nu(x)$")
# ax.set_xlim(left=min(x_values),right=max(x_values))
# ax.axvline(x=0,linestyle="dashed",color="black")
# ax.axvline(x=Nx-1,linestyle="dashed",color="black")
# ax.legend()


x=Nx//2
Vm_values=np.linspace(0,6,25)
km_values=np.linspace(0,np.pi,25)
invariant_values=np.zeros((len(Vm_values),len(km_values),2))
for disorder_indx,disorder in enumerate(disorder_values):

    
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        for km_indx,km in enumerate(km_values):
            invariant_values[Vm_indx,km_indx,disorder_indx]=topological_hamiltonian_class_D_invariant(x, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config[disorder_indx])
            
            
    plt.figure()
    sns.heatmap(invariant_values,cmap="viridis",vmin=-1,vmax=1)
    x_ticks=[]
    x_labels=[]
    y_ticks=[]
    y_labels=[]
    
    for i in range(5):
        y_ticks.append(i/4*len(Vm_values))
        y_labels.append(str(np.round(i/4*max(Vm_values),2)))
        
    for i in range(5):
        x_ticks.append(i/4*len(km_values))
        x_labels.append(str(np.round(i/4*max(km_values)/np.pi,2)))
        
    plt.yticks(ticks=y_ticks,labels=y_labels)
    plt.xticks(ticks=x_ticks,labels=x_labels)
    
    plt.ylabel("$V_m/t$")
    plt.xlabel("$k_m/\pi$")
    plt.gca().invert_yaxis()
    
    
    km,Vm=np.meshgrid(km_values,Vm_values)
    phase_boundaries_1=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1)
    phase_boundaries_2=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1,kx=np.pi)
    plt.contour(phase_boundaries_1,levels=[0],linestyles="dashed",linewidths=5,colors="black")
    plt.contour(phase_boundaries_2,levels=[0],linestyles="dashed",linewidths=5,colors="black")
