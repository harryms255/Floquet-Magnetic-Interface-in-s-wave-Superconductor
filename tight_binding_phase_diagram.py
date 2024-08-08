# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:00:13 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

t=1
mu=-3.6
Delta=0.1
km=0.65 #km=kf
B=0.1*Delta*0
theta=np.pi*0
y=0

B_values=np.linspace(-Delta,Delta,250)
theta_values=np.linspace(0,np.pi,251)

invariant_values=np.zeros((len(B_values),len(theta_values)))
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, 0, np.pi/2, 1)
Vm_values=[0.25*Vm_crit,0.5*Vm_crit,Vm_crit,1.25*Vm_crit]



fig,axs=plt.subplots(2,2,figsize=[12,12])
fig_placement=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
for Vm_indx,Vm in enumerate(Vm_values):  
    ax=fig_placement[Vm_indx]
    for B_indx,B in enumerate(tqdm(B_values)):
        for theta_indx,theta in enumerate(theta_values):
            parameters=[y, t, mu, Delta, km, B, Vm, theta]
            invariant_values[B_indx,theta_indx]=pfaffian_invariant(TB_topological_Hamiltonian , parameters,1,TB=True)
    
    sns.heatmap(invariant_values,cmap="viridis",vmax=1,vmin=-1,ax=ax)
    ax.invert_yaxis()
    
    theta,B=np.meshgrid(theta_values,B_values)
    
    phase_boundaries_1=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1)
    phase_boundaries_2=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1)
    phase_boundaries_3=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1,kx=np.pi)
    phase_boundaries_4=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1,kx=np.pi)
    ax.contour(phase_boundaries_1,levels=[0],linestyles="dashed",linewidths=5,colors="black")
    ax.contour(phase_boundaries_2,levels=[0],linestyles="dashed",linewidths=5,colors="black")
    ax.contour(phase_boundaries_3,levels=[0],linestyles="dashed",linewidths=5,colors="black")
    ax.contour(phase_boundaries_4,levels=[0],linestyles="dashed",linewidths=5,colors="black")
    
    x_ticks=[i*len(theta_values)/max(theta_values/(np.pi/4)) for i in range(5)]      
    x_labels=[str(i/4*max(theta_values)/np.pi) for i in range(5)]
    y_ticks=[i*len(B_values)/4 for i in range(5)]      
    y_labels=[str(np.round(np.min(B_values)/Delta+i/4*(max(B_values)-min(B_values))/Delta,2)) for i in range(5)]
        
    ax.set_yticks(ticks=y_ticks,labels=y_labels)
    ax.set_xticks(ticks=x_ticks,labels=x_labels)
    
    ax.set_ylabel("$B/\Delta$")
    ax.set_xlabel(r"$\theta/\pi$")
    ax.set_title("$V_m={:.2f}V_m^*$".format(Vm/Vm_crit))
plt.tight_layout()