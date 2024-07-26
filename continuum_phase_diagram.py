# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:25:26 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")
kf=0.65
Delta=0.1
B=Delta*0.9
theta=np.pi*0.3
km=1

Cm_crit=continuum_phase_boundaries(kf, km, Delta, 0, np.pi/2, 1)

km_values=np.linspace(0,2,251)
Cm_values=np.linspace(-5,5,251)
B_values=np.linspace(-Delta,Delta,501)
theta_values=np.linspace(0,2*np.pi,501)

invariant_values=np.zeros((len(B_values),len(theta_values)))

Cm_values=[0.25*Cm_crit,0.5*Cm_crit,0.75*Cm_crit,Cm_crit]

fig,axs=plt.subplots(2,2)
fig_placement=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
for Cm_indx,Cm in enumerate(Cm_values):  
    ax=fig_placement[Cm_indx]
    for B_indx,B in enumerate(tqdm(B_values)):
        for theta_indx,theta in enumerate(theta_values):
            parameters=[kf,km,Delta,B,Cm,theta]
            invariant_values[B_indx,theta_indx]=pfaffian_invariant(continuum_topological_Hamiltonian,parameters,1,TB=False)
    
    sns.heatmap(invariant_values,cmap="viridis",vmax=1,vmin=-1,ax=ax)
    ax.invert_yaxis()
    
    theta,B=np.meshgrid(theta_values,B_values)
    phase_boundaries_1=continuum_phase_boundaries_numpy(kf, km, Delta, B, Cm, theta, 1)
    ax.contour(phase_boundaries_1,levels=[0],colors="black",linestyles="dashed",linewidths=5)
    phase_boundaries_2=continuum_phase_boundaries_numpy(kf, km, Delta, B, Cm, theta, -1)
    ax.contour(phase_boundaries_2,levels=[0],colors="black",linestyles="dashed",linewidths=5)
    
    
    # x_ticks=[i*len(km_values)/10 for i in range(11)]      
    # x_labels=[str(i/10*max(km_values)) for i in range(11)]
    # y_ticks=[i*len(Cm_values)/6 for i in range(7)]      
    # y_labels=[str(np.round(np.min(Cm_values)+i/6*(max(Cm_values)-min(Cm_values)),2)) for i in range(7)]
        
    # plt.yticks(ticks=y_ticks,labels=y_labels)
    # plt.xticks(ticks=x_ticks,labels=x_labels)
    
    # plt.ylabel("$V_m/t$")
    # plt.xlabel("$k_m/k_F$")
    
    x_ticks=[i*len(theta_values)/max(theta_values/(2*np.pi/4)) for i in range(5)]      
    x_labels=[str(i/4*max(theta_values)/np.pi) for i in range(5)]
    y_ticks=[i*len(B_values)/4 for i in range(5)]      
    y_labels=[str(np.round(np.min(B_values)/Delta+i/4*(max(B_values)-min(B_values))/Delta,2)) for i in range(5)]
        
    ax.set_yticks(ticks=y_ticks,labels=y_labels)
    ax.set_xticks(ticks=x_ticks,labels=x_labels)
    
    ax.set_ylabel("$B/\Delta$")
    ax.set_xlabel(r"$\theta/\pi$")
    ax.set_title("$C_m={:.2f}C_m^*$".format(Cm/Cm_crit))
plt.tight_layout()