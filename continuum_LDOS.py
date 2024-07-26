# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:00:31 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

kf=0.65
km=1
Delta=0.1
B=0.5*Delta
y=0
theta=np.pi/2

Cm_crit=continuum_phase_boundaries(kf, km, Delta, B, theta, 1)
Cm_values=[0.7*Cm_crit,0.8*Cm_crit,0.9*Cm_crit,Cm_crit,1.1*Cm_crit,1.2*Cm_crit]
B_values=np.linspace(0,Delta,11)
theta_values=np.linspace(0,np.pi/2,11)


omega_values=np.linspace(-2*Delta,2*Delta,501)
kx_values=np.linspace(-1,1,501)
LDOS_values=np.zeros((len(omega_values),len(kx_values)))

fig,axs=plt.subplots(2,3)

figure_placement=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]

for Cm_indx,Cm in enumerate(Cm_values):
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        for omega_indx,omega in enumerate(omega_values):
            LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, y, kf, km, Delta, B, Cm, theta)
            
    ax=figure_placement[Cm_indx]  
    sns.heatmap(LDOS_values,cmap="inferno",ax=ax,cbar=False,vmax=10)
    ax.invert_yaxis()
    ax.set_title(r"$C_m={:.2f}C_m^*$".format(Cm/Cm_crit))
    x_ticks=[]
    x_labels=[]
    y_ticks=[]
    y_labels=[]
    
    for i in range(5):
        y_ticks.append(i/4*len(omega_values))
        y_labels.append(str(np.round(min(omega_values)/Delta+i/4*(max(omega_values)-min(omega_values))/(Delta),2)))
        
    for i in range(5):
        x_ticks.append(i/4*len(kx_values))
        x_labels.append(str(np.round(np.min(kx_values)+i/4*(max(kx_values)-min(kx_values)),2)))
    ax.set_xticks(x_ticks,labels=x_labels)
    ax.set_yticks(y_ticks,labels=y_labels)
    ax.set_xlabel("$k_x/k_F$")
    ax.set_ylabel("$\omega/\Delta$")
plt.tight_layout()