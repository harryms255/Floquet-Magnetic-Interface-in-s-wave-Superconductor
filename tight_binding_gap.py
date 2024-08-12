# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:52:11 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

Ny=101
t=1
mu=-3.6
km=0.65
Delta=0.1
Vm=TB_phase_boundaries(t, mu, Delta, km, 0, np.pi/2, 1)

theta_values=np.linspace(0,2*np.pi,251)
B_values=np.linspace(-Delta,Delta,251)

gap_values=np.loadtxt("tight_binding_gap.txt")
B_indx_init=221

#for B_indx,B in enumerate(tqdm(B_values)):
for B_indx in tqdm(range(B_indx_init,len(B_values))):
    B=B_values[B_indx]
    for theta_indx,theta in enumerate(theta_values):
        gap_values[B_indx,theta_indx]=TB_gap(Ny,t, mu, Delta, km, B, Vm, theta)
        #np.savetxt("tight_binding_gap.txt",gap_values)
        
plt.figure(figsize=[12,8])
sns.heatmap(gap_values/Delta,cmap="viridis",vmin=0,cbar_kws={"label": r"$Gap/\Delta$"})
plt.gca().invert_yaxis()

theta,B=np.meshgrid(theta_values,B_values)

phase_boundaries_1=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1)
phase_boundaries_2=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1)
phase_boundaries_3=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1,kx=np.pi)
phase_boundaries_4=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1,kx=np.pi)
plt.contour(phase_boundaries_1,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_2,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_3,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_4,levels=[0],linestyles="dashed",linewidths=5,colors="black")

x_ticks=[i*len(theta_values)/max(theta_values/(2*np.pi/4)) for i in range(5)]      
x_labels=[str(i/4*max(theta_values)/np.pi) for i in range(5)]
y_ticks=[i*len(B_values)/4 for i in range(5)]      
y_labels=[str(np.round(np.min(B_values)/Delta+i/4*(max(B_values)-min(B_values))/Delta,2)) for i in range(5)]
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)

plt.ylabel("$B/\Delta$")
plt.xlabel(r"$\theta/\pi$")

