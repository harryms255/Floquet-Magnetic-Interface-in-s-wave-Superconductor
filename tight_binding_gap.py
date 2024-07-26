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
B=0.9*Delta
theta=0

Vm_values=np.linspace(-6,6,501)
km_values=np.linspace(0,np.pi,501)
theta_values=np.linspace(0,np.pi/2,501)

gap_values=np.zeros((len(Vm_values),len(km_values)))

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    #for km_indx,km in enumerate(km_values):
    for theta_indx,theta in enumerate(theta_values):
        gap_values[Vm_indx,theta_indx]=TB_gap(Ny,t, mu, Delta, km, B, Vm, theta)
        np.savetxt("tight_binding_gap.txt",gap_values)
        
plt.figure(figsize=[12,8])
sns.heatmap(gap_values,cmap="viridis",vmax=1,vmin=0)
plt.gca().invert_yaxis()

theta,Vm=np.meshgrid(theta_values,Vm_values)

phase_boundaries_1=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1)
phase_boundaries_2=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1)
phase_boundaries_3=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, 1,kx=np.pi)
phase_boundaries_4=TB_phase_boundaries_numpy(t, mu, Delta, km, B, Vm, theta, -1,kx=np.pi)
plt.contour(phase_boundaries_1,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_2,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_3,levels=[0],linestyles="dashed",linewidths=5,colors="black")
plt.contour(phase_boundaries_4,levels=[0],linestyles="dashed",linewidths=5,colors="black")

x_ticks=[i*len(km_values)/max(km_values/(np.pi/10)) for i in range(11)]      
x_labels=[str(i/10) for i in range(int(max(km_values)/(np.pi/10))+1)]
y_ticks=[i*len(Vm_values)/6 for i in range(7)]      
y_labels=[str(np.round(np.min(Vm_values)+i/6*(max(Vm_values)-min(Vm_values)),2)) for i in range(7)]
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)

plt.ylabel("$V_m/t$")
plt.xlabel("$k_m/\pi$")