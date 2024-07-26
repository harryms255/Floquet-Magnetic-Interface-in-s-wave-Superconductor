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
B=0.1*Delta
theta=np.pi*0
y=0
Vm=TB_phase_boundaries(t, mu, Delta, km, 0, theta, 1)*0.25

km_values=np.linspace(0,np.pi,251)
Vm_values=np.linspace(-6,6,251)
B_values=np.linspace(-1.5*Delta,1.5*Delta,250)
theta_values=np.linspace(0,2*np.pi,251)

invariant_values=np.zeros((len(B_values),len(theta_values)))

for B_indx,B in enumerate(tqdm(B_values)):
    for theta_indx,theta in enumerate(theta_values):
        parameters=[y, t, mu, Delta, km, B, Vm, theta]
        invariant_values[B_indx,theta_indx]=pfaffian_invariant(TB_topological_Hamiltonian , parameters,1,TB=True)

plt.figure()
sns.heatmap(invariant_values,cmap="viridis")
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

x_ticks=[i*len(theta_values)/max(theta_values/(2*np.pi/10)) for i in range(11)]      
x_labels=[str(i/10*max(theta_values)/np.pi) for i in range(11)]
y_ticks=[i*len(B_values)/8 for i in range(9)]      
y_labels=[str(np.round(np.min(B_values)/Delta+i/8*(max(B_values)-min(B_values))/Delta,2)) for i in range(9)]
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)

plt.ylabel("$B/\Delta$")
plt.xlabel(r"$\theta/\pi$")