# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:35:45 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.1*Delta
theta=np.pi/2
y=0

Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=0)
Vm=Vm_crit
kx_values=np.linspace(-np.pi,np.pi,1001)
omega_values=np.linspace(-2*Delta,2*Delta,1001)
LDOS_values=np.zeros((len(omega_values),len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    for omega_indx,omega in enumerate(omega_values):
        LDOS_values[omega_indx,kx_indx]=TB_LDOS(omega, kx, y, t, mu, Delta, km, B, Vm, theta)

plt.figure()
sns.heatmap(LDOS_values,cmap="inferno",vmax=1)
plt.gca().invert_yaxis()

x_ticks=[]
x_labels=[]
y_ticks=[]
y_labels=[]

for i in range(11):
    y_ticks.append(i/10*len(omega_values))
    y_labels.append(str(np.round(min(omega_values)/Delta+i/10*(max(omega_values)-min(omega_values))/(Delta),2)))
    
for i in range(11):
    x_ticks.append(i/10*len(kx_values))
    x_labels.append(str(np.round(np.min(kx_values)/np.pi+i/10*(max(kx_values)-min(kx_values))/np.pi,2)))
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)
    
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.xticks(ticks=x_ticks,labels=x_labels)
plt.tight_layout()
plt.xlabel("$k_x/\pi$")
plt.ylabel("$\omega/\Delta$")