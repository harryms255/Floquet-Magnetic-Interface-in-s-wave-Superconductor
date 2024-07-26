# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:32:19 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.5*Delta
theta=np.pi/2
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1)
Vm_values=[0.7*Vm_crit,0.8*Vm_crit,0.9*Vm_crit,Vm_crit,1.1*Vm_crit,1.2*Vm_crit,1.3*Vm_crit]

kx_values=np.linspace(-0.3*np.pi,0.3*np.pi,1001)
band_structure=np.zeros((2,len(kx_values)))

cbar=plt.cm.inferno

for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
    for kx_indx,kx in enumerate(kx_values):
        band_structure[:,kx_indx]=TB_in_gap_band_structure(kx, t, mu, Delta, km, B, Vm, theta)
            
       
    for i in range(2):
        if i==0:
            plt.plot(kx_values/np.pi,band_structure[i,:],c=cbar(Vm_indx/(len(Vm_values)-1)),label="$V_m={:.2f}V_m^*$".format(Vm/Vm_crit))
        else:
            plt.plot(kx_values/np.pi,band_structure[i,:],c=cbar(Vm_indx/(len(Vm_values)-1)))
    
plt.legend()
plt.xlabel("$k_x/\pi$")
plt.ylabel("$\omega$") 
plt.ylim(top=Delta,bottom=-Delta)