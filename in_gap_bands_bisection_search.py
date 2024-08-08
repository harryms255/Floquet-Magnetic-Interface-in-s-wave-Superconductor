# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:47:04 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")
def pole_condition(omega,kx,t,mu,Delta,km,B,Vm,theta):
    return np.linalg.det(np.linalg.inv(TB_T_matrix(omega, kx, t, mu, Delta, km, B, Vm, theta)))

t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.5*Delta
theta=0.5*np.pi
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1)
Vm_values=[0.25*Vm_crit,0.5*Vm_crit,0.75*Vm_crit,Vm_crit]

kx_values=np.linspace(-np.pi,np.pi,1001)

cbar=plt.cm.viridis

fallback=Delta
plt.figure()
for Vm_indx,Vm in enumerate(Vm_values):
    band_solution=np.zeros(len(kx_values))
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        parameters=[kx,t,mu,Delta,km,B,Vm,theta]
        x0=0
        x1=0.999*(Delta+B)
        
            
        if kx_indx==0:
            band_solution[kx_indx]=bisection_search(pole_condition, parameters, x0, x1,fallback_solution=x1)
        else:
            band_solution[kx_indx]=bisection_search(pole_condition, parameters, x0, x1,fallback_solution=band_solution[kx_indx-1])


    plt.plot(kx_values/np.pi,band_solution/Delta,c=cbar(Vm_indx/(len(Vm_values)-1)),label="$V_m={:.2f}V_m^*$".format(Vm/Vm_crit))
    
    band_solution=np.zeros(len(kx_values))
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        parameters=[kx,t,mu,Delta,km,B,Vm,theta]
        
        x0=-0.999*(Delta+B)
        x1=0
        
        if kx_indx==0:
            band_solution[kx_indx]=bisection_search(pole_condition, parameters, x0, x1,fallback_solution=x0)
        else:
            band_solution[kx_indx]=bisection_search(pole_condition, parameters, x0, x1,fallback_solution=band_solution[kx_indx-1])


    plt.plot(kx_values/np.pi,band_solution/Delta,c=cbar(Vm_indx/(len(Vm_values)-1)))
    
plt.xlabel(r"$k_x/\pi$")
plt.ylabel(r"$\omega/\Delta$")
plt.legend()