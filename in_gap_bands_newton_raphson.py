# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:43:47 2024

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
Vm=0.5*TB_phase_boundaries(t, mu, Delta, km, B, theta, 1)
kx=0.2*np.pi

parameters=[kx, t, mu, Delta, km, B, Vm, theta]

omega_values=np.linspace(-0.99*Delta,0.99*Delta,1001)

pole_condition_values=np.zeros(len(omega_values))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    pole_condition_values[omega_indx]=pole_condition(omega, kx, t, mu, Delta, km, B, Vm, theta)
    
plt.figure()
plt.plot(omega_values,pole_condition_values)
plt.axhline(y=0,linestyle="dashed",color="black")
plt.ylim(top=100,bottom=-100)

omega_0=-0.99*(Delta-B)

NR_steps=np.zeros(100)
NR_det_values=np.zeros(len(NR_steps))

for i in tqdm(range(len(NR_steps))):
    if i==0:
        NR_steps[i]=omega_0
        NR_det_values[i]=pole_condition(NR_steps[i], kx, t, mu, Delta, km, B, Vm, theta)
    else:
        x0=NR_steps[i-1]
        NR_steps[i]=Newton_Raphson_update(x0, pole_condition,parameters)
        NR_det_values[i]=pole_condition(NR_steps[i], kx, t, mu, Delta, km, B, Vm, theta)
    
plt.plot(NR_steps,NR_det_values,"r")