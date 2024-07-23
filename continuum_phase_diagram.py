# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:25:26 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")
kf=0.65
Delta=0.1
B=Delta*0.75
theta=np.pi/2
km=1

km_values=np.linspace(0,2,51)
Cm_values=np.linspace(-2,2,51)
B_values=np.linspace(0,Delta,51)
theta_values=np.linspace(0,np.pi,51)

invariant_values=np.zeros((len(Cm_values),len(km_values)))


for Cm_indx,Cm in enumerate(tqdm(Cm_values)):
    for km_indx,km in enumerate(km_values):
    #for B_indx,B in enumerate(B_values):
    #for theta_indx,theta in enumerate(theta_values):
        parameters=[kf,km,Delta,B,Cm,theta]
        invariant_values[Cm_indx,km_indx]=pfaffian_invariant(continuum_topological_Hamiltonian,parameters,1,TB=False)[0]
plt.figure()
sns.heatmap(invariant_values,cmap="viridis",vmax=1,vmin=-1)
plt.gca().invert_yaxis()

km,Cm=np.meshgrid(km_values,Cm_values)
phase_boundaries_1=continuum_phase_boundaries_numpy(kf, km, Delta, B, Cm, theta, 1)
plt.contour(phase_boundaries_1,levels=[0],colors="black",linestyles="dashed",linewidths=5)
phase_boundaries_2=continuum_phase_boundaries_numpy(kf, km, Delta, B, Cm, theta, -1)
plt.contour(phase_boundaries_2,levels=[0],colors="black",linestyles="dashed",linewidths=5)


# gapless_values=np.zeros((len(Cm_values),len(km_values)))
# for Cm_indx,Cm in enumerate(tqdm(Cm_values)):
#     for km_indx,km in enumerate(km_values):
#         gapless_values[Cm_indx,km_indx]=gapless(kf, km, Delta, B, Cm, theta)
        
# plt.figure()
# sns.heatmap(gapless_values,cmap="viridis",vmax=10)
# plt.gca().invert_yaxis()


# plt.contour(phase_boundaries_1,levels=[0],colors="black",linestyles="dashed",linewidths=5)
# plt.contour(phase_boundaries_2,levels=[0],colors="black",linestyles="dashed",linewidths=5)