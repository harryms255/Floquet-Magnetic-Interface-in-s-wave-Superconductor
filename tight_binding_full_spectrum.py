# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:58:37 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

plt.close("all")

Ny=251
t=1
mu=-3.6
km=0.65
Delta=0.1
B=0.5*Delta
theta=np.pi/3
Vm_crit=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=0)

kx_values=np.linspace(-np.pi,np.pi,1001)

spectrum=np.zeros((4*Ny,len(kx_values)))

# for kx_indx,kx in enumerate(tqdm(kx_values)):
#     spectrum[:,kx_indx]=np.linalg.eigvalsh(static_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta))

# plt.figure(figsize=[12,8])
# for i in range(4*Ny):
#     plt.plot(kx_values/np.pi, spectrum[i,:]/Delta,c="k")
# plt.ylim(top=2,bottom=-2)
# plt.xlabel("$k_x/\pi$")
# plt.ylabel("$E/\Delta$")
# plt.tight_layout()


Vm_values=[0.7*Vm_crit,0.8*Vm_crit,0.9*Vm_crit,Vm_crit,1.1*Vm_crit,1.2*Vm_crit]
omega_values=np.linspace(-2*Delta,2*Delta,1001)
kx_values=np.linspace(-0.5*np.pi,0.5*np.pi,1001)


fig,axs=plt.subplots(2,3)

figure_placement=[axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]

for Vm_indx,Vm in enumerate(Vm_values):
    kx_values=np.linspace(-0.5*np.pi,0.5*np.pi,1001)

    spectrum=np.zeros((4*Ny,len(kx_values)))

    for kx_indx,kx in enumerate(tqdm(kx_values)):
        spectrum[:,kx_indx]=np.linalg.eigvalsh(static_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta))
    
    
    ax=figure_placement[Vm_indx]  
    for i in range(4*Ny):
        ax.plot(kx_values/np.pi, spectrum[i,:]/Delta,c="k")
    ax.invert_yaxis()
    ax.set_title(r"$V_m={:.2f}V_m^*$".format(Vm/Vm_crit))
    ax.set_xlabel("$k_x/\pi$")
    ax.set_ylabel("$\omega/\Delta$")
    ax.set_ylim(top=2,bottom=-2)
plt.tight_layout()