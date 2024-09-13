# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:03:01 2024

@author: Harry MullineauxSanders
"""

from functions_file import *

#plt.close("all")

Nx=250
Ny=101
Nev=20
t=1
mu=-3.6
km=np.arccos(-mu/2-1)
Delta=0.1
B=0.25*Delta
theta_values=[0.5*np.pi,0.3*np.pi]
sparse=True

Vm_values=np.linspace(0,6,251)
B_values=np.linspace(-Delta,Delta,101)

if sparse==True:
    x_position=np.zeros((Nev,len(Vm_values)))
    spectrum=np.zeros((Nev,len(Vm_values)))
else:
    x_position=np.zeros((4*Nx*Ny,len(Vm_values)))
    spectrum=np.zeros((4*Nx*Ny,len(Vm_values)))

fig,axs=plt.subplots(1,2,figsize=[15,8])
for theta_indx,theta in enumerate(theta_values):
    ax=axs[theta_indx]
    for Vm_indx,Vm in enumerate(tqdm(Vm_values)):
        parameters=[t,mu,Delta,km,B,Vm,theta]
        if sparse==True:
            spectrum[:,Vm_indx],x_position[:,Vm_indx]=sparse_real_space_spectrum(Nx, Ny, Nev, t, mu, Delta, km, B, Vm, theta)
            #spectrum[:,Vm_indx]=sparse_real_space_spectrum(Nx, Ny, Nev, t, mu, Delta, km, B, Vm, theta)
        else:
            spectrum[:,Vm_indx],x_position[:,Vm_indx]=real_space_spectrum(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
        
    
    
    
    if sparse==True:
        for i in range(Nev):
            sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position),cmap="plasma")
            #sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c="blue")
    else:
        for i in range(4*Nx*Ny):
            sc=ax.scatter(Vm_values,spectrum[i,:]/Delta,c=x_position[i,:],vmax=np.max(x_position),cmap="plasma")
    
    if theta_indx==1:
        cbar=plt.colorbar(sc)
        cbar.ax.get_yaxis().labelpad = 40
        cbar.ax.set_ylabel("$\sqrt{<x^2>}$", rotation=270)
    ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1),linewidth=5,linestyle="dashed",color="black")
    ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1),linewidth=5,linestyle="dashed",color="black")
    ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
    ax.axvline(x=TB_phase_boundaries(t, mu, Delta, km, B, theta, -1,kx=np.pi),linewidth=5,linestyle="dashed",color="black")
    ax.set_xlabel("$V_m/t$")
    ax.set_ylabel("$E/\Delta$")
    #ax.set_ylim(bottom=-5,top=5)
    ax.set_xlim(left=0,right=6)
    ax.set_title(r"$\theta={:.1f}\pi$".format(theta/np.pi))
plt.tight_layout()
plt.savefig("real_space_spectrum_Nx_Ny_250_101_kf=km=0.65_Delta=0.1_B=0.25Delta.png")