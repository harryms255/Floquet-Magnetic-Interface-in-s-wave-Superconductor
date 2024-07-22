# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:06:20 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

kf=0.65
km=kf
Delta=0.1
B=0.1*Delta*0
Cm=Delta**(1/2)
theta=np.pi*0

if __name__ == "__main__":

    cbar=[plt.cm.plasma(x) for x in np.linspace(0,1,6)]
    Cm_values=np.linspace(0.1,1,6)
    plt.figure()
    for Cm_indx,Cm in enumerate(tqdm(Cm_values)):
    
        kx_values=np.linspace(-1,1,251)
        band_structure=np.zeros((2,len(kx_values)))
        
        for kx_indx,kx in enumerate(kx_values):
            band_structure[:,kx_indx]=in_gap_band_structure(kx,kf,km,Delta,B,Cm,theta)
            
       
        for i in range(2):
            if i==0:
                plt.plot(kx_values,band_structure[i,:],c=cbar[Cm_indx],label="$C_m={:.2f}$".format(Cm))
            else:
                plt.plot(kx_values,band_structure[i,:],c=cbar[Cm_indx])

    plt.legend()
    plt.xlabel("$k_x/k_F$")
    plt.ylabel("$\omega$") 
    plt.ylim(top=2*Delta,bottom=-2*Delta)
    
    
    # kx_values=np.linspace(-5,5,101)
    # omega_values=np.linspace(-0.5*Delta,0.5*Delta,101)
    # det_values=np.zeros((len(omega_values),len(kx_values)))
    # for kx_indx,kx in enumerate(tqdm(kx_values)):
    #     for omega_indx,omega in enumerate(omega_values):
    #         det_values[omega_indx,kx_indx]=np.linalg.det(np.linalg.inv(T_matrix(omega, kx, kf, m, km, Delta, B, Vm, theta)))
            
    # plt.figure()
    # sns.heatmap(det_values,cmap="plasma")
