# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:57:33 2024

@author: Harry MullineauxSanders
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({'font.size': 15})
def poles(omega,kx,B,Delta,sigma,pm1,pm2):
    omega+=0.00001j
    pole=pm2*np.sqrt(1-kx**2+pm1*1j*np.sqrt(Delta**2-(omega+sigma*B)**2))
    return pole

Delta=1
B=0.9*Delta
omega=0.5*(Delta-B)
kx_values=np.linspace(-10,10,1001)

fig,axs=plt.subplots(2,4)

for m in range(2):
    sigma=(-1)**m
    column=0
    for i in range(2):
        pm1=(-1)**i
        for j in range(2):
            pm2=(-1)**j
            ax=axs[m,column]
            pole_values=poles(omega,kx_values,B,Delta,sigma,pm1,pm2)
            
            ax.plot(kx_values,np.imag(pole_values),"b")
            ax.set_title("$\sigma={},\pm_1={},\pm_2={}$".format(sigma,pm1,pm2))
            ax.axhline(y=0,color="black",linestyle="dashed")
            column+=1
            
            
plt.tight_layout()