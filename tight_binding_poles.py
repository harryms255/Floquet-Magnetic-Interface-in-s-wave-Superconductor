# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:47:01 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from functions_file import *
plt.close("all")
plt.rcParams.update({'font.size': 15})


Delta=0.1
B=1.5*Delta
omega=0
mu_values=np.linspace(-4,4,1001)

fig,axs=plt.subplots(2,4)

for m in range(2):
    sigma=(-1)**m
    column=0
    for i in range(2):
        pm1=(-1)**i
        for j in range(2):
            pm2=(-1)**j
            ax=axs[m,column]
            pole_values=tight_binding_poles(omega,mu_values,Delta,B,sigma,pm1,pm2)
            
            ax.plot(mu_values,np.sign(1-abs(pole_values)),"b")
            ax.set_title("$\sigma={},\pm_1={},\pm_2={}$".format(sigma,pm1,pm2))
            ax.axhline(y=1,color="black",linestyle="dashed")
            column+=1
            
plt.tight_layout()