# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:23:21 2024

@author: Harry MullineauxSanders
"""

from functions_file import *
plt.close("all")

Nx=11
Ny=11
Nev=20
t=1
mu=-3.6
km=0.65
Delta=1
B=0.5*Delta
theta=np.pi/2

Vm=(TB_phase_boundaries(t, mu, Delta, km, B, theta, 1,kx=np.pi)+TB_phase_boundaries(t, mu, Delta, km, B, theta, 1))/2

x_values=np.linspace(0,Nx-1,Nx)
y_values=np.linspace(0,Ny-1,Ny)

spectral_localiser_values=np.zeros((len(y_values),len(x_values)))

for x_indx, x in enumerate(tqdm(x_values)):
    for y_indx,y in enumerate(y_values):
        spectral_localiser_values[y_indx,x_indx]=localiser_gap(x, y, 0, Nx, Ny, t, mu, Delta, km, B, Vm, theta)
