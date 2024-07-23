# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:54:28 2024

@author: Harry MullineauxSanders
"""

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pfapack import pfaffian as pf


def partially_FT_spiral_chain_SC(k,T,Ny,t,mu,Delta,Vm,km,omega):
    """
    Tight Binding Hamiltonian of a 2D s-wave superconductor with a chain of magnetic 
    moments at y=Ny//2, spiralling in the xy plane with wavevector km
    
    The chosen basis is spin x nambu x space so the ordering is 
    (y1- particle-upspin, y1-particle-downspin,y1- hole-upspin, y1-hole-downspin,y2- particle-upspin, y2-particle-downspin,y2- hole-upspin, y2-hole-downspin)

    Parameters
    ----------
    k : float in [a,a+2pi] for some float a
        Momentum along x direction. Between -pi and pi mod(2pi)
    Ny : Integer
        System size in non-translationally invariant direction
    t : float
        Hopping integral in the x and y direction, generically set to 1
    mu : float
        Chemical potentiel
    Delta : array of floats
        S-Wave Superconductor pairing at each y site
    Vm : Float
        Magnetic scattering strength
    km : float in [0,2pi]
        Wavevector of spiral of magnetic moments.

    Returns
    -------
    H: Float64 Numpy Array
        Matrix representation of Bloch Hamiltonian

    """

    H=np.zeros((4*Ny,4*Ny),dtype=float)
    
    #y hopping terms
    for y in range(Ny):
        #up spin particle
        H[4*y,4*((y+1)%Ny)]=-t
        #down spin particle
        H[4*y+1,4*((y+1)%Ny)+1]=-t
        #up spin hole
        H[4*y+2,4*((y+1)%Ny)+2]=t
        #down spin hole
        H[4*y+3,4*((y+1)%Ny)+3]=t
        
        
    #S-Wave Pairing Terms
    for y in range(Ny):
        H[4*y,4*y+3]=Delta[y]
        H[4*y+1,4*y+2]=-Delta[y]
        #H[4*y+1,4*y+2]=-Delta[y]
    
    #Magnetic Scattering Terms
    
    y=Ny//2
    
    H[4*y,4*y+1]=Vm*np.e**(1j*omega*T)
    H[4*y+2,4*y+3]=-Vm*np.e**(-1j*omega*T)
    
    #Hamiltonian is made Hermitian
    H+=np.matrix.getH(H)
    
    #Onsite terms
    for y in range(Ny):
        H[4*y,4*y]=-2*t*np.cos(k+km)-mu
        H[4*y+1,4*y+1]=-2*t*np.cos(k-km)-mu
        H[4*y+2,4*y+2]=2*t*np.cos(-k+km)+mu
        H[4*y+3,4*y+3]=2*t*np.cos(-k-km)+mu
    
    return H

def floquet_Hamiltonian(k,Hamiltonian,parameters,omega):
    H=lambda k,t:Hamiltonian(k,t,*parameters)
    period=2*np.pi/omega
    t_values=np.linspace(0,period,1001)
    
    H_eff=lambda k:sum(H(k,t_values))/period
    
    U=lambda k:sl.expm(1j*H_eff(k)*period)
    
    return -1j*period*sl.logm(U(k))

def pfaffian_invariant(Hamiltonian,parameters,system_size):
    H=lambda k:Hamiltonian(H,*parameters)
    
    U=np.kron(np.array(([1,-1j],[1,1j])),np.identity(2))
    U_tot=np.kron(np.identity(system_size),U)
    
    H_majorana_0=np.conj(U_tot.T)@H(0)@U_tot
    H_majorana_pi=np.conj(U_tot.T)@H(np.pi)@U_tot
    
    invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)*pf.pfaffian(H_majorana_0)))
    
    return invariant
    
    
    