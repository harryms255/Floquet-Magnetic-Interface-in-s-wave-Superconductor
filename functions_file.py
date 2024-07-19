# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:28:42 2024

@author: Harry MullineauxSanders
"""
#Set of functions to calculate the electronic structure of a spiral magnetic in face rotating at a constant frequency embedded into a s-wave superconductor

import numpy as np
import scipy.linalg as sl
from pfapack import pfaffian as pf
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


#Tight Binding-----------------------------------------------------------------
def kx_tight_binding_Hamiltonian(k,T,Ny,t,mu,Delta,Vm,km,omega):
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



#Greens Function---------------------------------------------------------------

def poles(omega,kx,kf,m,km,Delta,B,sigma,pm):
    p=np.sqrt(1-(kx+sigma*km)**2+pm*1j*np.sqrt(Delta**2-(omega+sigma*B)**2))
    
    return p

def sigma_substrate_Greens_function(omega,kx,y,kf,m,km,Delta,B,sigma):
    
    omega+=0.000001j
    
    p_plus=poles(omega,kx,kf,m,km,Delta,B,sigma,1)
    p_min=poles(omega,kx,kf,m,km,Delta,B,sigma,-1)
    g_sigma=np.zeros((2,2),dtype=complex)
    
    g_sigma+=np.e**(1j*kf*p_plus*abs(y))/p_plus*np.array(([omega+sigma*B+kx**2+p_plus**2-1,sigma*Delta],[sigma*Delta,omega+sigma*B+1-kx**2-p_plus**2]))
    
    g_sigma+=np.e**(-1j*kf*p_min*abs(y))/p_min*np.array(([omega+sigma*B+kx**2+p_min**2-1,sigma*Delta],[sigma*Delta,omega+sigma*B+1-kx**2-p_min**2]))
    
    g_sigma*=-1j*m/(2*kf*np.sqrt((omega+sigma*B)**2-Delta**2))
    
    return g_sigma

def substrate_Greens_function(omega,kx,y,kf,m,km,Delta,B):
    g=np.zeros((4,4),dtype=complex)
    
    g_up=sigma_substrate_Greens_function(omega,kx,y,kf,m,km,Delta,B,1)
    g_down=sigma_substrate_Greens_function(omega,kx,y,kf,m,km,Delta,B,-1)
    
    g[0,0]=g_up[0,0]
    g[0,3]=g_up[0,1]
    g[3,0]=g_up[1,0]
    g[3,3]=g_up[1,1]
    
    g[1:3,1:3]=g_down
    
    return g

def T_matrix(omega,kx,kf,m,km,Delta,B,Vm,theta):
    Hm=Vm*np.kron(np.array(([1,0],[0,-1])),np.array(([np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)])))
    if Vm==0:
        T=np.zeros((4,4))
    else:
        
        T_inv=np.linalg.inv(Hm)-substrate_Greens_function(omega, kx, 0, kf, m, km, Delta, B)
        
        T=np.linalg.inv(T_inv)
    
    return T

def Greens_function(omega,kx,y1,y2,kf,m,km,Delta,B,Vm,theta):
    
    g_y1y2=substrate_Greens_function(omega, kx, y1-y2, kf, m, km, Delta, B)
    g_y1=substrate_Greens_function(omega, kx, y1, kf, m, km, Delta, B)
    g_y2=substrate_Greens_function(omega, kx, -y2, kf, m, km, Delta, B)
    T=T_matrix(omega, kx, kf, m, km, Delta, B, Vm, theta)
    
    G=g_y1y2+g_y1@T@g_y2
    
    return G

#Electronic Structure----------------------------------------------------------

def LDOS(omega,kx,y,kf,m,km,Delta,B,Vm,theta):
    G=Greens_function(omega,kx,y,y,kf,m,km,Delta,B,Vm,theta)
    
    LDOS=-1/np.pi*np.imag(np.trace(G))
    
    return LDOS

def DOS(omega,kx,y,kf,m,km,Delta,B,Vm,theta):
    
    y_values=np.linspace(-50/kf,50/kf,10001)
    
    DOS=0
    for y in y_values:
        DOS+=LDOS(omega, kx, y, kf, m, km, Delta, B, Vm, theta)
        
    return DOS

def in_gap_band_structure(kx,kf,m,km,Delta,B,Vm,theta):
    effective_gap=abs(Delta)-abs(B)
    
    pole_condition=lambda omega:1/np.linalg.det(T_matrix(omega, kx, kf, m, km, Delta, B, Vm, theta))
    
    positive_energy_mode=fsolve(pole_condition,x0=0.9*effective_gap)
    negative_enery_mode=fsolve(pole_condition,x0=-0.9*effective_gap)
    
    return negative_enery_mode,positive_energy_mode
    
    
    
    
    
#Topological Properties--------------------------------------------------------    
    
    
    
def pfaffian_invariant(Hamiltonian,parameters,system_size):
    H=lambda k:Hamiltonian(H,*parameters)
    
    U=np.kron(np.array(([1,-1j],[1,1j])),np.identity(2))
    U_tot=np.kron(np.identity(system_size),U)
    
    H_majorana_0=np.conj(U_tot.T)@H(0)@U_tot
    H_majorana_pi=np.conj(U_tot.T)@H(np.pi)@U_tot
    
    invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)*pf.pfaffian(H_majorana_pi)))
    
    return invariant

    
    
    
    
    
    
    
    
    
    
    
