# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:28:42 2024

@author: Harry MullineauxSanders
"""

"""
Set of functions to calculate the electronic structure of a spiral magnetic in face rotating at a constant frequency embedded into a s-wave superconductor

Parameters used in functions

Continuum Model

omega: Float, energy in Green's function
kx: Float, momentum along the interface, in units of kf
kf: Float, fermi momentum
km: Float, spiral pitch of interface, in units of kf
Delta: Float, superconductor pairing, should be real
B: Float, effective zeeman field, should be less than Delta for a gapped system
sigma: +-1, index for the spin sectors
Cm: Float, unitless magnetic scattering
theta: Float between -pi and pi, out of plane angle of the spiral

Tight Binding Model

omega: Float, energy in Green's function
kx: Float between -pi and pi, momentum along the interface
t: Float, bandwidth, set to 1 to make everything unitless
mu: Float, chemical potential, in units of t
km: Float between - pi and pi, spiral pitch of interface
Delta: Float, superconductor pairing, should be real, in units of t
B: Float, effective zeeman field, should be less than Delta for a gapped system, in units of t
sigma: +-1, index for the spin sectors
Vm: Float, magnetic scattering, in units of t
theta: Float between -pi and pi, out of plane angle of the spiral


Function notes:
    
Any Greens function will always have its first 2 parameters be omega and kx
Any Hamiltonian will always have its first paremeter be kx


"""


import numpy as np
import scipy.linalg as sl
from pfapack import pfaffian as pf
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


#Tight Binding-----------------------------------------------------------------
def driven_tight_binding_Hamiltonian(kx,T,Ny,t,mu,Delta,Vm,km,omega):
    """
    Tight Binding Hamiltonian of a 2D s-wave superconductor with a chain of magnetic 
    moments at y=Ny//2, spiralling in the xy plane with wavevector km
    
    The chosen basis is spin x nambu x space so the ordering is 
    (y1- particle-upspin, y1-particle-downspin,y1- hole-upspin, y1-hole-downspin,y2- particle-upspin, y2-particle-downspin,y2- hole-upspin, y2-hole-downspin)

    Parameters
    ----------
    kx : float in [a,a+2pi] for some float a
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
        H[4*y,4*y]=-2*t*np.cos(kx+km)-mu
        H[4*y+1,4*y+1]=-2*t*np.cos(kx-km)-mu
        H[4*y+2,4*y+2]=2*t*np.cos(-kx+km)+mu
        H[4*y+3,4*y+3]=2*t*np.cos(-kx-km)+mu
    
    return H


def floquet_Hamiltonian(k,Hamiltonian,parameters,omega):
    H=lambda k,t:Hamiltonian(k,t,*parameters)
    period=2*np.pi/omega
    t_values=np.linspace(0,period,1001)
    
    H_eff=lambda k:sum(H(k,t_values))/period
    
    U=lambda k:sl.expm(1j*H_eff(k)*period)
    
    return -1j*period*sl.logm(U(k))

def static_tight_binding_Hamiltonian(kx,Ny,t,mu,Delta,km,B,Vm,theta):
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
        H[4*y,4*y+3]=Delta
        H[4*y+1,4*y+2]=-Delta
        #H[4*y+1,4*y+2]=-Delta[y]
    
    #Magnetic Scattering Terms
    
    y=Ny//2
    
    H[4*y,4*y+1]=Vm*np.sin(theta)
    H[4*y+2,4*y+3]=-Vm*np.sin(theta)
    
    #Hamiltonian is made Hermitian
    H+=np.matrix.getH(H)
    
    #Onsite terms
    for y in range(Ny):
        H[4*y,4*y]=-2*t*np.cos(kx+km)-mu-B
        H[4*y+1,4*y+1]=-2*t*np.cos(kx-km)-mu+B
        H[4*y+2,4*y+2]=2*t*np.cos(-kx+km)+mu+B
        H[4*y+3,4*y+3]=2*t*np.cos(-kx-km)+mu-B
    
    y=Ny//2
    H[4*y,4*y]+=Vm*np.cos(theta)
    H[4*y+1,4*y+1]+=-Vm*np.cos(theta)
    H[4*y+2,4*y+2]+=-Vm*np.cos(theta)
    H[4*y+3,4*y+3]+=Vm*np.cos(theta)
    
    return H






#Continuum Greens Function---------------------------------------------------------------

def poles(omega,kx,kf,km,Delta,B,sigma,pm):
    p=np.sqrt(1-(kx+sigma*km)**2+pm*1j*np.emath.sqrt(Delta**2-(omega+sigma*B)**2))
    
    return p

def sigma_substrate_Greens_function(omega,kx,y,kf,km,Delta,B,sigma):
    #The greens function is written in units of m/kf so we ignore this factor out the front
    
    omega=np.add(omega,0.0001j,casting="unsafe")
    
    p1=poles(omega,kx,kf,km,Delta,B,sigma,1)
    p2=poles(omega,kx,kf,km,Delta,B,sigma,-1)
    
    e_plus=np.e**(1j*kf*p1*abs(y))/p1+np.e**(-1j*kf*p2*abs(y))/p2
    e_min=np.e**(1j*kf*p1*abs(y))/p1-np.e**(-1j*kf*p2*abs(y))/p2
    
    tau_x=np.array(([0,1],[1,0]),dtype=complex)
    tau_z=np.array(([1,0],[0,-1]),dtype=complex)
    iden=np.identity(2)
    
    g_sigma=-1/(2*np.emath.sqrt(Delta**2-(omega+sigma*B)**2))*(e_plus*(omega+sigma*B)*iden+sigma*Delta*e_plus*tau_x+1j*e_min*np.emath.sqrt(Delta**2-(omega+sigma*B)**2)*tau_z)
    
    return g_sigma

def substrate_Greens_function(omega,kx,y,kf,km,Delta,B):
    g=np.zeros((4,4),dtype=complex)
    
    g_up=sigma_substrate_Greens_function(omega,kx,y,kf,km,Delta,B,1)
    g_down=sigma_substrate_Greens_function(omega,kx,y,kf,km,Delta,B,-1)
    
    g[0,0]=g_up[0,0]
    g[0,3]=g_up[0,1]
    g[3,0]=g_up[1,0]
    g[3,3]=g_up[1,1]
    
    g[1:3,1:3]=g_down
    
    return g

def T_matrix(omega,kx,kf,km,Delta,B,Cm,theta):
    Hm=Cm*np.kron(np.array(([1,0],[0,-1])),np.array(([np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)])))
    if Cm==0:
        T=np.zeros((4,4))
    else:
        
        T_inv=np.linalg.inv(Hm)-substrate_Greens_function(omega, kx, 0, kf, km, Delta, B)
        
        T=np.linalg.inv(T_inv)
    
    return T

def Greens_function(omega,kx,y1,y2,kf,km,Delta,B,Cm,theta):
    
    g_y1y2=substrate_Greens_function(omega,kx,y1-y2,kf,km,Delta,B)
    g_y1=substrate_Greens_function(omega,kx,y1,kf,km,Delta,B)
    g_y2=substrate_Greens_function(omega,kx,-y2,kf,km,Delta,B)
    T=T_matrix(omega,kx,kf,km,Delta,B,Cm,theta)
    
    G=g_y1y2+g_y1@T@g_y2
    
    return G











#Tight Binding Green's Function------------------------------------------------

def tight_binding_poles(omega,mu,Delta,B,sigma,pm1,pm2):
    #omega=np.add(omega,0.0001j,casting="unsafe")
    a=1/2*(mu-pm1*np.emath.sqrt((omega+sigma*B)**2-Delta**2))
    
    # if omega==0:
    #     a=1/2*(-mu+pm1*1j*abs(Delta))
    
    return -a+pm2*np.emath.sqrt(a**2-1)

def sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,sigma):
    #omega=np.add(omega,0.00000001j,casting="unsafe")
    mu_kx=mu+2*t*np.cos(kx+sigma*km)
    z1=tight_binding_poles(omega,mu_kx,Delta,B,sigma,1,-1)
    z2=tight_binding_poles(omega,mu_kx,Delta,B,sigma,-1,-1)
    z3=tight_binding_poles(omega,mu_kx,Delta,B,sigma,1,1)
    z4=tight_binding_poles(omega,mu_kx,Delta,B,sigma,-1,1)
    
    tau_x=np.array(([0,1],[1,0]),dtype=complex)
    tau_z=np.array(([1,0],[0,-1]),dtype=complex)
    iden=np.identity(2)
    
    xi_plus=z1**(abs(y)+1)/((z1-z3)*(z1-z4))+z2**(abs(y)+1)/((z2-z3)*(z2-z4))
    xi_min=z1**(abs(y)+1)/((z1-z3)*(z1-z4))-z2**(abs(y)+1)/((z2-z3)*(z2-z4))
    
    
    
    GF=xi_min*((omega+sigma*B)*iden+sigma*Delta*tau_x)-1j*xi_plus*np.sqrt(Delta**2-(omega+sigma*B)**2)*tau_z
            
    return -GF/(t*(z1-z2))

def TB_SC_GF(omega,y,kx,t,mu,Delta,km,B):
    
    GF_up=sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,1)
    GF_down=sigma_TB_GF(omega,y,kx,t,mu,Delta,km,B,-1)
    
    GF=np.zeros((4,4),dtype=complex)
    GF[0,0]=GF_up[0,0]
    GF[0,3]=GF_up[0,1]
    GF[3,0]=GF_up[1,0]
    GF[3,3]=GF_up[1,1]
    
    GF[1:3,1:3]=GF_down
    
    return GF

def TB_T_matrix(omega,kx,t,mu,Delta,km,B,Vm,theta):
    g=TB_SC_GF(omega,0,kx, t, mu, Delta, km,B)

    
    tau_z=np.array(([1,0],[0,-1]))
    hm=np.array(([np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)]))
    
    if Vm==0:
        T=np.zeros((4,4))
    if Vm!=0:
        T=np.linalg.inv(np.linalg.inv(Vm*np.kron(tau_z,hm))-g)
    
    return T


def TB_GF(omega,kx,y1,y2,t,mu,Delta,km,B,Vm,theta):
    g_y1_y2=TB_SC_GF(omega,y1-y2,kx,t,mu,Delta,km,B)
    g_y1=TB_SC_GF(omega,y1,kx,t,mu,Delta,km,B)
    g_y2=TB_SC_GF(omega,-y2,kx,t,mu,Delta,km,B)
     
    T=TB_T_matrix(omega,kx,t,mu,Delta,km,B,Vm,theta)
        
    GF=g_y1_y2+g_y1@T@g_y2
    
    return GF










#Continuum Electronic Structure----------------------------------------------------------

def LDOS(omega,kx,y,kf,km,Delta,B,Cm,theta):
    G=Greens_function(omega,kx,y,y,kf,km,Delta,B,Cm,theta)
    
    LDOS=-1/np.pi*np.imag(np.trace(G))
    
    return LDOS

def DOS(omega,kx,kf,km,Delta,B,Cm,theta):
    
    y_values=np.linspace(-50/kf,50/kf,10001)
    
    DOS=0
    for y in y_values:
        DOS+=LDOS(omega, kx, y, kf, km, Delta, B, Cm, theta)
        
    return DOS

def in_gap_band_structure(kx,kf,km,Delta,B,Cm,theta):
    effective_gap=abs(Delta)-abs(B)
    
    pole_condition=lambda omega:np.linalg.det(np.linalg.inv(T_matrix(omega,kx,kf,km,Delta,B,Cm,theta)))
    
    positive_energy_mode=fsolve(pole_condition,x0=0.99*effective_gap)
    negative_enery_mode=fsolve(pole_condition,x0=-0.99*effective_gap)
    
    return negative_enery_mode[0],positive_energy_mode[0]
    
    
def gapless(kf,km,Delta,B,Cm,theta):
    kx_values=np.linspace(-10,10,251)
    zero_energy_DOS=0
    for kx in kx_values:
        zero_energy_DOS+=LDOS(0, kx, 0, kf, km, Delta, B, Cm, theta)
        
    return zero_energy_DOS/len(kx_values)
  

#TB Electronic Structure-------------------------------------------------------

def TB_LDOS(omega,kx,y,t,mu,Delta,km,B,Vm,theta):
    G=TB_GF(omega, kx, y, y, t, mu, Delta, km, B, Vm, theta)
    LDOS=-1/np.pi*np.imag(np.trace(G))
    
    return LDOS

def TB_DOS(omega,kx,y,t,mu,Delta,km,B,Vm,theta):
    
    kf=np.arccos(-mu/(2*t)-1)
    
    y_values=np.linspace(-50/kf,50/kf,10001)
    
    DOS=0
    for y in y_values:
        DOS+=TB_LDOS(omega,kx,y,t,mu,Delta,km,B,Vm,theta)
        
    return DOS

def TB_in_gap_band_structure(kx,t,mu,Delta,km,B,Vm,theta):
    effective_gap=abs(Delta)-abs(B)
    
    pole_condition=lambda omega:np.linalg.det(np.linalg.inv(TB_T_matrix(omega,kx,t,mu,Delta,km,B,Vm,theta)))
    
    positive_energy_mode=fsolve(pole_condition,x0=0.99*effective_gap)
    negative_enery_mode=fsolve(pole_condition,x0=-0.99*effective_gap)
    
    return negative_enery_mode[0],positive_energy_mode[0]
    
    
    
    
    
    
    
    
#Continuum Topological Properties--------------------------------------------------------    


def continuum_topological_Hamiltonian(kx,kf,km,Delta,B,Cm,theta):
    top_ham=-Greens_function(0,kx,0,0,kf,km,Delta,B,Cm,theta)
    
    top_ham+=np.conj(top_ham.T)
    return top_ham/2
    





def continuum_phase_boundaries(kf,km,Delta,B,theta,pm):
    p1=poles(0,0,kf,km,Delta,B,1,1)
    p2=poles(0,0,kf,km,Delta,B,1,-1)
    
    e_plus=1/p1+1/p2
    e_min=1/p1-1/p2
    
    g_11=-1j*e_min/2
    g_12=-e_plus*Delta/(2*np.emath.sqrt(Delta**2-B**2))
    g_B=-B*e_plus/(2*np.emath.sqrt(Delta**2-B**2))
    
    Cm=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Cm  
def continuum_phase_boundaries_numpy(kf,km,Delta,B,Cm,theta,pm):
    p1=poles(0,0,kf,km,Delta,B,1,1)
    p2=poles(0,0,kf,km,Delta,B,1,-1)
    
    e_plus=1/p1+1/p2
    e_min=1/p1-1/p2
    
    g_11=-1j*e_min/2
    g_12=-e_plus*Delta/(2*np.emath.sqrt(Delta**2-B**2))
    g_B=-B*e_plus/(2*np.emath.sqrt(Delta**2-B**2))
    
    return Cm-np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))



#Tight Binding Topological Properties------------------------------------------


def TB_topological_Hamiltonian(kx,y,t,mu,Delta,km,B,Vm,theta):
    top_ham=-np.linalg.inv(TB_GF(0,kx, y, y, t, mu, Delta, km, B, Vm, theta))
    
    top_ham+=np.conj(top_ham.T)
    
    top_ham*=1/2
    
    return top_ham


def TB_phase_boundaries(t,mu,Delta,km,B,theta,pm,kx=0):
    mu_kx=mu+2*t*np.cos(kx+km)
    z1=tight_binding_poles(0,mu_kx,Delta,B,1,1,-1)
    z2=tight_binding_poles(0,mu_kx,Delta,B,1,-1,-1)
    z3=tight_binding_poles(0,mu_kx,Delta,B,1,1,1)
    z4=tight_binding_poles(0,mu_kx,Delta,B,1,-1,1)
    
    xi_plus=z1/((z1-z3)*(z1-z4))+z2/((z2-z3)*(z2-z4))
    xi_min=z1/((z1-z3)*(z1-z4))-z2/((z2-z3)*(z2-z4))
    
    g_11=-1/(t*(z1-z2))*(-xi_plus*1j*np.emath.sqrt(Delta**2-B**2))
    g_12=-1/(t*(z1-z2))*(xi_min*Delta)
    g_B=-1/(t*(z1-z2))*(B*xi_min)
    
    Vm=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Vm

def TB_phase_boundaries_numpy(t,mu,Delta,km,B,Vm,theta,pm,kx=0):
    mu_kx=mu+2*t*np.cos(kx+km)
    z1=tight_binding_poles(0,mu_kx,Delta,B,1,1,-1)
    z2=tight_binding_poles(0,mu_kx,Delta,B,1,-1,-1)
    z3=tight_binding_poles(0,mu_kx,Delta,B,1,1,1)
    z4=tight_binding_poles(0,mu_kx,Delta,B,1,-1,1)
    
    xi_plus=z1/((z1-z3)*(z1-z4))+z2/((z2-z3)*(z2-z4))
    xi_min=z1/((z1-z3)*(z1-z4))-z2/((z2-z3)*(z2-z4))
    
    g_11=-1/(t*(z1-z2))*(-xi_plus*1j*np.emath.sqrt(Delta**2-B**2))
    g_12=-1/(t*(z1-z2))*(xi_min*Delta)
    g_B=-1/(t*(z1-z2))*(B*xi_min)
    
    Vm_crit=np.real(1/(g_B*np.cos(theta)+pm*np.emath.sqrt(g_B**2*np.cos(theta)**2+g_11**2+g_12**2-g_B**2)))
    
    return Vm-Vm_crit

def TB_gap(Ny,t,mu,Delta,km,B,Vm,theta):
    kx_values=np.linspace(-np.pi,np.pi,51)
    lowest_energy_value=np.zeros(len(kx_values))
    for kx_indx,kx in enumerate(kx_values):
        lowest_energy_value[kx_indx]=abs(np.min(abs(np.linalg.eigvalsh(static_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta)))))
        
    return np.min(abs(lowest_energy_value))

#General Topological Properties
  
def pfaffian_invariant(Hamiltonian,parameters,system_size,TB=True):
    try:
        H=lambda k:Hamiltonian(k,*parameters)
        
        U=np.kron(np.array(([1,-1j],[1,1j])),np.identity(2))
        U_tot=1/(np.sqrt(2))*np.kron(np.identity(system_size),U)
        
        if TB==True:
            H_majorana_0=np.round(np.conj(U_tot.T)@H(0)@U_tot,decimals=6)
            H_majorana_pi=np.round(np.conj(U_tot.T)@H(np.pi)@U_tot,decimals=6)
            
            invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)*pf.pfaffian(H_majorana_pi)))
        if TB==False:
            H_majorana_0=np.round((np.conj(U_tot.T)@H(0)@U_tot+np.conj((np.conj(U_tot.T)@H(0)@U_tot).T))/2,decimals=5)
            
            invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)))
    except AssertionError:
        invariant=0
        
    return invariant

    
    