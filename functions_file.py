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
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from pfapack import pfaffian as pf
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from random import uniform

plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


#Tight Binding-----------------------------------------------------------------
def driven_tight_binding_Hamiltonian(kx,Ny,T,period,t,mu,Delta,Vm,km):
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

    H=np.zeros((4*Ny,4*Ny),dtype=complex)
    
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
    omega=2*np.pi/period
    
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

def static_tight_binding_Hamiltonian(kx,Ny,t,mu,Delta,km,B,Vm,theta,sparse=False):
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
    if sparse==True:
        H=dok_matrix((4*Ny,4*Ny),dtype=float)
    else:
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
    
    if sparse==True:
        H=H.tocsc()
    
    return H

def real_space_static_tight_binding_Hamiltonian(Nx,Ny,t,mu,Delta,km,B,Vm,theta,sparse=False):
    
    sigma_x=np.array(([0,1],[1,0]),dtype=complex)
    sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
    sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
    if sparse==True:
        H=dok_matrix((4*Nx*Ny,4*Nx*Ny),dtype=complex)
    else:
        H=np.zeros((4*Nx*Ny,4*Nx*Ny),dtype=np.complex128)
    
    #(x,y) is the 4*x+4*Nx*y position in the spinor
    
    for x in range(Nx):
        for y in range(Ny):
            
            # x hopping
            if x!=Nx-1:
                H[4*x+4*Nx*y,4*(x+1)+4*Nx*y]=-t
                H[4*x+4*Nx*y+1,4*(x+1)+4*Nx*y+1]=-t
                H[4*x+4*Nx*y+2,4*(x+1)+4*Nx*y+2]=t
                H[4*x+4*Nx*y+3,4*(x+1)+4*Nx*y+3]=t
            
            #y-hopping
            if y!=Ny-1:
                H[4*x+4*Nx*y,4*(x)+4*Nx*(y+1)]=-t
                H[4*x+4*Nx*y+1,4*(x)+4*Nx*(y+1)+1]=-t
                H[4*x+4*Nx*y+2,4*(x)+4*Nx*(y+1)+2]=t
                H[4*x+4*Nx*y+3,4*(x)+4*Nx*(y+1)+3]=t
                
            #Pairing
            H[4*x+4*Nx*y,4*x+4*Nx*y+3]=Delta
            H[4*x+4*Nx*y+1,4*x+4*Nx*y+2]=-Delta
            
           
                
            
    H+=np.conj(H.T)
    
    for x in range(Nx):
        for y in range(Ny):
            #chemical potential
            
            H[4*x+4*Nx*y,4*(x)+4*Nx*y]=-mu
            H[4*x+4*Nx*y+1,4*(x)+4*Nx*y+1]=-mu
            H[4*x+4*Nx*y+2,4*(x)+4*Nx*y+2]=mu
            H[4*x+4*Nx*y+3,4*(x)+4*Nx*y+3]=mu
            
            #scattering
            
            if y==Ny//2:
                
                H[4*x+4*Nx*y:4*(x)+4*Nx*y+2,4*x+4*Nx*y:4*(x)+4*Nx*y+2]+=Vm*(np.cos(2*km*x)*np.sin(theta)*sigma_x+np.sin(2*km*x)*np.sin(theta)*sigma_y)
                H[4*x+4*Nx*y+2:4*(x)+4*Nx*y+4,4*x+4*Nx*y+2:4*(x)+4*Nx*y+4]+=-Vm*(np.conj(np.cos(2*km*x)*np.sin(theta)*sigma_x+np.sin(2*km*x)*np.sin(theta)*sigma_y))
                H[4*x+4*Nx*y:4*(x)+4*Nx*y+2,4*x+4*Nx*y:4*(x)+4*Nx*y+2]+=Vm*np.cos(theta)*sigma_z
                H[4*x+4*Nx*y+2:4*(x)+4*Nx*y+4,4*x+4*Nx*y+2:4*(x)+4*Nx*y+4]+=-Vm*np.conj(np.cos(theta)*sigma_z)
                
   
    
    if sparse==True:
        H=H.tocsc()
    
    return H
    
    
    
def position_operator(Nx,Ny,sparse=False):
   
    
    if sparse==False:
        X=np.zeros(4*Nx*Ny)
            
          
        for x in range(Nx):
            for y in range(Ny):
                X[4*x+4*Nx*y]=x-Nx/2
                X[4*x+4*Nx*y+1]=x-Nx/2
                X[4*x+4*Nx*y+2]=(x-Nx/2)
                X[4*x+4*Nx*y+3]=(x-Nx/2)
        X_operator=np.diagflat(X)
    
    if sparse==True:
        X=dok_matrix((4*Nx*Ny,4*Nx*Ny))
        for x in range(Nx):
            for y in range(Ny):
                X[4*x+4*Nx*y,4*x+4*Nx*y]=x-Nx/2
                X[4*x+4*Nx*y+1,4*x+4*Nx*y+1]=x-Nx/2
                X[4*x+4*Nx*y+2,4*x+4*Nx*y+2]=(x-Nx/2)
                X[4*x+4*Nx*y+3,4*x+4*Nx*y+3]=(x-Nx/2)
        X_operator=X.tocsc()
    return X_operator

def y_position_operator(Nx,Ny,sparse=False):
    
    if sparse==False:
        Y=np.zeros(4*Nx*Ny)
            
          
        for x in range(Nx):
            for y in range(Ny):
                Y[4*x+4*Nx*y]=y-Ny/2
                Y[4*x+4*Nx*y+1]=y-Ny/2
                Y[4*x+4*Nx*y+2]=(y-Ny/2)
                Y[4*x+4*Nx*y+3]=(y-Ny/2)
        Y_operator=np.diagflat(Y)
    
    if sparse==True:
        Y=dok_matrix((4*Nx*Ny,4*Nx*Ny))
        for x in range(Nx):
            for y in range(Ny):
                Y[4*x+4*Nx*y,4*x+4*Nx*y]=y-Ny/2
                Y[4*x+4*Nx*y+1,4*x+4*Nx*y+1]=y-Ny/2
                Y[4*x+4*Nx*y+2,4*x+4*Nx*y+2]=(y-Ny/2)
                Y[4*x+4*Nx*y+3,4*x+4*Nx*y+3]=(y-Ny/2)
        Y_operator=Y.tocsc()
    return Y_operator

def real_space_spectrum(Nx,Ny,t,mu,Delta,km,B,Vm,theta):
    C=real_space_static_tight_binding_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
    
    spectrum,U=np.linalg.eigh(C)
    
    position_average=np.real(np.diag(np.sqrt(np.conj(U.T)@(position_operator(Nx, Ny)**2)@U)))
    return spectrum,position_average

def sparse_real_space_spectrum(Nx,Ny,Nev,t,mu,Delta,km,B,Vm,theta):
    C=real_space_static_tight_binding_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta,sparse=True)
    
    spectrum,U=spl.eigsh(C,k=Nev,return_eigenvectors=True,sigma=0, which ='LM')
    
    position_average=np.real(np.diag(np.sqrt(np.conj(U.T)@(position_operator(Nx, Ny,sparse=True)**2)@U)))
    return spectrum,position_average
    
    # spectrum=spl.eigsh(C,k=Nev,sigma=0, which ='LM',return_eigenvectors=False)
    # return spectrum


#Floquet Hamiltonians----------------------------------------------------------

def floquet_Hamiltonian(kx,Ny,t,mu,Delta,km,B,Vm,theta):
    
    period=np.pi/B
    
    
                    
    U=-sl.expm(-1j*period*static_tight_binding_Hamiltonian(kx, Ny, t, mu, Delta, km, B, Vm, theta))
    
    HF=1j/period*sl.logm(U)
    
    return HF

def real_space_floquet_Hamiltonian(Nx,Ny,t,mu,Delta,km,B,Vm,theta):
    

       
   period=np.pi/B
   
   H=real_space_static_tight_binding_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
   
   positive_eigenvalues,positive_eigenstates=spl.eigsh(H,k=2*Nx*Ny,sigma=0,which="LA")
   negative_eigenvalues,negative_eigenstates=spl.eigsh(-H,k=2*Nx*Ny,sigma=0,which="LA")
   
   eigenvalues=np.concatenate((-negative_eigenvalues, positive_eigenvalues))
   
   eigenstates=np.concatenate((negative_eigenstates, positive_eigenstates),axis=1)
   
   Hf=eigenstates@(np.diag(np.log(np.exp(-1j*period*eigenvalues))))@np.conj(eigenstates.T)
   
   return Hf

def real_space_floquet_spectrum(Nx, Ny, Nev, t, mu, Delta, km, B, Vm, theta):
    C=real_space_floquet_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
    
    
    spectrum=spl.eigsh(C,k=Nev,sigma=0, which ='LM',return_eigenvectors=False)
    return spectrum




#Floquet Spectral Localiser------------------------------------------------------------

def spectral_localiser(x,y,E,Nx,Ny,t,mu,Delta,km,B,Vm,theta):
    
    X=position_operator(Nx, Ny,sparse=True)
    Y=y_position_operator(Nx, Ny,sparse=True)
    H=real_space_floquet_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
    
    k=0.001
    
    L=sp.csc_matrix(sp.array([H-E*sp.identity(4*Nx*Ny),k*(X-1j*Y-(x-1j*y)*sp.identity(4*Nx*Ny))],[k*(X+1j*Y-(x+1j*y)*sp.identity(4*Nx*Ny)),E*sp.identity(4*Nx*Ny)-H]))
    return L

def localiser_gap(x,y,E,Nx,Ny,t,mu,Delta,km,B,Vm,theta):
    L=spectral_localiser(x, y, E, Nx, Ny, t, mu, Delta, km, B, Vm, theta)
    
    eigenvalues=spl.eigsh(L,k=1,sigma=0,which="LM")
    
    gap=np.min(abs(eigenvalues))
    
    return gap

def class_D_invariant(x,Nx,Ny,t,mu,Delta,km,B,Vm,theta):
    X=position_operator(Nx, Ny,sparse=True)
    H=real_space_floquet_Hamiltonian(Nx, Ny, t, mu, Delta, km, B, Vm, theta)
    
    k=0.001
    
    C=k(X-x*sp.identity(4*Nx*Ny))+1j*H
    
    sgn,logdet=np.slogdet(C.A)
    
    return sgn
    
    


#projector---------------------------------------------------------------------
def momentum_projector(kx,Ny,Hamiltonian,parameters):
    H=lambda kx:Hamiltonian(kx,Ny,*parameters)
    E,U=np.linalg.eigh(H(kx))
    E_occ=E<0
    U_occ=U[:,E_occ]
    
    P=U_occ@np.conj(U_occ.T)
    
    return P

def real_space_correlator(Nx,Ny,Hamiltonian,parameters):
    #Function that calculates real space correlator matrix
    #It does this by calculating the fourier transform of the momentum space correlators at each integer x between -Nx//2 to Nx//2
    #As the system is bounded from 0 to Nx//2 when an entanglement cut is made through the middle of it
    #Then the block of the correlator matrix that measures correlations from a lattice site m to lattice n is simply the previously calculated
    #element with x=m-n
    #So a double loop then just iterates through the whole real space correlator placing the blocks in their correct places

    real_space_correlator_matrix=np.zeros((4*Nx//2*Ny,4*Nx//2*Ny),dtype=np.complex128)
    
    correlator_elements={}
    
    for m in range(-Nx//2,Nx//2):
        correlator_element=np.zeros((4*Ny,4*Ny),dtype=complex)
        for i in range(Nx):
            kx=2*np.pi*i/Nx
            momentum_correlator=momentum_projector(kx, Ny,Hamiltonian, parameters)
            correlator_element+=np.e**(1j*kx*(m))*momentum_correlator/Nx
        correlator_elements["{}".format(m)]=correlator_element
    
    for m in range(Nx//2):
        for n in range(Nx//2):
            lower_x_index=m*4*Ny
            upper_x_index=(m+1)*4*Ny
            lower_y_index=n*4*Ny
            upper_y_index=(n+1)*4*Ny
            
            real_space_correlator_matrix[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=correlator_elements["{}".format(m-n)]
    
    return real_space_correlator_matrix

    
def entanglement_spectrum(Nx,Ny,Hamiltonian,parameters):
    C=real_space_correlator(Nx, Ny, Hamiltonian, parameters)
    
    spectrum,U=np.linalg.eigh(C)
    
    position_average=np.real(np.diag(np.sqrt(np.conj(U.T)@(position_operator(Nx//2, Ny)**2)@U)))
    return spectrum,position_average



#Continuum Greens Function---------------------------------------------------------------

def poles(omega,kx,kf,km,Delta,B,sigma,pm):
    p=np.sqrt(1-(kx+sigma*km)**2+pm*1j*np.emath.sqrt(Delta**2-(omega+sigma*B)**2))
    
    return p

def sigma_substrate_Greens_function(omega,kx,y,kf,km,Delta,B,sigma):
    #The greens function is written in units of m/kf so we ignore this factor out the front
    
    
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
    omega=np.add(omega,0.0001j,casting="unsafe")
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
    omega=np.add(omega,0.0001j,casting="unsafe")
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
    
    positive_energy_mode=fsolve(pole_condition,x0=0.99*effective_gap*0)
    negative_enery_mode=fsolve(pole_condition,x0=-0.99*effective_gap*0)
    
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
def pfaffian(operator):
    n=len(operator[:,0])//2
    sigma_y=np.array(([0,-1j],[1j,0]))
    A=np.kron(sigma_y,np.identity(n))
    
    pfaffian_value=(1j)**(n**2)*np.exp(1/2*np.trace(sl.logm(((A.T)@operator))))
    
    return pfaffian_value
    
  
def pfaffian_invariant(Hamiltonian,parameters,system_size,TB=True):
    try:
        H=lambda k:Hamiltonian(k,*parameters)
        
        U=np.kron(np.array(([1,-1j],[1,1j])),np.identity(2))
        U_tot=1/(np.sqrt(2))*np.kron(np.identity(system_size),U)
        
        if TB==True:
            H_majorana_0=np.round(np.conj(U_tot.T)@H(0)@U_tot,decimals=6)
            H_majorana_pi=np.round(np.conj(U_tot.T)@H(np.pi)@U_tot,decimals=6)
            
            # H_majorana_0-=H_majorana_0.T
            # H_majorana_pi-=H_majorana_pi.T
            # H_majorana_0*=0.5
            # H_majorana_pi*=0.5
            
            invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)*pf.pfaffian(H_majorana_pi)))
            #invariant=np.real(np.sign(pfaffian(H_majorana_0)*pfaffian(H_majorana_pi)))
        if TB==False:
            H_majorana_0=np.round((np.conj(U_tot.T)@H(0)@U_tot+np.conj((np.conj(U_tot.T)@H(0)@U_tot).T))/2,decimals=5)
            
            invariant=np.real(np.sign(pf.pfaffian(H_majorana_0)))
    except AssertionError:
        invariant=0
        #print("Operator not anti-symmetric")
        
    return invariant

#Numerical Methods-------------------------------------------------------------


def Newton_Raphson_update(x0,function,parameters):
    dx=0.00000001
    
    f=lambda x:function(x,*parameters)
    
    x1=x0-f(x0)/((f(x0+dx)-f(x0-dx))/(2*dx))
    
    return x1

def bisection_search(function,parameters,x0,x1,fallback_solution=0):
    threshold=10**(-8)
    complete=False
    
    f=lambda x:function(x,*parameters)
    
    while complete==False:
        if x1-x0<threshold:
            solution=(x1+x0)/2
            complete=True
        else:
            function_0=np.real(f(x0))
            function_1=np.real(f(x1))
            function_mid=np.real(f((x1+x0)/2))
            #print(function_0,function_mid,function_1,"\n")
            
            if np.sign(function_0*function_mid)<=0:
                x0=x0
                x1=(x1+x0)/2
                #print(x0,x1, "\n")
                continue
            elif np.sign(function_1*function_mid)<=0:
                
                x0=(x1+x0)/2
                x1=x1
                #print(x0,x1,"\n")
                continue
            else:
                #print("No solution in given region")
                solution=fallback_solution
                break
    return solution


#Real Space topological Hamiltonian--------------------------------------------
        

def real_space_topological_hamiltonian(Nx,t,mu,Delta,km,B,Vm,theta,disorder=0,disorder_config=0):
    
    
    H=np.zeros((4*Nx,4*Nx),dtype=complex)
    
    hamiltonian_elements={}
    Nx_eff=2*Nx
    
    for m in range(-Nx,Nx):
        h_element=np.zeros((4,4),dtype=complex)
        for i in range(Nx_eff):
            kx=2*np.pi*i/Nx_eff
            kx_top_ham=TB_topological_Hamiltonian(kx, 0, t, mu, Delta, km, B, Vm, theta)
            h_element+=np.e**(1j*kx*(m))*kx_top_ham/Nx_eff
        hamiltonian_elements["{}".format(m)]=h_element
    
    for m in range(Nx):
        for n in range(Nx):
            lower_x_index=m*4
            upper_x_index=(m+1)*4
            lower_y_index=n*4
            upper_y_index=(n+1)*4
            
            if m-n==1 or m-n==-1:
                if disorder_config==0:
            
                    H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]+disorder*uniform(-1,1)*np.array(([1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]))
                
                else:
                    H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]+disorder*disorder_config[m]*np.array(([1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]))
            
            else:
                H[lower_x_index:upper_x_index,lower_y_index:upper_y_index]=hamiltonian_elements["{}".format(m-n)]
    return H

def real_space_floquet_topological_hamiltonian(Nx,t,mu,Delta,km,B,Vm,theta,disorder=0,disorder_config=0):
    H=real_space_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config)
    
    period=np.pi/B
    U=sl.expm(-1j*period*H)
    
    HF=1j*sl.logm(U)
    
    return HF


def one_D_position_operator(Nx):
    X=np.zeros(4*Nx)
    for x in range(Nx):
        X[4*x]=x
        X[4*x+1]=x
        X[4*x+2]=x
        X[4*x+3]=x
    X_operator=np.diagflat(X)

    return X_operator

def topological_hamiltonian_spectral_localiser(x,E,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0,disorder_config=0):
    X=one_D_position_operator(Nx)
    H=real_space_floquet_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config)
    
    k=10**(-4)
    
    L=np.zeros((8*Nx,8*Nx),dtype=complex)
    
    L[:4*Nx,4*Nx:]=k*(X-x*np.identity(4*Nx))-1j*(H-E*np.identity(4*Nx))
    L[4*Nx:,:4*Nx]=k*(X-x*np.identity(4*Nx))+1j*(H-E*np.identity(4*Nx))
   
    return L
    


def topological_hamiltonian_localiser_gap(x,E,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0,disorder_config=0):
    L=dok_matrix(topological_hamiltonian_spectral_localiser(x, E, Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config)).tocsc()
    
    #eigenvalues=np.linalg.eigvalsh(L.A)
    eigenvalues=spl.eigsh(L,k=1,sigma=0,return_eigenvectors=False,which="LM")
    
    gap=np.min(abs(eigenvalues))
    
    return gap

def topological_hamiltonian_class_D_invariant(x,Nx,t,mu,Delta,km,B,Vm,theta,disorder=0,disorder_config=0):
    X=one_D_position_operator(Nx)
    H=real_space_floquet_topological_hamiltonian(Nx, t, mu, Delta, km, B, Vm, theta,disorder=disorder,disorder_config=disorder_config)
    
    k=10**(-4)
    
    C=k*(X-x*np.identity(4*Nx))+1j*H
    
    invariant,det=np.linalg.slogdet(C)
    
    return np.real(invariant)
