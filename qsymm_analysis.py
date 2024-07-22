# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:40:18 2024

@author: hm255
"""

import numpy as np
import qsymm
import sympy

ham_1D_chain="""
    (-2*t*cos(k+k_m)-mu)*(1/2*(kron(eye(2),sigma_z)+kron(sigma_z,eye(2))))+
    (-2*t*cos(k-k_m)-mu)*(1/2*(-kron(eye(2),sigma_z)+kron(sigma_z,eye(2))))+
    V_m*kron(sigma_z,sigma_x)-D*kron(sigma_y,sigma_y)-B*kron(sigma_z,sigma_z)
    """
# ham_1D_chain="""
#             m*sigma_z+(t1+t2*cos(k))*sigma_x+sin(k)*sigma_y
#             """

H_1D_chain=qsymm.Model(ham_1D_chain, momenta=['k'])
discrete_symm, continuous_symm = qsymm.symmetries(H_1D_chain)

for i in range(len(discrete_symm)):
    display(discrete_symm[i])
    print(np.round(discrete_symm[i].U,decimals=10))
    print("Conjugate={}".format(discrete_symm[i].conjugate))