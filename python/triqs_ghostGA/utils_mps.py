# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: [Benedikt Kloss] Olivier Gingras and Tsung-Han Lee

import numpy as np
from itertools import product as itp

def setup_MPS(M, Utensor, Norb, Nbath, schedule,tolerances,use_Sz=True,use_Ntot=True,spin_pen=0.0):
    """
    Given the embedded Hamiltonian and parameters, run MPS and return density matrix.
        M               : Embedded Hamiltonian in a matrix of size (Norb+Nbath)x(Norb+Nbath), containing local hamiltonian,
                          hybridization with the bath, and bath degrees of freedom. The bath should be diagonal.
        Norb            : Number of physical orbitals.
        Nbath           : Number of bath degrees of freedom.
        UTensor          : quartic interaction tensor (Triqs convention).
    """


    #assumes that import has happened before (when initializing MPS solver class)
    kwarg_names=["use_Sz","use_Ntot","spin_pen"]
    kwarg_vals=(use_Sz,use_Ntot,spin_pen)
    converged, Eint, Cuu,Cdd=jl.solve(Utensor,M,schedule,tolerances,[kwarg_names,kwarg_vals]
           )

    return  Cuu,Cdd, Eint

def rotateBath(M, Norb, Nbath):
    """
    Diagonalizes the Bath part of the M matrix. Also rotates the hybridization.
    This function returns the rotated M matrix, along with the vectors to
    rotate it back.
        M : np.array((Norb*(Nbath+1), Norb*(Nbath+1))) : M matrix describing the 
            embedded Hamiltonian.
        Norb : int : Number of orbital degrees of freedom.
        Nbath : int : Number of bath per orbital.
    """
    # Preparing the rotated Embedded Hamiltonian
    M_rot = {"up": np.copy(M["up"]),
             "dn": np.copy(M["dn"])}

    # Obtained the eigenvectors of the bath sites to rotate the matrix
    v_all = {"up": [], "dn": []}
    for name in ["up", "dn"]:
        B = M[name][Norb:, Norb:] # Bath sites
        # W = M[name][:Norb, Norb:]
        # Wd = M[name][Norb:, :Norb]

        # Diagonalization of the bath
        w, v = np.linalg.eig(B)
        # Keep the eigenvectors in memory
        v_all[name] = np.block([[np.eye(Norb), np.zeros((Norb, Nbath*Norb))],
                                [np.zeros((Norb*Nbath, Norb)), v]])

        # Rotate the embedded Hamiltonian
        M_rot[name] = np.linalg.inv(v_all[name]) @ M_rot[name] @ v_all[name]
    return M_rot, v_all

def rotateDensityMatrix(singlePup,singlePdn, v):
    """
    Rotate back the density matrix. Used with rotateBath to minimize the entropy
    in MPS
        singleP : Density matrix obtained by forkTPS.
        Norb : int : Number of orbital degrees of freedom.
        Nbath : int : Number of baths per orbital.
        v : Eigenvectors of the Bath obtained from rotateBath.
    """
    # Extract density matrix for up and rotate back to the original basis,
    # before the bath was diagonalized.
    single_up = singlePup
    single_dn = singlePdn
    
    v_up = v["up"]
    v_dn = v["dn"]

    single_up = v_up @ single_up @ np.linalg.inv(v_up)  ##inv is the wrong thing to do here! it's a unitary rotation after all
    single_dn = v_dn @ single_dn @ np.linalg.inv(v_dn)

    return np.block([[single_up,np.zeros(single_up.shape)],[np.zeros(single_up.shape),single_dn]])

def rotateToTsungHanConvention(rho_CDC, Norb, Nbath):
    #assert False
    ##FIXME: not checked/implemented yet
    A = list(range(2*Norb*(Nbath+1)))
    B = []
    for a in range(Norb*(Nbath+1)):
        B.append(a)
        B.append(a+Norb*(Nbath+1))
    # B = list(np.arange(0, 2*Norb*(1+Nbath), (1+Nbath)))
    # B = list(np.arange(0, 2*Norb*(1+Nbath), 2))
    # B += list(np.arange(1, 2*Norb*(1+Nbath), 2))
    # for a, b in itp(range(2*Norb), range(Nbath)):
    #     B.append(1+b+a*(1+Nbath))

    rho_CDC[A, :] = rho_CDC[B, :]
    rho_CDC[:, A] = rho_CDC[:, B]
    return rho_CDC

