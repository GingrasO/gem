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
# Authors: Olivier Gingras and Tsung-Han Lee

import numpy as np
import triqs.utility.mpi as mpi
import forktps as ftps
from forktps.solver_core import Bath
from itertools import product as itp

def ConstructBath(gfstruc_ , Nbath_, SpinOrbCoup_, hopping_, eps_):
    """
    Construct the bath for the impurity solver.
        gfstruc_ : the structure of the impurity model
        Nbath_ : the number of bath sites for each band or impurity degree of freedom
        SpinOrbCoup_ : the spin-orbital coupling
        hopping_ : np.zeros([size, size, Nbath_], dtype=complex), hopping_[ i_imp, j_bath, ib] : the hopping amptitute from (j_bath orbital of the ib bath) to (i_imp orbital of the impurity)
        eps_ : np.zeros([size, Nbath_])
    """
    bath = Bath(gfstruc_, SpinOrbCoup_)
    for name,size in gfstruc_:
        for iorb in range(size):
            for ib in np.arange(Nbath_):
                indx = [name, iorb]
                eps = eps_[name][iorb, Nbath_ - 1 - ib]
                hop = hopping_[name][:, iorb, Nbath_ - 1 - ib]
                bath.addSite(indx, eps, hop) # note that the first added bath site is placed at the end of the fork
    return bath

def setup_forkTPS(M, Norb, Nbath, gf_struct, int_params, w_grid, maxm, tw,
                  other_params={"dt": 0.1, "time_steps": 1, "sweeps": 10, "prep_napph": 5, "DMRGMethod": "TwoSite"}):
    """
    Given the embedded Hamiltonian and parameters, run ForkTPS and return density matrix.
        M               : Embedded Hamiltonian in a matrix of size (Norb+Nbath)x(Norb+Nbath), containing local hamiltonian,
                        hybridization with the bath, and bath degrees of freedom. The bath should be diagonal.
        Norb            : Number of physical orbitals.
        Nbath           : Number of bath degrees of freedom.
        gf_struct       : Structure of the impurity model.
        int_params      : Parameters of the interacting Hamiltonian.
        w_grid          : Bandwidth of the w-grid.
        maxm            : Maximal bound dimension.
        tw              : Cutoff parameter for the bound dimension.
        other_params    : Other parameters for ForkTPS.
    """

    # Construct the real time ForkTPS solver.
    S = ftps.Solver(gf_struct = gf_struct , nw = w_grid["nw"], wmin=w_grid["window"][0], wmax=w_grid["window"][1])

    # Fix the interacting Hamiltonian
    Hint = ftps.solver_core.HInt(u=int_params["U"], j=int_params["J"], up=int_params["Up"], dd=int_params["dd"])

        # Construct the local Hamiltonian and extract from M matrix
    e0 = ftps.solver_core.Hloc(gf_struct) #give the local Hamiltonian the right block structure
    e0.Fill("up", M["up"][:Norb, :Norb])
    e0.Fill("dn", M["dn"][:Norb, :Norb])

    # Construct the bath:
    # the bath sites are assigned to an orbital. Since it should be diagonal,
    # there is only one bath orbital, which is (orb, bath).
    eps = {"up": np.zeros((Norb, Nbath)),
           "dn": np.zeros((Norb, Nbath))}
    # Construct the hybridization:
    # now this couples a bath site and an orbital, so (orb, bath)->(orb).
    hopping = {"up": np.zeros((Norb, Norb, Nbath), dtype=complex),
               "dn": np.zeros((Norb, Norb, Nbath), dtype=complex)}
    # Mapping M to the correct shape for ForkTPS:
    for a in range(Norb):
        eps["up"][a, :] = np.diag(M["up"][Norb+a*Nbath:Norb+(a+1)*Nbath, Norb+a*Nbath:Norb+(a+1)*Nbath])
        eps["dn"][a, :] = np.diag(M["dn"][Norb+a*Nbath:Norb+(a+1)*Nbath, Norb+a*Nbath:Norb+(a+1)*Nbath])
        hopping["up"][:, a, :] = M["up"][:Norb, Norb+a*Nbath:Norb+(a+1)*Nbath]
        hopping["dn"][:, a, :] = M["dn"][:Norb, Norb+a*Nbath:Norb+(a+1)*Nbath]

    # Assigning in the solver object
    S.b = ConstructBath(gf_struct, Nbath, False, hopping, eps)
    S.e0 = e0

    # Setting up the time-evolution solver
    tevo = ftps.solver.TevoParams(dt = other_params["dt"], time_steps = other_params["time_steps"])
    # Setting up the DMRG parameters
    dmrg = ftps.solver.DMRGParams(sweeps = other_params["sweeps"],
                                  prep_napph = other_params["prep_napph"],
                                  maxm=maxm, tw=tw, DMRGMethod=other_params["DMRGMethod"])

    # Solve the impurity model
    S.solve(h_int = Hint,
            tevo = tevo,
            params_partSector = dmrg,
            params_GS = dmrg
           )

    # Extract the single-particle density matrix and reshape it
    singleP = S.singleParticleDensity
    print('singleP=')
    print(singleP)
    rho_CDC = singleP[:(Norb*2*(Nbath+1))**2]
    rho_CDC = np.reshape(rho_CDC, (Norb*2*(Nbath+1), Norb*2*(Nbath+1)))

    A = list(range(2*Norb*(Nbath+1)))
    B = list(np.arange(0, 2*Norb*(1+Nbath), (1+Nbath)))
    for a, b in itp(range(2*Norb), range(Nbath)):
        B.append(1+b+a*(1+Nbath))

    rho_CDC[A, :] = rho_CDC[B, :]
    rho_CDC[:, A] = rho_CDC[:, B]

    # Return density matrix and interaction energy
    return rho_CDC, S.Ehint
