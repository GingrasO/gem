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
from forktps.solver_core import Bath

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
