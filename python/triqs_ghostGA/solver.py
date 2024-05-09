import numpy as np
from itertools import product

from triqs.gf import MeshImTime, MeshReTime, MeshReFreq, MeshLegendre, Gf, BlockGf, make_hermitian, Omega, iOmega_n, make_gf_from_fourier, fit_hermitian_tail
from triqs.gf.tools import inverse, make_zero_tail
from triqs.gf.descriptors import Fourier
from triqs.operators import c_dag, c, Operator
import triqs.utility.mpi as mpi
from h5 import HDFArchive

def get_n_orbitals(sum_k):
    """
    determines the number of orbitals within the
    solver block structure.

    Parameters
    ----------
    sum_k : dft_tools sumk object

    Returns
    -------
    n_orb : dict of int
        number of orbitals for up / down as dict for SOC calculation
        without up / down block up holds the number of orbitals
    """
    n_orbitals = [{'up': 0, 'down': 0} for i in range(sum_k.n_inequiv_shells)]
    for icrsh in range(sum_k.n_inequiv_shells):
        for block, n_orb in sum_k.gf_struct_solver[icrsh].items():
            if 'down' in block:
                n_orbitals[icrsh]['down'] += sum_k.gf_struct_solver[icrsh][block]
            else:
                n_orbitals[icrsh]['up'] += sum_k.gf_struct_solver[icrsh][block]

    return n_orbitals

class SolverStructure:

    r'''
    Handles all solid_dmft solver objects and contains TRIQS solver instance.

    Attributes
    ----------

    Methods
    -------
    solve(self, **kwargs)
        solve impurity problem
    '''

    def __init__(self, general_params, solver_params, advanced_params, sum_k, icrsh, h_int, iteration_offset, solver_struct_ftps):
        r'''
        Initialisation of the solver instance with h_int for impurity "icrsh" based on soliDMFT parameters.

        Parameters
        ----------
        general_paramuters: dict
                           general parameters as dict
        solver_params: dict
                           solver-specific parameters as dict
        sum_k: triqs.dft_tools.sumk object
               SumkDFT instance
        icrsh: int
               correlated shell index
        h_int: triqs.operator object
               interaction Hamiltonian of correlated shell
        iteration_offset: int
               number of iterations this run is based on
        '''

        self.general_params = general_params
        self.solver_params = solver_params
        self.advanced_params = advanced_params
        self.sum_k = sum_k
        self.icrsh = icrsh
        self.h_int = h_int
        self.iteration_offset = iteration_offset
        self.solver_struct_ftps = solver_struct_ftps
        # currently no solver requires random number
        #if solver_params.get("random_seed") is None:
        #    self.random_seed_generator = None

        if self.general_params['solver_type'] == 'fci':

            # sets up solver
            self.triqs_solver = self._create_fci_solver()
            #self.git_hash = triqs_hubbardI_hash
            #self.version = version

        elif self.general_params['solver_type'] == 'block2_dmrg':
            raise NotImplementedError("block2 DMRG solver not implemeted!")
            # sets up solver
            #self.triqs_solver = self._create_block2_solver()
            #self.git_hash = triqs_hartree_fock_hash
            #self.version = version

    # ********************************************************************
    # solver-specific solve() command
    # ********************************************************************

    def solve(self, **kwargs):
        r'''
        solve impurity problem with current solver
        '''

        # No solver requires random number
        #if self.random_seed_generator is None:
        #    random_seed = {}
        #else:
        #    random_seed = { "random_seed": int(self.random_seed_generator(it=kwargs["it"], rank=mpi.rank)) }

        if self.general_params['solver_type'] == 'fci':

            mpi.report('\n Using the full configuration interaction solver.')

            # Solve the impurity problem for icrsh shell
            # *************************************
            self.triqs_solver.solve(h_int=self.h_int, **{ **self.solver_params, **random_seed })
            # *************************************

            # call postprocessing
            self._fci_postprocessing()

        elif self.general_params['solver_type'] == 'block2_dmrg':
            raise NotImplementedError("block2 DMRG solver not implemeted!")
            # Solve the impurity problem for icrsh shell
            # *************************************
            # this is done on every node due to very slow bcast of the AtomDiag object as of now
            #self.triqs_solver.solve(h_int=self.h_int, calc_gtau=self.solver_params['measure_G_tau'],
            #                        calc_gw=True, calc_gl=self.solver_params['measure_G_l'],
            #                        calc_dm=self.solver_params['measure_density_matrix'])
            # if density matrix is measured, get this too. Needs to be done here,
            # because solver property 'dm' is not initialized/broadcastable
            #if self.solver_params['measure_density_matrix']:
            #    self.density_matrix = self.triqs_solver.dm
            #    self.h_loc_diagonalization = self.triqs_solver.ad
            # *************************************

            # call postprocessing
            #self._block2_postprocessing()

        return

    # ********************************************************************
    # create solvers objects
    # ********************************************************************

    def _create_fci_solver(self):
        r'''
        Initialize cthyb solver instance
        '''
        from triqs_ghostGA.ci import CI
        triqs_solver = CI(2*(self.general_params['norb_bath']+self.sum_k.corr_shells[self.icrsh]['dim']), use_Ntot=True, use_Sz=True, dtype=np.complex128)

        return triqs_solver

    def _create_block2_solver(self):
        r'''
        Initialize cthyb solver instance
        '''
        from triqs_cthyb.solver import Solver as cthyb_solver
        raise NotImplementedError("block2 DMRG solver not implemeted!")
        return 

    #def _make_spin_equal(self, Sigma):
    #
    #    # if not SOC than average up and down
    #    if not self.general_params['magnetic'] and not self.sum_k.SO == 1:
    #        Sigma['up_0'] = 0.5*(Sigma['up_0'] + Sigma['down_0'])
    #        Sigma['down_0'] = Sigma['up_0']
    #
    #    return Sigma

    # ********************************************************************
    # post-processing of solver output
    # ********************************************************************

    def _fci_postprocessing(self):
        r'''
        Organize G_freq, G_time, Sigma_freq and G_l from hubbardI solver
        '''

        # get everything from solver
        #self.Sigma_freq << self.triqs_solver.Sigma_iw
        #self.G0_freq << self.triqs_solver.G0_iw
        #self.G0_Refreq << self.triqs_solver.G0_w
        #self.G_freq << make_hermitian(self.triqs_solver.G_iw)
        #self.G_freq_unsym << self.triqs_solver.G_iw
        #self.sum_k.symm_deg_gf(self.G_freq, ish=self.icrsh)
        #self.G_freq << self.G_freq
        #self.G_Refreq << self.triqs_solver.G_w
        #self.Sigma_Refreq << self.triqs_solver.Sigma_w

        return
