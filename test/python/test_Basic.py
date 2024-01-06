#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utils_TH import get_semicircle_e_list, U_matrix_kanamori
from h5 import *
import numpy as np
from triqs.utility import mpi
from triqs.lattice.tight_binding import TBLattice
from triqs.gf import *
from triqs.lattice import *
from triqs.operators import *

class test_lattice_solver(unittest.TestCase):

    # Basic test just loading the lattice solver
    def test_loading(self):

        BL = BravaisLattice(units=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        BZ = BrillouinZone(BL)
        nk = 10
        mk = MeshBrZone(BZ, nk)
        ekup = Gf(mesh=mk, target_shape=[1, 1])
        ekdn = Gf(mesh=mk, target_shape=[1, 1])

        for k in mk:
            ekup[k][0, 0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))
            ekdn[k][0, 0] = 1*(np.cos(k[0]) + np.cos(k[1]) + np.cos(k[2]))

        h0_k = BlockGf(name_list=['up', 'down'], block_list=[ekup, ekdn])

        gf_struct = [('up', 1), ('down', 1)]
        h_int = 3*n('up', 0)*n('down', 0)

        S = LatticeSolver(h0_k=h0_k, gf_struct=gf_struct)
        S.solve_ForkTPS(h_int=h_int)

        pass

    def test_grisb_1o3_ftps_and_ci(self):
        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        nimp, nbath, ntot = 2, 6, 8

        # construct ek with semicircular DOS
        e_list = get_semicircle_e_list(nmesh=5000)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        eks = np.array(eks)

        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0 = np.diag([0.6, 0, -0.6])
        Lambda0 = np.kron(Lambda0, np.eye(2))

        U = 2.4
        eloc = np.zeros((nimp, nimp))
        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        eloc[0,0] = -U/2.
        eloc[1,1] = -U/2.
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        # test ForkTPS solver
        ed_params = {"solver": "ftps", "maxM": 300}
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, ed_params=ed_params)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)

        # test CI solver
        ed_params = {"solver": "ci", "use_Sz": True, "use_Ntot": True}
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, ed_params=ed_params)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)

if __name__ == '__main__':
    unittest.main()
