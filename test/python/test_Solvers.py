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
from triqs_ghostGA.version import *


class test_hemb_solver_1o3(unittest.TestCase):

    def setUp(self):
        # 1 orbital with 2 spins, 3 bath per orbital, total 8
        self.nimp, self.nbath, self.ntot = 2, 6, 8
        nimp, nbath, ntot = 2, 6, 8

        # construct ek with semicircular DOS
        e_list = get_semicircle_e_list(nmesh=5000)
        eks = []
        for e in e_list:
            tmp = np.array([[1.0*e]],dtype=np.complex128)
            tmp = np.kron(tmp,np.eye(2))
            eks.append(tmp)
        self.eks = np.array(eks)

        # random initial value for hybridization
        R0 = np.random.rand(nbath//2, nimp//2)
        self.R0 = np.kron(R0, np.eye(2))

        Lambda0 = np.zeros((nbath//2, nbath//2))
        Lambda0 = np.diag([0.6, 0, -0.6])
        self.Lambda0 = np.kron(Lambda0, np.eye(2))

        U = 2.4
        self.eloc = np.zeros((nimp, nimp))
        self.Utensor = np.zeros((nimp, nimp, nimp, nimp))
        self.eloc[0,0] = -U/2.
        self.eloc[1,1] = -U/2.
        self.Utensor[0,0,1,1] = U
        self.Utensor[1,1,0,0] = U

        self.docc, self.denMat = dict(), dict()

    def test_grisb_ftps(self):

        # test ForkTPS solver
        from triqs_ghostGA.ftps import FTPS

        nimp, nbath, ntot = self.nimp, self.nbath, self.ntot
        eks, eloc, Utensor = self.eks, self.eloc, self.Utensor
        R0, Lambda0 = self.R0, self.Lambda0

        edsolver = FTPS(ntot, nimp, nbath, 300)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)

        self.docc['ftps'] = grisb.docc
        self.denMat['ftps'] = grisb.denMat

    def test_grisb_ci(self):

        # test CI solver
        from triqs_ghostGA.ci import CI

        nimp, nbath, ntot = self.nimp, self.nbath, self.ntot
        eks, eloc, Utensor = self.eks, self.eloc, self.Utensor
        R0, Lambda0 = self.R0, self.Lambda0

        edsolver = CI(ntot, use_Ntot=True,
                      use_Sz=True, dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)

        self.docc['ci'] = grisb.docc
        self.denMat['ci'] = grisb.denMat

    def test_grisb_mps(self):

        # test julia MPS solver
        from triqs_ghostGA.mps import ITensorMPSSolver as MPS

        nimp, nbath, ntot = self.nimp, self.nbath, self.ntot
        eks, eloc, Utensor = self.eks, self.eloc, self.Utensor
        R0, Lambda0 = self.R0, self.Lambda0

        params={"use_Sz":True, "use_Ntot":True, "spin_pen":0.05}
        edsolver = MPS(ntot, nimp, nbath, params=params)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)

        self.docc['mps'] = grisb.docc
        self.denMat['mps'] = grisb.denMat

    def test_grisb_with_diis_and_ci(self):

        from triqs_ghostGA.ci import CI

        nimp, nbath, ntot = self.nimp, self.nbath, self.ntot
        eks, eloc, Utensor = self.eks, self.eloc, self.Utensor
        R0, Lambda0 = self.R0, self.Lambda0

        edsolver = CI(ntot, use_Ntot=True, use_Sz=False,
                      dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=5, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05, diis=True)
        print(grisb.docc)
        print(grisb.denMat)


if __name__ == '__main__':
    unittest.main()
