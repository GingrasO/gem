#!/usr/bin/env python

import unittest

from triqs_ghostGA import LatticeSolver
from triqs_ghostGA.grisb import *
from triqs_ghostGA.utils_TH import get_semicircle_e_list, U_matrix_kanamori
import numpy as np
from triqs_ghostGA.ci import CI


class test_hemb_ci_1o3(unittest.TestCase):

    def test_grisb_ci(self):

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
        eloc[0,0] = -U/2.
        eloc[1,1] = -U/2.

        Utensor = np.zeros((nimp, nimp, nimp, nimp))
        Utensor[0,0,1,1] = U
        Utensor[1,1,0,0] = U

        # test CI solver
        edsolver = CI(ntot, use_Ntot=True,
                      use_Sz=True, dtype=np.complex128)
        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0,
                      Lambda=Lambda0, edsolver=edsolver)
        grisb.run(itmax=1, mix=0.5, tol=5e-2, beta=500,
                  silence=True, spin_pen=0.05)
        print(grisb.docc)
        print(grisb.denMat)


if __name__ == '__main__':
    unittest.main()
