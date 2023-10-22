# Author: Tsung-Han Lee henhans74716@gmail.com
import unittest
import numpy
import h5py
from grisb import *
from utils_TH import get_semicircle_e_list,U_matrix_kanamori

class TestGrisb(unittest.TestCase):
    def runTest(self):
        numpy.set_printoptions(suppress=True,precision=10)
        ntot = 16
        nimp = 4
        nbath= 12

        # construct ek with semicircular DOS 
        e_list = get_semicircle_e_list(nmesh=5000) 
        eks = []
        for e in e_list:
            tmp = numpy.array([[1.0*e, 0.0  ],
                               [0.0  , 1.0*e]],dtype=numpy.complex128)
            tmp = numpy.kron(tmp,numpy.eye(2))
            eks.append(tmp)
        eks = numpy.array(eks)
        R0 = numpy.random.rand(nbath//2,nimp//2)
        R0 = numpy.kron(R0,numpy.eye(2))
        Lambda0 = numpy.zeros((nbath//2,nbath//2))
        Lambda0[0,0] = 0.1
        Lambda0[1,1] = 0.1
        Lambda0[2,2] = 0.0
        Lambda0[3,3] = 0.0
        Lambda0[4,4] =-0.1
        Lambda0[5,5] =-0.1
        Lambda0 = numpy.kron(Lambda0,numpy.eye(2))

        U = 1.2
        J = U/4.
        eloc = numpy.zeros((nimp,nimp))
        nnom = 2.0
        eloc[0,0] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[1,1] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[2,2] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[3,3] =-(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
        eloc[0,2] = 0.0
        eloc[2,0] = 0.0
        eloc[1,3] = 0.0
        eloc[3,1] = 0.0
        Utensor = U_matrix_kanamori(2, U, J)

        grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'ci', 'use_Sz': True, 'use_Ntot': True})
        grisb.run(itmax=1000, mix=0.5, tol=1e-6, beta=500, silence=True, spin_pen=0.05)

        docc0 = grisb.docc[0]
        docc1 = grisb.docc[1]
        Z = grisb.R.conj().T.dot(grisb.R)
        docc = grisb.docc
        R0 = grisb.R
        Lambda0 = grisb.Lambda

        self.assertAlmostEqual(docc0.real , 0.1442486503727918, 4, 'incorrect double occupancy')

if __name__ == '__main__':
    unittest.main()
