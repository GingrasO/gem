#!/usr/bin/env python

import unittest

import numpy as np
import h5py
from triqs_ghostGA.sumk_grisb import SumkGRISB
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from qepack.grisb import Grisb_dp
from qepack.utils import get_semicircle_e_list,U_matrix_kanamori
np.set_printoptions(suppress=True, precision=6)

sumk_mesh = MeshImFreq(beta=200,
                       S='Fermion',
                       n_iw=1024)
sumk = SumkGRISB(hdf_file='nio.h5',
                 mesh=sumk_mesh, use_dft_blocks=False, h_field=0.0,
                 nbaths=[5])

u_trans = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 1, 0, 0, 0, 0],
                    [ 0, 1, 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 1, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 1, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 1, 0, 0],
                    [ 0, 0, 0, 0, 0, 0, 1, 0],
                    [ 0, 0, 0, 0, 0, 0, 0, 1],
                    ], dtype=float)
mu0 = -0.0#13.8931

# 2 orbital with 2 spins, 3 bath per orbital, total 16
nimp, ndp, nbath, ntot = 4, 16, 4, 8
#nimp, ndp, nbath, ntot = 4, 16, 12, 16

# construct ek with semicircular DOS
#e_list = get_semicircle_e_list(nmesh=100)
eks = []
eloc_orig = np.zeros((8,8),dtype=complex)
for ik in range(sumk.n_k):
#for e in e_list:
#    tmp = e*np.eye(3,dtype=complex)
    tmp = u_trans.dot(sumk.hopping[ik,0,:,:]).dot(u_trans.conj().T)
    eloc_orig += tmp
    #print(tmp)
    tmp = np.kron(tmp,np.eye(2))
    eks.append(tmp)
eks = np.array(eks)
eloc_orig = eloc_orig[:,:]/sumk.n_k

print('eloc_orig=')
print(eloc_orig)

for ik in range(sumk.n_k):
    eks[ik,:4,:4] -= np.kron(eloc_orig[:2,:2], np.eye(2))

test = np.zeros((16,16),dtype=complex)
for ik in range(sumk.n_k):
    test += eks[ik,:,:]
print('test=')
print(test[::2,::2]/sumk.n_k)

eloc_orig = np.kron(eloc_orig[:2,:2], np.eye(2))

#U, J = 0.5, 0.0
#nfix = 1.0
#nnom = 1.0
#tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
#eloc[0,0] += tmp_e
#eloc[1,1] += tmp_e
#eloc[2,2] += tmp_e
#eloc[3,3] += tmp_e
#eloc[4,4] += tmp_e
#eloc[5,5] += tmp_e

np.random.seed(1234)
# random initial value for hybridization
R0 = np.eye(nimp)*0.8
#R0 = np.random.rand(nbath//2, nimp//2)
#R0 = np.kron(R0, np.eye(2))

Lambda0 = eloc_orig[:4,:4] - mu0*np.eye(nimp)
#Lambda0 = np.zeros((nbath//2, nbath//2))
#Lambda0[0, 0], Lambda0[1, 1], Lambda0[2, 2] = 0.5, 0.5, 0.5
#Lambda0[3, 3], Lambda0[4, 4], Lambda0[5, 5] = 0.0, 0.0, 0.0
#Lambda0[6, 6], Lambda0[7, 7], Lambda0[8, 8] = -0.5, -0.5, -0.5
#Lambda0 = np.random.rand(nbath//2, nbath//2)
#Lambda0 = (Lambda0.conj().T+Lambda0)/2
#Lambda0 = np.kron(Lambda0, np.eye(2))
#Lambda0 = Lambda0 - mu0*np.eye(nbath)


Us = np.arange(0.4,3.1,0.1)

for U in [8.0]:#Us:
    J = 0.1*U
    nfix = 14.0
    nnom = 2.0
    eloc = np.zeros((ndp,ndp),dtype=complex)
    eloc[:4,:4] = eloc_orig.copy()[:4,:4]
    tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
    eloc[0,0] += tmp_e
    eloc[1,1] += tmp_e
    eloc[2,2] += tmp_e
    eloc[3,3] += tmp_e
    
    Utensor = U_matrix_kanamori(nimp//2, U, J)
   
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'pyscf_fci', 'restricted': True})
    grisb = Grisb_dp(ntot, nimp, ndp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'ci', 'use_Sz': True, 'use_Ntot': True})
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor)
    #grisb.run(itmax=200, mix=0.2, tol=1e-4, beta=500, silence=True, spin_pen=0.05, num_eig=5)
    mu = grisb.run_canonical(mu0 ,nfix=nfix, itmax=100, mix=0.2, tol=1e-4, beta=500,
                            silence=True, spin_pen=0.05, num_eig=5, dmu= 0.05)

    mu0 = mu 
    R0 = grisb.R
    Lambda0 = grisb.Lambda
    Z = R0.dot(R0.conj().T)

    ## compute Gf and Sig
    print('computing G and Sig ...')
    Nom = 500
    oms = np.linspace(-10,10,Nom)
    eta = 0.05
    grisb.compute_Gf_Sig(mu0, eks, oms, eta)

    import matplotlib.pyplot as plt
    plt.plot(oms, -grisb.Gf[:,0,0].imag/np.pi, 'b-')
    plt.plot(oms, -grisb.Gf[:,2,2].imag/np.pi, 'b-')
    plt.plot(oms, -grisb.Gf[:,4,5].imag/np.pi, 'r-')
    plt.plot(oms, -grisb.Gf[:,6,6].imag/np.pi, 'r-')
    plt.plot(oms, -grisb.Gf[:,8,8].imag/np.pi, 'r-')
    plt.plot(oms, -grisb.Gf[:,10,10].imag/np.pi, 'g-')
    plt.plot(oms, -grisb.Gf[:,12,12].imag/np.pi, 'g-')
    plt.plot(oms, -grisb.Gf[:,14,14].imag/np.pi, 'g-')
    plt.show()

    fo = open('U_Z.dat','a')
    print(U, Z[0,0].real, Z[2,2].real, file=fo)
    fo.close()
