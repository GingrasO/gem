#!/usr/bin/env python
# from pyblock2.driver.core import DMRGDriver, SymmetryTypes
# Doesn't work now. pyblock2 conflicts with triqs!
import unittest
from triqs_ghostGA.sumk_grisb import SumkGRISB
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq

import numpy as np
import h5py
from qepack.grisb import Grisb
from qepack.utils import get_semicircle_e_list,U_matrix_kanamori
import os
from pyscf import dmrgscf
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''#mpirun -n 1 --bind-to none'

sumk_mesh = MeshImFreq(beta=200,
                       S='Fermion',
                       n_iw=1024)
sumk = SumkGRISB(hdf_file='svo.h5',
                 mesh=sumk_mesh, use_dft_blocks=False, h_field=0.0,
                 nbaths=[3])

mu0 = 0.0

# 2 orbital with 2 spins, 3 bath per orbital, total 16
#nimp, nbath, ntot = 6, 6, 12
nimp, nbath, ntot = 6, 18, 24

U, J = 0.5, 0.0
nfix = 1.0
nnom = 1.0
#eloc = np.kron(sumk.eloc_orig[0]['up'],np.eye(2)) - mu0*np.eye(nimp)
eloc_orig = np.zeros((nimp,nimp))
eloc = eloc_orig.copy()
tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
eloc[0,0] += tmp_e
eloc[1,1] += tmp_e
eloc[2,2] += tmp_e
eloc[3,3] += tmp_e
eloc[4,4] += tmp_e
eloc[5,5] += tmp_e
mu0 = 0.23802627291276476

# construct ek with semicircular DOS
e_list = get_semicircle_e_list(nmesh=100)
eks = []
test = np.zeros((3,3),dtype=complex)
for ik in range(sumk.n_k):
#for e in e_list:
#    tmp = e*np.eye(3,dtype=complex)
    tmp = sumk.hopping_nloc[ik,0,:,:]
    test += tmp
    #print(tmp)
    tmp = np.kron(tmp,np.eye(2))
    eks.append(tmp)
eks = np.array(eks)
print('test=')
print(test/sumk.n_k)

print('eloc=')
print(eloc)
#quit() 

np.random.seed(1234)
# random initial value for hybridization
#R0 = np.eye(nimp)*0.8
R0 = np.random.rand(nbath//2, nimp//2)
R0 = np.kron(R0, np.eye(2))

#Lambda0 = eloc - mu0*np.eye(nimp)
Lambda0 = np.zeros((nbath//2, nbath//2))
#Lambda0[0, 0], Lambda0[1, 1], Lambda0[2, 2] = 0.5, 0.5, 0.5
#Lambda0[3, 3], Lambda0[4, 4], Lambda0[5, 5] = 0.0, 0.0, 0.0
#Lambda0[6, 6], Lambda0[7, 7], Lambda0[8, 8] = -0.5, -0.5, -0.5
Lambda0 = np.random.rand(nbath//2, nbath//2)
Lambda0 = (Lambda0.conj().T+Lambda0)/2
Lambda0 = np.kron(Lambda0, np.eye(2))
Lambda0 = Lambda0 - mu0*np.eye(nbath)


Us = np.arange(1.0,3.1,0.2)

for U in [3.0]:#Us:
    J = 0.1*U
    nfix = 1.0
    nnom = 1.0
    eloc = np.kron(sumk.eloc_orig[0]['up'],np.eye(2)) #- mu*np.eye(nimp)
    tmp_e = -(U+(nimp//2-1)*(U-2*J)+(nimp//2-1)*(U-3*J))*(nnom-0.5)/(2*nimp//2-1)
    eloc[0,0] += tmp_e
    eloc[1,1] += tmp_e
    eloc[2,2] += tmp_e
    eloc[3,3] += tmp_e
    eloc[4,4] += tmp_e
    eloc[5,5] += tmp_e
    
    Utensor = U_matrix_kanamori(nimp//2, U, J)
   
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'pyscf_fci', 'restrict': True})
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'ci', 'use_Sz': True, 'use_Ntot': True})
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'pyscf_ccsd', 'restrict': True})
    grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'pyscf_dmrg', 'maxM': 200, 'restrict': True})
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor, R=R0, Lambda=Lambda0, ed_params={"solver":'pyblock2', 'use_Sz': True, 'use_Ntot': True, 'maxM': 500})
    #grisb = Grisb(ntot, nimp, nbath, eks, eloc, Utensor)
    #grisb.run(mu=mu0, itmax=2, mix=0.2, tol=1e-4, beta=500, silence=True, spin_pen=0.05, num_eig=5)
    mu = grisb.run_canonical(mu0 ,nfix=nfix, itmax=100, mix=0.4, tol=1e-3, beta=200,
                            silence=True, spin_pen=0.05, num_eig=5)

    #mu0 = mu 
    R0 = grisb.R
    Lambda0 = grisb.Lambda
    Z = R0.conj().T.dot(R0)

    ## compute Gf and Sig
    print('computing G and Sig ...')
    Nom = 500
    oms = np.linspace(-10,10,Nom)
    eta = 0.05
    grisb.compute_Gf_Sig(mu0, eks, oms, eta)

    Gloc = np.sum(grisb.Gf, axis=0)/sumk.n_k

    Z0test = 1/(1-(grisb.Sig[Nom//2,0,0]-grisb.Sig[Nom//2-1,0,0]).real/(oms[Nom//2]-oms[Nom//2-1]))
    Z1test = 1/(1-(grisb.Sig[Nom//2,2,2]-grisb.Sig[Nom//2-1,2,2]).real/(oms[Nom//2]-oms[Nom//2-1]))
    Z2test = 1/(1-(grisb.Sig[Nom//2,4,4]-grisb.Sig[Nom//2-1,4,4]).real/(oms[Nom//2]-oms[Nom//2-1]))

    if np.abs(U-3.0) < 1e-5:
        import matplotlib.pyplot as plt
        plt.plot(oms, -Gloc[:,0,0].imag/np.pi, 'b-')
        plt.plot(oms, -Gloc[:,2,2].imag/np.pi, 'b-')
        plt.plot(oms, -Gloc[:,4,4].imag/np.pi, 'b-')
        plt.show()


    fo = open('U_Z.dat','a')
    print(U, Z[0,0].real, Z[2,2].real, Z[4,4].real, Z0test, Z1test, Z2test, file=fo)
    fo.close()
