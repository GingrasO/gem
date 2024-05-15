import numpy as np
import matplotlib.pyplot as plt
import triqs.utility.mpi as mpi
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.gf.tools import inverse
from triqs_ghostGA.sumk_grisb import SumkGRISB
from triqs_ghostGA.utils_TH import calc_nf
from triqs.plot.mpl_interface import oplot
from h5 import HDFArchive
from copy import deepcopy
import time
np.set_printoptions(suppress=True,precision=6)

t_start = time.time()

beta = 200.

# first we have to determine the mesh
#sumk_mesh = MeshReFreq(window=[-20,20], n_w=2000)
sumk_mesh = None

sumk = SumkGRISB(hdf_file='nio.h5',
                mesh=sumk_mesh, use_dft_blocks=False, beta=beta, h_field=0.0, nbath=5)
icrsh = 0
eloc = [{}]
eloc_full = {}
enloc_full = {}
for sp, isp in sumk.spin_names_to_ind[sumk.SO].items():
    ind = sumk.spin_names_to_ind[
                        sumk.corr_shells[icrsh]['SO']][sp]
    eloc[icrsh][sp] = np.zeros((5,5),dtype=complex)
    eloc_full[sp] = np.zeros((8,8),dtype=complex)
    enloc_full[sp] = np.zeros((8,8),dtype=complex)
    for ik in range(sumk.n_k):
        eloc[icrsh][sp] += sumk.bz_weights[ik] * sumk.rot_mat[icrsh].conjugate().transpose().dot(sumk.hopping[ik,ind,:5,:5]).dot(sumk.rot_mat[icrsh])
        eloc_full[sp] += sumk.bz_weights[ik] * sumk.hopping[ik,ind,:8,:8]
        enloc_full[sp] += sumk.bz_weights[ik] * sumk.hopping_nloc[ik,ind,:8,:8]

print('eloc=')
print(eloc)
print('Hsumk=')
print(sumk.Hsumk)
print('eloc_full=')
print(eloc_full)
print('enloc_full=')
print(enloc_full)
#quit()

mu = sumk.calc_mu(precision=0.001,beta=beta)
print('mu=',mu)
dm_test = sumk.density_matrix(method='using_gf')
#print(sumk.gf_struct_sumk)
#print(sumk.gf_struct_sumk)
#print(sumk.hopping.shape)
print('dm_test=')
print(dm_test)

#Gloc = sumk.extract_G_loc(broadening=0.1)
#
#mesh = Gloc[icrsh].mesh
#mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))
#
#plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,0,0].imag/np.pi,'b-',label='t2g')
#plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,1,1].imag/np.pi,'b-')
#plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,2,2].imag/np.pi,'b-')
#plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,3,3].imag/np.pi,'r-',label='eg')
#plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,4,4].imag/np.pi,'r-')
#plt.axvline(0)
#plt.legend()
#plt.show()

#sumk.eff_atomic_levels()
#print('Hsumk=')
#print(sumk.Hsumk)
#print('rot_mat=')
#print(sumk.rot_mat)
R = [{"up":np.eye(5,dtype=complex),"down":np.eye(5,dtype=complex)}]
Lambda = sumk.eloc_orig#np.zeros((3,3),dtype=complex) - mu*np.eye(3)
print('Hsumk=')
print(sumk.Hsumk)
T = 1/beta
sumk.calc_rhoks( R,Lambda,T)
#print(sumk.rhoks['up'][0,:,:])
#print(sumk.rhoks['down'][0,:,:])
sumk.calc_Delta()
print(sumk.Delta[icrsh]['up'])
print(sumk.Delta[icrsh]['down'])
#print(np.dot( np.dot(np.conjugate(sumk.rot_mat[icrsh]).transpose(), sumk.Delta[icrsh]['up']), sumk.rot_mat[icrsh] ) )
#print(np.dot( np.dot(np.conjugate(sumk.rot_mat[icrsh]).transpose(), sumk.Delta[icrsh]['down']), sumk.rot_mat[icrsh] ) )
sumk.calc_D(R, Lambda)
print(sumk.D[icrsh]['up'])
print(sumk.D[icrsh]['down'])

##ikarray = np.array(list(range(sumk.n_k)))
##print(len(sumk.spin_names_to_ind[1]))#sumk.SO])
##
##for icrsh in range(sumk.n_corr_shells):
##    dim = sumk.corr_shells[icrsh]['dim']
##    print('icrsh=',icrsh, 'dim=',dim)
##    for sp, isp in sumk.spin_names_to_ind[sumk.SO].items():
##        for ik in mpi.slice_array(ikarray):
##            print('ik=', ik, 'sp=', sp, 'isp=', isp)
##            print(sumk.hopping[ik,isp])
###    G_latt_w = sumk.lattice_gf(ik=ik, mu=sumk.chemical_potential)
###    print(G_latt_w["up"].data[:,:])
###    print(G_latt_w["down"].data[:,:])
#
#t_end = time.time()
#t_elapse = t_end - t_start
#
#print('time elapsed=', t_elapse)
