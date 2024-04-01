import numpy as np
import triqs.utility.mpi as mpi
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.gf.tools import inverse
from triqs_ghostGA.sumk_grisb import SumkGRISB
import time

t_start = time.time()

# first we have to determine the mesh
sumk_mesh = MeshReFreq(window=[-10,10], n_w=200)

sumk = SumkGRISB(hdf_file='svo.h5',
                mesh=sumk_mesh, use_dft_blocks=False, h_field=0.0, nbath=6)

print(sumk.gf_struct_sumk)
print(sumk.gf_struct_sumk)

R = np.eye(3,dtype=complex)
Lambda = np.zeros((3,3),dtype=complex)
T = 0.005
sumk.calc_rhoks(R,Lambda,T)

#ikarray = np.array(list(range(sumk.n_k)))
#for ik in mpi.slice_array(ikarray):
#    print('ik=', ik)
#    print(sumk.hopping[ik])
##    G_latt_w = sumk.lattice_gf(ik=ik, mu=sumk.chemical_potential)
##    print(G_latt_w["up"].data[:,:])
##    print(G_latt_w["down"].data[:,:])
#
#t_end = time.time()
#t_elapse = t_end - t_start
#
#print('time elapsed=', t_elapse)
