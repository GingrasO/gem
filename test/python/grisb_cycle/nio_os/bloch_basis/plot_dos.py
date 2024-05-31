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
sumk_mesh = MeshReFreq(window=[-20,20], n_w=2000)

sumk = SumkGRISB(hdf_file='nio.h5',
                mesh=sumk_mesh, use_dft_blocks=False, beta=beta, h_field=0.0, nbath=5)
icrsh = 0
R = [{}]
R[icrsh]['up'] = np.eye(5,dtype=complex)
R[icrsh]['down'] = np.eye(5,dtype=complex)
Lambda = sumk.eloc_orig
Gloc = sumk.extract_G_phy(R, Lambda, broadening=0.1)

mesh = Gloc.mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gloc['up'].data[:,0,0].imag/np.pi,'b-',label='Ni-eg')
plt.plot(mesh_values,-Gloc['up'].data[:,1,1].imag/np.pi,'r-',label='Ni-t2g')
plt.plot(mesh_values,-Gloc['up'].data[:,2,2].imag/np.pi,'r-')
plt.plot(mesh_values,-Gloc['up'].data[:,3,3].imag/np.pi,'b-')
plt.plot(mesh_values,-Gloc['up'].data[:,4,4].imag/np.pi,'r-')
plt.plot(mesh_values,-Gloc['up'].data[:,5,5].imag/np.pi,'g-',label='o-p')
plt.plot(mesh_values,-Gloc['up'].data[:,6,6].imag/np.pi,'g-')
plt.plot(mesh_values,-Gloc['up'].data[:,7,7].imag/np.pi,'g-')
plt.axvline(13.8931)
plt.xlim(0,20)
plt.legend()
plt.show()


