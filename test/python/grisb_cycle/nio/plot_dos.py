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
Gloc = sumk.extract_G_loc(broadening=0.1)

mesh = Gloc[icrsh].mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,0,0].imag/np.pi,'b-',label='t2g')
plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,1,1].imag/np.pi,'b-')
plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,2,2].imag/np.pi,'b-')
plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,3,3].imag/np.pi,'r-',label='eg')
plt.plot(mesh_values,-Gloc[icrsh]['up'].data[:,4,4].imag/np.pi,'r-')
plt.axvline(13.8931)
plt.xlim(0,20)
plt.legend()
plt.show()


