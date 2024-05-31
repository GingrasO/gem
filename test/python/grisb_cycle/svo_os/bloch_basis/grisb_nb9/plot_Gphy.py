import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.plot.mpl_interface import oplot

archive = HDFArchive('out/svo.h5', 'r')
Gphy = archive['gGA_results']['Gphy']
mesh = Gphy.mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gphy['up'].data[:,0,0].imag/np.pi)
plt.plot(mesh_values,-Gphy['up'].data[:,1,1].imag/np.pi)
plt.plot(mesh_values,-Gphy['up'].data[:,2,2].imag/np.pi)
plt.show()
