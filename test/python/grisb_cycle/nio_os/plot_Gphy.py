import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.plot.mpl_interface import oplot

archive = HDFArchive('out/nio.h5', 'r')
Gphy = archive['gGA_results']['Gphy']
mesh = Gphy.mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gphy['up'].data[:,0,0].imag/np.pi, 'b-', label='Ni-eg')
plt.plot(mesh_values,-Gphy['up'].data[:,1,1].imag/np.pi, 'r-', label='Ni-t2g')
plt.plot(mesh_values,-Gphy['up'].data[:,2,2].imag/np.pi, 'r-')
plt.plot(mesh_values,-Gphy['up'].data[:,3,3].imag/np.pi, 'b-')
plt.plot(mesh_values,-Gphy['up'].data[:,4,4].imag/np.pi, 'r-')
plt.plot(mesh_values,-Gphy['up'].data[:,5,5].imag/np.pi, 'g-', label='O-p')
plt.plot(mesh_values,-Gphy['up'].data[:,6,6].imag/np.pi, 'g-')
plt.plot(mesh_values,-Gphy['up'].data[:,7,7].imag/np.pi, 'g-')
plt.legend()
plt.show()
