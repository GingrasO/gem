import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.plot.mpl_interface import oplot

archive = HDFArchive('out/vo2.h5', 'r')
Gphy = archive['gGA_results']['Gphy']
mesh = Gphy.mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

plt.plot(mesh_values,-Gphy['up'].data[:,0,0].imag/np.pi, 'b-', label='V1_dz2')
plt.plot(mesh_values,-Gphy['up'].data[:,1,1].imag/np.pi, 'r-', label='V1_dxz')
plt.plot(mesh_values,-Gphy['up'].data[:,2,2].imag/np.pi, 'g-', label='V1_dyz')
plt.plot(mesh_values,-Gphy['up'].data[:,3,3].imag/np.pi, 'c-', label='V1_dx2-y2')
plt.plot(mesh_values,-Gphy['up'].data[:,4,4].imag/np.pi, 'm-', label='V1_dxy')
plt.plot(mesh_values,-Gphy['up'].data[:,5,5].imag/np.pi, 'b--', label='V2_dz2')
plt.plot(mesh_values,-Gphy['up'].data[:,6,6].imag/np.pi, 'r--', label='V2_dxz')
plt.plot(mesh_values,-Gphy['up'].data[:,7,7].imag/np.pi, 'g--', label='V2_dyz')
plt.plot(mesh_values,-Gphy['up'].data[:,8,8].imag/np.pi, 'c--', label='V2_dx2-u2')
plt.plot(mesh_values,-Gphy['up'].data[:,9,9].imag/np.pi, 'm--', label='V2_dxy')
plt.plot(mesh_values,-Gphy['up'].data[:,10,10].imag/np.pi, '--', color='grey', label='O')
plt.plot(mesh_values,-Gphy['up'].data[:,11,11].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,12,12].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,13,13].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,14,14].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,15,15].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,16,16].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,17,17].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,18,18].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,19,19].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,20,20].imag/np.pi, '--', color='grey')
plt.plot(mesh_values,-Gphy['up'].data[:,21,21].imag/np.pi, '--', color='grey')

plt.legend()
plt.show()
