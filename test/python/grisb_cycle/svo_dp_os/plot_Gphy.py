import matplotlib.pyplot as plt
import numpy as np
from h5 import HDFArchive
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from triqs.plot.mpl_interface import oplot

archive = HDFArchive('out/svo.h5', 'r')
Gphy = archive['gGA_results']['Gphy']
mesh = Gphy.mesh
mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))

dos_t2g = -(Gphy['up'].data[:,0,0] + Gphy['up'].data[:,1,1] + Gphy['up'].data[:,2,2]).imag/np.pi
dos_uncorr = -(Gphy['up'].data[:,3,3] + Gphy['up'].data[:,4,4] + Gphy['up'].data[:,5,5]
              +Gphy['up'].data[:,6,6] + Gphy['up'].data[:,7,7] + Gphy['up'].data[:,8,8]
              +Gphy['up'].data[:,9,9] + Gphy['up'].data[:,10,10] + Gphy['up'].data[:,11,11]
              +Gphy['up'].data[:,12,12] + Gphy['up'].data[:,13,13]
              ).imag/np.pi

plt.plot(mesh_values,dos_t2g, 'b-', label='V-t2g')
plt.plot(mesh_values,dos_uncorr, 'g-', label='uncorr')
plt.plot(mesh_values,dos_t2g+dos_uncorr, 'k-', label='total')

plt.axvline(0,color='k',ls='-')
plt.ylim(0,)
plt.xlim(-8,8)
plt.xlabel(r'$\omega$ (eV)',size=15)
plt.ylabel('DOS (1/eV)',size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(loc='best',fontsize=15)
plt.tight_layout()
#plt.savefig('DOS.pdf')
plt.show()

#plt.plot(mesh_values,-Gphy['up'].data[:,0,0].imag/np.pi, 'b-')
#plt.plot(mesh_values,-Gphy['up'].data[:,1,1].imag/np.pi, 'b-')
#plt.plot(mesh_values,-Gphy['up'].data[:,2,2].imag/np.pi, 'b-')
#plt.plot(mesh_values,-Gphy['up'].data[:,3,3].imag/np.pi, 'g-')
#plt.plot(mesh_values,-Gphy['up'].data[:,4,4].imag/np.pi, 'g-')
#plt.plot(mesh_values,-Gphy['up'].data[:,5,5].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,6,6].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,7,7].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,8,8].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,9,9].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,10,10].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,11,11].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,12,12].imag/np.pi, '-', color='grey')
#plt.plot(mesh_values,-Gphy['up'].data[:,13,13].imag/np.pi, '-', color='grey')


