import h5py
import numpy as np
np.set_printoptions(suppress=True, precision=6)

fh5_w = h5py.File('nio.h5','r')
fh5_b = h5py.File('bloch_basis/nio.h5','r')

hopping_w = fh5_w['/dft_input/hopping'][...]
hopping_b = fh5_b['/dft_input/hopping'][...]
u_total = fh5_b['/dft_input/u_total'][...]

#print(hopping_w[0,0,:,:,0])#.shape)
#print(hopping_b[0,0,:,:,0])#.shape)
#print(hopping_w[0,0,:,:,1])#.shape)
#print(hopping_b[0,0,:,:,1])#.shape)
#print(u_total[0,0,:,:,1])#.shape)

h_w = hopping_w[0,0,:,:,0] + 1j*hopping_w[0,0,:,:,1]
h_b = hopping_b[0,0,:,:,0] + 1j*hopping_b[0,0,:,:,1]
u = u_total[0,0,:,:,0] + 1j*u_total[0,0,:,:,1]
print(h_w)
print(u.dot(h_b).dot(u.conj().T)+0.201382*np.eye(8))

fh5_w.close()
fh5_b.close()
