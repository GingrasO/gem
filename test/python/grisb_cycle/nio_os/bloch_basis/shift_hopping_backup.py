#import h5py
from h5 import HDFArchive
import numpy as np
np.set_printoptions(suppress=True, precision=6)

#fh5 = h5py.File('nio.h5','r+')

with HDFArchive('nio.h5', 'a') as archive:
    hopping = archive['dft_input']['hopping'][...]
    u_total = archive['dft_input']['u_total'][...]
    
    hopping_shift = hopping.copy()
    
    #print(hopping.shape) #(125, 1, 8, 8)
    for ik in range(125):
        hopping_shift[ik,0,:,:] = hopping[ik,0,:,:]+0.201382*np.eye(8)
        
    #print(hopping[0,0,:,:])
    #print(hopping_shift[0,0,:,:])
    
    #h_b = hopping_shift[0,0,:,:]
    #u = u_total[0,0,:,:]
    #print(u.dot(h_b).dot(u.conj().T))
    ##print(u.dot(hopping_shift[0,0,:,:,0]).dot(u.conj().T)+0.201382*np.eye(8))
    #
    del archive['dft_input']['hopping']
    
    archive['dft_input']['hopping'] = hopping_shift


