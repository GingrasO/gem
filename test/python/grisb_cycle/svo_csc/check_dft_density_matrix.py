from h5 import HDFArchive
import numpy as np

def nf(x,beta):
    return 1./(np.exp(beta*x)+1)

with HDFArchive('svo.h5', 'r') as archive:
    dft_fermi_energy = archive['dft_misc_input']['dft_fermi_energy']
    dft_fermi_weights = archive['dft_misc_input']['dft_fermi_weights'][...]
    hopping = archive['dft_input']['hopping'][...]

beta = (1./0.0015)/13.605703976#3.67
#ik = 1

for ik in range(124):
    print('==================== ik=%d ======================='%(ik))
    print('dft_fermi_energy=', dft_fermi_energy)
    print('hopping=')
    print(hopping[ik,0,:,:])
    print('nf(hopping.diagonal())')
    print(nf(hopping[ik,0,0,0],beta), nf(hopping[ik,0,1,1],beta), nf(hopping[ik,0,2,2],beta))
    print('dft_fermi_weights=')
    print(dft_fermi_weights[ik,0,:])

