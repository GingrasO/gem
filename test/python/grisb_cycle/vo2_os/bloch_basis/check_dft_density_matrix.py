from h5 import HDFArchive
import numpy as np
from triqs_ghostGA.utils_TH import calc_nf
np.set_printoptions(suppress=True)

def nf(x,beta):
    return 1./(np.exp(beta*x)+1)

with HDFArchive('vo2.h5', 'r') as archive:
    dft_fermi_energy = archive['dft_misc_input']['dft_fermi_energy']
    dft_fermi_weights = archive['dft_misc_input']['dft_fermi_weights'][...]
    hopping = archive['dft_input']['hopping'][...]
    u_total = archive['dft_input']['u_total'][...]

beta = (1./0.0015)/13.605703976#3.67
#ik = 1

for ik in range(124):
    #print('==================== ik=%d ======================='%(ik))
    #print('dft_fermi_energy=', dft_fermi_energy)
    #print('hopping=')
    #print(hopping[ik,0,:,:])
    fermi_weights_calc = []
    for io in range(22):
        fermi_weights_calc.append(nf(hopping[ik,0,io,io],beta).real)
    Hkorb = np.dot(np.dot(u_total[0,ik,:,:], hopping[ik,0,:,:]), u_total[0,ik,:,:].conj().T)
    rhok_orb = calc_nf(Hkorb,1./beta)
    #print('fermi_weights_calc=')
    #print(fermi_weights_calc)
    #print('dft_fermi_weights=')
    #print(dft_fermi_weights[ik,0,:])
    #print('rhok_orb=')
    #print(rhok_orb)
    #print('rhok_bnd=')
    #print(np.dot(np.dot(u_total[0,ik,:,:].conj().T, rhok_orb), u_total[0,ik,:,:]))
    rhok_bnd = calc_nf(hopping[ik,0,:,:],1./beta)
    #print('rhok_bnd2orb=')
    #print(np.dot(np.dot(u_total[0,ik,:,:], rhok_bnd), u_total[0,ik,:,:].conj().T))
    assert(np.allclose(rhok_orb, np.dot(np.dot(u_total[0,ik,:,:], rhok_bnd), u_total[0,ik,:,:].conj().T), atol=1e-4))
    assert(np.allclose(dft_fermi_weights[ik,0,:],fermi_weights_calc, atol=1e-3))
