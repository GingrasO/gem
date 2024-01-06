from triqs_ghostGA.ftps import *
from triqs_ghostGA.ci import *
import numpy as np
import h5py

np.set_printoptions(suppress=True, precision=6)


if __name__ == "__main__":

    # fh5 = h5py.File('data/hemb_test_1orb3bath.h5')
    fh5 = h5py.File('data/hemb_test.h5')
    D = fh5['D'][...]
    Lambda_c = fh5['Lambda_c'][...]
    Utensor = fh5['Utensor'][...]
    eloc = fh5['eloc'][...]
    mu = fh5['mu'][...]
    fh5.close()

    assert(np.allclose(Lambda_c[::2,::2],Lambda_c[1::2,1::2]))
    assert(np.allclose(Lambda_c.imag,np.zeros(Lambda_c.shape)))
    assert(np.allclose(D.imag,np.zeros(D.shape)))

    # Number of physical orbitals, bath sites and interaction parameter
    #Norb, Nbath, U = eloc.shape[0]//2, Lambda_c.shape[0]//2, Utensor[0,0,1,1] #1, 3, 1.0
    nimp = eloc.shape[0]
    nbath = Lambda_c.shape[0]
    ntot = nimp + nbath

    solver = FTPS(ntot, nimp, nbath, maxM=300)
    solver.build_Hemb(D, eloc, Lambda_c, Utensor)
    solver.solve_Hemb()
    denMat_ftps = solver.calc_density_matrix()
    print('denMat_ftps=')
    print(denMat_ftps)

    edsolver = CI(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)

    h1e = np.zeros((ntot//2,ntot//2),dtype=complex)
    h1e[:nimp//2,:nimp//2] = eloc[::2,::2]
    h1e[:nimp//2,nimp//2:] = D[::2,::2].T
    h1e[nimp//2:,:nimp//2] = D[::2,::2].conj()
    h1e[nimp//2:,nimp//2:] = -Lambda_c[::2,::2]
    h1e = np.kron(h1e, np.eye(2))
    print('h1e=')
    print(h1e)

    edsolver.build_Hemb(h1e, Utensor, spin_pen=0.0, sz_pen=0.0)
    edsolver.solve_Hemb(num_eig=1, verbose=True )
    denMat = edsolver.calc_density_matrix()
    np.set_printoptions(precision=3, threshold=np.inf, linewidth=np.inf)
    print('density matrix CI=')
    print(denMat)
    print('density matrix FTPS=')
    print(denMat_ftps)


    print('diff in density matrix=')
    print(denMat-denMat_ftps)
