import numpy as np
#import scipy.linalg as lg
#from scipy.optimize import minimize
#
#from triqs.gf import *
# import forktps as ftps
#from forktps.solver import DMRGParams, TevoParams
#
#from forktps.DiscreteBath import *
#from forktps.Helpers import getX,MakeGFstruct
#
## from h5 import *
#
#from itertools import product as itp
#import triqs_ghostGA
#from triqs_ghostGA.utils_forktps import ConstructBath, setup_forkTPS, rotateBath, rotateDensityMatrix, rotateToTsungHanConvention
from triqs.ghostGA.forktps import *

np.set_printoptions(suppress=True, precision=6)


if __name__ == "__main__":

    import h5py
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

    #print("Norb: %i, Nbath: %i, U: %f" % (Norb, Nbath, U))
    #
    ## Local Hamiltonian
    #E = {"up": np.zeros((Norb, Norb)),
    #     "dn": np.zeros((Norb, Norb))}
    #
    ## Hybridization matrix
    #W = {"up": np.zeros((Norb, Nbath)),
    #     "dn": np.zeros((Norb, Nbath))}
    ##W["up"][:,:] = np.array([[0.5,0.4,0.5]])
    ##W["dn"][:,:] = np.array([[0.5,0.4,0.5]])
    #W["up"][:,:] = D[::2,::2].conj().T
    #W["dn"][:,:] = D[1::2,1::2].conj().T
    #
    ## Bath parameters
    #B = {"up": np.zeros((Nbath, Nbath)),
    #     "dn": np.zeros((Nbath, Nbath))}
    #B["up"][:,:] = Lambda_c[::2,::2]
    #B["dn"][:,:] = Lambda_c[1::2,1::2]
    ##B["up"][:,:] = np.array([[ 1.0, 0.0, 0.2],
    ##                         [ 0.0, 0.0, 0.0],
    ##                         [ 0.2, 0.0,-1.0]]) 
    ##B["dn"][:,:] = np.array([[ 1.0, 0.0, 0.2],
    ##                         [ 0.0, 0.0, 0.0],
    ##                         [ 0.2, 0.0,-1.0]]) 

   
    ## Half filled case: mu = -U/2 for local Hamiltonian
    #Filling=1.0
    #mu = 0.5 * U
    #E["up"] = E["up"] - mu * np.eye(Norb)
    #E["dn"] = E["dn"] - mu * np.eye(Norb)
    #
    ## Set up the M matrix which has all local Ham, hybridization and bath
    #M = {"up": np.block([[E["up"], W["up"]],
    #                     [W["up"].T, B["up"]]]),
    #     "dn": np.block([[E["dn"], W["dn"]],
    #                     [W["dn"].T, B["dn"]]])}
    #np.set_printoptions(precision=5, threshold=np.inf, linewidth=np.inf)
    #print('M["up"] before rotating the bath:')
    #print(M["up"])
    #print()
    #LAMBDA = B["up"]

    ## Rotate the Bath and Hybridization for smaller entropy
    #M, v = rotateBath(M, Norb, Nbath)
    #print(v)

    #print("M['up'] after rotating to the basis in which the bath is diagonal:")
    #print(M['up'])
    #print()

    ## Setting up some parameters for ForkTPS
    #gfstruct = [("up", Norb), ("dn", Norb)] # Structure of the Green's function
    ## Interaction parameters for Kanamori
    #int_params = {"U": U, "J": 0, "Up": 0, "dd": True}
    #
    #nw = 3001 # Number of real frequencies
    #window = [-3., 3.] # Bandwidth of the spectral function
    #w_grid = {"nw": 3001, "window": [-3., 3.]} # Resulting grid
    #
    #maxm = 300 # Maximum dimension bond for DMRG
    #
    ## Criteria for the bound dimension of the DMRG, just be converged
    #tw = 1e-20
    #
    ## Set up and run ForkTPS using the useful_func.py
    #singleP_rot, EHint = setup_forkTPS(M, Norb, Nbath, gfstruct, int_params,
    #                                   w_grid, maxm, tw)
    #
    #print(singleP_rot)
    #print(v)
    #singleP = rotateDensityMatrix(singleP_rot, Norb, Nbath, v)
    #print(singleP)
    #singleP = rotateToTsungHanConvention(singleP, Norb, Nbath)
   
    #print('density matrix=')
    #print(singleP)
 
    # Write the density matrix and interaction energy in a file that the ghost-GA code will read
    #with open("data_from_python_inv.dat", "w") as A:
    #    for a, b in itp(range(2*(Norb+Nbath)), repeat=2):
    #        A.write("%.8f\t%.8f\n" % (singleP[0][a, b].real, singleP[0][a, b].imag))
    #    # A.write("# Ehint")
    #    A.write("%.8f" % EHint[0])


    from ci import *
    edsolver = CI(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)
    
    h1e = np.zeros((ntot//2,ntot//2),dtype=complex)
    h1e[:nimp//2,:nimp//2] = eloc[::2,::2]
    h1e[:nimp//2,nimp//2:] = D[::2,::2].T
    h1e[nimp//2:,:nimp//2] = D[::2,::2].conj()
    h1e[nimp//2:,nimp//2:] = -Lambda_c[::2,::2]
    h1e = np.kron(h1e, np.eye(2))
    print('h1e=')
    print(h1e)
    #Utensor = np.zeros((2,2,2,2),dtype=complex)
    #Utensor[0,0,1,1] = U
    #Utensor[1,1,0,0] = U
    
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
