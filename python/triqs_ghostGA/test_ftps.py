import numpy as np
import scipy.linalg as lg
from scipy.optimize import minimize

from triqs.gf import *
import forktps as ftps
from forktps.solver import DMRGParams, TevoParams

from forktps.DiscreteBath import *
from forktps.Helpers import getX,MakeGFstruct

# from h5 import *

from itertools import product as itp
import triqs_ghostGA
from triqs_ghostGA.utils_forktps import ConstructBath, setup_forkTPS, rotateBath, rotateDensityMatrix

np.set_printoptions(suppress=True, precision=6)


if __name__ == "__main__":
    print("\tWelcome in the python script.")
    
    # Number of physical orbitals, bath sites and interaction parameter
    Norb, Nbath, U = 1, 3, 1.0
    
    print("Norb: %i, Nbath: %i, U: %f" % (Norb, Nbath, U))
    
    # Local Hamiltonian
    E = {"up": np.zeros((Norb, Norb)),
         "dn": np.zeros((Norb, Norb))}
    
    # Hybridization matrix
    W = {"up": np.zeros((Norb, Nbath)),
         "dn": np.zeros((Norb, Nbath))}
    W["up"][:,:] = np.array([[0.5,0.4,0.5]])
    W["dn"][:,:] = np.array([[0.5,0.4,0.5]])
    
    # Bath parameters
    B = {"up": np.zeros((Nbath, Nbath)),
         "dn": np.zeros((Nbath, Nbath))}
    B["up"][:,:] = np.array([[ 1.0, 0.0, 0.2],
                             [ 0.0, 0.0, 0.0],
                             [ 0.2, 0.0,-1.0]]) 
    B["dn"][:,:] = np.array([[ 1.0, 0.0, 0.2],
                             [ 0.0, 0.0, 0.0],
                             [ 0.2, 0.0,-1.0]]) 
   
    # Half filled case: mu = -U/2 for local Hamiltonian
    Filling=1.0
    mu = 0.5 * U
    E["up"] = E["up"] - mu * np.eye(Norb)
    E["dn"] = E["dn"] - mu * np.eye(Norb)
    
    # Set up the M matrix which has all local Ham, hybridization and bath
    M = {"up": np.block([[E["up"], W["up"]],
                         [W["up"].T, B["up"]]]),
         "dn": np.block([[E["dn"], W["dn"]],
                         [W["dn"].T, B["dn"]]])}
    print('M["up"] before rotating the bath:')
    print(M["up"])
    print()
    LAMBDA = B["up"]

    # Rotate the Bath and Hybridization for smaller entropy
    M, v = rotateBath(M, Norb, Nbath)

    print("M['up'] after rotating to the basis in which the bath is diagonal:")
    print(M['up'])
    print()

    # Setting up some parameters for ForkTPS
    gfstruct = [("up", Norb), ("dn", Norb)] # Structure of the Green's function
    # Interaction parameters for Kanamori
    int_params = {"U": U, "J": 0, "Up": 0, "dd": True}
    
    nw = 3001 # Number of real frequencies
    window = [-3., 3.] # Bandwidth of the spectral function
    w_grid = {"nw": 3001, "window": [-3., 3.]} # Resulting grid
    
    maxm = 300 # Maximum dimension bond for DMRG
    
    # Criteria for the bound dimension of the DMRG, just be converged
    tw = 1e-20
    
    # Set up and run ForkTPS using the useful_func.py
    singleP_rot, EHint = setup_forkTPS(M, Norb, Nbath, gfstruct, int_params,
                                       w_grid, maxm, tw)
    
    singleP = rotateDensityMatrix(singleP_rot, Norb, Nbath, v)
   
    print('density matrix=')
    print(singleP)
 
    # Write the density matrix and interaction energy in a file that the ghost-GA code will read
    #with open("data_from_python_inv.dat", "w") as A:
    #    for a, b in itp(range(2*(Norb+Nbath)), repeat=2):
    #        A.write("%.8f\t%.8f\n" % (singleP[0][a, b].real, singleP[0][a, b].imag))
    #    # A.write("# Ehint")
    #    A.write("%.8f" % EHint[0])


    from ci import *
    print(LAMBDA)
    edsolver = CI(2*(Norb+Nbath), use_Ntot=True, use_Sz=True, dtype=np.complex128)
    
    h1e = np.zeros((Norb+Nbath,Norb+Nbath),dtype=complex)
    h1e[:Norb,:Norb] = E['up'] 
    h1e[:Norb,Norb:] = W['up']
    h1e[Norb:,:Norb] = W['up'].T
    h1e[Norb:,Norb:] = LAMBDA
    h1e = np.kron(h1e, np.eye(2))
    print('h1e=')
    print(h1e)
    Utensor = np.zeros((2,2,2,2),dtype=complex)
    Utensor[0,0,1,1] = U
    Utensor[1,1,0,0] = U
    
    edsolver.build_Hemb(h1e, Utensor, spin_pen=0.0, sz_pen=0.0)
    edsolver.solve_Hemb(num_eig=1, verbose=True )
    denMat = edsolver.calc_density_matrix()
    np.set_printoptions(precision=3, threshold=np.inf, linewidth=np.inf)
    print('density matrix CI=')
    print(denMat)
    print('density matrix FTPS=')
    print(singleP)
    
    
    print('diff in density matrix=')
    print(denMat-singleP)
