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
from triqs_ghostGA.utils_forktps import ConstructBath, setup_forkTPS

np.set_printoptions(suppress=True, precision=6)


if __name__ == "__main__":
    print("\tWelcome in the python script.")
    
    # Number of physical orbitals, bath sites and interaction parameter
    Norb, Nbath, U = 1, 3, 1.0
    
    print("Norb: %i, Nbath: %i, U: %f" % (Norb, Nbath, U))
    
    # Local Hamiltonian
    E = {"up": np.zeros((Norb, Norb)),
         "dn": np.zeros((Norb, Norb))}
    #E["up"][0,0] = -U/2.
    #E["dn"][0,0] = -U/2.
    
    # Hybridization matrix
    W = {"up": np.zeros((Norb, Nbath)),
         "dn": np.zeros((Norb, Nbath))}
    W["up"][:,:] = np.array([[0.3,0.4,0.5]])
    W["dn"][:,:] = np.array([[0.3,0.4,0.5]])
    
    # Bath parameters
    B = {"up": np.zeros((Nbath, Nbath)),
         "dn": np.zeros((Nbath, Nbath))}
    B["up"][:,:] = np.array([[ 1.0, 0.0, 0.0],
                             [ 0.0, 0.0, 0.0],
                             [ 0.0, 0.0,-1.0]]) 
    B["dn"][:,:] = np.array([[ 1.0, 0.0, 0.0],
                             [ 0.0, 0.0, 0.0],
                             [ 0.0, 0.0,-1.0]]) 
   
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
    
    # Setting up some parameters for ForkTPS
    gfstruct = [("up", Norb), ("dn", Norb)] # Structure of the Green's function
    int_params = {"U": U, "J": 0, "Up": 0, "dd": True} # Interaction parameters for Kanamori
    
    nw = 3001 # Number of real frequencies
    window = [-3., 3.] # Bandwidth of the spectral function
    w_grid = {"nw": 3001, "window": [-3., 3.]} # Resulting grid
    
    maxm = 300 # Maximum dimension bond for DMRG
    
    # Full Hamiltonian (was useful for check ups?)
    ##########
    M_full = np.block([[E["up"], np.zeros((Norb, Norb)), W["up"], np.zeros((Norb, Norb*Nbath))],
                       [np.zeros((Norb, Norb)), E["dn"], np.zeros((Norb, Norb*Nbath)), W["dn"]],
                       [W["up"].T, np.zeros((Norb*Nbath, Norb)), B["up"], np.zeros((Norb*Nbath, Norb*Nbath))],
                       [np.zeros((Norb*Nbath, Norb)), W["dn"].T, np.zeros((Norb*Nbath, Norb*Nbath)), B["dn"]]])
    
    A = list(range(0, 2*Norb))
    B = list(range(0, 2*Norb, 2)) + list(range(1, 2*Norb, 2))
    M_full[A, :] = M_full[B, :]
    M_full[:, A] = M_full[:, B]
    
    A = list(range(2*Norb, 2*Norb+2*Norb*Nbath))
    B = []
    for a in range(Norb):
        B += list(range(2*Norb+2*a*Nbath, 2*Norb+(1+2*a)*Nbath))
    for a in range(Norb):
        B += list(range(2*Norb+(1+2*a)*Nbath, 2*Norb+(2+2*a)*Nbath))
    M_full[[A], :] = M_full[[B], :]
    M_full[:, [A]] = M_full[:, [B]]
    ##########
    
    # Preparing the rotated Embedded Hamiltonian
    M_rot = {"up": np.copy(M["up"]),
             "dn": np.copy(M["dn"])}
    
    # Obtained the eigenvectors of the bath sites to rotate the matrix 
    v_all = {"up": [], "dn": []}
    for name in ["up", "dn"]:
        B = M[name][Norb:, Norb:] # Bath sites
        # W = M[name][:Norb, Norb:]
        # Wd = M[name][Norb:, :Norb]
    
        w, v = np.linalg.eig(B) # Diagonalization of the bath
        v_all[name] = np.block([[np.eye(Norb), np.zeros((Norb, Nbath*Norb))],
                                [np.zeros((Norb*Nbath, Norb)), v]]) # Keep the eigenvectors in memory
    
        M_rot[name] = np.linalg.inv(v_all[name]) @ M_rot[name] @ v_all[name] # Rotate the embedded Hamiltonian
    
    print("M['up'] after rotating to the basis in which the bath is diagonal:")
    print(M_rot['up'])
    print()
    
    # Criteria for the bound dimension of the DMRG, just be converged
    tws = [1e-20] # , 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    singleP = []
    EHint = []
    
    for tw in tws:
        # ForkTPS writes the density matrix in a basis that mixes spin up and down.
        # These list help convert to separate up and down.
        list_up = list(range(0, 2*Norb, 2))
        list_dn = list(range(1, 2*Norb, 2))
        for a in range(Norb):
            list_up += list(range(Norb*2+2*a*Nbath, Norb*2+(2*a+1)*Nbath))
            list_dn += list(range(Norb*2+(2*a+1)*Nbath, Norb*2+(2*a+2)*Nbath))
    
        # Set up and run ForkTPS using the useful_func.py
#        singleP_tmp, EHint_tmp = setup_forkTPS(M_rot, Norb, Nbath, gfstruct, int_params,
#                                               w_grid, maxm, tw)
        singleP_tmp, EHint_tmp = setup_forkTPS(M, Norb, Nbath, gfstruct, int_params,
                                               w_grid, maxm, tw)
        
        # Extract density matrix for up and rotate back to the original basis,
        # before the bath was diagonalized.
#        single_up = singleP_tmp[list_up, :][:, list_up]
#        v_up = v_all["up"]
#        single_up = np.linalg.inv(v_up).T @ single_up @ v_up.T
#        
#        # Same for down
#        single_dn = singleP_tmp[list_dn, :][:, list_dn]
#        v_dn = v_all["dn"]
#        single_dn = np.linalg.inv(v_dn).T @ single_dn @ v_dn.T
#        
#        # Replace in the density matrix
#        for a, A in enumerate(list_up):
#            for b, B in enumerate(list_up):
#                singleP_tmp[A, B] = single_up[a, b]
#        for a, A in enumerate(list_dn):
#            for b, B in enumerate(list_dn):
#                singleP_tmp[A, B] = single_dn[a, b]
#        
#        singleP.append(singleP_tmp)
#        EHint.append(EHint_tmp)
#   
#    print('density matrix=')
#    print(singleP[0])
    print('density matrix=')
    print(singleP_tmp)
 
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
print(singleP_tmp)


print('diff in density matrix=')
print(denMat-singleP_tmp)
