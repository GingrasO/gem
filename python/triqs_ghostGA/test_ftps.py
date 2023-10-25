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
np.set_printoptions(suppress=True, precision=6)

def setup_forkTPS(M, Norb, Nbath, gf_struct, int_params, w_grid, maxm, tw,
                  other_params={"dt": 0.1, "time_steps": 1, "sweeps": 10, "prep_napph": 5, "DMRGMethod": "TwoSite"}):
    """
    Given the embedded Hamiltonian and parameters, run ForkTPS and return density matrix.
        M               : Embedded Hamiltonian in a matrix of size (Norb+Nbath)x(Norb+Nbath), containing local hamiltonian,
                        hybridization with the bath, and bath degrees of freedom. The bath should be diagonal.
        Norb            : Number of physical orbitals.
        Nbath           : Number of bath degrees of freedom.
        gf_struct       : Structure of the impurity model.
        int_params      : Parameters of the interacting Hamiltonian.
        w_grid          : Bandwidth of the w-grid.
        maxm            : Maximal bound dimension.
        tw              : Cutoff parameter for the bound dimension.
        other_params    : Other parameters for ForkTPS.
    """
    
    def ConstructBath(gfstruc_ , Nbath_, SpinOrbCoup_, hopping_, eps_):
        """
        Construct the bath for the impurity solver.
            gfstruc_ : the structure of the impurity model
            Nbath_ : the number of bath sites for each band or impurity degree of freedom
            SpinOrbCoup_ : the spin-orbital coupling
            hopping_ : np.zeros([size, size, Nbath_], dtype=complex), hopping_[ i_imp, j_bath, ib] : the hopping amptitute from (j_bath orbital of the ib bath) to (i_imp orbital of the impurity)
            eps_ : np.zeros([size, Nbath_])
        """
        bath = Bath(gfstruc_, SpinOrbCoup_)
        for name,size in gfstruc_:
            for iorb in range(size):
                for ib in np.arange(Nbath_):
                    indx = [name, iorb]        
                    bath.addSite(indx, eps_[name][iorb, Nbath_- 1 - ib], hopping_[name][:, iorb, Nbath_ - 1 - ib]) # note that the first added bath site is placed at the end of the fork
        return bath
    
    # Construct the real time ForkTPS solver.
    S = ftps.Solver(gf_struct = gf_struct , nw = w_grid["nw"], wmin=w_grid["window"][0], wmax=w_grid["window"][1])

    # Fix the interacting Hamiltonian
    Hint = ftps.solver_core.HInt(u=int_params["U"], j=int_params["J"], up=int_params["Up"], dd=int_params["dd"])
    
    # Construct the local Hamiltonian and extract from M matrix
    e0 = ftps.solver_core.Hloc(gf_struct) #give the local Hamiltonian the right block structure
    e0.Fill("up", M["up"][:Norb, :Norb])
    e0.Fill("dn", M["dn"][:Norb, :Norb])

    # Construct the bath:
    # the bath sites are assigned to an orbital. Since it should be diagonal,
    # there is only one bath orbital, which is (orb, bath).
    eps = {"up": np.zeros((Norb, Nbath)),
           "dn": np.zeros((Norb, Nbath))}
    # Construct the hybridization:
    # now this couples a bath site and an orbital, so (orb, bath)->(orb).
    hopping = {"up": np.zeros((Norb, Norb, Nbath), dtype=complex),
               "dn": np.zeros((Norb, Norb, Nbath), dtype=complex)}
    # Mapping M to the correct shape for ForkTPS:
    for a in range(Norb):
        eps["up"][a, :] = np.diag(M["up"][Norb+a*Nbath:Norb+(a+1)*Nbath, Norb+a*Nbath:Norb+(a+1)*Nbath])
        eps["dn"][a, :] = np.diag(M["dn"][Norb+a*Nbath:Norb+(a+1)*Nbath, Norb+a*Nbath:Norb+(a+1)*Nbath])
        hopping["up"][:, a, :] = M["up"][:Norb, Norb+a*Nbath:Norb+(a+1)*Nbath]
        hopping["dn"][:, a, :] = M["dn"][:Norb, Norb+a*Nbath:Norb+(a+1)*Nbath]
    
    # Assigning in the solver object
    S.b = ConstructBath(gf_struct, Nbath, False, hopping, eps)
    S.e0 = e0
    
    # Setting up the time-evolution solver
    tevo = ftps.solver.TevoParams(dt = other_params["dt"], time_steps = other_params["time_steps"])
    # Setting up the DMRG parameters
    dmrg = ftps.solver.DMRGParams(sweeps = other_params["sweeps"],   
                                  prep_napph = other_params["prep_napph"],
                                  maxm=maxm, tw=tw, DMRGMethod=other_params["DMRGMethod"])
    
    # Solve the impurity model
    S.solve(h_int = Hint, 
            tevo = tevo, 
            params_partSector = dmrg,
            params_GS = dmrg
           )
    
    # Extract the single-particle density matrix and reshape it
    singleP = S.singleParticleDensity
    print('singleP=')
    print(singleP)
    rho_CDC = singleP[:(Norb*2*(Nbath+1))**2]
    rho_CDC = np.reshape(rho_CDC, (Norb*2*(Nbath+1), Norb*2*(Nbath+1)))

    A = list(range(2*Norb*(Nbath+1)))
    B = list(np.arange(0, 2*Norb*(1+Nbath), (1+Nbath)))
    for a, b in itp(range(2*Norb), range(Nbath)):
        B.append(1+b+a*(1+Nbath))
        
    rho_CDC[A, :] = rho_CDC[B, :]
    rho_CDC[:, A] = rho_CDC[:, B]
    
    # Return density matrix and interaction energy
    return rho_CDC, S.Ehint


if __name__ == "__main__":
    #import numpy as np
    #import scipy.linalg as lg
    #from scipy.optimize import minimize
    
    # from triqs.gf import *
    # import forktps as ftps
    # from forktps.solver import DMRGParams, TevoParams
    
    
    # from forktps.DiscreteBath import *
    # from forktps.Helpers import getX,MakeGFstruct
    
    #from h5 import *
    
    #from itertools import product as itp
    #from useful_func import setup_forkTPS
    
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
print('density matrix CI=')
print(denMat)
print('density matrix FTPS=')
print(singleP_tmp)


print('diff in density matrix=')
print(denMat-singleP_tmp)
