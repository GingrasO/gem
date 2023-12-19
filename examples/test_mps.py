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

##import julia interface
import juliacall
from juliacall import Main as jl
from juliacall import Pkg
GGA_base_dir="../"
julia_project_dir=GGA_base_dir+"julia/"
#from juliacall import Pkg
Pkg.activate(julia_project_dir)
include_str="include(\""+julia_project_dir+"src/driver.jl"+"\")"
jl.seval(include_str)

from triqs.ghostGA.mps import *
import triqs.operators.util
np.set_printoptions(suppress=True, precision=6)
def promote_to_spinful(up,dn):
    assert up.shape == dn.shape
    full=np.zeros(up.dtype,(up.shape[0]*2,up.shape[1]*2))
    full[::2,::2]=up
    full[1::2,1::2]=dn
    return full

def U_matrix_kanamori(n_orb, U_int, J_hund, Up_int=None, full_Uijkl=False, Jc_hund=None):
    r"""
    Calculate the Kanamori two-index interaction matrix for parallel spins:

    .. math:: U_{m m'}^{\sigma \sigma} \equiv U_{m m' m m'} - J_{m m'}

    with:

    .. math:: J_{m m'} \equiv U_{m m' m' m} ,

    and the two-index interaction matrix for anti-parallel spins:

    .. math:: U_{m m'}^{\sigma \bar{\sigma}} \equiv U_{m m' m m'}

    If full_Uijkl=True is specified the full four index
    Uijkl tensor is returned instead:

        .. math:: U_{m m m m} = U, \\
                  U_{m m' m m'} = U', \\
                  U_{m m' m' m} = J, \\
                  U_{m m m' m'} = J_C,

    with :math:`m \neq m'`.

    Parameters
    ----------
    n_orb : integer
            Number of orbitals in basis.
    U_int : float
            Value of the screened Hubbard interaction.
    J_hund : float
             Value of the Hund's coupling.
    Up_int : float, optional
            Value of the screened U prime parameter
            defaults to U_int-2*J_hund if not given.
            (fully rotationally invariant form)
    full_Uijkl : bool, optional
            retunr instead the full four-index Uijkl tensor
            default is False
    Jc_hund : foat, optional
            only used if full_Uijkl=True, defaults to J_hund

    Returns
    -------
    U : float numpy array
        The two-index interaction matrix for parallel spins or
        the four-index Uijkl tensor if full_Uijkl=True
    Uprime : float numpy array
        The two-index interaction matrix for anti-parallel spins.

    """

    # Jc_hund can only be used if the full tensor is returned
    if Jc_hund is not None and not full_Uijkl:
        raise ValueError('Jc_hund can only be specified if the full four index tensor is returned')

    if Up_int is None:
        Up_int = U_int-2*J_hund
    if Jc_hund is None:
        Jc_hund = J_hund

    m_range = range(n_orb)

    if not full_Uijkl:
        U = np.zeros((n_orb, n_orb), dtype=float)      # matrix for same spin
        Uprime = np.zeros((n_orb, n_orb), dtype=float)  # matrix for opposite spin

        for m, mp in product(m_range, m_range):
            if m == mp:
                Uprime[m, mp] = U_int
            else:
                U[m, mp] = Up_int - J_hund
                Uprime[m, mp] = Up_int

        return U, Uprime
    else:
        U_kan = np.zeros((n_orb, n_orb, n_orb, n_orb))

        for i, j, k, l in product(m_range, m_range, m_range, m_range):
            if i == j == k == l:  # Uiiii
                U_kan[i, j, k, l] = U_int
            elif i == k and j == l:  # Uijij
                U_kan[i, j, k, l] = Up_int
            elif i == l and j == k:  # Uijji
                U_kan[i, j, k, l] = J_hund
            elif i == j and k == l:  # Uiijj
                U_kan[i, j, k, l] = Jc_hund
        return U_kan

if __name__ == "__main__":

    import h5py
    # fh5 = h5py.File('data/hemb_test_1orb3bath.h5')
    #fh5 = h5py.File('data/hemb_test.h5')
    #D = fh5['D'][...]
    #Lambda_c = fh5['Lambda_c'][...]
    #Utensor = fh5['Utensor'][...]
    #eloc = fh5['eloc'][...]
    #mu = fh5['mu'][...]
    #fh5.close()
    Nimp=5
    Nbath=5*Nimp
    N=Nimp+Nbath
    U=3.0
    J=0.4
    #Build quartic part
    Utype="Slater"    #"Kanamori" or "Slater"
    if Utype=="Kanamori":
        try:
            Utensor=triqs.operators.util.U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=True)
        except:
            Utensor=U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=True)
        finally:
            print("Kanamori UTensor construction breaks")
        #Utensor=U_matrix_kanamori(Nimp,U_int=U,J_hund=)
        #Um,Ump=triqs.operators.U_matrix_kanamori(Nimp,U_int=U,J_hund=J)
    elif Utype=="Slater":
        Utensor=triqs.operators.U_matrix_slater(Nimp//2,U_int=U,J_hund=J)

    
    assert(np.allclose(Lambda_c[::2,::2],Lambda_c[1::2,1::2]))
    assert(np.allclose(Lambda_c.imag,np.zeros(Lambda_c.shape)))
    assert(np.allclose(D.imag,np.zeros(D.shape)))
    
    #Build quadratic part
    
    np.random.seed(1234)
    eloc_spinless=np.random.rand(Nimp,Nimp)
    eloc_spinless+=0.5*eloc_spinless.conjugate().t
    #eloc=np.zeros((2*Nimp,2*Nimp))
    #eloc[::2,::2]=eloc_spinless
    #eloc[1::2,1::2]=eloc_spinless
    eloc=promote_to_spinful(eloc_spinless)
    
    W=3.0
    bathenergies=np.sort(2*W*(np.random.rand(Nbath)-0.5))
    Lambda_spinless=np.diag(bathenergies)
    Lambda_c=promote_to_spinful(Lambda_spinless,Lambda_spinless)

    D_spinless=np.random.rand(Nbath,Nimp)
    D=promote_to_spinful(D_spinless,D_spinless)
    #R0 = np.random.rand(nbath//2,nimp//2)
    #Gamma=diagm(sort( 2*W *(rand(Nbath).-0.5)))
    # Number of physical orbitals, bath sites and interaction parameter
    #Norb, Nbath, U = eloc.shape[0]//2, Lambda_c.shape[0]//2, Utensor[0,0,1,1] #1, 3, 1.0

    solver = MPS(ntot, nimp, nbath)
    solver.make_kwargs(use_Sz=True,use_Ntot=True,spin_pen=0.0)
    solver.make_schedule()  ##default schedule, probably overkill for 3-orbital model
    solver.set_tolerances(["E","rho"],[1e-5,5e-3])
    solver.build_Hemb(D, eloc, Lambda_c, Utensor)

    solver.solve_Hemb()
    denMat_ftps = solver.calc_density_matrix()
    print('denMat_mps=')
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
