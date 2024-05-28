# ghostGA
from triqs_ghostGA.read_config import read_config
from triqs_ghostGA.sumk_grisb import SumkGRISB
# solid_dmft
#from solid_dmft.dmft_tools import interaction_hamiltonian
# triqs
import triqs.utility.mpi as mpi
from triqs.gf import Gf, make_hermitian, MeshReFreq, MeshImFreq
from h5 import HDFArchive
from triqs.operators import util, n, c, c_dag, Operator
from solid_dmft.dmft_tools import solver

# scipy sparse
from scipy import sparse

# system
import os
import numpy as np
from itertools import product

def build_fermion_op(ntot):
    ''' 
    Build fermionic operators for each spin+orbitals, alpha, with size 
    of Hilberspace A and B.
    Input:
        no: number of orbital
    Output:
        FH_list: a list of fermionic operator <A|[f^dagger_alpha]|B>
    '''
    strb = '{0:0'+str(ntot)+'b}' # string to save binary configuration
    hsize = 2**ntot
    FH_list = []
    print('Hibert space size:', hsize)
    for o in range(ntot): #calculate FH for each orbital
        row = []
        col = []
        data = []
        for b in range(hsize): # col |B>
            config_b = strb.format(b) # cofiguration correspond to B
            if config_b[o] == '0':
                col.append(b)
                config_a = ''
                #calculate the corresponding <A|
                for i in range(ntot):
                    if i == o:
                        config_a += '1'
                    else:
                        config_a += config_b[i]
                a = int(config_a,2)
                row.append(a)
                #calculate the exponent for minus sign
                expo = 0
                for i in range(o):
                    expo += int(config_b[i])
                data.append((-1.)**(expo))
        #assign values into sparse matrix
        FH = sparse.csr_matrix((data, (row, col)), shape=(hsize, hsize), dtype=np.float64)
        FH_list.append(FH)
    return FH_list

def build_Hint(V2E, FH_list):
    ''' 
    Build Hemb matrix.
    Input:  
        V2E: local two-body interaction
        FH_list: Fermionic operators c^dagger_{i}
    Return:
        Hint: embedding Hamiltonian
    '''
    ntot = len(FH_list)
    hsize = FH_list[0].shape[0]
    Hint = sparse.csr_matrix((hsize,hsize), dtype=np.complex128)
    #build local two-body part
    for i in range(ntot):
        for j in range(ntot):
            for k in range(ntot):
                for l in range(ntot):
                    if np.abs(V2E[i,k,j,l]) > 1e-6:
                        Hint += 0.5*V2E[i,k,j,l]*FH_list[i].dot(FH_list[j]).dot(FH_list[l].H).dot(FH_list[k].H)
    return Hint

from solid_dmft.dmft_tools import interaction_hamiltonian

config_file_name = "grisb_config.ini"
general_params, solver_params, dft_params, advanced_params = read_config(config_file_name)
sumk_mesh = MeshImFreq(beta=general_params['beta'],
                       S='Fermion',
                       n_iw=general_params['n_iw'])
sum_k = SumkGRISB(hdf_file=general_params['jobname']+'/'+general_params['seedname']+'.h5',
                  mesh=sumk_mesh, use_dft_blocks=False, h_field=general_params['h_field'])

h_int = interaction_hamiltonian.construct(sum_k, general_params, advanced_params)

print(h_int)

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

fundamental_operators = [
        c("up",0), c("down",0), c("up",1), c("down",1), c("up",2), c("down",2)]
    
ed = TriqsExactDiagonalization(h_int[0], fundamental_operators, 100.0)
print('ed.ed.ed=',ed.ed.E)
print('ed.ed.E0=',ed.ed.E0)
#egs = ed.get_ground_state_energy()
#print('ground state energy=', egs)

#from triqs_ghostGA.utils_TH import U_matrix_kanamori
def U_matrix_kanamori(n_orb, U_int, J_hund):
    r"""
    Calculate the Kanamori U and Uprime matrices.
    Parameters
    ----------
    n_orb : integer
            Number of orbitals in basis.
    U_int : scalar
            Value of the screened Hubbard interaction.
    J_hund : scalar
             Value of the Hund's coupling.
    Returns
    -------
    U_matrix : float numpy array
               The four-index interaction matrix in the chosen basis.
    """
    import itertools as it
    # TODO: Use the native TRIQS function.

    U_matrix = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.float64)
    m_range = range(n_orb)
    for m, mp in it.product(m_range, m_range):
        if m == mp:
            U_matrix[m, m, mp, mp] = U_int
        else:
            U_matrix[m, m, mp, mp] = U_int - 2.0 * J_hund
            U_matrix[m, mp, mp, m] = J_hund
            U_matrix[m, mp, m, mp] = J_hund
    print(U_matrix)
    norb = U_matrix.shape[0]
    norb2 = norb * 2
    Ufull_matrix = np.zeros((norb2, norb2, norb2, norb2), dtype=np.complex128)
    Ufull_matrix[::2, ::2, ::2, ::2] = U_matrix  # up, up
    Ufull_matrix[1::2, 1::2, 1::2, 1::2] = U_matrix  # dn, dn
    Ufull_matrix[::2, ::2, 1::2, 1::2] = U_matrix  # up, dn
    Ufull_matrix[1::2, 1::2, ::2, ::2] = U_matrix  # dn, up
    return Ufull_matrix#, u_avg, j_avg

Utensor = [ U_matrix_kanamori(3, general_params['U'][0], general_params['J'][0]) ]
#print(Utensor)

import triqs_ghostGA.interaction_hamiltonian as gGA_interaction_hamiltonian
Utensor = gGA_interaction_hamiltonian.construct(sum_k, general_params, advanced_params)[0]

FH_list = build_fermion_op(2*3)
Hint = build_Hint(Utensor, FH_list)

from scipy.linalg import eigh
evals, evecs = eigh(Hint.todense())
print(evals)

assert(np.allclose(evals, ed.ed.E))
