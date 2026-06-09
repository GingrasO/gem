import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

# 1 orbital with 2 spins, B bath per orbital
B = 3
nimp = 2
nbath = nimp * B
ntot = nimp + nbath

Nk = 200
Kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
Ky = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

t = 0.25

ek_list = []
for ix, kx in enumerate(Kx):
    for iy, ky in enumerate(Ky):
        gamma_k = -t * (1.0
                    + np.exp(-1j * (kx + ky))
                    + np.exp(-1j * ky)
                    + np.exp(-1j * kx))
        Hk_spinless = np.array([[0.0, gamma_k], [gamma_k.conj(), 0.0]], dtype=np.complex128)
        ek_list.append(np.kron(Hk_spinless, np.eye(2)))

eks = np.array(ek_list)
wks = np.ones(len(ek_list)) / len(ek_list)

lattice = Lattice(eks, wk_list=wks)

itmax = 100
mix = 0.01
tol = 1e-3
spin_pen= 0.0
mu = 0.0
bfield = 1e-2

U_list = np.array([0.5,1.5,2.0,2.5,4.0]) #np.array([0.5,2.0,3.5])
T_list = np.linspace( 0, 0.10, 51)

# mag_grid[iU, iT] = |m_A|, order parameter on sublattice A
mag_grid = np.zeros((len(U_list), len(T_list)))

h5_file = f'data_Square_1orb_2frag_B{B}_phase.h5'

# create a fresh file; each U group will be appended as it completes
with h5py.File(h5_file, 'a') as _: # 'a' is append, 'w' is write if you want to delete previous
    pass

L_t0 = np.kron( np.tanh( np.diag(np.linspace(-1,1,B+1,endpoint=False)[1:])) , np.eye(2) ) 
#adding a small magnetization to stabilize AFM
L_t0A= L_t0+1e-1*np.kron( np.eye(B), np.diag([1.0,-1.0]) )
L_t0B= -L_t0A
R_t0 = np.kron( np.ones((B,1))/np.sqrt(B) , np.eye(2) )

magA=0.0;magB=0.0

fig, axs = plt.subplots(1, 1, figsize=(6,5))
for iU, U in enumerate(U_list):
    eloc = np.zeros((nimp, nimp))
    eloc[0, 0] = -U / 2.
    eloc[1, 1] = -U / 2.

    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0, 0, 1, 1] = U
    Utensor[1, 1, 0, 0] = U

    # warm-start: carry converged Lambda/R from one T to the next (low to high T)
    Lambda_A = L_t0A.copy()
    R_A = R_t0.copy()
    Lambda_c_A = L_t0A.copy()
    D_A = R_t0.copy()

    
    
    Lambda_B = L_t0B.copy()
    R_B = R_t0.copy()
    Lambda_c_B = L_t0B.copy()
    D_B = R_t0.copy()

    nT = len(T_list)
    arr_denMat_A = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_denMat_B = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_R_A      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
    arr_R_B      = np.zeros((nT, nbath, nimp), dtype=np.complex128)
    arr_Lambda_A = np.zeros((nT, nbath, nbath), dtype=np.complex128)
    arr_Lambda_B = np.zeros((nT, nbath, nbath), dtype=np.complex128)
    arr_Z_A      = np.zeros((nT, nimp, nimp), dtype=np.complex128)
    arr_Z_B      = np.zeros((nT, nimp, nimp), dtype=np.complex128)

    for iT, T in enumerate(T_list):
        print(f'Doing U={U:.2f} - T={T:.3f}')

        edsolverA = SimpleED(ntot, use_Ntot=True, use_Sz=True,dtype=np.complex128)
        edsolverB = SimpleED(ntot, use_Ntot=True, use_Sz=True,dtype=np.complex128)
        fragmentA = Fragment(nimp, nbath, eloc, Utensor, edsolverA,
                             Lambda=Lambda_A, R=R_A, Lambda_c=Lambda_c_A, D=D_A, verbose=0)
        fragmentB = Fragment(nimp, nbath, eloc, Utensor, edsolverB,
                             Lambda=Lambda_B, R=R_B, Lambda_c=Lambda_c_B, D=D_B, verbose=0)

        for it in range(itmax):
            print(f'it={it} ')
            # Enforce no magnetization along X and Y
            fragmentA.R[::2,1::2]=0.0; fragmentA.R[1::2,::2]=0.0
            fragmentB.R[::2,1::2]=0.0; fragmentB.R[1::2,::2]=0.0
            fragmentA.Lambda[::2,1::2]=0.0; fragmentA.Lambda[1::2,::2]=0.0
            fragmentB.Lambda[::2,1::2]=0.0; fragmentB.Lambda[1::2,::2]=0.0


            Delta_tot, ERD_tot =lattice.solve_qp([fragmentA, fragmentB], T=T)


            fragmentA.update_hybridization(T=T,move_pen=1e-7)
            fragmentB.update_hybridization(T=T,move_pen=1e-7)

            # seed AFM only at lowest T for the first few iterations
            if iT==0 and it < 3:
                fragmentA.eloc = eloc + bfield * np.diag([-1, 1])
                fragmentB.eloc = eloc - bfield * np.diag([-1, 1])
            else:
                fragmentA.eloc = eloc.copy()
                fragmentB.eloc = eloc.copy()

            # Enforce no magnetization along X and Y
            fragmentA.D[::2,1::2]=0.0; fragmentA.D[1::2,::2]=0.0
            fragmentB.D[::2,1::2]=0.0; fragmentB.D[1::2,::2]=0.0
            fragmentA.Lambda_c[::2,1::2]=0.0; fragmentA.Lambda_c[1::2,::2]=0.0
            fragmentB.Lambda_c[::2,1::2]=0.0; fragmentB.Lambda_c[1::2,::2]=0.0

            fragmentA.solve_impurity(mu, T=T)
            fragmentB.solve_impurity(mu, T=T)

            Lambda_old_A = fragmentA.Lambda.copy()
            R_old_A = fragmentA.R.copy()
            Lambda_old_B = fragmentB.Lambda.copy()
            R_old_B = fragmentB.R.copy()


            fragmentA.update_self_energy(T=T,move_pen=1e-7)
            fragmentB.update_self_energy(T=T,move_pen=1e-7)

            diff_LR = max(
                np.abs(fragmentA.Lambda - Lambda_old_A).max(),
                np.abs(np.abs(fragmentA.R) - np.abs(R_old_A)).max(),
                np.abs(fragmentB.Lambda - Lambda_old_B).max(),
                np.abs(np.abs(fragmentB.R) - np.abs(R_old_B)).max(),
            )

            fragmentA.Lambda = (1 - mix) * fragmentA.Lambda + mix * Lambda_old_A
            fragmentA.R = (1 - mix) * fragmentA.R + mix * R_old_A

            fragmentB.Lambda = (1 - mix) * fragmentB.Lambda + mix * Lambda_old_B
            fragmentB.R = (1 - mix) * fragmentB.R + mix * R_old_B

            magA_old = magA*1.0
            magB_old = magB*1.0

            magA = (fragmentA.denMat[0, 0].real - fragmentA.denMat[1, 1].real)
            magB = (fragmentB.denMat[0, 0].real - fragmentB.denMat[1, 1].real)

            diff_A = abs(magA - magA_old)
            diff_B = abs(magB - magB_old)
            diff = max( diff_LR, 10*diff_A, 10*diff_B )

            print(f'diff={diff:.2e} diffA={diff_A:.4f} diffB={diff_B:.4f} '
                  f'magA={magA:.4f} magB={magB:.4f} '
                  f'nfillA={fragmentA.nfill:.4f} nfillB={fragmentB.nfill:.4f} ')
            if (diff < tol and it > 1) or it == itmax - 1:
                print(f'U={U:.2f} T={T:.2f} converged at it={it} diff={diff:.2e}')
                break

        # save converged Lambda/R as warm start for next T
        Lambda_A = fragmentA.Lambda.copy()
        R_A = fragmentA.R.copy()
        Lambda_c_A = fragmentA.Lambda_c.copy()
        D_A = fragmentA.D.copy()

        Lambda_B = fragmentB.Lambda.copy()
        R_B = fragmentB.R.copy()
        Lambda_c_B = fragmentB.Lambda_c.copy()
        D_B = fragmentB.D.copy()

        if(iT==0):
            L_t0 = Lambda_A.copy()
            R_t0 = R_A.copy()

        dm_A = fragmentA.denMat[:nimp, :nimp].real
        mag_grid[iU, iT] = abs(dm_A[0, 0] - dm_A[1, 1])

        arr_denMat_A[iT] = fragmentA.denMat[:nimp,:nimp]
        arr_denMat_B[iT] = fragmentB.denMat[:nimp,:nimp]
        arr_R_A[iT]      = fragmentA.R
        arr_R_B[iT]      = fragmentB.R
        arr_Lambda_A[iT] = fragmentA.Lambda
        arr_Lambda_B[iT] = fragmentB.Lambda
        arr_Z_A[iT]      = fragmentA.compute_Z()
        arr_Z_B[iT]      = fragmentB.compute_Z()

    # flush this U to disk immediately so partial runs are not lost
    with h5py.File(h5_file, 'a') as h5f:
        grp = h5f.create_group(f'U{U:.2f}_B{B}')
        grp.create_dataset('T_list',   data=T_list)
        grp.create_dataset('denMat_A', data=arr_denMat_A)
        grp.create_dataset('denMat_B', data=arr_denMat_B)
        grp.create_dataset('R_A',      data=arr_R_A)
        grp.create_dataset('R_B',      data=arr_R_B)
        grp.create_dataset('Lambda_A', data=arr_Lambda_A)
        grp.create_dataset('Lambda_B', data=arr_Lambda_B)
        grp.create_dataset('Z_A',      data=arr_Z_A)
        grp.create_dataset('Z_B',      data=arr_Z_B)
    print(f'Written U={U:.2f} to {h5_file}')
    
    axs.plot(T_list, mag_grid[iU], marker='o',label=f'U={U:.2f}')
axs.set_xlabel('T')
axs.set_ylabel('|m_A|')
axs.set_title(f'Square lattice, B={B} bath sites per impurity orbital')
axs.legend()
plt.tight_layout()
plt.savefig('Square_1orb_2frag_phase.png', dpi=300)
plt.show()