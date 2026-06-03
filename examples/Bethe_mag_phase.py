"""
Bethe lattice (B=3) magnetic phase diagram example.

Protocol:
  1. Warm-up at U=1.5 with Zeeman field h: heat from T=0.01 to T=0.2
     (20 log-spaced points).  R, Lambda, D, Lambda_c are kept diagonal throughout
     (spin-up and spin-down values may differ; no off-diagonal bath mixing).
  2. Sweep U: 1.5 -> 3.0 at T=0.2, recording magnetisation.
  3. Sweep T: 0.2 -> 0.01 at U=3.0, recording magnetisation.
  4. Plot magnetisation and convergence residual along both paths.
"""

import numpy as np
import matplotlib.pyplot as plt

from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import h5py

# ── system dimensions ─────────────────────────────────────────────────────────
B     = 3
nimp  = 2        # one orbital, two spins  (index 0 = up, index 1 = down)
nbath = nimp * B  # = 6,  ordering: (bath0_up, bath0_dn, bath1_up, bath1_dn, ...)
ntot  = nimp + nbath

# ── Bethe lattice: semicircular DOS ───────────────────────────────────────────
e_list = np.linspace(-1, 1, 5001)
wks    = np.sqrt(1 - e_list**2)
wks   /= np.sum(wks)
eks    = np.array([np.kron(np.array([[e]], dtype=np.complex128), np.eye(2))
                   for e in e_list])
lattice = Lattice(eks, wk_list=wks)

# ── helpers ───────────────────────────────────────────────────────────────────
def make_eloc(U, h=0.0):
    eloc = np.zeros((nimp, nimp))
    eloc[0, 0] = -U / 2. - h / 2.   # spin-up  energetically favoured
    eloc[1, 1] = -U / 2. + h / 2.
    return eloc

def make_Utensor(U):
    Utensor = np.zeros((nimp, nimp, nimp, nimp))
    Utensor[0, 0, 1, 1] = U
    Utensor[1, 1, 0, 0] = U
    return Utensor

def make_solver():
    return SimpleED(ntot, use_Ntot=True, use_Sz=True, dtype=np.complex128)

def impose_diagonal(frag):
    """
    Project self-energy and hybridization parameters onto the diagonal subspace.

    Spin convention (interleaved): rows/cols 0,2,4 are spin-up bath levels,
    rows/cols 1,3,5 are spin-down bath levels.

    - R, D  (nbath x nimp): keep only same-spin coupling:
        even rows <-> col 0 (up),  odd rows <-> col 1 (dn)
    - Lambda, Lambda_c  (nbath x nbath): keep only the main diagonal
    """
    R_new = np.zeros_like(frag.R)
    R_new[::2,  0] = frag.R[::2,  0]
    R_new[1::2, 1] = frag.R[1::2, 1]
    frag.R = R_new

    frag.Lambda = np.diag(np.diag(frag.Lambda))

    D_new = np.zeros_like(frag.D)
    D_new[::2,  0] = frag.D[::2,  0]
    D_new[1::2, 1] = frag.D[1::2, 1]
    frag.D = D_new

    frag.Lambda_c = np.diag(np.diag(frag.Lambda_c))

def impose_ph_symmetry(frag):
    """
    Enforce particle-hole symmetry between the two spin sectors.

    Bath-index pairing (b_up <-> B-1-b_dn):
      Lambda_up[b] = -Lambda_dn[B-1-b]   (opposite eigenvalues)
      R_up[b]      =  R_dn[B-1-b]        (R_up = R_dn[::-1] in diagonal basis)

    Enforced by averaging each pair of constraints.
    Must be called after impose_diagonal (operates on diagonal entries only).
    """
    # read current diagonal values (copies to avoid update-order issues)
    l_up = np.diag(frag.Lambda)[::2].copy()   # [Lambda[0,0], Lambda[2,2], Lambda[4,4]]
    l_dn = np.diag(frag.Lambda)[1::2].copy()  # [Lambda[1,1], Lambda[3,3], Lambda[5,5]]
    r_up = frag.R[::2,  0].copy()
    r_dn = frag.R[1::2, 1].copy()

    new_l_up = l_up.copy()
    new_l_dn = l_dn.copy()
    new_r_up = r_up.copy()
    new_r_dn = r_dn.copy()

    for b in range(B):
        j = B - 1 - b
        avg_l = (l_up[b] - l_dn[j]) / 2.0   # enforce l_up[b] = -l_dn[j]
        new_l_up[b] =  avg_l
        new_l_dn[j] = -avg_l
        avg_r = (r_up[b] + r_dn[j]) / 2.0   # enforce r_up[b] = r_dn[j]
        new_r_up[b] = avg_r
        new_r_dn[j] = avg_r

    diag = np.diag(frag.Lambda).copy()
    diag[::2]  = new_l_up
    diag[1::2] = new_l_dn
    frag.Lambda = np.diag(diag)
    frag.R[::2,  0] = new_r_up
    frag.R[1::2, 1] = new_r_dn

def magnetization(frag):
    dm = frag.denMat[:nimp, :nimp]
    return (dm[0, 0] - dm[1, 1]).real

def _update_with_retry(update_fn, frag, T, move_pen_start=1e-8):
    """Call update_fn(T, move_pen) with exponentially increasing move_pen on SVD failure."""
    move_pen = move_pen_start
    for _ in range(6):
        try:
            update_fn(T=T, move_pen=move_pen)
            return
        except np.linalg.LinAlgError:
            print(f"    SVD failed, retrying with move_pen={move_pen*10:.1e}")
            move_pen *= 10
    raise RuntimeError(f"update failed after retries (last move_pen={move_pen:.1e})")


def run_scf(frag, T, mu=0.0, itmax=150, mix=0.02, tol=1e-4, num_eig=1):
    """
    One self-consistent field cycle at temperature T.

    Convergence is measured by the change in eigenvalues of the spin-up and
    spin-down blocks of Lambda, together with the change in the absolute values
    of the nonzero R elements.
    """
    diff = np.inf
    for it in range(itmax):
        lattice.solve_qp([frag], T=T)

        _update_with_retry(frag.update_hybridization, frag, T)
        impose_diagonal(frag)
        impose_ph_symmetry(frag)

        frag.solve_impurity(mu, T=T, num_eig=num_eig)

        R_old      = frag.R.copy()
        Lambda_old = frag.Lambda.copy()

        _update_with_retry(frag.update_self_energy, frag, T)
        impose_diagonal(frag)
        impose_ph_symmetry(frag)

        # convergence: eigenvalues of both spin blocks of Lambda
        leval_up_new = np.linalg.eigvalsh(frag.Lambda[::2,  ::2])
        leval_up_old = np.linalg.eigvalsh(Lambda_old[::2,  ::2])
        leval_dn_new = np.linalg.eigvalsh(frag.Lambda[1::2, 1::2])
        leval_dn_old = np.linalg.eigvalsh(Lambda_old[1::2, 1::2])
        dL_up = np.abs(leval_up_new - leval_up_old)
        dL_dn = np.abs(leval_dn_new - leval_dn_old)
        diff_Lambda  = max(dL_up.max(), dL_dn.max())

        # R: compare absolute values of the nonzero (same-spin) elements
        dR_up = np.abs(np.abs(frag.R[::2,  0]) - np.abs(R_old[::2,  0]))
        dR_dn = np.abs(np.abs(frag.R[1::2, 1]) - np.abs(R_old[1::2, 1]))
        diff_R = max(dR_up.max(), dR_dn.max())

        diff = max(diff_R, diff_Lambda)

        frag.R      = (1 - mix) * frag.R      + mix * R_old
        frag.Lambda = (1 - mix) * frag.Lambda + mix * Lambda_old
        impose_diagonal(frag)
        impose_ph_symmetry(frag)

        # identify largest-change elements for diagnostics
        iL_up = np.argmax(dL_up); iL_dn = np.argmax(dL_dn)
        iR_up = np.argmax(dR_up); iR_dn = np.argmax(dR_dn)
        print(f"    it={it:3d}  diff={diff:.3e}  (dL={diff_Lambda:.3e}, dR={diff_R:.3e})")
        print(f"      Lambda_up: {np.diag(frag.Lambda)[::2]}   dL_up={dL_up}  max@bath{iL_up}")
        print(f"      Lambda_dn: {np.diag(frag.Lambda)[1::2]}  dL_dn={dL_dn}  max@bath{iL_dn}")
        print(f"      R_up:      {frag.R[::2,  0].real}  dR_up={dR_up}  max@bath{iR_up}")
        print(f"      R_dn:      {frag.R[1::2, 1].real}  dR_dn={dR_dn}  max@bath{iR_dn}")
        if diff < tol and it > 2:
            break

    if diff >= tol:
        print(f"    WARNING: not converged, diff={diff:.3e}")
    return diff

def checkpoint(phase1, phase2, phase3):
    """Overwrite the checkpoint file with all data accumulated so far."""
    kw = dict(h_field=h_field, T_high=T_high, U_final=U_final)
    if phase1 is not None:
        kw.update(T_warmup=phase1[0], mag_warmup=phase1[1], diff_warmup=phase1[2])
    if phase2 is not None:
        kw.update(U_list=phase2[0], mag_U=phase2[1], diff_U=phase2[2])
    if phase3 is not None:
        kw.update(T_cool=phase3[0], mag_T=phase3[1], diff_T=phase3[2])
    np.savez("Bethe_mag_phase.npz", **kw)

# ── run parameters ────────────────────────────────────────────────────────────
U_init  = 1.5
U_final = 12.0
h_field = 2e-2
mu      = 0.0
itmax   = 150
tol     = 1e-4
mix     = 0.02

# ── initial diagonal bath parameters ─────────────────────────────────────────
np.random.seed(42)
lvals = np.tanh(np.arange(B) - (B - 1) / 2.)   # tanh([-1, 0, 1])
Lambda0 = np.kron(np.diag(lvals), np.eye(nimp))  # diag(l0,l0, l1,l1, l2,l2)

rvals = np.array([0.2, 0.6, 0.2]) + np.random.rand(B) * 0.05
R0 = np.zeros((nbath, nimp))
R0[::2,  0] = rvals   # up-bath -> up-imp
R0[1::2, 1] = rvals   # dn-bath -> dn-imp  (equal to up initially)

# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 – warm-up: heat T from 0.01 to 0.2 at U=1.5 with field h
# ═════════════════════════════════════════════════════════════════════════════
T_warmup = np.logspace(-2, np.log10(0.2), 20)
T_high   = 0.2

print("\n" + "="*60)
print(f"PHASE 1  –  warm-up (U={U_init}, h={h_field})")
print("="*60)

fragment = Fragment(nimp, nbath,
                    make_eloc(U_init, h=h_field), make_Utensor(U_init),
                    make_solver(), Lambda=Lambda0, R=R0)
impose_ph_symmetry(fragment)   # ensure initial conditions satisfy the symmetry

T_warmup_done  = []
mag_warmup     = []
diff_warmup    = []

for iT, T in enumerate(T_warmup):
    print(f"\n  T = {T:.4e}  ({iT+1}/{len(T_warmup)})")
    d = run_scf(fragment, T, mu=mu, itmax=itmax, mix=mix, tol=tol)
    m = magnetization(fragment)
    T_warmup_done.append(T)
    mag_warmup.append(m)
    diff_warmup.append(d)
    print(f"  Magnetisation: {m:.6f}  diff: {d:.3e}")
    checkpoint((np.array(T_warmup_done), np.array(mag_warmup), np.array(diff_warmup)),
               None, None)

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 – sweep U: 1.5 → 3.0  at T=0.2  (track magnetisation)
# ═════════════════════════════════════════════════════════════════════════════
U_list  = np.linspace(1.5, 3.0, 20)
mag_U   = []
diff_U  = []

print("\n" + "="*60)
print(f"PHASE 2  –  U sweep  (T={T_high}, h={h_field})")
print("="*60)

for iU, U in enumerate(U_list):
    print(f"\n  U = {U:.4f}  ({iU+1}/{len(U_list)})")
    fragment.eloc    = make_eloc(U, h=h_field)
    fragment.Utensor = make_Utensor(U)
    d = run_scf(fragment, T_high, mu=mu, itmax=itmax, mix=mix, tol=tol)
    m = magnetization(fragment)
    mag_U.append(m)
    diff_U.append(d)
    print(f"  Magnetisation: {m:.6f}  diff: {d:.3e}")
    checkpoint((np.array(T_warmup_done), np.array(mag_warmup), np.array(diff_warmup)),
               (U_list[:iU+1],          np.array(mag_U),       np.array(diff_U)),
               None)

mag_U  = np.array(mag_U)
diff_U = np.array(diff_U)

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 – sweep T: 0.2 → 0.01  at U=3.0  (track magnetisation)
# ═════════════════════════════════════════════════════════════════════════════
T_cool  = np.logspace(np.log10(0.2), -2, 100)
mag_T   = []
diff_T  = []

print("\n" + "="*60)
print(f"PHASE 3  –  T sweep  (U={U_final}, h={h_field})")
print("="*60)

for iT, T in enumerate(T_cool):
    print(f"\n  T = {T:.4e}  ({iT+1}/{len(T_cool)})")
    d = run_scf(fragment, T, mu=mu, itmax=itmax, mix=mix, tol=tol)
    m = magnetization(fragment)
    mag_T.append(m)
    diff_T.append(d)
    print(f"  Magnetisation: {m:.6f}  diff: {d:.3e}")
    if(abs(d)>1e-4): break
    checkpoint((np.array(T_warmup_done), np.array(mag_warmup), np.array(diff_warmup)),
               (U_list,                  mag_U,                diff_U),
               (T_cool[:iT+1],           np.array(mag_T),      np.array(diff_T)))

mag_T  = np.array(mag_T)
diff_T = np.array(diff_T)

with h5py.File(f'U{U_final:.1f}_dataset.hdf5', 'w') as f:
    f.create_dataset('T_list', data=T_cool[:len(mag_T)])
    f.create_dataset('m_list', data=mag_T )
    f.create_dataset('diff_list', data=diff_T )


print("\nFinal results saved to Bethe_mag_phase.npz")

# ═════════════════════════════════════════════════════════════════════════════
# Plot: 2x2 grid  –  top row: magnetisation  /  bottom row: convergence diff
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ── mag vs U ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(U_list, mag_U, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel(r'$U$', fontsize=13)
ax.set_ylabel(r'$m = n_\uparrow - n_\downarrow$', fontsize=13)
ax.set_title(f'Magnetisation vs $U$  ($T={T_high}$, $h={h_field}$)', fontsize=12)
ax.grid(True, alpha=0.3)

# ── mag vs T ──────────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.semilogx(T_cool, mag_T, 's-', color='tomato', linewidth=2, markersize=6)
ax.invert_xaxis()
ax.set_xlabel(r'$T$', fontsize=13)
ax.set_ylabel(r'$m = n_\uparrow - n_\downarrow$', fontsize=13)
ax.set_title(f'Magnetisation vs $T$  ($U={U_final}$, $h={h_field}$)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)

# ── diff vs U ─────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.semilogy(U_list, diff_U, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axhline(tol, color='k', linestyle='--', linewidth=1, label=f'tol={tol}')
ax.set_xlabel(r'$U$', fontsize=13)
ax.set_ylabel('Convergence residual', fontsize=13)
ax.set_title(f'Residual vs $U$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

# ── diff vs T ─────────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.loglog(T_cool, diff_T, 's-', color='tomato', linewidth=2, markersize=6)
ax.invert_xaxis()
ax.axhline(tol, color='k', linestyle='--', linewidth=1, label=f'tol={tol}')
ax.set_xlabel(r'$T$', fontsize=13)
ax.set_ylabel('Convergence residual', fontsize=13)
ax.set_title(f'Residual vs $T$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.suptitle(f'Bethe lattice, $B={B}$, $h={h_field}$', fontsize=13)
plt.tight_layout()
plt.savefig('Bethe_mag_phase.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to Bethe_mag_phase.png")
