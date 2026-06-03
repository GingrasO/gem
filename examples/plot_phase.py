import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

h5_file = 'Square_1orb_2frag_phase.h5'

# ── load all U groups ──────────────────────────────────────────────────────────
data = {}
with h5py.File(h5_file, 'r') as f:
    for key in sorted(f.keys()):
        U = float(key.split('_')[1])
        grp = f[key]
        data[U] = {
            'T':        grp['T_list'][:],
            'denMat_A': grp['denMat_A'][:],
            'denMat_B': grp['denMat_B'][:],
            'R_A':      grp['R_A'][:],
            'R_B':      grp['R_B'][:],
            'Lambda_A': grp['Lambda_A'][:],
            'Lambda_B': grp['Lambda_B'][:],
            'Z_A':      grp['Z_A'][:],
            'Z_B':      grp['Z_B'][:],
        }

U_list = np.array(sorted(data.keys()))
T_list = data[U_list[0]]['T']
nimp   = data[U_list[0]]['denMat_A'].shape[1] - data[U_list[0]]['R_A'].shape[2]
# nimp from Lambda shape
nimp   = data[U_list[0]]['Lambda_A'].shape[1]

colors = plt.cm.viridis(np.linspace(0, 0.9, len(U_list)))

# ── derived quantities ─────────────────────────────────────────────────────────
# impurity 1bdm diagonal: spin-up = [0,0], spin-down = [1,1]
# magnetisation |m| = |n_up - n_down| on fragment A
mag  = {U: np.abs(data[U]['denMat_A'][:, 0, 0].real - data[U]['denMat_A'][:, 1, 1].real)
        for U in U_list}
n_up = {U: data[U]['denMat_A'][:, 0, 0].real for U in U_list}
n_dw = {U: data[U]['denMat_A'][:, 1, 1].real for U in U_list}

# quasiparticle weight: take diagonal mean of Z (real part)
Z_up = {U: data[U]['Z_A'][:, 0, 0].real for U in U_list}
Z_dw = {U: data[U]['Z_A'][:, 1, 1].real for U in U_list}

# Lambda diagonal (crystal-field / self-energy shift)
Lam_up = {U: data[U]['Lambda_A'][:, 0, 0].real for U in U_list}
Lam_dw = {U: data[U]['Lambda_A'][:, 1, 1].real for U in U_list}

# ── MF criticality fit: m(T) = A * (Tc - T)^beta  for T < Tc ─────────────────
def mf_order(T, A, Tc, beta):
    return A * np.maximum(Tc - T, 0.0) ** beta

fit_results = {}
plt.figure() #sa
for U in U_list:
    m = mag[U]
    m_max = m.max()
    if m_max < 0.02:
        fit_results[U] = None
        continue

    # Tc estimate: first T where m drops below m_max/4
    below_quarter = np.where(m < 0.25 * m_max)[0]
    if len(below_quarter) == 0:
        fit_results[U] = None
        continue
    Tc_est = T_list[below_quarter[0]]

    thresh = max(0.005, 0.02 * m_max)
    A_est  = m_max / np.sqrt(Tc_est)
    plt.plot(T_list[below_quarter[0]],mf_order(T_list[below_quarter[0]], A_est, Tc_est, 0.5))

    print(f'for U={U} - Tc_est={Tc_est}')

    # restrict to the critical region: m must be small enough that we are not
    # yet in the saturated regime (m < 0.4 * m_max) but above the noise floor.
    # The MF power law m ~ (Tc-T)^beta only holds close to Tc.
    mask = (m > thresh) & (m < 0.7 * m_max)
    print(U,mask)
    if mask.sum() < 4:
        mask = (m > thresh) & (m < 0.9 * m_max)
    if mask.sum() < 3:
        fit_results[U] = None
        continue

    try:
        popt, pcov = curve_fit(
            mf_order, T_list[mask], m[mask],
            p0=[A_est, Tc_est, 0.5],
            bounds=([0.0, 0.0, 0.05], [4.0* m_max/np.sqrt(Tc_est), T_list.max() * 2.0, 3.0]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        fit_results[U] = dict(A=popt[0], Tc=popt[1], beta=popt[2],
                              A_err=perr[0], Tc_err=perr[1], beta_err=perr[2])
        print(f'U={U:.2f}  Tc={popt[1]:.4f}±{perr[1]:.4f}  β={popt[2]:.4f}±{perr[2]:.4f}')
    except Exception as e:
        print(f'Fit failed for U={U:.2f}: {e}')
        fit_results[U] = None
plt.show()
# ── Figure 1: magnetisation vs T with MF fits ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
for U, c in zip(U_list, colors):
    res = fit_results[U]
    if res is not None:
        label = f'U={U:.1f}  Tc={res["Tc"]:.3f}  β={res["beta"]:.3f}'
        T_fit = np.linspace(0, res['Tc'], 300)
        ax1.plot(T_fit, mf_order(T_fit, res['A'], res['Tc'], res['beta']),
                 color=c, linestyle='--', linewidth=1.2)
    else:
        label = f'U={U:.1f}'
    ax1.plot(T_list, mag[U], color=c, marker='o', markersize=3, label=label)
ax1.set_xlabel('T')
ax1.set_ylabel('|m| = |n↑ - n↓|')
ax1.set_title('AFM order parameter vs temperature — dashed: MF fit')
ax1.legend(fontsize=7)
fig1.tight_layout()
fig1.savefig('plot_mag_vs_T.png', dpi=150)

# ── Figure 2: phase diagram colormap ──────────────────────────────────────────
mag_grid = np.array([mag[U] for U in U_list])   # (nU, nT)
UU, TT   = np.meshgrid(U_list, T_list, indexing='ij')

fig2, ax2 = plt.subplots(figsize=(7, 5))
levels = np.linspace(0, mag_grid.max(), 51)
cf = ax2.contourf(UU, TT, mag_grid, levels=levels, cmap='RdBu_r')
fig2.colorbar(cf, ax=ax2, label='|m|')
ax2.set_xlabel('U')
ax2.set_ylabel('T')
ax2.set_title('Phase diagram: AFM order parameter')
fig2.tight_layout()
fig2.savefig('plot_phase_diagram.png', dpi=150)

# ── Figure 3: quasiparticle weight Z vs T ─────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for U, c in zip(U_list, colors):
    axes3[0].plot(T_list, Z_up[U], color=c, marker='o', markersize=3, label=f'U={U:.1f}')
    axes3[1].plot(T_list, Z_dw[U], color=c, marker='o', markersize=3, label=f'U={U:.1f}')
for ax, spin in zip(axes3, ['↑', '↓']):
    ax.set_xscale('log')
    ax.set_xlabel('T')
    ax.set_ylabel('Z')
    ax.set_title(f'Quasiparticle weight Z{spin} (fragment A)')
    ax.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig('plot_Z_vs_T.png', dpi=150)

# ── Figure 4: occupancies and Lambda diagonal vs T ────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4))
for U, c in zip(U_list, colors):
    axes4[0].plot(T_list, n_up[U], color=c, linestyle='-',  marker='o', markersize=3,
                  label=f'U={U:.1f} ↑')
    axes4[0].plot(T_list, n_dw[U], color=c, linestyle='--', marker='x', markersize=3,
                  label=f'U={U:.1f} ↓')
    axes4[1].plot(T_list, Lam_up[U], color=c, linestyle='-',  marker='o', markersize=3,
                  label=f'U={U:.1f} ↑')
    axes4[1].plot(T_list, Lam_dw[U], color=c, linestyle='--', marker='x', markersize=3,
                  label=f'U={U:.1f} ↓')
axes4[0].set_xlabel('T')
axes4[0].set_ylabel('n')
axes4[0].set_title('Impurity occupancies (fragment A)')
axes4[0].legend(fontsize=6, ncol=2)
axes4[1].set_xlabel('T')
axes4[1].set_ylabel('Λ diagonal')
axes4[1].set_title('Lambda diagonal (fragment A)')
axes4[1].legend(fontsize=6, ncol=2)
fig4.tight_layout()
fig4.savefig('plot_occ_lambda_vs_T.png', dpi=150)

# ── Figure 5: Tc and β vs U ───────────────────────────────────────────────────
fit_U    = [U for U in U_list if fit_results[U] is not None]
fit_Tc   = [fit_results[U]['Tc']       for U in fit_U]
fit_Tc_e = [fit_results[U]['Tc_err']   for U in fit_U]
fit_beta = [fit_results[U]['beta']     for U in fit_U]
fit_be_e = [fit_results[U]['beta_err'] for U in fit_U]

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 4))
ax5a.errorbar(fit_U, fit_Tc,   yerr=fit_Tc_e,   fmt='o-', capsize=4)
ax5a.set_xlabel('U')
ax5a.set_ylabel('$T_c$')
ax5a.set_title('Critical temperature vs U')
ax5a.set_ylim(0,None)
ax5a.set_xlim(0,None)

ax5b.errorbar(fit_U, fit_beta, yerr=fit_be_e, fmt='s-', capsize=4, color='tab:orange')
ax5b.axhline(0.5, linestyle='--', color='grey', linewidth=0.8, label='MF β = 0.5')
ax5b.set_xlabel('U')
ax5b.set_ylabel('β')
ax5b.set_title('Critical exponent β vs U')
ax5b.legend(fontsize=8)

fig5.tight_layout()
fig5.savefig('plot_Tc_beta_vs_U.png', dpi=150)
print('Saved plot_Tc_beta_vs_U.png')

plt.show()
print('Done.')
