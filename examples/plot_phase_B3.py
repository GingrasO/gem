import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit


DMFT_extracted=np.array([
[0.40186762778505897, 0.020986984815618222],
[0.6000982961992136, 0.03535791757049892],
[0.8019331585845347, 0.04495661605206074],
[1.0019659239842726, 0.04934924078091107],
[1.2019986893840104, 0.0472885032537961],
[1.5011467889908257, 0.04002169197396963],
[2.000327653997379, 0.030639913232104125],
])
DMFT_U=DMFT_extracted[:,0]
DMFT_T=DMFT_extracted[:,1]


h5_file = 'data_Square_1orb_2frag_B3_phase.h5'

# ── load ───────────────────────────────────────────────────────────────────────
data = {}
with h5py.File(h5_file, 'r') as f:
    for key in sorted(f.keys()):
        U = float(key.split('_')[0][1:])   # 'U0.50_B3' -> 0.5
        grp = f[key]
        data[U] = {
            'T':        grp['T_list'][:],
            'denMat_A': grp['denMat_A'][:],
        }

U_list = np.array(sorted(data.keys()))
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(U_list)))

# ── magnetisation |m| = |n_up - n_down| on fragment A ─────────────────────────
mag = {U: np.abs(data[U]['denMat_A'][:, 0, 0].real - data[U]['denMat_A'][:, 1, 1].real)
       for U in U_list}

# ── MF fit: m(T) = A * (Tc - T)^beta ─────────────────────────────────────────
def mf_order(T, A, Tc, beta):
    return A * np.maximum(Tc - T, 0.0) ** beta

fit_results = {}
for U in U_list:
    T = data[U]['T']
    m = mag[U]
    m_max = m.max()
    if m_max < 0.02:
        fit_results[U] = None
        continue

    below_quarter = np.where(m < 0.25 * m_max)[0]
    if len(below_quarter) == 0:
        fit_results[U] = None
        continue
    Tc_est = T[below_quarter[0]]

    thresh = max(0.005, 0.02 * m_max)
    A_est  = m_max / np.sqrt(max(Tc_est, 1e-6))

    mask = (m > thresh) & (m < 0.7 * m_max)
    if mask.sum() < 4:
        mask = (m > thresh) & (m < 0.9 * m_max)
    if mask.sum() < 3:
        fit_results[U] = None
        continue

    try:
        popt, pcov = curve_fit(
            mf_order, T[mask], m[mask],
            p0=[A_est, Tc_est, 0.5],
            bounds=([0.0, 0.0, 0.05],
                    [4.0 * m_max / np.sqrt(max(Tc_est, 1e-6)), T.max() * 2.0, 3.0]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        fit_results[U] = dict(A=popt[0], Tc=popt[1], beta=popt[2],
                              A_err=perr[0], Tc_err=perr[1], beta_err=perr[2])
        print(f'U={U:.2f}  Tc={popt[1]:.4f}±{perr[1]:.4f}  β={popt[2]:.4f}±{perr[2]:.4f}')
    except Exception as e:
        print(f'Fit failed for U={U:.2f}: {e}')
        fit_results[U] = None

fit_U    = [U for U in U_list if fit_results[U] is not None]
fit_Tc   = np.array([fit_results[U]['Tc']       for U in fit_U])
fit_Tc_e = np.array([fit_results[U]['Tc_err']   for U in fit_U])
fit_beta = np.array([fit_results[U]['beta']     for U in fit_U])
fit_be_e = np.array([fit_results[U]['beta_err'] for U in fit_U])

# ── Figure 1: magnetisation vs T with fits ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
for U, c in zip(U_list, colors):
    T  = data[U]['T']
    res = fit_results[U]
    label = (f'U={U:.2f}  $T_c$={res["Tc"]:.3f}  β={res["beta"]:.3f}'
             if res else f'U={U:.2f}')
    ax1.plot(T, mag[U], marker='o', markersize=4, color=c, label=label)
    if res is not None:
        T_fit = np.linspace(0, res['Tc'], 400)
        ax1.plot(T_fit, mf_order(T_fit, res['A'], res['Tc'], res['beta']),
                 color=c, linestyle='--', linewidth=1.2)
ax1.set_xlabel('T', fontsize=13)
ax1.set_ylabel('|m| = |n↑ − n↓|', fontsize=13)
ax1.set_title('AFM order parameter vs T  (B=3, dashed = fit)', fontsize=12)
ax1.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig('B3_mag_vs_T.png', dpi=150)

# ── Figure 2: Tc in the T–U plane ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.errorbar(fit_U, fit_Tc, yerr=fit_Tc_e,label=r'\mathcal{B}=3',
             fmt='o-', capsize=5, color='tab:blue', linewidth=1.5, markersize=7)
ax2.fill_between(fit_U, fit_Tc - fit_Tc_e, fit_Tc + fit_Tc_e,
                 alpha=0.2, color='tab:blue')
ax2.plot(2*DMFT_U,2*DMFT_T,marker='x',color='black',label='DMFT', markersize=10)
ax2.set_xlabel('U', fontsize=13)
ax2.set_ylabel('$T_c$', fontsize=13)
ax2.set_title('Critical temperature vs U  (B=3)', fontsize=12)
ax2.set_xlim(0, None)
ax2.set_ylim(0, None)
ax2.legend()
fig2.tight_layout()
fig2.savefig('B3_Tc_vs_U.png', dpi=150)

# ── Figure 3: fitted β exponent vs U ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.errorbar(fit_U, fit_beta, yerr=fit_be_e,
             fmt='s-', capsize=5, color='tab:orange', linewidth=1.5, markersize=7)
ax3.fill_between(fit_U, fit_beta - fit_be_e, fit_beta + fit_be_e,
                 alpha=0.2, color='tab:orange')
ax3.axhline(0.5, linestyle='--', color='grey', linewidth=0.9, label='MF β = 0.5')
ax3.set_xlabel('U', fontsize=13)
ax3.set_ylabel('β', fontsize=13)
ax3.set_title('Critical exponent β vs U  (B=3)', fontsize=12)
ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig('B3_beta_vs_U.png', dpi=150)

print('Saved B3_mag_vs_T.png, B3_Tc_vs_U.png, B3_beta_vs_U.png')
plt.show()
