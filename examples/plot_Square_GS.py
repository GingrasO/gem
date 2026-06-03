import numpy as np
import matplotlib.pyplot as plt
import h5py

fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(10, 4))

colors = {1: 'C1', 3: 'C0'}
labels = {1: 'GA ($B=1$)', 3: 'ghost-GA ($B=3$)'}

for B in [1, 3]:
    with h5py.File(f'Square_GS_B{B}.h5', 'r') as h5f:
        U_list = h5f['U_list'][:]
        docc_A = h5f['docc_A'][:]
        docc_B = h5f['docc_B'][:]
        m_A    = h5f['m_A'][:]
        m_B    = h5f['m_B'][:]

    docc = 0.5 * (docc_A + docc_B)
    m    = 0.5 * (m_A - m_B)       # staggered magnetisation

    ax_d.plot(U_list, docc, color=colors[B], label=labels[B], marker='o', ms=3)
    ax_m.plot(U_list, m,    color=colors[B], label=labels[B], marker='o', ms=3)

ax_d.set_xlabel('$U$')
ax_d.set_ylabel('$\\langle n_\\uparrow n_\\downarrow \\rangle$')
ax_d.set_title('Double occupancy')
ax_d.set_ylim(bottom=0)
ax_d.legend()

ax_m.set_xlabel('$U$')
ax_m.set_ylabel('$m = (m_A - m_B)/2$')
ax_m.set_title('Staggered magnetisation')
ax_m.set_ylim(0, 1.0)
ax_m.legend()

fig.suptitle('Square lattice — AFM order ($T=0$)')
fig.tight_layout()
fig.savefig('Square_GS_docc_m.png', dpi=150)
print('Saved Square_GS_docc_m.png')
plt.show()
