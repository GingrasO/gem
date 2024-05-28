#%%
from triqs_ghostGA.wannier90 import Wannier90Converter
Converter = Wannier90Converter(seedname='nio', bloch_basis=True, rot_mat_type='none')
Converter.convert_dft_input()
# %%
