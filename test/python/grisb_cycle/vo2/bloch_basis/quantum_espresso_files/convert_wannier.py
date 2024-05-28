#%%
from triqs_ghostGA.wannier90 import Wannier90Converter
Converter = Wannier90Converter(seedname='vo2', rot_mat_type='none', bloch_basis=True)
Converter.convert_dft_input()
# %%
