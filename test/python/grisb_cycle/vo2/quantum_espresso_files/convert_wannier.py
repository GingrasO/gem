#%%
from triqs_dft_tools.converters import Wannier90Converter
Converter = Wannier90Converter(seedname='vo2',rot_mat_type="none")
Converter.convert_dft_input()
# %%
