#%%
from triqs_dft_tools.converters import Wannier90Converter
Converter = Wannier90Converter(seedname='vo2')
Converter.convert_dft_input()
# %%
