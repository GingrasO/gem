# module purge
# module load triqs/3.2.x_nix2.2_llvm julia
###your julia binary
export PYTHON_JULIAPKG_EXE="/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/bin/julia"
###the julia folder in ghostGA
export PYTHON_JULIAPKG_PROJECT="/mnt/home/ogingras/Work/Libs_2.2-20230808/TRIQS/ghostGA/julia"
##your python venv
export JULIA_CONDAPKG_BACKEND="Null"
export JULIA_PYTHONCALL_EXE="/mnt/home/ogingras/.py_3.9.15_2.1-20230222/bin/python"
# source /mnt/home/bkloss/projects/ghostGA/.triqs/bin/activate
###Threading layer defaults to sequential after sourcing these modules, so we set it to INTEL manually --- may be breaking things elsewhere!
export MKL_THREADING_LAYER="INTEL"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
