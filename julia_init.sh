module purge
module load triqs/3.2.x_nix2.2_llvm julia
###your julia binary
export PYTHON_JULIAPKG_EXE="/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/bin/julia"
###the julia folder in ghostGA
export PYTHON_JULIAPKG_PROJECT="/mnt/home/bkloss/projects/ghostGA/julia"
##your python venv
export "JULIA_CONDAPKG_BACKEND"="Null"
export "JULIA_PYTHONCALL_EXE"="/mnt/home/bkloss/projects/ghostGA/.triqs/bin/python"
source /mnt/home/bkloss/projects/ghostGA/.triqs/bin/activate
