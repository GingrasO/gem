##set these environment variables before installing JuliaCall into your venv
##ToDo replace the explicit path by piping which julia 
export PYTHON_JULIAPKG_EXE="/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/bin/julia"
##ToDo replace the absolute path by the path relative to ghostGA basedir
export PYTHON_JULIAPKG_PROJECT="/home/bkloss/projects/ghostGA/julia"

##setup venv that uses site-packages
#virtualenv --system-site-packages .triqs
#source .triqs/bin/activate
#python3 -m pip install JuliaCall

##set these environment variables before setting up PythonCall in julia (maybe this works automatically upon import of juliacall?
#ENV["JULIA_CONDAPKG_BACKEND"]="Null"
#ENV["JULIA_PYTHONCALL_EXE"]=".triqs/bin/python"
