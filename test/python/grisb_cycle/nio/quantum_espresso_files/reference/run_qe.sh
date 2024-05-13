
# run the scf calculation
#mpirun -n 4 pw.x -i nio.scf.in | tee nio.scf.out


# OPTIONAL: run the DFT bandstructure calculation
#mpirun -n 4 pw.x -i nio.bnd.in | tee nio.bnd.out
#mpirun -n 4 bands.x -i nio.bands.in | tee nio.bands.out


# run the nscf calculation
#mpirun -n 4 pw.x -i nio.nscf.in | tee nio.nscf.out

# Wannierize

## pre-process wannier90
wannier90.x -pp nio
## run interface wannier90-quantum espresso
mpirun -n 4 pw2wannier90.x -i nio.pw2wan.in | tee nio.pw2wan.out
## run wannier90
wannier90.x nio

# Interface with TRIQS, this will create the nio.h5 archive to start the calculation
#python3 ./convert_wannier.py
