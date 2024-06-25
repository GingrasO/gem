
# run the scf calculation
mpirun -n 4 pw.x -i nio.scf.in | tee nio.scf.out


# run the DFT bandstructure calculation
mpirun -n 4 pw.x -i nio.bnd.in | tee nio.bnd.out
mpirun -n 4 bands.x -i nio.bands.in | tee nio.bands.out


# run the nscf calculation
mpirun -n 4 pw.x -i nio.nscf.in | tee nio.nscf.out


# run the mod_scf calculation
mpirun -n 4 pw.x -i nio.mod_scf.in | tee nio.nscf.out


# run the DFT bandstructure calculation
mpirun -n 4 pw.x -i nio.bnd.in | tee nio.bnd.out
mpirun -n 4 bands.x -i nio.bands.in | tee nio.bands.out

