###FIRST load your julia module (v1.9 is the CCQ one that is available with modules/2.2 --- juliaup is probably a better solution to be flexible regarding the version)
##also setup a python venv, unless you already have one. e.g.:
#virtualenv --system-site-packages .triqs
#source .triqs/bin/activate 
###SECOND edit julia_init.sh with the correct absolute paths on your system and source it.
#source init_julia.sh
#pip install juliacall

###This is it. Run examples/test_mps.py to check (takes a while).
###Also make sure that the crorect env_variables are set before running, otherwise executing the code will initialize their own conda environments and also fail.
###You may want to set env_variables for multithreading (both Julia level and BLAS)
###export JULIA_NUM_THREADS=4
###export MKL_NUM_THREADS=4
###this would be using 16 threads for example.

