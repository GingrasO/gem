import triqs
import subprocess

mpi_arguments = ['/opt/homebrew/bin/mpirun', '-np', '4', 'pw.x', '-nk', '4']
env_vars = {'PATH': '/opt/homebrew/opt/llvm@16/bin:/opt/homebrew/bin/:/opt/local/bin:/opt/local/sbin:/Users/henhans/Softwares/WIEN2k_18:/Users/henhans/Softwares/WIEN2k_18/SRC_structeditor/bin:/Users/henhans/Softwares/WIEN2k_18/SRC_IRelast/script-elastic:/Users/henhans/Git/block2-preview/pyblock2/driver/:/opt/homebrew/bin/:/Users/henhans/Git/triqs/install/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Library/TeX/texbin:.:/Users/henhans/Softwares/WIEN2k_18:.:/Users/henhans/Git/q-e/install/bin:/Users/henhans/Git/pwtk-2.0/pwtk-2.0/:/Users/henhans/Git/wannier90/:/Users/henhans/GWIEN/bin', 'LD_LIBRARY_PATH': '/Users/henhans/Git/triqs/install/lib:', 'SHELL': '/bin/zsh', 'PWD': '/Users/henhans/Git/ghostGA/test/python/grisb_cycle/nio_csc', 'HOME': '/Users/henhans', 'OMP_NUM_THREADS': '1'} 

inp = open(f'nio.mod_scf.in', 'r')
qe_result = subprocess.run(mpi_arguments, stdin=inp, env=env_vars, capture_output=True,
                                   text=True, shell=False)
print(qe_result.stderr)
print(qe_result.stdout)

out = open(f'nio.mod_scf.out', 'w')
err = open(f'nio.mod_scf.err', 'w')

output = qe_result.stdout
error = qe_result.stderr
out.writelines(output)
err.writelines(error)
