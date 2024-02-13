#from pyscf import fci, gto, scf, ao2mo, ci, cc, hci, lib, dmrgscf
from pyscf import fci, gto, scf, ao2mo, ci, cc, lib # dmrgscf
from pyscf.scf import diis
import numpy
import os

class Pyscf_ccsd(object):
    """ Wrapper for pyscf ccsd solvers
    """
    def __init__(self, ntot, nimp, nbath):
        """Constructor method
        """
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.hsize = 2**ntot
        self.type= 'PySCFCCSD'
        # initialize pyscf solvers

    def build_Hemb(self, D, H1E, LAMBDA, V2E, spin_pen=0.0):
        tmat = numpy.zeros((self.ntot,self.ntot))
        tmat[:self.nimp,:self.nimp] = H1E
        tmat[:self.nimp,self.nimp:] = D.T
        tmat[self.nimp:,self.nimp:] = -LAMBDA
        tmat[self.nimp:,:self.nimp] = D.conj()
        self.h1 = tmat[::2,::2]
        self.h2 = numpy.zeros((self.ntot//2,self.ntot//2,self.ntot//2,self.ntot//2))
        self.h2[:self.nimp//2,:self.nimp//2,:self.nimp//2,:self.nimp//2] = V2E[::2,::2,1::2,1::2] # spin symmetric

    def solve_Hemb(self, num_eig=10, verbose=0, restrict=True):
        self.restrict = restrict # restrict CCSD? True: for spin symmetry
        if restrict:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.RHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8#1e-12
            mf.diis_space = 20
            mf.diis_start_cycle = 5
            mf.chkfile = 'hf.chk'
            if os.path.isfile('hf.chk'):
                mol = lib.chkfile.load_mol('hf.chk')
                #mf = scf.HF(mol)
                mf.__dict__.update(lib.chkfile.load('hf.chk', 'scf'))
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = '1e'
            mf.diis = diis.EDIIS
            mf = mf.run()
            self.C = mf.mo_coeff
            self.mycc = cc.RCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 10000
            self.mycc.diis_space = 20#15
            self.mycc.diis_start_cycle = 5
            self.mycc.diis = diis.EDIIS
            self.mycc.iterative_damping = 0.05
            #self.mycc.diis = False
            self.mycc.diis_file = 'ccdiis.h5'
            if os.path.isfile('ccdiis.h5'):
                self.mycc.restore_from_diis_('ccdiis.h5')
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()
        else:
            mol = gto.M()
            mol.nelectron = self.ntot//2
            mol.incore_anyway = True
            mf = scf.UHF(mol)
            mf.max_cycle = 1000
            mf.conv_tol = 1e-8
            mf.diis_space = 20
            mf.init_guess_breaksym = True
            mf.get_hcore = lambda *args: self.h1
            mf.get_ovlp = lambda *args: numpy.eye(self.ntot//2)
            mf._eri = ao2mo.restore(8, self.h2, self.ntot//2) # 8-fold symmetry
            mf.init_guess = 'minao'
            mf = mf.run()
            self.Cup, self.Cdn = mf.mo_coeff
            self.mycc = cc.UCCSD(mf)
            self.mycc.conv_tol = 1e-8
            self.mycc.conv_tol_normt = 1e-5
            self.mycc.max_cycle = 500
            self.mycc.diis_space = 20
            self.mycc.diis_start_cycle = 4
            #self.mycc.diis = False
            self.mycc.iterative_damping = 0.001
            self.mycc.diis = diis.ADIIS
            #self.eccsd, t1, t2 = self.mycc.kernel()
            self.eccsd, t1, t2 = self.mycc.ccsd()
            #self.mycc.nroots = 3
            #e, v = self.mycc.ipccsd()
            #self.mycc.solve_lambda()
            self.e0 = self.eccsd + mf.energy_tot()

    def calc_density_matrix(self):
        if self.restrict:
            dmup = self.mycc.make_rdm1()/2.
            dmup = numpy.einsum('ai,bj,ij->ab',self.C, self.C, dmup)
            #print('dmup=')
            #print(dmup)
            #assert(numpy.allclose(dmup,dmdn))
            dm = numpy.kron(dmup,numpy.eye(2))
            self.dm = dm
            return dm
        else:
            dmup, dmdn = self.mycc.make_rdm1()
            dmup = numpy.einsum('ai,bj,ij->ab',self.Cup, self.Cup, dmup)
            dmdn = numpy.einsum('ai,bj,ij->ab',self.Cdn, self.Cdn, dmdn)
            #print('dmup=')
            #print(dmup)
            #print('dmdn=')
            #print(dmdn)
            dm = (dmup + dmdn)/2. # average over spin to recover spin symmetry. Don't use it with magnetism
            dm = numpy.kron(dm,numpy.eye(2))
            self.dm = dm
            return dm

    def compute_E2loc(self):
        eone = 2*numpy.einsum('ij,ij',self.h1,self.dm[::2,::2])
        etwo = self.e0 - eone
        return etwo

    def calc_double_occ(self,idx):
        print('double occupancy not implemented')
        return 0.25
