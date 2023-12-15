###############################################
# ghost-RISB/GA algorithm
# Author: Tsung-Han Lee 
# Email:  henhans74716@gmail.com
###############################################
import scipy
from scipy.linalg import sqrtm
import h5py
import numpy
import numba
from triqs_ghostGA.ci import *
from triqs_ghostGA.ftps import *
from triqs_ghostGA.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination\
     , Hermitian_list, get_blocks, funcMat, calc_nf, dF
from triqs_ghostGA.DIIS import *

def calc_rhoks(R, Lambda, eks, T):
    return [calc_nf( numpy.dot(R, numpy.dot(x, R.conj().T ) ) + Lambda ,T).T for x in eks]

def calc_Delta_p(rhok_list):
    return sum(rhok_list)/len(rhok_list)

def calc_D(R, Lambda, Delta_p, eks, rhoks):
    """ Compute D matrix
    """
    Left=[numpy.dot( numpy.dot(eks[x], R.conj().T ), rhoks[x].T ) for x in range(len(rhoks))]
    Left=sum(Left)/float(len(rhoks))
    Right=funcMat(Delta_p, denR)
    return numpy.dot(Right,numpy.transpose(Left))

def calc_Lambda_c(R, Lambda, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda.shape[0]
    l=inverse_realHcombination(Lambda,H_list)
    lc=numpy.copy(l)*0.0
    MM=numpy.dot(D,numpy.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T 
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=numpy.trace(numpy.dot(MM,derivative))
        lc[k]=-l[k]-(tt+numpy.conjugate(tt)).real
    Lambda_c=realHcombination(lc,H_list)
    return Lambda_c

def calc_Lambda(R, Lambda_c, Delta_p, D, H_list):
    """ Compute Lambda_c matrix
    """
    no = Lambda_c.shape[0]
    lc=inverse_realHcombination(Lambda_c,H_list)
    l=numpy.copy(lc)*0.0
    MM=numpy.dot(D,numpy.transpose(R))
    for k in range(len(H_list)):
        AA=Delta_p
        HH=H_list[k].T 
        derivative=dF(AA,HH, denRm1, ddenRm1)
        tt=numpy.trace(numpy.dot(MM,derivative))
        l[k]=-lc[k]-(tt+numpy.conjugate(tt)).real
    Lambda=realHcombination(l,H_list)
    return Lambda

def cost_function(x, *args):
    ''' Cost function for find Lambda
    '''
    R, ffdagger, eks, Hspin_list, beta = args
    Lambda_spin = realHcombination(x, Hspin_list)
    Lambda = numpy.kron(Lambda_spin,numpy.eye(2))
    rhok_list=calc_rhoks(R, Lambda, eks, 1./beta)
    Delta_p=calc_Delta_p(rhok_list)
    diff = numpy.linalg.norm(Delta_p[::2,::2]-ffdagger.T[::2,::2])
    #diff = inverse_realHcombination(Delta_p[::2,::2]-ffdagger.T[::2,::2], Hspin_list)
    #print('diff=',diff)
    return diff

def find_Lambda(Lambda0, R, ffdagger, eks, Hspin_list, beta):
    """ Find Lambda for given ffdagger
    """
    Lambda0_spin = Lambda0[::2,::2]
    x = inverse_realHcombination(Lambda0_spin, Hspin_list)
    args = (R, ffdagger, eks, Hspin_list, beta)
    result = scipy.optimize.minimize( cost_function, x, args=args, tol=1e-12, method='L-BFGS-B', options={'eps':1e-12} )
    #result = scipy.optimize.root( cost_function, x, args=args, method='lm', options={'eps':1e-10} )
    if ( result.success==False ):
        print("   Minimize mesage ::",result.message)
    print("   Minimize :: Cost function after convergence =",result.fun)
    Lambda = numpy.kron(realHcombination(result.x, Hspin_list),numpy.eye(2))
    return Lambda

def svd_truncate_R(R, eps=0.5):
    "perform SVD truncation for the singular value of R greater than 1 and smaller than a threshold eps"
    from scipy.linalg import svd
    u, s, vh = svd(R)
    print('singular values of R:', s)
    sp = np.zeros(R.shape, dtype=s.dtype)
    for i,si in enumerate(s):
        if si > 1.0:
            sp[i,i] = 1.0
        elif si < (1.0 - eps):
            sp[i,i] = (1.0 - eps)
        else:
            sp[i,i] = si
    Rp = u @ sp @ vh
    return Rp

class Grisb(object):
    """This is a class representation of a ghost-RISB object.

    :param ntot: Total number of orbital.
    :type ntot: int

    :param nimp: Total number of imp orbital.
    :type ntot: int

    :param nbath: Total number of bath orbital.
    :type ntot: int

    :param eks: Momentum distribution.
    :type eks: numpy.ndarray

    :param eloc: Local one-body Hamtilonian.
    :type eloc: numpy.ndarray

    :param Utensor: Local two-obdy interaction.
    :type Utensor: numpy.ndarray

    :param R: R matrix.
    :type R: numpy.ndarray

    :param Lambda: Lambda matrix.
    :type Lambda: numpy.ndarray

    :param ed_params: Exact diagonalization solver parameters.
    :type ed_params: dic

    :param Hspin_list: Hermitian matrix basis in each spin block with spin symmetry.
    :type Hspin_list: list

    :param Hfull_list: Full Hermitian matrix basis.
    :type Hfull_list: list

    """
    def __init__(self, ntot, nimp, nbath, eks, eloc, Utensor, spin_sym=True, soc=False, R=None, Lambda=None, ed_params={'solver':'fed'}):
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        self.eks = eks
        self.eloc = eloc
        self.Utensor = Utensor
        self.soc = soc
        self.spin_sym = spin_sym
        # initialize R and Lambda
        if R is None:
            self.R = numpy.kron(numpy.ones((nbath//2,nimp//2)), numpy.eye(2))*0.5
        else:
            if R.shape != (nbath,nimp):
                raise ValueError("R has inconsistent shape. Should be (nbath,nimp)")
            self.R = R
        if Lambda is None:
            #self.Lambda = numpy.kron(numpy.random.rand(nbath//2,nbath//2)+1j*numpy.random.rand(nbath//2,nbath//2), numpy.eye(2))
            self.Lambda = numpy.zeros((nbath,nbath))
            self.Lambda[0,0] = 1.0
            self.Lambda[1,1] = 1.0
            self.Lambda[2,2] =-1.0
            self.Lambda[3,3] =-1.0
        else:
            if Lambda.shape != (nbath,nbath):
                raise ValueError("Lambda has inconsistent shape. Should be (nbath,nbath)")
            self.Lambda = Lambda
        # initialize edsolver
        self.initialize_edsolver(ed_params)
        # initialize single-particle basis
        self.Hspin_list,self.tHspin_list=Hermitian_list(nbath//2)
        self.Hfull_list,self.tHfull_list=Hermitian_list(nbath)
        print('initial R matirx =')
        print(self.R)
        print('initial Lambda matirx =')
        print(self.Lambda)

    def initialize_edsolver(self, ed_params):
        if ed_params["solver"] == 'ci':
            self.edsolver = CI(self.ntot, use_Ntot=ed_params["use_Ntot"], use_Sz=ed_params["use_Sz"], dtype=np.complex128)
        elif ed_params["solver"] == 'ftps':
            self.edsolver = FTPS(self.ntot, self.nimp, self.nbath, ed_params["maxM"])
        else:
            raise ValueError("impurity solver are supported")

    def build_h1e(self,mu):
        h1e = np.zeros((self.ntot,self.ntot), dtype=np.complex128)
        h1e[:self.nimp,:self.nimp] = self.eloc - mu*np.eye(self.nimp)
        h1e[:self.nimp,self.nimp:] = self.D.T
        h1e[self.nimp:,self.nimp:] = -self.Lambda_c
        h1e[self.nimp:,:self.nimp] = self.D.conj()
        return h1e

    def solve_embedding(self, mu, num_eig, ed_verbose, spin_pen, sz_pen=0.0):
        """ Solve embedding problem using a variety of impurity solver
        """
        fh5 = h5py.File('hemb_test.h5','w')
        fh5['eloc'] = self.eloc
        fh5['D'] = self.D
        fh5['Lambda_c'] = self.Lambda_c
        fh5['Utensor'] = self.Utensor
        fh5['mu'] = mu
        fh5.close()
        #print('Lambda_c=')
        #print(self.Lambda_c)
        #print('mu=',mu)
        #print('eloc=')
        #print(self.eloc)
        if type(self.edsolver) == CI:
            h1e = self.build_h1e(mu)
            #print('h1e=')
            #print(h1e)
            #print('spin_pen=',spin_pen)
            self.edsolver.build_Hemb(h1e, self.Utensor, spin_pen=spin_pen, sz_pen=sz_pen)
            self.edsolver.solve_Hemb(num_eig=num_eig, verbose=ed_verbose )
            self.denMat = self.edsolver.calc_density_matrix()
            self.E2loc = self.edsolver.compute_E2loc()
        elif type(self.edsolver) == FTPS:
            self.edsolver.build_Hemb(self.D, self.eloc- mu*np.eye(self.nimp), self.Lambda_c, self.Utensor, spin_pen=spin_pen)
            self.edsolver.solve_Hemb(num_eig=num_eig, verbose=ed_verbose )
            self.denMat = self.edsolver.calc_density_matrix()
            self.E2loc = self.edsolver.compute_E2loc()
        else:
            raise ValueError("only Full ED, CI, and HCI are supported")
        #print(self.denMat)
        #quit()

    def compute_energy(self,beta=200.,mu=0.0):
        """ Compute total energy, kinetic energy, and potential energy
        """
        #self.ekin = [numpy.sum(self.R.dot( self.eks[x] ).dot( self.R.conj().T )*self.rhok_list[x].T ) for x in range(len(self.rhok_list))]
        #self.ekin = sum(self.ekin)/float(len(self.rhok_list))
        self.ekin = sum([numpy.sum( ( numpy.dot(self.R, numpy.dot(x, self.R.conj().T )) ) * \
                    calc_nf( numpy.dot(self.R, numpy.dot(x, self.R.conj().T) ) + self.Lambda , 1./beta).T ) for x in self.eks] )/float(len(self.eks))
        self.epot = self.E2loc + np.trace(self.eloc.dot(self.denMat[:self.nimp,:self.nimp].T))
        self.etot = self.ekin + self.epot - mu*self.nfill

    def run(self, mu=0.0, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=2, ed_verbose=0, diis=False):
        """ Run ghost-RISB self-consistency

        :param itmax: Maxiumum iteraction for self-consistency.
        :type itmax: int
    
        :param tol: Tolerence for convergence
        :type tol: float
    
        :param beta: Inverse temperature (equivalent to smearing temperature).
        :type beta: float

        :param silence: Silence the printing.
        :type silence: bool
 
        :param spin_pen: Penalty for S2 conservation.
        :type spin_pen: float
    
        :param silence: Orbital index for computing double occupancy.
        :type idx: int

        """
        print("mu = ", mu)
        self.diff = 1e20
        if diis is True:
            #RDIIS = DIIS(7)
            LDIIS = DIIS(7)
            numNonDIIS = 4
        for it in range(itmax):
            # compute qp density matrix
            self.rhok_list=calc_rhoks(self.R, self.Lambda, self.eks, 1./beta)
            self.Delta_p=calc_Delta_p(self.rhok_list)
            self.D=calc_D(self.R, self.Lambda, self.Delta_p, self.eks, self.rhok_list)
            self.Lambda_c=calc_Lambda_c(self.R, self.Lambda, self.Delta_p, self.D, self.Hfull_list)
            if not silence:
                if not self.soc:
                    print("Delta_p=")
                    print(self.Delta_p[::2,::2])
                    print("D=")
                    print(self.D[::2,::2])
                    print("Lambda_c=")
                    print(self.Lambda_c[::2,::2])
                else:
                    print("Delta_p=")
                    print(self.Delta_p[:,:])
                    print("D=")
                    print(self.D[:,:])
                    print("Lambda_c=")
                    print(self.Lambda_c[:,:])
            # ED solvers
            self.solve_embedding(mu, num_eig, ed_verbose, spin_pen, sz_pen)
            #Update R and Update Lambda
            cdaggerf = self.denMat[:self.nimp,self.nimp:]
            ffdagger = self.denMat[self.nimp:,self.nimp:]
            ffdagger = (numpy.eye(self.nbath,dtype=numpy.complex128) - ffdagger).T
            #if not silence:
            print("norm(ffdagger.T-Delta_p)=", numpy.linalg.norm(ffdagger.T-self.Delta_p))
            self.Delta_p = ffdagger.T
            R_new = numpy.transpose(cdaggerf.dot(funcMat(self.Delta_p, denR)))
            if not self.soc:
                R_new = numpy.kron(R_new[::2,::2],numpy.eye(2))# symmetrize
            R_new = svd_truncate_R(R_new)
            #Lambda_new = find_Lambda(self.Lambda, R_new, ffdagger, self.eks, self.Hspin_list, beta)
            Lambda_new = calc_Lambda(R_new, self.Lambda_c, self.Delta_p, self.D, self.Hfull_list)
            if not self.soc:
                Lambda_new = numpy.kron(Lambda_new[::2,::2],numpy.eye(2)) # symmetryize
            diff_R = numpy.abs(self.R-R_new).max()
            diff_Lambda = numpy.abs(self.Lambda-Lambda_new).max()
            self.diff = max(diff_R,diff_Lambda)
            if diis and ( it >= numNonDIIS ):
                error = Lambda_new - self.Lambda
                error = np.reshape( error, error.shape[0]*error.shape[1] )
                LDIIS.append( error, Lambda_new )
                #error = R_new - self.R
                #error = np.reshape( error, error.shape[0]*error.shape[1] )
                #RDIIS.append( error, R_new )
                self.R = R_new#RDIIS.Solve()
                self.Lambda = LDIIS.Solve()
            else:
                self.R = (1.-mix)*numpy.copy(self.R) + mix*R_new
                self.Lambda = (1.-mix)*numpy.copy(self.Lambda) + mix*Lambda_new
#           Try fix a gague that R is non-zero only on the upper left Experiment!
#            tmp = numpy.zeros(self.R.shape,dtype=self.R.dtype)
#            tmp[:self.nimp,:self.nimp] = sqrtm(self.R.conj().T.dot(self.R)[:self.nimp,:self.nimp])
#            self.R = tmp
            # check point
            fh5 = h5py.File('checkpoint.h5','w')
            fh5['R'] = self.R
            fh5['Lambda'] = self.Lambda
            fh5['eks'] = self.eks
            fh5['Utensor'] = self.Utensor
            fh5['mu'] = mu
            fh5.close()
            if not silence:
                print("R_new=")
                print(R_new)
                print("R=")
                print(self.R)
                print("Lambda_new=")
                print(Lambda_new)
                print("Lambda=")
                print(self.Lambda)
                print("ffdagger.T")
                print(ffdagger.T)
                print("density matrix=")
                print(self.denMat[::2,::2])
            print("iteration:",it,'diff=',self.diff)
            if self.diff < tol or it == (itmax-1):
                print("--------------------- ghost-RISB converged with diff=%g ---------------------"%(self.diff))
                print("density matrix=")
                print(self.denMat)
                self.nfill = numpy.trace(self.denMat[:self.nimp,:self.nimp])
                self.docc = []
                for idx in range(0,self.nimp,2):
                    self.docc.append(self.edsolver.calc_double_occ(idx))
                print("double occupancy=", self.docc)
                break

    def func_mu(self, x, *args):
        mu = x
        #self.mu_tmp = mu
        nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis = args
        self.run(mu, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        diff = nfix - self.nfill 
        print('nfix-nfill=', diff, 'nfill=',self.nfill)
        return diff

    def run_canonical(self, mu0, nfix, itmax=200, mix=0.5, tol=1e-6, beta=200., silence=True, spin_pen=0.0, sz_pen=0.0, idx=0, num_eig=10, ed_verbose=0, diis=False, mu_tol=1e-2, dmu=0.05):
        print('canonical mu0=',mu0)
        self.run(mu0, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose, diis)
        print('nfix-nfill=', nfix - self.nfill, 'nfill=',self.nfill)
        if np.abs(nfix - self.nfill ) < mu_tol:
            return mu0
        else:
            args = ( nfix, itmax, mix, tol, beta, silence, spin_pen, sz_pen, idx, num_eig, ed_verbose , diis)
            #sols = scipy.optimize.root(self.func_mu,x0=mu0,args=args,method='lm',tol=1e-3,options={'eps':1e-5,'factor':0.1})
            #sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,args=args,method='bisect',bracket=(mu0-0.05,mu0+0.05),xtol=1e-3)
            sols = scipy.optimize.root_scalar(self.func_mu,x0=mu0,x1=mu0+dmu,args=args,method='secant',xtol=mu_tol)
            #sols = scipy.optimize.root_scalar(self.func_mu,args=args,xtol=tol)
            print('root solver for mu converged? ',sols.converged)
            mu = sols.root
            print('mu=',mu)
            return mu 

    def compute_Gf_Sig(self, mu, ek_path, oms, eta):
        self.oms = oms
        self.eta = eta
        self.Gf, self.Sig = self._compute_Gf_Sig(mu, ek_path, oms, eta, self.R, self.Lambda, self.eloc, self.nbath, self.nimp)

    @staticmethod
    #@numba.jit(nopython=True)
    def _compute_Gf_Sig(mu, ek_path, oms, eta, R, Lambda, eloc, nbath, nimp):
        Gf = np.zeros((ek_path.shape[0],oms.shape[0],nimp,nimp),dtype=numpy.complex128)#numba.complex128)
        Sig = np.zeros((oms.shape[0],nimp,nimp),dtype=numpy.complex128)#numba.complex128)
        for ik, ek in enumerate(ek_path):
            for iom, om in enumerate(oms):
                Gf[ik,iom,:,:] = R.conj().T.dot( numpy.linalg.inv( (om+1j*eta)*np.eye(nbath)
                                  - R.dot(ek).dot(R.conj().T) - Lambda ) ).dot(R)
                if ik == 0:
                    Sig[iom,:,:] = (om + 1j*eta + mu)*np.eye(nimp) - ek - eloc - numpy.linalg.inv(Gf[ik,iom]) #om + 1j*eta - ek - numpy.linalg.inv(Gf[ik,iom])
        return Gf, Sig

