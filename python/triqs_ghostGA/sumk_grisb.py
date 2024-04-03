from triqs_dft_tools.sumk_dft import *
from triqs_ghostGA.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list, get_blocks, funcMat, calc_nf, dF

class SumkGRISB(SumkDFT):
    '''
    Inherent from SumkDFT for GRISB k-summation
    '''
    def __init__(self, *args, **kwargs):
      '''
      Inherent all initial parameters from sumk_dft
      '''
      super().__init__(*args, **kwargs)
      # additional grisb parameters
      #self.nbath = nbath
      #print('number of bath orbital:', nbath)

    def calc_rhoks(self, R, Lambda, T):
        '''
        density matrix for each momentum. currently only consider one correlated shell.
        TODO: self.rhoks = [{} for icrsh in range(self.n_corr_shells)]
        '''
        self.rhoks = {}
        ikarray = np.array(list(range(self.n_k)))
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            self.rhoks[sp] = np.zeros((self.n_k,Lambda.shape[0],Lambda.shape[1]),dtype=complex)
            for ik in mpi.slice_array(ikarray):
                #print('ik=', ik, 'isp=', isp, 'sp=', sp, self.spin_names_to_ind[self.SO][sp])
                #print(self.hopping[ik,isp,:,:])
                self.rhoks[sp][ik,:,:] = calc_nf( np.dot(R, np.dot(self.hopping[ik,isp], R.conj().T ) ) + Lambda ,T).T

    def calc_Delta(self, R, Lambda):
        '''
        The first k-summation for quasiparticle density matrix. currently only consider one correlated shell
        TODO: self.Delta = [{} for icrsh in range(self.n_corr_shells)]
        '''
        self.Delta = {}
        ikarray = np.array(list(range(self.n_k)))
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            self.Delta[sp] = np.zeros((self.rhoks[sp].shape[1],self.rhoks[sp].shape[2]),dtype=complex)
            for ik in mpi.slice_array(ikarray):
                self.Delta[sp][:,:] += self.rhoks[sp][ik,:,:]
            self.Delta[sp][:,:] = self.Delta[sp][:,:]/self.rhoks[sp].shape[0]

    def calc_D(self, R, Lambda):
        '''
        The second k-sum for kinetic energy. currently only consider one correlated shell
        TODO:
        '''
        self.D = {}
        ikarray = np.array(list(range(self.n_k)))
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            sum_ek_Rdagger_rhoks = np.zeros((self.rhoks[sp].shape[1],self.rhoks[sp].shape[2]),dtype=complex)
            for ik in mpi.slice_array(ikarray):
                sum_ek_Rdagger_rhoks[:,:] += self.hopping[ik,isp].dot(R.conj().T).dot(self.rhoks[sp][ik,:,:].T)
            sum_ek_Rdagger_rhoks[:,:] = sum_ek_Rdagger_rhoks[:,:]/self.rhoks[sp].shape[0]
            sqrt_Delta=funcMat(self.Delta[sp], denR)
            self.D[sp] = sum_ek_Rdagger_rhoks.dot(np.transpose(sqrt_Delta))

    def calc_mu_grisb(self, R, Lambda):
        '''
        Override the sumk calc_mu for GRISB
        '''
        pass

    def calc_density_correction(self):
        '''
        Overide the density correction for GRISB
        '''
        pass

    @staticmethod
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
   
    @staticmethod 
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

