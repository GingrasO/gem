from triqs_dft_tools.sumk_dft import *
from triqs_ghostGA.utils_TH import denR, denRm1, ddenRm1, realHcombination, inverse_realHcombination, \
     Hermitian_list, get_blocks, funcMat, calc_nf, dF

class SumkGRISB(SumkDFT):
    '''
    Inherent from SumkDFT for GRISB k-summation
    '''
    def __init__(self, *args, nbath, **kwargs):
      '''
      Inherent all initial parameters from sumk_dft
      '''
      super().__init__(*args, **kwargs)
      # additional grisb parameters
      self.nbath = nbath
      print('number of bath orbital:', nbath)

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

    def ksum1(self, R, Lambda):
        '''
        The first k-summation for quasiparticle density matrix. currently only consider one correlated shell
        TODO: self.Delta_p = [{} for icrsh in range(self.n_corr_shells)]
        '''
        self.Delta_p = {}
        ikarray = np.array(list(range(self.n_k)))
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            self.Delta_p[sp] = np.zeros((self.rhoks[sp].shape[1],self.rhoks[sp].shape[2]),dtype=complex)
            for ik in mpi.slice_array(ikarray):
                self.Delta_p[sp][:,:] += self.rhoks[sp][ik,:,:]
            self.Delta_p[sp][:,:] = self.Delta_p[sp][:,:]/self.rhoks[sp].shape[0]

    def ksum2(self, R, Lambda):
        '''
        The second k-sum for kinetic energy
        '''
        pass

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
