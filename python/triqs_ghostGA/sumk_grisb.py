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
        self.rhoks = np.zeros((self.n_k,Lambda.shape[0],Lambda.shape[1]))
        ikarray = np.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            print('ik=', ik)
            for isp in [0]:
                print(self.hopping[ik])
                self.rhoks[ik,:,:] = calc_nf( np.dot(R, np.dot(self.hopping[ik,isp], R.conj().T ) ) + Lambda ,T).T

    def ksum1(self, R, Lambda):
        '''
        The first k-summation for density matrix
        '''
        Delta_p = np.zeros((self.nbath,self.nbath))
        

    def ksum2(self, R, Lambda):
        '''
        The second k-sum for kinetic energy
        '''
        pass

    def calc_mu(self, R, Lambda):
        '''
        Override the sumk calc_mu for GRISB
        '''
        pass

    def calc_density_correction(self):
        '''
        Overide the density correction for GRISB
        '''
        pass
