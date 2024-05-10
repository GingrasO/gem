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
        #print('number of bath orbital:', nbath)
        # Generic Hermitian matrix basis for ghostGA
        self.H_list = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.H_list[icrsh][sp] = Hermitian_list(self.nbath)[0]

    def calc_rhoks(self, R, Lambda, T):
        '''
        density matrix for each momentum. currently only consider one correlated shell.
        TODO: self.rhoks = [{} for icrsh in range(self.n_corr_shells)]
        '''
        self.rhoks = [{} for icrsh in range(self.n_corr_shells)]
        ikarray = np.array(list(range(self.n_k)))
        #icrsh = 0
        for icrsh in range(self.n_corr_shells):
            dim = self.corr_shells[icrsh]['dim']
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.rhoks[icrsh][sp] = np.zeros((self.n_k,Lambda[icrsh][sp].shape[0],
                                           Lambda[icrsh][sp].shape[1]),dtype=complex)
                ind = self.spin_names_to_ind[
                            self.corr_shells[icrsh]['SO']][sp]
                for ik in mpi.slice_array(ikarray):
                    #print('ik=', ik, 'isp=', isp, 'sp=', sp, self.spin_names_to_ind[self.SO][sp])
                    #print(self.hopping[ik,isp,:,:])
                    n_orb = self.n_orbitals[ik, ind]
                    #MMat = np.identity(n_orb, complex)
                    MMat = self.hopping[
                                ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
                    self.rhoks[icrsh][sp][ik,:,:] = calc_nf(np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) ) 
                                                     + Lambda[icrsh][sp]
                                                     - self.chemical_potential*np.eye(Lambda[icrsh][sp].shape[0]) ,T).T

    def calc_Delta(self):
        '''
        The first k-summation for quasiparticle density matrix. currently only consider one correlated shell
        TODO: self.Delta = [{} for icrsh in range(self.n_corr_shells)]
        '''
        self.Delta = [{} for icrsh in range(self.n_corr_shells)]
        ikarray = np.array(list(range(self.n_k)))
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.Delta[icrsh][sp] = np.zeros((self.rhoks[icrsh][sp].shape[1],
                                                  self.rhoks[icrsh][sp].shape[2]),dtype=complex)
                for ik in mpi.slice_array(ikarray):
                    self.Delta[icrsh][sp][:,:] += self.bz_weights[ik] * self.rhoks[icrsh][sp][ik,:,:]
                #self.Delta[sp][:,:] = self.Delta[sp][:,:]/self.rhoks[sp].shape[0]

    def calc_D(self, R):
        '''
        The second k-sum for kinetic energy. currently only consider one correlated shell
        TODO:
        '''
        self.D = [{} for icrsh in range(self.n_corr_shells)]
        ikarray = np.array(list(range(self.n_k)))
        for icrsh in range(self.n_corr_shells):
            dim = self.corr_shells[icrsh]['dim']
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                ind = self.spin_names_to_ind[
                self.corr_shells[icrsh]['SO']][sp]
                sum_ek_Rdagger_rhoks = np.zeros((self.rhoks[icrsh][sp].shape[1],
                                                 self.rhoks[icrsh][sp].shape[2]),dtype=complex)
                for ik in mpi.slice_array(ikarray):
                    n_orb = self.n_orbitals[ik, ind]
                    #MMat = np.identity(n_orb, complex)
                    MMat = self.hopping[
                                ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
                    sum_ek_Rdagger_rhoks[:,:] += self.bz_weights[ik]*MMatproj_nloc.dot(R[icrsh][sp].conj().T).dot(
                                                 self.rhoks[icrsh][sp][ik,:,:].T)
                #sum_ek_Rdagger_rhoks[:,:] = sum_ek_Rdagger_rhoks[:,:]/self.rhoks[sp].shape[0]
                sqrt_Delta=funcMat(self.Delta[icrsh][sp], denR)
                self.D[icrsh][sp] = sum_ek_Rdagger_rhoks.dot(np.transpose(sqrt_Delta))

    def calc_Lambdac(self, R, Lambda):
        '''
        Calculate Lambdac matrix
        '''
        self.Lambdac = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.Lambdac[icrsh][sp] = self.calc_Lambdac_icrsh_isp(R[icrsh][sp], Lambda[icrsh][sp], 
                                        self.Delta[icrsh][sp], self.D[icrsh][sp], self.H_list[icrsh][sp])

    def calc_Lambda(self, R, Lambdac):
        '''
        Calculate Lambda matrix
        '''
        Lambda = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                Lambda[icrsh][sp] = self.calc_Lambda_icrsh_isp(R[icrsh][sp], Lambdac[icrsh][sp], 
                                        self.Delta[icrsh][sp], self.D[icrsh][sp], self.H_list[icrsh][sp])
        return Lambda

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
    def calc_Lambdac_icrsh_isp(R, Lambda, Delta_p, D, H_list):
        """ Compute Lambda_c matrix for a specific shell icrsh and spin isp
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
    def calc_Lambda_icrsh_isp(R, Lambda_c, Delta_p, D, H_list):
        """ Compute Lambda matrix for a specific shell icrsh and spin isp 
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

