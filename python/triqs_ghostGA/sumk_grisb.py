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

    def lattice_gf_qp(self, R, Lambda, ik, mu=None, broadening=None, mesh=None):
        r"""
        Calculates the Quasiparticle lattice Green function for a given k-point from the DFT Hamiltonian and the self energy.
        Currently only consider a single correlated shell and no ghost orbital.
        
        Parameters
        ----------
        ik : integer
             k-point index.
        mu : real, optional
             Chemical potential for which the Green's function is to be calculated.
             If not provided, self.chemical_potential is used for mu.
        broadening : real, optional
                     Imaginary shift for the axis along which the real-axis GF is calculated.
                     If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
        mesh : MeshReFreq or MeshImFreq, optional
                    Mesh to be used if with_Sigma=False. If with Sigma=False and mesh is none then self.mesh is used.

        Returns
        -------
        G_latt_qp : BlockGf
                Quasiparticle Lattice Green's function.

        """
        if mu is None:
            mu = self.chemical_potential
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        if not hasattr(self, "Sigma_imp"):
            with_Sigma = False
        if broadening is None:
            if mesh is None:
                broadening = 0.01
            else:  # broadening = 2 * \Delta omega, where \Delta omega is the spacing of omega points
                broadening = 2.0 * ((mesh.w_max - mesh.w_min) / (len(mesh) - 1))

        # Check if G_latt_qp is present
        set_up_G_latt_qp = False                       # Assume not
        if not hasattr(self, "G_latt_qp" ):
            # Need to create G_latt_(i)w
            set_up_G_latt_qp = True
        else:                                       # Check that existing GF is consistent
            G_latt_qp = self.G_latt_qp
            GFsize = [gf.target_shape[0] for bname, gf in G_latt_qp]
            unchangedsize = all([self.n_orbitals[ik, ntoi[spn[isp]]] == GFsize[
                                isp] for isp in range(self.n_spin_blocks[self.SO])])
            if (not mesh is None) or (not unchangedsize):
                set_up_G_latt_qp = True

        if not mesh is None:
            assert isinstance(mesh, MeshReFreq) or isinstance(mesh, MeshImFreq),  "mesh must be a triqs MeshReFreq or MeshImFreq"
            if isinstance(mesh, MeshImFreq):
                mesh_values = np.linspace(mesh(mesh.first_index()), mesh(mesh.last_index()), len(mesh))
            else:
                mesh_values = np.linspace(mesh.w_min, mesh.w_max, len(mesh))
        else:
            mesh = self.mesh
            mesh_values = self.mesh_values

        # Set up G_latt
        if set_up_G_latt_qp:
            block_structure = [
                list(range(self.n_orbitals[ik, ntoi[sp]])) for sp in spn]
            gf_struct = [(spn[isp], block_structure[isp])
                         for isp in range(self.n_spin_blocks[self.SO])]
            block_ind_list = [block for block, inner in gf_struct]
            if isinstance(mesh, MeshImFreq):
                glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                                 for block, inner in gf_struct]
            else:
                glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                                 for block, inner in gf_struct]
            G_latt_qp = BlockGf(name_list=block_ind_list,
                             block_list=glist(), make_copies=False)
            G_latt_qp.zero()

        idmat = [np.identity(
            self.n_orbitals[ik, ntoi[sp]], complex) for sp in spn]

        #print('mesh:',mesh)
        #print('gf=')
        #print(G_latt_qp['up'].data.shape)
        #quit()

        # fill Glatt
        for ibl, (block, gf) in enumerate(G_latt_qp):
            ind = ntoi[spn[ibl]]
            sp = spn[ibl]
            n_orb = self.n_orbitals[ik, ind]
            for icrsh in range(self.n_corr_shells):
                dim = self.corr_shells[icrsh]['dim']
                MMat = self.hopping[
                                ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
                if isinstance(mesh, MeshImFreq):
                    gf.data[:, :, :] = (idmat[ibl] * (mesh_values[:, None, None] + mu) #+ self.h_field*(1-2*ibl))
                                        - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) ) 
                                        - Lambda[icrsh][sp] ) 
                else:
                    gf.data[:, :, :] = (idmat[ibl] *
                                        (mesh_values[:, None, None] + mu + 1j*broadening)# + self.h_field*(1-2*ibl)
                                        - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) ) 
                                        - Lambda[icrsh][sp] ) 

        G_latt_qp.invert()
        self.G_latt_qp = G_latt_qp

        return G_latt_qp
    
    def extract_G_phy(self, R, Lambda, mu=None, broadening=None, mesh=None, show_warnings=True):
        r"""
        Extracts the local downfolded Green function by the Brillouin-zone integration of the lattice Green's function.
        Currently only consider a single correlated shell and no ghost orbital.

        Parameters
        ----------
        mu : real, optional
            Input chemical potential. If not provided the value of self.chemical_potential is used as mu.
        broadening : float, optional
            Imaginary shift for the axis along which the real-axis GF is calculated.
            If not provided, broadening will be set to double of the distance between mesh points in 'mesh'.
            Only relevant for real-frequency GF.
        show_warnings : bool, optional
            Displays warning messages during transformation
            (Only effective if transform_to_solver_blocks = True

        Returns
        -------
        G_loc : list of BlockGf (Green's function) objects
            List of the local Green's functions for all (inequivalent) correlated shells,
            rotated into the corresponding local frames.
            If ``transform_to_solver_blocks`` is True, it will be one per inequivalent correlated shell, else one per
            correlated shell.
        """

        if mu is None:
            mu = self.chemical_potential

        if mesh is None:
            mesh = self.mesh

        # create G_loc to be returned in sumk space for all correlated shells. Trafo to solver block structure done later
        G_loc = [self.block_structure.create_gf(ish=ish, mesh=mesh, space='sumk') for ish in range(self.n_corr_shells)]
        #print(G_loc[0]['up'].data.shape)

        ikarray = np.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            if isinstance(mesh, MeshImFreq):
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu)
            else:
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu, broadening=broadening, mesh=mesh)
            G_latt_qp *= self.bz_weights[ik]
            
            for icrsh in range(self.n_corr_shells):
                # init temporary storage
                for bname, gf in G_loc[icrsh]:
                    #print(G_latt_qp[bname].data.shape)
                    gf.data[:,:,:] += np.einsum('ij,kjl,lm->kim', R[icrsh][bname].conj().T, G_latt_qp[bname].data, R[icrsh][bname])

        # Collect data from mpi
        for icrsh in range(self.n_corr_shells):
            G_loc[icrsh] << mpi.all_reduce(G_loc[icrsh])
        mpi.barrier()

        return G_loc

