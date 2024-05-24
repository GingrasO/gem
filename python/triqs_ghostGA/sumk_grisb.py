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
        # read additional data u_total transformation matrix from bloch to wannier90 orbitals
        if not isinstance(self.hdf_file, str):
            mpi.report("Give a string for the hdf5 filename to read the input!")
        else:
            # additional properties to load
            # soon bz_weights is depraced and replaced by kpt_weights, kpts_basis and kpts will become required to read soon
            additional_things_to_read = ['u_total']
            subgroup_present_additional, self.additional_values_not_read = self.read_input_from_hdf(subgrp=self.dft_data, 
                                                                                things_to_read=additional_things_to_read)
        #print(self.hdf_file)
        #print('u_total=')
        #print(self.u_total)
        #print(self.additional_values_not_read)

        #print('number of bath orbital:', nbath)
        # Generic Hermitian matrix basis for ghostGA
        self.H_list = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.H_list[icrsh][sp] = Hermitian_list(self.nbath)[0]
        # Compute local atomic levels
        self.eff_atomic_levels()
        # non-local part of the hopping matrix
        self.calc_nonlocal_hopping()

    def calc_nonlocal_hopping(self):
        '''
        Compute the non-local part of the hopping term
        TODO: this routine will only work for a single-correlated shell. We stil need to generalize
        the code below.
        '''
        ikarray = np.array(list(range(self.n_k)))
        self.hopping_nloc = np.zeros(self.hopping.shape,dtype=self.hopping.dtype)
        # we need the local potential in the original basis
        self.eloc_orig = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.eloc_orig[icrsh][sp] = np.dot( np.dot( self.rot_mat[icrsh], self.Hsumk[icrsh][sp] ), self.rot_mat[icrsh].conj().T)
        if mpi.is_master_node():        
            print('eloc_orig=')
            print(self.eloc_orig)
            print('Hsumk=')
            print(self.Hsumk)
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for ik in mpi.slice_array(ikarray):
                n_orb = self.n_orbitals[ik, isp]
                # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                u = self.u_total[0,ik,:n_orb,:n_orb]
                # rotate to orbital basis
                self.hopping_nloc[ik, isp, :, :] = self.hopping[ik, isp, 0:n_orb, 0:n_orb].copy()
                #self.hopping_nloc[ik, isp, :, :] = np.dot( np.dot( u, self.hopping[ik, isp, 0:n_orb, 0:n_orb]), u.conj().T)
                index = 0
                for icrsh in range(self.n_corr_shells):
                    # local one-body in the original basis: Hsumk has been rotated to local coordinate
                    #eloc_orig = np.dot( np.dot( self.rot_mat[icrsh], self.Hsumk[icrsh][sp] ), self.rot_mat[icrsh].conj().T)
                    #print(eloc_orig)
                    dim = self.corr_shells[icrsh]['dim']
                    # TODO: the two lines below needs to be generalized to multicorrelated shell.
                    # specifically we need to take care of the index:
                    # icrsh*dim:icrsh*dim+dim,icrsh*dim:icrsh*dim+dim 
                    # which we have to arange the starting slice of the matrix icrsh*dim properly.
                    #hmat = self.hopping_nloc[ik, isp, index:index+dim,index:index+dim].copy()
                    #self.hopping_nloc[ik, isp, index:index+dim,index:index+dim] = hmat - self.eloc_orig[icrsh][sp]
                    #print(self.hopping_nloc[ik,ind,:,:])
                    projmat = self.proj_mat[ik, isp, icrsh, 0:dim, 0:n_orb]
                    self.hopping_nloc[ik,isp,:,:] -= np.dot( np.dot(projmat.conj().T, self.eloc_orig[icrsh][sp]),projmat) 
                    index += dim
                #rotate back to bloch basis
                #self.hopping_nloc[ik, isp, :, :] = np.dot( np.dot( u.conj().T,self.hopping_nloc[ik, isp, 0:n_orb, 0:n_orb]), u)


    def calc_R_Lambda_full(self, R, Lambda, ik, ind, sp):
        '''
        Construct full R and Lambda matrix in the full Wannier90 projection space.
        TODO: this routine will only work for a single-correlated shell. We stil need to generalize
        the code below.
        '''
        n_orb = self.n_orbitals[ik, ind]
        R_full = np.eye(n_orb,dtype=complex)
        Lambda_full = np.zeros((n_orb,n_orb),dtype=complex)
        index = 0
        for icrsh in range(self.n_corr_shells):
            dim = self.corr_shells[icrsh]['dim']
            # TODO: the two lines below needs to be generalized to multicorrelated shell.
            # specifically we need to take care of the index:
            # icrsh*dim:icrsh*dim+dim,icrsh*dim:icrsh*dim+dim 
            # which we have to arange the starting slice of the matrix icrsh*dim properly.
            R_full[index:index+dim,index:index+dim] = R[icrsh][sp]
            Lambda_full[index:index+dim,index:index+dim] = Lambda[icrsh][sp]
            index += dim
        #print(R_full)
        #print(Lambda_full)
        return R_full, Lambda_full

    def calc_rhoks(self, R, Lambda, T):
        '''
        density matrix for each momentum. NOTE: doesn't work for ghostGA yet.
        '''
        self.rhoks = [{} for icrsh in range(self.n_corr_shells)]
        self.rhoks_full = {}
        ikarray = np.array(list(range(self.n_k)))
        #icrsh = 0
        for icrsh in range(self.n_corr_shells):
            dim = self.corr_shells[icrsh]['dim']
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                self.rhoks[icrsh][sp] = np.zeros((self.n_k,Lambda[icrsh][sp].shape[0],
                                           Lambda[icrsh][sp].shape[1]),dtype=complex)
                self.rhoks_full[sp] = np.zeros((self.n_k,self.hopping.shape[2],self.hopping.shape[3]),dtype=complex)
                ind = self.spin_names_to_ind[
                            self.corr_shells[icrsh]['SO']][sp]
                for ik in mpi.slice_array(ikarray):
                    #print('ik=', ik, 'isp=', isp, 'sp=', sp, self.spin_names_to_ind[self.SO][sp])
                    #print(self.hopping[ik,isp,:,:])
                    R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
                    n_orb = self.n_orbitals[ik, ind]
                    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                    u = self.u_total[0,ik,:n_orb,:n_orb]
                    MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    # rotate to orbital basis
                    MMat = np.dot(np.dot(u, MMat), u.conj().T)
                    self.rhoks_full[sp][ik,:,:] = calc_nf(np.dot(R_full, np.dot(MMat, R_full.conj().T ) )
                                                         + Lambda_full - self.chemical_potential*np.eye(n_orb) , T ).T
                    # TODO: the line below only works for normal GA. We need to think about how to
                    # construct projmat that project out the correlated quasiparticle space.
                    rhoks_bloch = np.dot(np.dot(u.conj().T,self.rhoks_full[sp][ik,:,:]), u)
                    self.rhoks[icrsh][sp][ik,:,:] = np.dot(np.dot(projmat, rhoks_bloch), projmat.conjugate().transpose())

       # mpi reduce:
        for ik in range(self.n_k):
            for sp, isp in self.spin_names_to_ind[self.SO].items():
                for icrsh in range(self.n_corr_shells):
                    self.rhoks[icrsh][sp][ik,:,:] = mpi.all_reduce(self.rhoks[icrsh][sp][ik,:,:])
                self.rhoks_full[sp][ik,:,:] = mpi.all_reduce(self.rhoks_full[sp][ik,:,:])


    def calc_Delta(self):
        '''
        The first k-summation for quasiparticle density matrix. NOTE: doesn't work for ghostGA yet.
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

        # mpi reduce:
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for icrsh in range(self.n_corr_shells):
                self.Delta[icrsh][sp][:,:] = mpi.all_reduce(self.Delta[icrsh][sp][:,:])           

    def calc_D(self, R, Lambda):
        '''
        The second k-sum for kinetic energy. NOTE: doesn't work fo ghostGA yet.
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
                    R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
                    n_orb = self.n_orbitals[ik, ind]
                    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
                    # u_total wannier90 transformation matrix from bloch to orbital with index [orbital, bloch]
                    u = self.u_total[0,ik,:n_orb,:n_orb]
                    MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
                    # rotate to orbital basis
                    MMat = np.dot(np.dot(u, MMat), u.conj().T)
                    #MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
                    #sum_ek_Rdagger_rhoks[:,:] += self.bz_weights[ik]*MMatproj_nloc.dot(R[icrsh][sp].conj().T).dot(self.rhoks[icrsh][sp][ik,:,:].T)
                    # TODO: Below line only works for RISB where the correlated quasiparticle part has # the same size as the correlated physical part. We need to take care of the second
                    # projmat acting on the right of rhoks_full, when we added ghost orbitals.
                    tmp = self.bz_weights[ik]*np.dot(np.dot(np.dot(np.dot(np.dot( np.dot(projmat, u.conj().T), MMat), 
                                                     R_full.conj().T) , self.rhoks_full[sp][ik,:,:].T), u), projmat.conj().T ) 
                    sum_ek_Rdagger_rhoks[:,:] += tmp
                sqrt_Delta=funcMat(self.Delta[icrsh][sp], denR)
                self.D[icrsh][sp] = sum_ek_Rdagger_rhoks.dot(np.transpose(sqrt_Delta))
        # mpi reduce:
        for sp, isp in self.spin_names_to_ind[self.SO].items():
            for icrsh in range(self.n_corr_shells):
                self.D[icrsh][sp][:,:] = mpi.all_reduce(self.D[icrsh][sp])  

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
            R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, sp)
            u = self.u_total[0,ik,:n_orb,:n_orb]
            MMat = self.hopping_nloc[ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
            # rotate to orbital basis
            MMat = np.dot(np.dot(u, MMat), u.conj().T)
            
            if isinstance(mesh, MeshImFreq):
                gf.data[:, :, :] = (idmat[ibl] * (mesh_values[:, None, None] + mu) #+ self.h_field*(1-2*ibl))
                                    - np.dot(R_full, np.dot(Mmat, R_full.conj().T ) ) 
                                    - Lambda_full ) 
            else:
                gf.data[:, :, :] = (idmat[ibl] *
                                    (mesh_values[:, None, None] + mu + 1j*broadening)# + self.h_field*(1-2*ibl)
                                    - np.dot(R_full, np.dot(MMat, R_full.conj().T ) ) 
                                    - Lambda_full ) 
            #for icrsh in range(self.n_corr_shells):
            #    dim = self.corr_shells[icrsh]['dim']
            #    MMat = self.hopping[
            #                    ik, ind, 0:n_orb, 0:n_orb] #- (1 - 2 * isp) * self.h_field * MMat
            #    projmat = self.proj_mat[ik, ind, icrsh, 0:dim, 0:n_orb]
            #    #MMatproj_nloc = np.dot(np.dot(projmat, MMat), projmat.conjugate().transpose()) - self.Hsumk[icrsh][sp]
            #    if isinstance(mesh, MeshImFreq):
            #        gf.data[:, :, :] = (idmat[ibl] * (mesh_values[:, None, None] + mu) #+ self.h_field*(1-2*ibl))
            #                            - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) ) 
            #                            - Lambda[icrsh][sp] ) 
            #    else:
            #        gf.data[:, :, :] = (idmat[ibl] *
            #                            (mesh_values[:, None, None] + mu + 1j*broadening)# + self.h_field*(1-2*ibl)
            #                            - np.dot(R[icrsh][sp], np.dot(MMatproj_nloc, R[icrsh][sp].conj().T ) ) 
            #                            - Lambda[icrsh][sp] ) 

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
        #G_loc = [self.block_structure.create_gf(ish=ish, mesh=mesh, space='sumk') for ish in range(self.n_corr_shells)]
        #print(G_loc[0]['up'].data.shape)
        # create G_loc in the Wannierized space including correlated and non-correlated orbitals
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        # n_orbital doesn't depend on ik when using wannier90
        block_structure = [list(range(self.n_orbitals[0, ntoi[sp]])) for sp in spn]
        gf_struct = [(spn[isp], block_structure[isp])
                     for isp in range(self.n_spin_blocks[self.SO])]
        block_ind_list = [block for block, inner in gf_struct]
        if isinstance(mesh, MeshImFreq):
            glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                             for block, inner in gf_struct]
        else:
            glist = lambda: [Gf(mesh=mesh, target_shape=[len(inner),len(inner)])
                             for block, inner in gf_struct]
        G_loc = BlockGf(name_list=block_ind_list,
                            block_list=glist(), make_copies=False)
        G_loc.zero()


        ikarray = np.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            if isinstance(mesh, MeshImFreq):
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu)
            else:
                G_latt_qp = self.lattice_gf_qp( R, Lambda, ik=ik, mu=mu, broadening=broadening, mesh=mesh)
            G_latt_qp *= self.bz_weights[ik]

            #n_orb = self.n_orbitals[ik, ind]
            for bname, gf in G_loc:
                #print('bname=',bname)
                #print(G_latt_qp[bname].data.shape)
                ind = ntoi[bname]
                R_full, Lambda_full = self.calc_R_Lambda_full(R, Lambda, ik, ind, bname)
                #sp = spn[ibl]
                gf.data[:,:,:] += np.einsum('ij,kjl,lm->kim', R_full.conj().T, G_latt_qp[bname].data, R_full)
            #for icrsh in range(self.n_corr_shells):
            #    # init temporary storage
            #    for bname, gf in G_loc[icrsh]:
            #        #print(G_latt_qp[bname].data.shape)
            #        gf.data[:,:,:] += np.einsum('ij,kjl,lm->kim', R[icrsh][bname].conj().T, G_latt_qp[bname].data, R[icrsh][bname])

        # Collect data from mpi
        G_loc << mpi.all_reduce(G_loc)
        #for icrsh in range(self.n_corr_shells):
        #    G_loc[icrsh] << mpi.all_reduce(G_loc[icrsh])
        mpi.barrier()

        return G_loc

