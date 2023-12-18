using MKL
using PyCall
using ITensors
using NamedTupleTools
using Random
using LinearAlgebra
include("model.jl")
include("util.jl")

"""
Computes the GS of impurity model using DMRG and returns correlation matrix and scalar <Himp>
"""
function compute_gs_corrmatrix(U::Number,J::Number,Jmat::AbstractMatrix,bath_energies::AbstractVector,hybs::AbstractMatrix)
    #setup sites

    #setup hamiltonian

    #setup initial state

    #
end

function setup_H(sites,U,J,Jmat,bath_energies,hybs)
    N=length(sites)
    Utensor=triqsutils.U_matrix(2,U_int=U,J_hund=J)
    Nimp=size(Utensor,1)
    Nbath=N-Nimp
    perm=get_perm(Nimp,Nbath,bath_energies;mu=0.0)
    os_imp=get_H_imp(N,J,U;perm=perm)
    os_bath=get_H_bath(N,diagm(bath_energies),bathoffset=Nimp;perm=perm)
    os_hyb=get_H_hyb(N,hybs;perm=perm)
    Himp=MPO(os_imp,sites,)
    H=MPO(os_imp+os_bath+os_hyb,sites)
    return H,Himp
end

###assumes:
###half filling, zero magnetization sector
###spin rotational symmetry
###
function solve(Utensor,H1E,schedule,tolerances,kwargs)
    # kwargs
    #   sweep schedule as a list of Dictionaries or zipped key value pairs
    #   flags: permute sites, diagonalize_bath, min_iters etc.
    #   conserve_qns=true
    #   
    dmrg_params=convert_schedule(schedule)
    kwargs=convert_schedule(kwargs)
    conserve_sz=get(kwargs, :use_Sz, True)
    conserve_N=get(kwargs, :use_Ntot, True)
    spin_pen=get(kwargs,:spin_pen,0.0)
    
    
    #extract the relevant quantities
    Nimp=size(Utensor,1)
    H1Eup=H1E["up"]
    H1Edn=H1E["dn"]
    N=size(H1Eup,1)
    Nbath=N-Nimp
    hbath_up=H1Eup[Nimp+1:end,Nimp+1:end]
    hbath_dn=H1Edn[Nimp+1:end,Nimp+1:end]
    hybs_up=H1Eup[1:Nimp,Nimp+1:end]
    hybs_dn=H1Edn[1:Nimp,Nimp+1:end]
    hloc_1e_up=H1Eup[1:Nimp,1:Nimp]
    hloc_1e_dn=H1Edn[1:Nimp,1:Nimp]
    
    
    ###diag and determine perm?
    perm=collect(1:N)
    ##ToDo: Implement permutation either on julia side or python side ...
    ##If implemented on python side, generalize the Hamiltonian constructors s.t. they accept siteinds for the impurity etc.
    #create observables and Hamiltonians
    os_bath=get_H_bath(N,hbath_up,hbath_dn;bathoffset=Nimp,perm=perm)
    os_hyb=get_H_hyb(N,hybs_up,hybs_dn;perm=perm)
    os_imp=get_H_imp(N,hloc_1e_up,hloc_1e_dn,Utensor;perm=perm)
    os_S2=get_Ssquared(N)
       
    #make sites
    sites=siteinds("Electron", N; conserve_nf=conserve_nf,conserve_sz=conserve_sz))
    
    S2=MPO(os_S2,sites)
    if !iszero(spin_pen)
        H=MPO(os_imp+os_bath+os_hyb+spin_pen*os_S2,sites)
    else
        MPO(os_imp+os_bath+os_hyb,sites)
    end
    #TODO: verify whether <Eint> (quartic only) or <Eimp> to be returned
    Himp=MPO(os_imp,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
    @assert compute_commutator(H,S2m)<1e-3
    #make starting MPS
    ##potentially trigger different behaviour via kwarg
    ##assumes that the total system size is even, otherwise not half filled and zero mag
    @assert iseven(length(sites))
    psi=MPS(sites,x -> isodd(x) ? "Up" : "Dn")
    psi=psi+MPS(sites,x -> isodd(x) ? "Dn" : "Up")
    oldCuu=nothing
    oldCdd=nothing
    Eold=nothing
    #run dmrg loop, terminate when tolerances are satisfied
    for (iteration,pars) in enumerate(dmrg_params)
        #we should be passing all these
        #dmrg_kwargs = (nsweeps=Nsweeps[i], reverse_step=false, normalize=true, maxdim=D, cutoff=cutoffs[i], noise=noise[i], outputlevel=1, nsites = 2,)
        E,psi=dmrg(H,psi; pars...)
        Eimp=inner(psi',Himp,psi)
        Cuu = correlation_matrix(psi, "Cdagup", "Cup")[perm,perm]
        Cdd = correlation_matrix(psi, "Cdagdn", "Cdn")[perm,perm]
        converged=false
        if !isnothing(oldCuu)
            @show E,Eold
            @show Maximum(abs.(oldCuu .- Cuu))
            @show Maximum(abs.(oldCdd .- Cdd))            
            converged=check_convergence(E,Cuu,Cdd,Eold,oldCuu,oldCdd,tolerances)
        end
        if converged
            return True, E,Cuu,Cdd
        end
        oldCuu=deepcopy(Cuu)
        oldCdn=deepcopy(Cdn)
    end
    return False, Eimp, Cuu, Cdd
end
    #eventually implement logging via Observers, pass in an iteration id, so we can save separate HDF5 files for every iteration
    