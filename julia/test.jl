using ITensors
using PyCall
using PyPlot
using LinearAlgebra
using MKL
using Random
#matplotlib.use("QtAgg")
include("src/model.jl")
include("src/util.jl")
triqsutils=pyimport("triqs.operators.util")
atomdiag=pyimport("triqs.atom_diag")

let
    ITensors.Strided.disable_threads()
    BLAS.set_num_threads(4)
    @show Threads.nthreads()
    @show ITensors.blas_get_num_threads()
    ITensors.enable_threaded_blocksparse()
    Nimp=5
    Nbath=3*Nimp
    N=Nimp+Nbath
    U=3.0
    J=0.4

    Utensor=triqsutils.U_matrix_slater(2,U_int=U,J_hund=J)#,full_Uijkl=true)
    #Utensor=triqsutils.U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=true)
    #Umatrix,Upmatrix,=triqsutils.U_matrix_kanamori(3,U_int=U,J_hund=J)
    #n_orb=Nimp
    #S2triqs=triqsutils.observables.S2_op(("up","dn"),n_orb,off_diag=true)
    #S2triqs=triqsutils.hamiltonians.make_operator_real(S2triqs)
    #Hkanamori_triqs=triqsutils.hamiltonians.h_int_kanamori(("up","dn"), n_orb, Umatrix, Upmatrix, J, off_diag=true)
    #Hkanamori_triqs=atomdiag.AtomDiag(Hkanamori_triqs, ("up","dn"))#("up_0","dn_0","up_1","dn_1","up_2","dn_2"))
    
    #@show atomdiag.atom_diag.quantum_number_eigenvalues(S2triqs, Hkanamori_triqs)
    #Utensor=permutedims(Utensor, (3, 2, 1, 4)) ###copy paste from ITensorChemistry (Conversion from PySCF)
    #@show reshape(Utensor, 5^4)
    #Utensor.=0.0
    #@show Utensor
    Random.seed!(1234)


    sites_imp=siteinds("Electron", Nimp; conserve_qns=true)
    sites_bath=siteinds("Electron", Nbath;conserve_qns=true)
    sites=vcat(sites_imp,sites_bath)
    J=symmetrize(rand(Nimp,Nimp))
    #J.=0.0
    U=Utensor
    W=0.5
    Gamma=diagm(sort( 2*W *(rand(Nbath).-0.5))) #start with diagonal bath
    if iseven(length(Gamma))
        Gamma=sort(rand(div(Nbath,2))*W)
        Gamma=diagm(vcat(-Gamma[end:-1:1],Gamma))
    else
        Gamma=sort(rand(Int(floor(Nbath/2)))*W)
        Gamma=diagm(vcat(-Gamma[end:-1:1],[0.0,],Gamma))
    end
    @show diag(Gamma)
        #Gamma=
    Ds=rand(Nimp,Nbath)
    ##perm: first bath sites with E<0, then imp, then bath sites with E>0
    perm=get_perm(Nimp,Nbath,diag(Gamma);mu=0.0)
    #perm=collect(1:N)
    #os_imp=get_H_imp(N,J,U;perm=perm)
    os_imp=get_H_imp(N,J,U;perm=perm)
    
    os_bath=get_H_bath(N,Gamma,bathoffset=Nimp;perm=perm)
    os_hyb=get_H_hyb(N,Ds;perm=perm)
    S2=get_Ssquared(N)
    
    Himp=MPO(os_imp,sites,)
    #Himp_imp=MPO(os_imp,sites_imp,)
    #@show Matrix(mapreduce(*,Himp),sites_imp
    
    Hbath=MPO(os_bath,sites,)
    Hhyb=MPO(os_hyb,sites,)
    
    @show norm(Himp), norm(Hbath), norm(Hhyb)
    H=MPO(os_imp+os_bath+os_hyb,sites)
    @show norm(H)
    alpha=1e0
    S2m=MPO(S2,sites)
    
    @show maxlinkdim(S2m)
    Hpen=MPO(os_imp+os_bath+os_hyb+alpha*S2,sites)
    @show compute_commutator(Himp,S2m)
    
    H=Hpen

    
    @show maxlinkdim(H)
    @show maxlinkdim(Hpen)
    
    @show maxlinkdim(Himp)
    H=Hpen

    #optionally diagonalize bath and transform hybridization accordingly
    #think about best arrangement for star-geometry bath

    #get pieces of Hamiltonian and assemble Hamiltonian

    #generate initial state for DMRG (potentially with GMPS and setting quartic terms to zero)
    ##think about correct sector
    psi=MPS(sites,x -> isodd(x) ? "Up" : "Dn")
    psi=psi+MPS(sites,x -> isodd(x) ? "Up" : "Dn")
    

    #psi=MPS(sites,x -> isodd(x) ? "Up" : "Dn")

    #parameter schedule for DMRG
    #Sweeps(10,32,1e-12,1e-5,50)
    sw1=Sweeps(10;maxdim=32,cutoff=1e-12,mindim=1,noise=1e-5)
    sw2=Sweeps(10;maxdim=64,cutoff=1e-12,mindim=1,noise=1e-6)
    
    sws=[sw1,sw2]
    Ddmrgs=[32,64,128,256,512,1024,2048,4096,8192]
    Nsweeps=[20,20,10,10,10,10,5,5,5]
    cutoffs=[1e-12,1e-12,1e-12,1e-12,1e-14,1e-14,1e-14,1e-14,1e-14]
    noise=[1e-5,1e-6,1e-6,1e-7,1e-8,1e-8,1e-9,0.0,0.0]
    eigsolve_krylovdim=[50,50,50,30,20,10,10,5,5]
    eigsolve_maxiter=[2,2,2,1,1,1,1,1,1,1]
    oldCuu=nothing
    oldCdd=nothing
    Eold=nothing
    for (i,D) in enumerate(Ddmrgs)
        dmrg_kwargs = (nsweeps=Nsweeps[i], reverse_step=false, normalize=true, maxdim=D, cutoff=cutoffs[i], noise=noise[i], outputlevel=1, nsites = 2,)
        #E,psi=dmrg(H,psi; dmrg_kwargs...)
        E,psi=dmrg(H,psi,sws[i]; nsites=2,reverse_step=false,normalize=true)
        @show inner(psi',S2m,psi)
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
    return False, E,Cuu,Cdd
end

#compute expectation values
##probably easiest to just do correlation matrix
##also setup MPO for Himp, so we can calculate <Himp>

            

