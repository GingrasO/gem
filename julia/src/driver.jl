function solve(Utensor,H1E,schedule,tolerances,kwargs;outfile="data")
    # kwargs
    #   sweep schedule as a list of Dictionaries or zipped key value pairs
    #   flags: permute sites, diagonalize_bath, min_iters etc.
    #   conserve_qns=true
    #
    #outfile="data"
    @show outfile
    dmrg_params=GGMPSSolver.convert_schedule(schedule)
    kwargs=GGMPSSolver.convert_schedule(kwargs)[]
    tolerances=GGMPSSolver.convert_schedule(tolerances)[]
    conserve_sz=get(kwargs, :use_Sz, true)
    conserve_N=get(kwargs, :use_Ntot,true)
    spin_pen=get(kwargs,:spin_pen,0.0)
    #@show typeof(tolerances)
    #@show typeof(kwargs)
    #@show typeof(dmrg_params[1])
    #@show dmrg_params[1]
    
    #extract the relevant quantities

    Utensor=GGMPSSolver.PythonCall.pyconvert(Array,Utensor)
    Nimp=size(Utensor,1)
    
    H1Eup=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["up"])
    H1Edn=GGMPSSolver.PythonCall.pyconvert(Matrix,H1E["dn"])
    N=size(H1Eup,1)
    Nbath=N-Nimp
    ###diag and determine perm? 
    perm=collect(1:N)
    os_quadratic=GGMPSSolver.get_H_quadratic(N,H1Eup, H1Edn;perm=perm)
    os_quartic=GGMPSSolver.get_H_quartic(N,Utensor;perm=perm)
    os_S2=GGMPSSolver.get_Ssquared(N)
       
    #make sites
    sites=GGMPSSolver.ITensors.siteinds("Electron", N; conserve_nf=conserve_N,conserve_sz=conserve_sz)
    
    S2=MPO(os_S2,sites)
    if !iszero(spin_pen)
        H=GGMPSSolver.ITensors.MPO(os_quadratic + os_quartic + spin_pen*os_S2,sites)
    else
        H=GGMPSSolver.ITensors.MPO(os_quadratic + os_quartic,sites)
    end
    #TODO: verify whether <Eint> (quartic only) or <Eimp> to be returned
    Hint=GGMPSSolver.ITensors.MPO(os_quartic,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
    #Hint=GGMPSSolver.ITensors.MPO(os_int,sites)  ##for <Eimp>        ###FIXME: most likely we'll want to use only the quartic part here
    
    @assert GGMPSSolver.compute_commutator(H,S2)<1e-3
    #make starting MPS
    ##potentially trigger different behaviour via kwarg
    ##assumes that the total system size is even, otherwise not half filled and zero mag
    @assert iseven(length(sites))
    psi=GGMPSSolver.ITensors.MPS(sites,x -> isodd(x) ? "Up" : "Dn")
    psi=psi+GGMPSSolver.ITensors.MPS(sites,x -> isodd(x) ? "Dn" : "Up")
    oldCuu=nothing
    oldCdd=nothing
    Eold=nothing
    @show GGMPSSolver.ITensors.maxlinkdim(H)
    #run dmrg loop, terminate when tolerances are satisfied
    internal_obs = GGMPSSolver.Observers.Observer(
        "sweepnumber"=>get_total_sweep,
        "maxdim"=>get_maxdim,
        "energy"=>get_energy,
        "corr_dn"=>get_corr_dn,
        "corr_up"=>get_corr_up,
    )

    #@show internal_obs
    obs = GGMPSSolver.MyDMRGObserver(0,internal_obs,perm)
    #update!(obs.the_observer;nsweep=1,psi=psi)  #energy_tol,last_energy 
    #@show obs.the_observer
    for (iteration,pars) in enumerate(dmrg_params)
        #we should be passing all these
        #dmrg_kwargs = (nsweeps=Nsweeps[i], reverse_step=false, normalize=true, maxdim=D, cutoff=cutoffs[i], noise=noise[i], outputlevel=1, nsites = 2,)
        #@show typeof(H)
        E,psi=GGMPSSolver.ITensors.dmrg(H,psi; observer=obs,pars...)
        #@show obs
        savedata(outfile,obs.the_observer)
        GC.gc()
        Eint=GGMPSSolver.ITensors.inner(psi',Hint,psi)
        S2val=GGMPSSolver.ITensors.inner(psi',S2,psi)
        Cuu = GGMPSSolver.ITensors.correlation_matrix(psi, "Cdagup", "Cup")[perm,perm]
        Cdd = GGMPSSolver.ITensors.correlation_matrix(psi, "Cdagdn", "Cdn")[perm,perm]
        GC.gc()
        converged=false
        if !isnothing(oldCuu)
            @show E,Eold
            @show maximum(abs.(oldCuu .- Cuu))
            @show maximum(abs.(oldCdd .- Cdd))
            @show S2val            
            converged=GGMPSSolver.check_convergence(E,Cuu,Cdd,Eold,oldCuu,oldCdd,tolerances)
        end
        if converged
            return true, Eint,Cuu,Cdd
        end
        oldCuu=deepcopy(Cuu)
        oldCdd=deepcopy(Cdd)
        Eold=deepcopy(E)
    end
    return false, Eint, Cuu, Cdd
end
    #eventually implement logging via Observers, pass in an iteration id, so we can save separate HDF5 files for every iteration
    
