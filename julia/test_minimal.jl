using MKL
using Random
using PythonCall
triqsutils=pyimport("triqs.operators.util")

using GGMPSSolver
using ITensors
using LinearAlgebra

let
    ITensors.Strided.disable_threads()
    BLAS.set_num_threads(4)
    @show BLAS.get_num_threads()
    @show Threads.nthreads()
    @show ITensors.blas_get_num_threads()
    ITensors.enable_threaded_blocksparse()
    Nimp=3
    Nbath=3*Nimp
    N=Nimp+Nbath
    U=3.0
    J=0.4
    W=3.0
    Utensor=triqsutils.U_matrix_slater(1,U_int=U,J_hund=J)#,full_Uijkl=true)
    #Utensor=triqsutils.U_matrix_kanamori(Nimp,U_int=U,J_hund=J,full_Uijkl=true)
    #Umatrix,Upmatrix,=triqsutils.U_matrix_kanamori(3,U_int=U,J_hund=J)
    #Utensor.=0.0
    Random.seed!(1234)


    J=GGMPSSolver.symmetrize(rand(Nimp,Nimp))	#symmetrize is from src/util.jl
    #J.=0.0
    U=Utensor
    
    Gamma=diagm(sort( 2*W *(rand(Nbath).-0.5))) #start with diagonal bath
    if iseven(length(Gamma))
        Gamma=sort(rand(div(Nbath,2))*W)
        Gamma=diagm(vcat(-Gamma[end:-1:1],Gamma))
    else
        Gamma=sort(rand(Int(floor(Nbath/2)))*W)
        Gamma=diagm(vcat(-Gamma[end:-1:1],[0.0,],Gamma))
    end
    #@show diag(Gamma)
    Ds=rand(Nimp,Nbath)
    ##perm: first bath sites with E<0, then imp, then bath sites with E>0
    perm=GGMPSSolver.get_perm(Nimp,Nbath,diag(Gamma);mu=0.0)
    H1E_spinless=zeros(Float64,(N,N))
    H1E_spinless[1:Nimp,1:Nimp].=J
    H1E_spinless[1:Nimp,Nimp+1:end].=Ds
    H1E_spinless[Nimp+1:end,1:Nimp].=transpose(conj.(Ds))
    H1E_spinless[Nimp+1:end,Nimp+1:end].=Gamma
    H1E=Dict("up"=>H1E_spinless,"dn"=>H1E_spinless)
    
    #perm=collect(1:N)
    param_names=["nsweeps","maxdim","cutoff","noise","outputlevel"]
    schedule=Vector{Pair{Vector{String},Tuple}}()
    push!(schedule,param_names=>(15,32,1e-10,1e-5,1))
    push!(schedule,param_names=>(15,64,1e-10,1e-5,1))
    push!(schedule,param_names=>(15,128,1e-12,1e-6,1))
    push!(schedule,param_names=>(10,256,1e-12,1e-6,1))
    push!(schedule,param_names=>(10,512,1e-12,1e-7,1))
    push!(schedule,param_names=>(10,1024,1e-12,1e-8,1))
    push!(schedule,param_names=>(10,2048,1e-14,1e-9,1))
    push!(schedule,param_names=>(5,4096,1e-14,1e-10,1))
    push!(schedule,param_names=>(3,4096,1e-14,0.0,1))
    tolerances=Vector{Pair{Vector{String},Tuple}}()
    push!(tolerances,["E","rho"]=>(1e-5,5e-3))
    kwargs=Vector{Pair{Vector{String},Tuple}}()
    push!(kwargs,["use_Sz","use_Ntot","spin_pen"]=>(true,true,1.0))
    is_converged,Eimp,Gamma_up,Gamma_dn=solve(Utensor,
	   H1E,
	   schedule,
	   tolerances,
	   kwargs)
    @show is_converged
end

#compute expectation values
##probably easiest to just do correlation matrix
##also setup MPO for Himp, so we can calculate <Himp>

            

