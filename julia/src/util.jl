function check_convergence(E::Number,Cuu::AbstractMatrix,Cdd::AbstractMatrix,Eold::Number,Cuuold::AbstractMatrix,Cddold::AbstractMatrix,tolerances)
    Etol=tolerances[:E]
    rhotol=tolerances[:rho]
    converged=false
    if maximum(abs.(Cuuold.-Cuu))<=rhotol && maximum(abs.(Cddold.-Cdd))<=rhotol && abs((E-Eold)/E)<=Etol
        converged=true
    end
    return converged
end

function  convert_schedule(schedule)::Vector{NamedTuple}
    param_vec=NamedTuple[]
    
    for apair in schedule
        keys,values=apair
         
        #pyconvert(Vector{Any},values)
        @show typeof(values)
        @show typeof(keys)
        
        push!(param_vec,namedtuple(keys,values))
    end
    @show param_vec
    return param_vec
end

function get_perm(Nimp,Nbath,es;mu=0.0)
    normal=1:(Nimp+Nbath)
    impnormal=1:Nimp
    bathnormal=Nimp+1:(Nimp+Nbath)
    left=bathnormal[es .< mu]
    right=bathnormal[es .>= mu]
    return sortperm(vcat(left,impnormal,right))
end

function get_perm_bybathabs(Nimp,Nbath,es;mu=0.0)
    return vcat(1:Nimp,sortperm(abs(es)) .+Nimp)
end

function compute_commutator(A,B)
    Bp=prime(siteinds,B)
    Ap=prime(siteinds,A)
    return norm(A*Bp - B*Ap)
end



function symmetrize(A)
    n=length(size(A))
    A .+= conj.(permutedims(A,n:-1:1))
    return A
end
