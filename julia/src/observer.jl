mutable struct MyDMRGObserver <: AbstractObserver
    nsweep_total::Int
    the_observer::DataFrame  #OBSERVER?
    perm
    MyDMRGObserver(an_observer) = new(0,an_observer,nothing)
    MyDMRGObserver(an_observer,a_perm) = new(0,an_observer,a_perm)
    MyDMRGObserver(nsweep_initial,an_observer,a_perm) = new(nsweep_initial,an_observer,a_perm)
end

ITensors.checkdone!(o::MyDMRGObserver;kwargs...) = false



function ITensors.measure!(o::MyDMRGObserver; kwargs...)
    energy = kwargs[:energy]
    sweep = kwargs[:sweep]
    bond = kwargs[:bond]
    outputlevel = kwargs[:outputlevel]
    psi = kwargs[:psi]
    bond = kwargs[:bond]
    half_sweep = kwargs[:half_sweep]
    
    if bond==1 && half_sweep==2
        GC.gc()
        o.nsweep_total+=1
        update!(o.the_observer;nsweep=o.nsweep_total,perm=o.perm,psi=psi,energy=energy)
    end 
    #if outputlevel > 0
    #  println("Sweep $sweep at bond $bond, the energy is $energy")
    #end
    end
 
function get_corr_up(;psi,perm)
if !isnothing(perm)
    Cuu=correlation_matrix(psi, "Cdagup", "Cup")[perm,perm]
else
    Cuu=correlation_matrix(psi, "Cdagup", "Cup")
end
return Cuu
end

function get_corr_dn(;psi,perm)
if !isnothing(perm)
    Cdd=correlation_matrix(psi, "Cdagdn", "Cdn")[perm,perm]
else
    Cdd=correlation_matrix(psi, "Cdagdn", "Cdn")
end
return Cdd
end

function get_corr_dn(;psi,perm)
if !isnothing(perm)
    Cdd=correlation_matrix(psi, "Cdagdn", "Cdn")[perm,perm]
else
    Cdd=correlation_matrix(psi, "Cdagdn", "Cdn")
end
return Cdd
end

function get_energy(;energy)
    return energy
end

function get_maxdim(;psi)
    #@show maxlinkdim(psi)
    return maxlinkdim(psi)
end

function get_total_sweep(;nsweep)
    #@show nsweep
    return nsweep
end

function savedata(name::String, obs)
    name == "" && return
    h5open(name*".h5", "w") do file
      # iterate through the fields of obs and append the data to the dataframe
      for n in names(obs)
        create_group(file, n)
        for (i,data) in enumerate(obs[:,n])
          file[n][string(i)] = [data[i] for i in 1:length(data)]
        end
      end
    end
  end
  