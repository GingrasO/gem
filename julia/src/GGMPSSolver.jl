module GGMPSSolver

    using PythonCall
    using MKL
    using ITensors
    using NamedTupleTools
    using Random
    using LinearAlgebra
    using Observers
    using ITensors.HDF5
    using DataFrames

    ##backend functionality
    include("model.jl")
    include("util.jl")
    include("observer.jl")
    ##interface functionality
    include("driver.jl")

    export
        #from driver.jl
        solve
end
