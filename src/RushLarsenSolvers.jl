module RushLarsenSolvers

    using Reexport, MuladdMacro
    @reexport using DiffEqBase
    using LinearAlgebra
    using Parameters
    using ConcreteStructs
    using OrdinaryDiffEq

    abstract type AbstractRushLarsenAlgorithm <: SciMLBase.AbstractODEAlgorithm end

    include("simple.jl")

    is_gating_variable(var) = false

    export RushLarsen, RushLarsenFunction
    export GeneralizedRushLarsen

end
