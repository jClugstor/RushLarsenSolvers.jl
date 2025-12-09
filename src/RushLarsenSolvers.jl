module RushLarsenSolvers

    using Reexport, MuladdMacro
    @reexport using DiffEqBase
    using LinearAlgebra
    using Parameters
    using ConcreteStructs
    using OrdinaryDiffEq

    abstract type AbstractRushLarsenAlgorithm <: SciMLBase.AbstractODEAlgorithm end

    include("simple.jl")

    # Fallback methods - return false for non-symbolic types
    is_gating_variable(::Any) = false
    is_tau_variable(::Any) = false
    is_alphabeta_variable(::Any) = false

    # Define abstract type or function stub for extension
    # The concrete type RushLarsenSystem is defined in the ModelingToolkit extension
    function RushLarsenSystem end
    function RushLarsenProblem end

    export RushLarsen, RushLarsenFunction
    export GeneralizedRushLarsen
    export is_gating_variable, is_tau_variable, is_alphabeta_variable
    export RushLarsenSystem, RushLarsenProblem

end
