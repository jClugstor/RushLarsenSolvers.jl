module RushLarsenSolvers

    using Reexport, MuladdMacro
    @reexport using DiffEqBase
    using LinearAlgebra
    using Parameters
    using ConcreteStructs
    using OrdinaryDiffEq
    using BlockDiagonals
    using ExponentialUtilities

    abstract type AbstractRushLarsenAlgorithm <: SciMLBase.AbstractODEAlgorithm end

    include("simple.jl")
    include("generalized_rl_functions.jl")
    include("exponential_utils.jl")
    include("blockdiagonal_mul.jl")
    include("split_ode_form.jl")
    include("algs/RL1.jl")
    include("algs/RL2.jl")
    include("algs/RL3.jl")
    include("algs/RL4.jl")

    # Fallback methods - return false for non-symbolic types
    is_gating_variable(::Any) = false
    is_tau_variable(::Any) = false
    is_alphabeta_variable(::Any) = false
    is_markov_gate(::Any) = false
    get_markov_chain_id(::Any) = nothing

    # Define abstract type or function stub for extension
    # The concrete type RushLarsenSystem is defined in the ModelingToolkit extension
    function RushLarsenSystem end
    function RushLarsenProblem end

    export RushLarsen, RushLarsenFunction
    export GRLFunction
    export RL1, RL2, RL3, RL4
    export phi, rl_phi, rl_phi!
    export split_ode_problem, split_ode_problem_iip, matrix_operator
    export is_gating_variable, is_tau_variable, is_alphabeta_variable, is_markov_gate
    export get_markov_chain_id
    export RushLarsenSystem, RushLarsenProblem

end
