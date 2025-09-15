struct GeneralizedRushLarsen{A} <: AbstractRushLarsenAlgorithm where {I}
    inner_alg::A 
end

mutable struct GeneralizedRushLarsenIntegrator{IIP,S,T,P,F,G,I} <: DiffEqBase.AbstractODEIntegrator{RushLarsen,IIP,S,T}
    f::F
    uprev::S
    u::S
    tmp::S
    gating_vars::G
    prev_gating_vars::G
    tprev::T
    t::T
    t0::T
    dt::T
    tdir::T
    p::P
    u_modified::Bool
    inner_integrator::I
end

@inline function generalized_rushlarsen_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
    p::P, alg::GeneralizedRushLarsen) where
{F,P,T,S}

    if f isa RushLarsenFunction
        rl_f = f
    else
        rl_f = f.f
    end
    gating_vars_prototype = fill(ntuple(_ -> zero(eltype(u0)),2), length(rl_f.gating_idxs))

    # Create a subproblem for the non-gating variables to use with the inner algorithm
    non_gating_u0 = view(u0, rl_f.non_gating_idxs)
    non_gating_prob = ODEProblem(rl_f.non_gating_f, non_gating_u0, (t0, t0 + dt), p)
    inner_integrator = DiffEqBase.__init(non_gating_prob, alg.inner_alg, dt=dt)

    integ = GeneralizedRushLarsenIntegrator{IIP,S,T,P,F,typeof(gating_vars_prototype),typeof(inner_integrator)}(f,
        copy(u0),
        copy(u0),
        copy(u0),
        #gating_vars
        gating_vars_prototype,
        #prev_gating_vars
        gating_vars_prototype,
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true,
        inner_integrator)

    return integ
end