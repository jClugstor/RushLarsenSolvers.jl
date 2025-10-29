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
    # non_gating_u0 = view(u0, rl_f.non_gating_idxs)
    non_gating_u0 = u0
    non_gating_prob = ODEProblem(rl_f.non_gating_f, non_gating_u0, (t0, t0 + dt), p)
    inner_integrator = DiffEqBase.init(non_gating_prob, alg.inner_alg, dt=dt)

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

@inline @muladd function DiffEqBase.step!(integ::GeneralizedRushLarsenIntegrator{true, S, T}) where {T,S}
    integ.uprev .= integ.u
    integ.prev_gating_vars .= integ.gating_vars

    @unpack tmp, f, p, t, dt, uprev, u, prev_gating_vars, gating_vars, inner_integrator = integ

    if f isa RushLarsenFunction
        rl_f = f
    else
        rl_f = f.f
    end

    # Update gating variables using Rush-Larsen method
    rl_f.gating_f(gating_vars, u, p, t)

    for (i, k) in enumerate(rl_f.gating_idxs)
        alpha_i = gating_vars[i][1]
        beta_i = gating_vars[i][2]
        # Rush-Larsen formula: u_new = u_inf + (u_old - u_inf) * exp(-dt/tau)
        # where u_inf = alpha/(alpha+beta) and tau = 1/(alpha+beta)
        if alpha_i + beta_i < 1e-14
            # Handle special case to avoid division by zero
            u[k] = uprev[k]
        else
            u_inf = alpha_i / (alpha_i + beta_i)
            tau = 1 / (alpha_i + beta_i)
            u[k] = u_inf + (uprev[k] - u_inf) * exp(-dt / tau)
        end
    end

    # Update non-gating variables using inner integrator
    # Update the inner integrator's state with current values
    inner_integrator.u[rl_f.gating_idxs] .= u[rl_f.gating_idxs]
    inner_integrator.t = t
    inner_integrator.tprev = integ.tprev

    # Step the inner integrator
    DiffEqBase.step!(inner_integrator)

    # Copy results back to main state
    u[rl_f.non_gating_idxs] .= inner_integrator.u[rl_f.non_gating_idxs]

    integ.tprev = t
    integ.t += dt

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::GeneralizedRushLarsenIntegrator{false,S,T}) where {T,S}
    integ.uprev = integ.u
    @unpack tmp, f, p, t, dt, uprev, u, inner_integrator = integ

    if f isa RushLarsenFunction
        rl_f = f
    else
        rl_f = f.f
    end

    gating_idxs = rl_f.gating_idxs
    non_gating_idxs = rl_f.non_gating_idxs

    # Update gating variables using Rush-Larsen method
    gating_vars = rl_f.gating_f(uprev, p, t)

    for (i, k) in enumerate(gating_idxs)
        alpha_i = gating_vars[i][1]
        beta_i = gating_vars[i][2]
        # Rush-Larsen formula: u_new = u_inf + (u_old - u_inf) * exp(-dt/tau)
        # where u_inf = alpha/(alpha+beta) and tau = 1/(alpha+beta)
        if alpha_i + beta_i < 1e-14
            # Handle special case to avoid division by zero
            u = setindex(u, uprev[k], k)
        else
            u_inf = alpha_i / (alpha_i + beta_i)
            tau = 1 / (alpha_i + beta_i)
            u = setindex(u, u_inf + (uprev[k] - u_inf) * exp(-dt / tau), k)
        end
    end

    # Update non-gating variables using inner integrator
    # Update the inner integrator's state with current values
    inner_integrator.u = view(u, non_gating_idxs)
    inner_integrator.uprev = view(uprev, non_gating_idxs)
    inner_integrator.t = t
    inner_integrator.tprev = integ.tprev

    # Step the inner integrator
    DiffEqBase.step!(inner_integrator)

    # Copy results back to main state
    for (i, k) in enumerate(non_gating_idxs)
        u = setindex(u, inner_integrator.u[i], k)
    end

    integ.tprev = t
    integ.t += dt

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::GeneralizedRushLarsen;
    dt=error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = generalized_rushlarsen_init(DiffEqBase.unwrapped_f(prob.f), DiffEqBase.isinplace(prob), prob.u0,
        prob.tspan[1], dt, prob.p, alg)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error=false)

    return sol
end