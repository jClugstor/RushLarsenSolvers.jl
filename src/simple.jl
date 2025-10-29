struct RushLarsen<:AbstractRushLarsenAlgorithm end

mutable struct RushLarsenIntegrator{IIP, S, T, P, F, G} <: DiffEqBase.AbstractODEIntegrator{RushLarsen, IIP, S, T}
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
end

DiffEqBase.isinplace(::RushLarsenIntegrator{IIP}) where {IIP} = IIP

function DiffEqBase.__init(prob::ODEProblem, alg::RushLarsen; dt = error("dt is required for this algorithm"))
    rushlarsen_init(DiffEqBase.unwrapped_f(prob.f), DiffEqBase.isinplace(prob),
    prob.u0,
    prob.tspan[1],
    dt,
    prob.p)
end

@inline function rushlarsen_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
    p::P) where
{F,P,T,S}

    if f isa RushLarsenFunction
        rl_f = f
    else
        rl_f = f.f
    end
    gating_vars_prototype = fill(ntuple(_ -> zero(eltype(u0)),2), length(rl_f.gating_idxs))
    integ = RushLarsenIntegrator{IIP,S,T,P,F,typeof(gating_vars_prototype)}(f,
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
        true)

    return integ
end

@inline @muladd function DiffEqBase.step!(integ::RushLarsenIntegrator{true, S, T}) where {T,S}
    integ.uprev .= integ.u
    integ.prev_gating_vars .= integ.gating_vars

    @unpack tmp, f, p, t, dt, uprev, u, prev_gating_vars, gating_vars = integ

    if f isa RushLarsenFunction
        rl_f = f
    else
        rl_f = f.f
    end

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

    tmp .= u

    # Use Euler for non-gating equations
    #Evaluate non gating variables with updated gating values
    rl_f.non_gating_f(u, tmp, p, t)
    for (i, k) in enumerate(rl_f.non_gating_idxs)
        u[k] = uprev[k] + dt * u[k]
    end
    integ.tprev = t
    integ.t += dt 
    
    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::RushLarsenIntegrator{false,S,T}) where {T,S}
    integ.uprev = integ.u
    @unpack tmp, f, p, t, dt, uprev, u = integ

    if f isa ODEFunction
        f = f.f
    end

    gating_idxs = f.gating_idxs
    non_gating_idxs = f.non_gating_idxs
    
    gating_vars = f.gating_f(uprev,p,t)

    for (i,k) in enumerate(gating_idxs)
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

    # Use Euler for non-gating equations
    # First need to do a function evaluation with updated gating variables
    dudt = f.non_gating_f(integ.u,p,t)

    for (i,k) in enumerate(non_gating_idxs)
        u[i] = uprev[i] + dt*dudt[k]
    end

    integ.tprev = t
    integ.t += dt

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::RushLarsen;
    dt=error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = rushlarsen_init(DiffEqBase.unwrapped_f(prob.f), DiffEqBase.isinplace(prob), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error=false)

    return sol
end

@concrete struct RushLarsenFunction
    gating_f
    non_gating_f

    gating_idxs
    non_gating_idxs
end

function RushLarsenFunction(gating_f, non_gating_f; gating_idxs = nothing, non_gating_idxs = nothing)
    RushLarsenFunction(gating_f, non_gating_f, gating_idxs, non_gating_idxs)
end

# For compatibility with other ODE solvers
# if gating_f returns a list of tuples of the alpha and beta values
# calling just "f" will calculate the derivatives with those 
function (f::RushLarsenFunction)(u,p,t)
    du = similar(u)
    gating_vars = f.gating_f(u,p,t)
    for (i,k) in enumerate(f.gating_idxs)
        # alpha, beta = gating_vars[i]
        du[k] = gating_var[i][1]*(1 - u[k]) - gating_vars[i][2]*u[k]
    end
    
    du[f.non_gating_idxs] .= f.non_gating_f(u,p,t)
    
    du
end

function (f::RushLarsenFunction)(du,u,p,t)

end