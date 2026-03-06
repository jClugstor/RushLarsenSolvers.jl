"""
    RL2(exponential_alg=nothing)

Second-order generalized Rush-Larsen algorithm for ODEs of the form:
    du/dt = a(u,p,t)*u + b(u,p,t)

Uses a multi-step formula with α_n and β_n computed from previous time steps.
For RL2:
    α_n = (3/2)*u_n - (1/2)*u_{n-1}
    β_n = (3/2)*b_n - (1/2)*b_{n-1}

# Arguments
- `exponential_alg`: ExponentialUtilities algorithm for matrix phi computation (optional)

# Example
```julia
alg = RL2()
```
"""
struct RL2 <: AbstractRushLarsenAlgorithm
    exponential_alg
end

RL2() = RL2(nothing)

mutable struct RL2Integrator{IIP, S, T, P, F, PhiType} <: DiffEqBase.AbstractODEIntegrator{RL2, IIP, S, T}
    f::F
    uprev::S
    u::S
    uprev2::S  # u_{n-1} for multi-step
    tmp::S
    phi_cache::PhiType
    bprev::S  # b_{n-1} for multi-step
    tprev::T
    t::T
    t0::T
    dt::T
    tdir::T
    p::P
    u_modified::Bool
    step_number::Int  # Track which step we're on
end

DiffEqBase.isinplace(::RL2Integrator{IIP}) where {IIP} = IIP

function DiffEqBase.__init(prob::ODEProblem, alg::RL2; dt = error("dt is required for this algorithm"))
    rl2_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

@inline function rl2_init(f::F, ::Val{IIP}, u0::S, t0::T, dt::T, p::P) where {F, P, T, S, IIP}
    grl_f = _get_grl_function(f)
    a_sample = grl_f.a(u0, p, t0)
    phi_cache = similar(a_sample)

    integ = RL2Integrator{IIP, S, T, P, F, typeof(phi_cache)}(
        f,
        copy(u0),
        copy(u0),
        copy(u0),  # uprev2
        copy(u0),  # tmp
        phi_cache,
        copy(u0),  # bprev
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true,
        0)  # step_number

    return integ
end

@inline @muladd function DiffEqBase.step!(integ::RL2Integrator{true, S, T, P, F}) where {T, S, P, F}
    @unpack tmp, phi_cache, f, p, t, dt, uprev, uprev2, bprev, u, step_number = integ

    grl_f = _get_grl_function(f)

    # Evaluate a_n and b_n at current state
    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL2)!")
            println("  Time: ", t, " ms")
            println("  Step number: ", step_number)
            println("  First NaN at a_n linear index: ", j)
            println("  a_n type: ", typeof(a_n))
            println("  a_n size: ", size(a_n))
            row, col = divrem(j-1, size(a_n, 1))
            row += 1
            col += 1
            println("  a_n[row=$row, col=$col] = NaN")
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in a_n at row $row, col $col (linear index $j) at time $t")
        end
    end

    # Check for NaNs in b_n
    for (j, val) in enumerate(b_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL2)!")
            println("  Time: ", t, " ms")
            println("  Step number: ", step_number)
            println("  First NaN at b_n element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in b_n at element $j at time $t")
        end
    end

    if step_number == 0
        # First step: use RL1 formula
        mul!(tmp, a_n, uprev)
        @. tmp = tmp + b_n

        rl_phi!(phi_cache, a_n, dt)
        mul!(u, phi_cache, tmp)
        @. u = uprev + dt * u
    else
        # Second step onwards: use RL2 formula
        # α_n = (3/2)*u_n - (1/2)*u_{n-1}
        # β_n = (3/2)*b_n - (1/2)*b_{n-1}

        # Compute α_n (store in tmp temporarily)
        @. tmp = 1.5 * uprev - 0.5 * uprev2

        # Compute a_n * α_n + β_n
        mul!(u, a_n, tmp)  # u = a_n * α_n
        @. u = u + 1.5 * b_n - 0.5 * bprev

        # Apply phi
        rl_phi!(phi_cache, a_n, dt)
        mul!(tmp, phi_cache, u)
        @. u = uprev + dt * tmp
    end

    # Update history
    integ.uprev2 .= uprev
    integ.uprev .= u
    integ.bprev .= b_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::RL2Integrator{false, S, T, P, F}) where {T, S, P, F}
    @unpack f, p, t, dt, uprev, uprev2, bprev, step_number = integ

    grl_f = _get_grl_function(f)

    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL2 out-of-place)!")
            println("  Time: ", t, " ms")
            println("  Step number: ", step_number)
            println("  First NaN at a_n linear index: ", j)
            println("  a_n type: ", typeof(a_n))
            println("  a_n size: ", size(a_n))
            row, col = divrem(j-1, size(a_n, 1))
            row += 1
            col += 1
            println("  a_n[row=$row, col=$col] = NaN")
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in a_n at row $row, col $col (linear index $j) at time $t")
        end
    end

    # Check for NaNs in b_n
    for (j, val) in enumerate(b_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL2 out-of-place)!")
            println("  Time: ", t, " ms")
            println("  Step number: ", step_number)
            println("  First NaN at b_n element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in b_n at element $j at time $t")
        end
    end

    if step_number == 0
        # First step: use RL1 formula
        rhs = a_n * uprev .+ b_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    else
        # Second step onwards: use RL2 formula
        α_n = 1.5 * uprev .- 0.5 * uprev2
        β_n = 1.5 * b_n .- 0.5 * bprev

        rhs = a_n * α_n .+ β_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    end

    # Update history
    integ.uprev2 = uprev
    integ.uprev = integ.u
    integ.bprev = b_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::RL2;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = rl2_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    return sol
end
