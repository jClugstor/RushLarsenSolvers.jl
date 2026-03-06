"""
    RL3(exponential_alg=nothing)

Third-order generalized Rush-Larsen algorithm for ODEs of the form:
    du/dt = a(u,p,t)*u + b(u,p,t)

Uses a multi-step formula with α_n and β_n computed from previous time steps.
For RL3:
    α_n = (1/12)*(23*u_n - 16*u_{n-1} + 5*u_{n-2})
    β_n = (1/12)*(23*b_n - 16*b_{n-1} + 5*b_{n-2}) + (1/12)*(a_n*b_{n-1} - a_{n-1}*b_n)

# Arguments
- `exponential_alg`: ExponentialUtilities algorithm for matrix phi computation (optional)

# Example
```julia
alg = RL3()
```
"""
struct RL3 <: AbstractRushLarsenAlgorithm
    exponential_alg
end

RL3() = RL3(nothing)

mutable struct RL3Integrator{IIP, S, T, P, F, PhiType, AType} <: DiffEqBase.AbstractODEIntegrator{RL3, IIP, S, T}
    f::F
    uprev::S
    u::S
    uprev2::S  # u_{n-1}
    uprev3::S  # u_{n-2}
    tmp::S
    phi_cache::PhiType
    bprev::S   # b_{n-1}
    bprev2::S  # b_{n-2}
    aprev::AType  # a_{n-1}
    tprev::T
    t::T
    t0::T
    dt::T
    tdir::T
    p::P
    u_modified::Bool
    step_number::Int
end

DiffEqBase.isinplace(::RL3Integrator{IIP}) where {IIP} = IIP

function DiffEqBase.__init(prob::ODEProblem, alg::RL3; dt = error("dt is required for this algorithm"))
    rl3_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

@inline function rl3_init(f::F, ::Val{IIP}, u0::S, t0::T, dt::T, p::P) where {F, P, T, S, IIP}
    grl_f = _get_grl_function(f)
    a_sample = grl_f.a(u0, p, t0)
    phi_cache = similar(a_sample)

    integ = RL3Integrator{IIP, S, T, P, F, typeof(phi_cache), typeof(a_sample)}(
        f,
        copy(u0),
        copy(u0),
        copy(u0),  # uprev2
        copy(u0),  # uprev3
        copy(u0),  # tmp
        phi_cache,
        copy(u0),  # bprev
        copy(u0),  # bprev2
        copy(a_sample),  # aprev
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true,
        0)

    return integ
end

@inline @muladd function DiffEqBase.step!(integ::RL3Integrator{true, S, T, P, F}) where {T, S, P, F}
    @unpack tmp, phi_cache, f, p, t, dt, uprev, uprev2, uprev3, bprev, bprev2, aprev, u, step_number = integ

    grl_f = _get_grl_function(f)

    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL3)!")
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
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL3)!")
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

    if step_number < 2
        # First two steps: use lower order method (RL1)
        mul!(tmp, a_n, uprev)
        @. tmp = tmp + b_n

        rl_phi!(phi_cache, a_n, dt)
        mul!(u, phi_cache, tmp)
        @. u = uprev + dt * u
    else
        # Third step onwards: use RL3 formula
        # α_n = (1/12)*(23*u_n - 16*u_{n-1} + 5*u_{n-2})
        @. tmp = (23.0 * uprev - 16.0 * uprev2 + 5.0 * uprev3) / 12.0

        # β_n = (1/12)*(23*b_n - 16*b_{n-1} + 5*b_{n-2}) + (1/12)*(a_n*b_{n-1} - a_{n-1}*b_n)
        # Compute a_n * α_n first
        mul!(u, a_n, tmp)

        # Add β_n components
        # First part: (1/12)*(23*b_n - 16*b_{n-1} + 5*b_{n-2})
        @. u = u + (23.0 * b_n - 16.0 * bprev + 5.0 * bprev2) / 12.0

        # Second part: (1/12)*(a_n*b_{n-1} - a_{n-1}*b_n)
        # Compute a_n * b_{n-1}
        mul!(tmp, a_n, bprev)
        @. u = u + tmp / 12.0

        # Subtract a_{n-1} * b_n
        mul!(tmp, aprev, b_n)
        @. u = u - tmp / 12.0

        # Apply phi
        rl_phi!(phi_cache, a_n, dt)
        mul!(tmp, phi_cache, u)
        @. u = uprev + dt * tmp
    end

    # Update history
    integ.uprev3 .= uprev2
    integ.uprev2 .= uprev
    integ.uprev .= u
    integ.bprev2 .= bprev
    integ.bprev .= b_n
    integ.aprev .= a_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::RL3Integrator{false, S, T, P, F}) where {T, S, P, F}
    @unpack f, p, t, dt, uprev, uprev2, uprev3, bprev, bprev2, aprev, step_number = integ

    grl_f = _get_grl_function(f)

    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL3)!")
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
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL3)!")
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

    if step_number < 2
        # First two steps: use RL1
        rhs = a_n * uprev .+ b_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    else
        # Third step onwards: use RL3 formula
        α_n = (23.0 * uprev .- 16.0 * uprev2 .+ 5.0 * uprev3) / 12.0

        β_n = (23.0 * b_n .- 16.0 * bprev .+ 5.0 * bprev2) / 12.0 .+
              (a_n * bprev .- aprev * b_n) / 12.0

        rhs = a_n * α_n .+ β_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    end

    # Update history
    integ.uprev3 = uprev2
    integ.uprev2 = uprev
    integ.uprev = integ.u
    integ.bprev2 = bprev
    integ.bprev = b_n
    integ.aprev = a_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::RL3;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = rl3_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    return sol
end
