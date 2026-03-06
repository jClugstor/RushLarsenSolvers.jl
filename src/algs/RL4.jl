"""
    RL4(exponential_alg=nothing)

Fourth-order generalized Rush-Larsen algorithm for ODEs of the form:
    du/dt = a(u,p,t)*u + b(u,p,t)

Uses a multi-step formula with α_n and β_n computed from previous time steps.
For RL4:
    α_n = (1/24)*(55*u_n - 59*u_{n-1} + 37*u_{n-2} - 9*u_{n-3})
    β_n = (1/24)*(55*b_n - 59*b_{n-1} + 37*b_{n-2} - 9*b_{n-3})
          + (h/12)*(a_n*(3*b_{n-1} - b_{n-2}) - (3*a_{n-1} - a_{n-2})*b_n)

where h is the time step (dt).

# Arguments
- `exponential_alg`: ExponentialUtilities algorithm for matrix phi computation (optional)

# Example
```julia
alg = RL4()
```
"""
struct RL4 <: AbstractRushLarsenAlgorithm
    exponential_alg
end

RL4() = RL4(nothing)

mutable struct RL4Integrator{IIP, S, T, P, F, PhiType, AType} <: DiffEqBase.AbstractODEIntegrator{RL4, IIP, S, T}
    f::F
    uprev::S
    u::S
    uprev2::S  # u_{n-1}
    uprev3::S  # u_{n-2}
    uprev4::S  # u_{n-3}
    tmp::S
    tmp2::S    # Additional temp storage for RL4
    phi_cache::PhiType
    bprev::S   # b_{n-1}
    bprev2::S  # b_{n-2}
    bprev3::S  # b_{n-3}
    aprev::AType  # a_{n-1}
    aprev2::AType # a_{n-2}
    tprev::T
    t::T
    t0::T
    dt::T
    tdir::T
    p::P
    u_modified::Bool
    step_number::Int
end

DiffEqBase.isinplace(::RL4Integrator{IIP}) where {IIP} = IIP

function DiffEqBase.__init(prob::ODEProblem, alg::RL4; dt = error("dt is required for this algorithm"))
    rl4_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

@inline function rl4_init(f::F, ::Val{IIP}, u0::S, t0::T, dt::T, p::P) where {F, P, T, S, IIP}
    grl_f = _get_grl_function(f)
    a_sample = grl_f.a(u0, p, t0)
    phi_cache = similar(a_sample)

    integ = RL4Integrator{IIP, S, T, P, F, typeof(phi_cache), typeof(a_sample)}(
        f,
        copy(u0),
        copy(u0),
        copy(u0),  # uprev2
        copy(u0),  # uprev3
        copy(u0),  # uprev4
        copy(u0),  # tmp
        copy(u0),  # tmp2
        phi_cache,
        copy(u0),  # bprev
        copy(u0),  # bprev2
        copy(u0),  # bprev3
        copy(a_sample),  # aprev
        copy(a_sample),  # aprev2
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

@inline @muladd function DiffEqBase.step!(integ::RL4Integrator{true, S, T, P, F}) where {T, S, P, F}
    @unpack tmp, tmp2, phi_cache, f, p, t, dt, uprev, uprev2, uprev3, uprev4 = integ
    @unpack bprev, bprev2, bprev3, aprev, aprev2, u, step_number = integ

    grl_f = _get_grl_function(f)

    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL4)!")
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
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL4)!")
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

    if step_number < 3
        # First three steps: use lower order method (RL1)
        mul!(tmp, a_n, uprev)
        @. tmp = tmp + b_n

        rl_phi!(phi_cache, a_n, dt)
        mul!(u, phi_cache, tmp)
        @. u = uprev + dt * u
    else
        # Fourth step onwards: use RL4 formula
        # α_n = (1/24)*(55*u_n - 59*u_{n-1} + 37*u_{n-2} - 9*u_{n-3})
        @. tmp = (55.0 * uprev - 59.0 * uprev2 + 37.0 * uprev3 - 9.0 * uprev4) / 24.0

        # Compute a_n * α_n
        mul!(u, a_n, tmp)

        # β_n = (1/24)*(55*b_n - 59*b_{n-1} + 37*b_{n-2} - 9*b_{n-3})
        #       + (h/12)*(a_n*(3*b_{n-1} - b_{n-2}) - (3*a_{n-1} - a_{n-2})*b_n)

        # First part: (1/24)*(55*b_n - 59*b_{n-1} + 37*b_{n-2} - 9*b_{n-3})
        @. u = u + (55.0 * b_n - 59.0 * bprev + 37.0 * bprev2 - 9.0 * bprev3) / 24.0

        # Second part: (h/12)*(a_n*(3*b_{n-1} - b_{n-2}) - (3*a_{n-1} - a_{n-2})*b_n)
        # Compute 3*b_{n-1} - b_{n-2}
        @. tmp = 3.0 * bprev - bprev2
        # Compute a_n * (3*b_{n-1} - b_{n-2})
        mul!(tmp2, a_n, tmp)
        @. u = u + (dt / 12.0) * tmp2

        # Compute 3*a_{n-1} - a_{n-2}
        @. tmp = 3.0 * aprev - aprev2
        # Compute (3*a_{n-1} - a_{n-2}) * b_n
        mul!(tmp2, tmp, b_n)
        @. u = u - (dt / 12.0) * tmp2

        # Apply phi
        rl_phi!(phi_cache, a_n, dt)
        mul!(tmp, phi_cache, u)
        @. u = uprev + dt * tmp
    end

    # Update history
    integ.uprev4 .= uprev3
    integ.uprev3 .= uprev2
    integ.uprev2 .= uprev
    integ.uprev .= u
    integ.bprev3 .= bprev2
    integ.bprev2 .= bprev
    integ.bprev .= b_n
    integ.aprev2 .= aprev
    integ.aprev .= a_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::RL4Integrator{false, S, T, P, F}) where {T, S, P, F}
    @unpack f, p, t, dt, uprev, uprev2, uprev3, uprev4 = integ
    @unpack bprev, bprev2, bprev3, aprev, aprev2, step_number = integ

    grl_f = _get_grl_function(f)

    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n (RL4)!")
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
            println("\n⚠️  FIRST NaN DETECTED in b_n (RL4)!")
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

    if step_number < 3
        # First three steps: use RL1
        rhs = a_n * uprev .+ b_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    else
        # Fourth step onwards: use RL4 formula
        α_n = (55.0 * uprev .- 59.0 * uprev2 .+ 37.0 * uprev3 .- 9.0 * uprev4) / 24.0

        β_n = (55.0 * b_n .- 59.0 * bprev .+ 37.0 * bprev2 .- 9.0 * bprev3) / 24.0 .+
              (dt / 12.0) .* (a_n * (3.0 * bprev .- bprev2) .- (3.0 * aprev .- aprev2) * b_n)

        rhs = a_n * α_n .+ β_n
        phi_a = rl_phi(a_n, dt)
        integ.u = uprev .+ dt .* (phi_a * rhs)
    end

    # Update history
    integ.uprev4 = uprev3
    integ.uprev3 = uprev2
    integ.uprev2 = uprev
    integ.uprev = integ.u
    integ.bprev3 = bprev2
    integ.bprev2 = bprev
    integ.bprev = b_n
    integ.aprev2 = aprev
    integ.aprev = a_n
    integ.tprev = t
    integ.t += dt
    integ.step_number += 1

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::RL4;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = rl4_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    return sol
end
