"""
    RL1(exponential_alg=nothing)

First-order generalized Rush-Larsen algorithm for ODEs of the form:
    du/dt = a(u,p,t)*u + b(u,p,t)

Uses the formula: y_{n+1} = y_n + dt * Φ(a_n * dt) * (a_n * y_n + b_n)
where Φ(x) = (e^x - 1) / x

# Arguments
- `exponential_alg`: ExponentialUtilities algorithm for matrix phi computation (optional)
  - If `nothing` (default), uses ExponentialUtilities.jl's default method
  - Can specify methods like `ExpMethodHigham2005()`, `ExpMethodGeneric()`, etc.

# Example
```julia
# Default algorithm
alg = RL1()

# With specific exponential method (requires ExponentialUtilities.jl)
using ExponentialUtilities
alg = RL1(ExpMethodHigham2005())
```
"""
struct RL1 <: AbstractRushLarsenAlgorithm
    exponential_alg
end

# Default constructor
RL1() = RL1(nothing)

# Type-stable helper functions to extract GRLFunction
@inline _get_grl_function(f::GRLFunction) = f
@inline _get_grl_function(f) = f.f

mutable struct RL1Integrator{IIP, S, T, P, F, PhiType} <: DiffEqBase.AbstractODEIntegrator{RL1, IIP, S, T}
    f::F
    uprev::S
    u::S
    tmp::S
    phi_cache::PhiType  # Cache for phi(a*dt) computation
    tprev::T
    t::T
    t0::T
    dt::T
    tdir::T
    p::P
    u_modified::Bool
end

DiffEqBase.isinplace(::RL1Integrator{IIP}) where {IIP} = IIP

function DiffEqBase.__init(prob::ODEProblem, alg::RL1; dt = error("dt is required for this algorithm"))
    rl1_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

@inline function rl1_init(f::F, ::Val{IIP}, u0::S, t0::T, dt::T, p::P) where {F, P, T, S, IIP}
    # Extract GRLFunction and compute initial a to determine phi_cache type
    grl_f = _get_grl_function(f)
    a_sample = grl_f.a(u0, p, t0)

    # Create phi_cache with same structure as a
    phi_cache = similar(a_sample)

    integ = RL1Integrator{IIP, S, T, P, F, typeof(phi_cache)}(
        f,
        copy(u0),
        copy(u0),
        copy(u0),
        phi_cache,
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true)

    return integ
end

@inline @muladd function DiffEqBase.step!(integ::RL1Integrator{true, S, T, P, F}) where {T, S, P, F}
    integ.uprev .= integ.u

    @unpack tmp, phi_cache, f, p, t, dt, uprev, u = integ

    # Extract GRLFunction
    grl_f = _get_grl_function(f)

    # Evaluate a_n and b_n at current state
    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n!")
            println("  Time: ", t, " ms")
            println("  First NaN at a_n linear index: ", j)
            println("  a_n type: ", typeof(a_n))
            println("  a_n size: ", size(a_n))
            # Convert linear index to row/column
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
            println("\n⚠️  FIRST NaN DETECTED in b_n!")
            println("  Time: ", t, " ms")
            println("  First NaN at b_n element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in b_n at element $j at time $t")
        end
    end

    # RL1 formula: y_{n+1} = y_n + dt * Φ(a_n * dt) * (a_n * y_n + b_n)
    # where Φ(x) = (e^x - 1) / x

    # Compute a_n * y_n + b_n (store in tmp)
    # Matrix case (including BlockDiagonal and diagonal matrices)
    mul!(tmp, a_n, uprev)
    @. tmp = tmp + b_n

    # Check for NaNs in tmp (a_n * y_n + b_n)
    for (j, val) in enumerate(tmp)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in tmp (a_n * y_n + b_n)!")
            println("  Time: ", t, " ms")
            println("  First NaN at tmp element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in tmp at element $j at time $t")
        end
    end

    # Compute phi in-place to avoid allocations
    rl_phi!(phi_cache, a_n, dt)
    mul!(u, phi_cache, tmp)
    @. u = uprev + dt * u

    integ.tprev = t
    integ.t += dt

    # Check for NaNs in final result
    for (j, val) in enumerate(u)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in final result!")
            println("  Time: ", integ.t, " ms")
            println("  First NaN at state index: ", j)
            println("\n  All state values at NaN time:")
            for (k, v) in enumerate(u)
                println("    [", k, "] = ", v)
            end
            println("\n  Previous timestep values:")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected at state index $j at time $(integ.t)")
        end
    end

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::RL1Integrator{false, S, T, P, F}) where {T, S, P, F}
    integ.uprev = integ.u

    @unpack tmp, f, p, t, dt, uprev = integ

    # Extract GRLFunction
    grl_f = _get_grl_function(f)

    # Evaluate a_n and b_n at current state
    a_n = grl_f.a(uprev, p, t)
    b_n = grl_f.b(uprev, p, t)

    # Check for NaNs in a_n
    for (j, val) in enumerate(a_n)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in a_n!")
            println("  Time: ", t, " ms")
            println("  First NaN at a_n linear index: ", j)
            println("  a_n type: ", typeof(a_n))
            println("  a_n size: ", size(a_n))
            # Convert linear index to row/column
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
            println("\n⚠️  FIRST NaN DETECTED in b_n!")
            println("  Time: ", t, " ms")
            println("  First NaN at b_n element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in b_n at element $j at time $t")
        end
    end

    # RL1 formula: y_{n+1} = y_n + dt * Φ(a_n * dt) * (a_n * y_n + b_n)
    # where Φ(x) = (e^x - 1) / x

    # Compute a_n * y_n + b_n (matrix-vector multiplication)
    rhs = a_n * uprev .+ b_n

    # Check for NaNs in rhs
    for (j, val) in enumerate(rhs)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in rhs (a_n * y_n + b_n)!")
            println("  Time: ", t, " ms")
            println("  First NaN at rhs element: ", j)
            println("\n  State values at current time (uprev):")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected in rhs at element $j at time $t")
        end
    end

    # Compute phi using dispatch based on type of a_n
    phi_a = rl_phi(a_n, dt)

    # Apply update
    integ.u = uprev .+ dt .* (phi_a * rhs)

    integ.tprev = t
    integ.t += dt

    # Check for NaNs in final result
    for (j, val) in enumerate(integ.u)
        if isnan(val)
            println("\n⚠️  FIRST NaN DETECTED in final result!")
            println("  Time: ", integ.t, " ms")
            println("  First NaN at state index: ", j)
            println("\n  All state values at NaN time:")
            for (k, v) in enumerate(integ.u)
                println("    [", k, "] = ", v)
            end
            println("\n  Previous timestep values:")
            for (k, v) in enumerate(uprev)
                println("    [", k, "] = ", v)
            end
            error("NaN detected at state index $j at time $(integ.t)")
        end
    end

    return nothing
end

function DiffEqBase.__solve(prob::ODEProblem, alg::RL1;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = copy(u0)

    integ = rl1_init(DiffEqBase.unwrapped_f(prob.f), Val(DiffEqBase.isinplace(prob)), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n-1)
        step!(integ)
        us[i+1] = copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    return sol
end