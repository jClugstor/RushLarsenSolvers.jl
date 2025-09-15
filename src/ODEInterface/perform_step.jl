function initialize!(integrator, cache::RushLarsenConstantCache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Pre-start fsal: compute both the gating parameters and non-gating derivatives
    gating_params = integrator.f.f1(integrator.uprev, integrator.p, integrator.t)
    non_gating_derivs = integrator.f.f2(integrator.uprev, integrator.p, integrator.t)

    # Store both components in fsalfirst
    integrator.fsalfirst = (gating_params, non_gating_derivs)
    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = (zero.(gating_params), zero.(non_gating_derivs))
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator, cache::RushLarsenConstantCache, repeat_step=false)
    @unpack t, dt, uprev, f, p = integrator

    # Unpack the components from fsalfirst
    gating_params, non_gating_derivs = integrator.fsalfirst

    # Initialize the new solution
    u = similar(uprev)

    # Apply Rush-Larsen method to gating variables
    # gating_params contains tuples of (alpha, beta) for each gating variable
    for i in eachindex(gating_params)
        alpha, beta = gating_params[i]
        # Rush-Larsen formula: u_new = u_inf + (u_old - u_inf) * exp(-dt/tau)
        # where u_inf = alpha/(alpha+beta) and tau = 1/(alpha+beta)
        if alpha + beta < 1e-14
            # Handle special case to avoid division by zero
            u[i] = uprev[i]
        else
            u_inf = alpha / (alpha + beta)
            tau = 1 / (alpha + beta)
            u[i] = u_inf + (uprev[i] - u_inf) * exp(-dt / tau)
        end
    end

    # Apply forward Euler to non-gating variables
    for i in eachindex(non_gating_derivs)
        idx = i + length(gating_params)
        u[idx] = @muladd uprev[idx] + dt * non_gating_derivs[i]
    end

    # Compute values at the new point for interpolation
    new_gating_params = f.f1(u, p, t + dt)
    new_non_gating_derivs = f.f2(u, p, t + dt)

    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)

    integrator.fsallast = (new_gating_params, new_non_gating_derivs)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
end

@inline function initialize!(integrator, cache::RushLarsenCache, f=integrator.f)

end

@inline function perform_step!(integrator, cache::RushLarsenCache, f=integrator.f)

end