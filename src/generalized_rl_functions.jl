"""
    GeneralizedRushLarsenFunction

A struct to represent ODEs of the form:
    du/dt = a(u,p,t) * u + b(u,p,t),  t ∈ (0, T]
    u(0) = u_0

where:
- `a`: Function representing the coefficient a(u,p,t)
- `b`: Function representing the additive term b(u,p,t)

"""
@concrete struct GeneralizedRushLarsenFunction
    a  # Function a(t, y) - the coefficient function
    b  # Function b(t, y) - the additive function
end

function GeneralizedRushLarsenFunction(a, b)
    GeneralizedRushLarsenFunction(a, b)
end

# Out-of-place functor: returns du/dt = a(t, u) * u + b(t, u)
function (f::GeneralizedRushLarsenFunction)(u, p, t)
    a_val = f.a(u, p, t)
    b_val = f.b(u, p, t)
    return a_val .* u .+ b_val
end

# In-place functor: computes du/dt = a(t, u) * u + b(t, u)
function (f::GeneralizedRushLarsenFunction)(du, u, p, t)
    a_val = f.a(u, p, t)
    b_val = f.b(u, p, t)
    @. du = a_val * u + b_val
    return nothing
end
