"""
    GRLFunction

A struct to represent ODEs of the form:
    du/dt = a(u,p,t) * u + b(u,p,t),  t ∈ (0, T]
    u(0) = u_0

where:
- `a`: Function representing the coefficient a(u,p,t)
- `b`: Function representing the additive term b(u,p,t)

This is the Generalized Rush-Larsen function structure that handles:
- HH gates: a(u,p,t) = -(α+β), b(u,p,t) = α
- Markov gates: a(u,p,t) = Q (transition matrix), b(u,p,t) = 0
- Mixed systems: a(u,p,t) as BlockDiagonal matrix
"""
@concrete struct GRLFunction
    a  # Function a(t, y) - the coefficient function
    b  # Function b(t, y) - the additive function
end

# Alias for backwards compatibility
const GeneralizedRushLarsenFunction = GRLFunction

# Out-of-place functor: returns du/dt = a(t, u) * u + b(t, u)
function (f::GRLFunction)(u, p, t)
    a_val = f.a(u, p, t)
    b_val = f.b(u, p, t)
    return a_val .* u .+ b_val
end

# In-place functor: computes du/dt = a(t, u) * u + b(t, u)
function (f::GRLFunction)(du, u, p, t)
    a_val = f.a(u, p, t)
    b_val = f.b(u, p, t)
    @. du = a_val * u + b_val
    return nothing
end
