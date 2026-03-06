using SciMLOperators: FunctionOperator
using LinearAlgebra: I, mul!

"""
    split_ode_problem(grl_f::GRLFunction, u0, tspan, p=nothing)

Convert a GRLFunction to a SplitODEProblem compatible with OrdinaryDiffEq's
exponential integrators (ETD, ExpRK, etc.).

The generalized Rush-Larsen form:
    du/dt = a(u,p,t)*u + b(u,p,t)

is converted to the split form:
    du/dt = Au + f(u,p,t)

where:
- A = a(u,p,t) is the linear operator (can be time/state dependent)
- f(u,p,t) = b(u,p,t) is the nonlinear part

# Arguments
- `grl_f::GRLFunction`: The generalized Rush-Larsen function
- `u0`: Initial condition
- `tspan`: Time span tuple (t0, tf)
- `p`: Parameters (optional)

# Returns
- `SplitODEProblem` that can be used with exponential integrators

# Example
```julia
using OrdinaryDiffEq

# Create GRLFunction from ModelingToolkit system
grl_f = GRLFunction(sys)

# Convert to split ODE problem
split_prob = split_ode_problem(grl_f, u0, tspan, p)

# Solve with exponential integrators
sol = solve(split_prob, ETDRK2(), dt=0.1)
sol = solve(split_prob, ETDRK4(), dt=0.1)
sol = solve(split_prob, Exp4(), dt=0.1)
sol = solve(split_prob, EPIRK4s3A(), dt=0.1)
```

# Notes
- The linear operator `A` is treated as time/state-dependent
- For constant coefficient problems, use simpler exponential methods
- For stiff problems with BlockDiagonal structure, this is very efficient
- Compatible with ETDRK2, ETDRK4, Exp4, EPIRK family, and other exponential RK methods
"""
function split_ode_problem(grl_f::GRLFunction, u0, tspan, p=nothing)
    # Linear part: du/dt = A(u,p,t) * u
    # A(u,p,t) returns the matrix a(u,p,t)
    function linear_operator(u, p, t)
        return grl_f.a(u, p, t)
    end

    # Nonlinear part: du/dt = f(u,p,t)
    # f(u,p,t) = b(u,p,t)
    function nonlinear_part(u, p, t)
        return grl_f.b(u, p, t)
    end

    # Create out-of-place split function
    split_f = SplitFunction(linear_operator, nonlinear_part)

    # Create and return SplitODEProblem
    return SplitODEProblem(split_f, u0, tspan, p)
end

"""
    split_ode_problem_iip(grl_f::GRLFunction, u0, tspan, p=nothing)

In-place version of split_ode_problem for better performance with large systems.

The linear operator is applied via matrix-vector multiplication, and the nonlinear
part is computed in-place.

# Example
```julia
using OrdinaryDiffEq

grl_f = GRLFunction(sys)
split_prob = split_ode_problem_iip(grl_f, u0, tspan, p)

# Solve with exponential integrators
sol = solve(split_prob, ETDRK2(), dt=0.1)
```
"""
function split_ode_problem_iip(grl_f::GRLFunction, u0, tspan, p=nothing)
    n = length(u0)

    # Linear part: wrap a(u,p,t) in FunctionOperator for proper size() method
    linear_op = FunctionOperator(
        (du, u, p, t) -> mul!(du, grl_f.a(u, p, t), u),
        u0, p=p, t=tspan[1], size=(n, n)
    )

    # Nonlinear part: b(u,p,t)
    nonlinear_part! = (du, u, p, t) -> (du .= grl_f.b(u, p, t); nothing)

    # Create split function and problem
    split_f = SplitFunction(linear_op, nonlinear_part!, mass_matrix=I)
    return SplitODEProblem(split_f, u0, tspan, p)
end

"""
    matrix_operator(grl_f::GRLFunction)

Create a DiffEqArrayOperator that wraps the `a(u,p,t)` function for use with
exponential integrators that require an operator interface.

This is useful for methods that need to compute matrix exponentials or phi functions.

# Example
```julia
A_op = matrix_operator(grl_f)

# The operator can be evaluated at any state
A = A_op(u, p, t)

# Use with exponential propagators
using ExponentialUtilities
expA = exp(A * dt)
```
"""
function matrix_operator(grl_f::GRLFunction)
    # Return a callable that computes A(u,p,t)
    return (u, p, t) -> grl_f.a(u, p, t)
end
