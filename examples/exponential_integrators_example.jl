# Example: Using GRLFunction with OrdinaryDiffEq's Exponential Integrators
#
# This example shows how to convert a GRLFunction to a SplitODEProblem
# and solve it with various exponential Runge-Kutta methods.

using RushLarsenSolvers
using OrdinaryDiffEq
using ModelingToolkit

# Define a simple cardiac gating model
@variables t
@variables begin
    m(t), [alpha_beta_gate = true]
    h(t), [alpha_beta_gate = true]
    V(t)  # Non-gating variable (voltage)
end

@parameters begin
    g_Na = 120.0
    E_Na = 50.0
end

# Alpha and beta functions for m gate
function alpha_m(V)
    return 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))
end

function beta_m(V)
    return 4.0 * exp(-(V + 65) / 18)
end

# Alpha and beta functions for h gate
function alpha_h(V)
    return 0.07 * exp(-(V + 65) / 20)
end

function beta_h(V)
    return 1.0 / (1 + exp(-(V + 35) / 10))
end

# Sodium current
I_Na(m, h, V) = g_Na * m^3 * h * (V - E_Na)

# Define the ODEs
@equations begin
    # m gate (alpha-beta form)
    D(m) ~ alpha_m(V) * (1 - m) - beta_m(V) * m

    # h gate (alpha-beta form)
    D(h) ~ alpha_h(V) * (1 - h) - beta_h(V) * h

    # Voltage (non-gating, simplified for example)
    D(V) ~ -I_Na(m, h, V) / 1.0  # Capacitance = 1.0
end

# Create the system
@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)

# Create GRLFunction
grl_f = GRLFunction(sys)

# Initial conditions and parameters
u0 = [0.05, 0.6, -65.0]  # [m, h, V]
tspan = (0.0, 50.0)
p = [g_Na, E_Na]

println("=== Testing Exponential Integrators with GRLFunction ===\n")

# Method 1: Convert to SplitODEProblem (out-of-place)
println("1. Creating out-of-place SplitODEProblem...")
split_prob = split_ode_problem(grl_f, u0, tspan, p)

# Solve with ETDRK2 (Exponential Time Differencing Runge-Kutta 2)
println("   Solving with ETDRK2...")
@time sol_etdrk2 = solve(split_prob, ETDRK2(), dt=0.1)
println("   Solution steps: ", length(sol_etdrk2.t))

# Solve with ETDRK4 (4th order)
println("   Solving with ETDRK4...")
@time sol_etdrk4 = solve(split_prob, ETDRK4(), dt=0.1)
println("   Solution steps: ", length(sol_etdrk4.t))

# Solve with Exp4 (Exponential 4th order)
println("   Solving with Exp4...")
@time sol_exp4 = solve(split_prob, Exp4(), dt=0.1)
println("   Solution steps: ", length(sol_exp4.t))

# Solve with EPIRK4s3A (Exponentially-fitted Peer 4-stage 3rd order)
println("   Solving with EPIRK4s3A...")
@time sol_epirk = solve(split_prob, EPIRK4s3A(), dt=0.1)
println("   Solution steps: ", length(sol_epirk.t))

# Method 2: In-place version for larger systems
println("\n2. Creating in-place SplitODEProblem...")
split_prob_iip = split_ode_problem_iip(grl_f, u0, tspan, p)

println("   Solving with ETDRK2 (in-place)...")
@time sol_etdrk2_iip = solve(split_prob_iip, ETDRK2(), dt=0.1)
println("   Solution steps: ", length(sol_etdrk2_iip.t))

# Compare with standard RL1
println("\n3. Comparing with RL1 (for reference)...")
rl_prob = ODEProblem(grl_f, u0, tspan, p)
@time sol_rl1 = solve(rl_prob, RL1(), dt=0.1)
println("   Solution steps: ", length(sol_rl1.t))

# Print final values
println("\n=== Final Values Comparison ===")
println("ETDRK2:      ", sol_etdrk2[end])
println("ETDRK4:      ", sol_etdrk4[end])
println("Exp4:        ", sol_exp4[end])
println("EPIRK4s3A:   ", sol_epirk[end])
println("ETDRK2 (IIP):", sol_etdrk2_iip[end])
println("RL1:         ", sol_rl1[end])

println("\n=== Benefits of Exponential Integrators ===")
println("✓ Excellent for stiff systems with linear stiffness")
println("✓ Can use larger time steps than explicit methods")
println("✓ BlockDiagonal structure is efficiently handled")
println("✓ Good for systems with separated time scales")
println("✓ Compatible with adaptive time stepping")

println("\n=== When to Use ===")
println("• Use exponential integrators (ETDRK, Exp4) for:")
println("  - Stiff gating dynamics")
println("  - Large systems with BlockDiagonal structure")
println("  - When you need high accuracy")
println("  - Problems where implicit methods would be too expensive")
println("\n• Use custom RL methods (RL1-RL4) for:")
println("  - Maximum control over the numerical scheme")
println("  - When you need specific multistep properties")
println("  - Very large systems where exponential computation is costly")
