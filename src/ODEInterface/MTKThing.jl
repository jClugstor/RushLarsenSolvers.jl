"""
    beeler_reuter_1977_with_stimulus(; periodic=true)

Create the Beeler-Reuter 1977 cardiac model with built-in stimulus functionality.

Arguments:
- periodic: If true, creates a periodic stimulus. If false, single pulse.

The model includes parameters for controlling the stimulus:
- stim_start: Start time of first stimulus [ms]
- stim_duration: Duration of each stimulus pulse [ms]
- stim_amplitude: Amplitude of stimulus current [μA/cm²]
- stim_period: Period between stimuli (only used if periodic=true) [ms]

Returns:
- sys: The ODESystem with built-in stimulus
- u0: Initial conditions
- p: Parameters (including stimulus parameters)
- tspan: Time span for simulation
"""
function beeler_reuter_1977_with_stimulus(; periodic=true)
    # Define independent variable
    @independent_variables t

    # Define state variables
    @variables V(t)     # Membrane potential [mV]
    @variables m(t)     # Sodium activation gate
    @variables h(t)     # Sodium inactivation gate
    @variables j(t)     # Sodium slow inactivation gate
    @variables d(t)     # Calcium activation gate
    @variables f(t)     # Calcium inactivation gate
    @variables x1(t)    # Potassium activation gate
    @variables Ca_i(t)  # Intracellular calcium concentration [mM]

    # Mark gating variables with @gating macro
    @gating m(t) h(t) j(t) d(t) f(t) x1(t)

    # Define differential operator
    D = Differential(t)

    # Define parameters including stimulus parameters
    @parameters begin
        # Membrane capacitance
        C_m = 1.0  # μF/cm²

        # Sodium current parameters
        g_Na = 4.0    # mS/cm²
        E_Na = 50.0   # mV
        g_Nac = 0.003 # mS/cm²

        # Slow inward current parameters
        g_s = 0.09    # mS/cm²
        E_s = -82.3   # mV

        # Potassium current parameters
        g_K1 = 0.35   # mS/cm²

        # Stimulus parameters
        stim_start = 10.0      # Start time of first stimulus [ms]
        stim_duration = 2.0    # Duration of stimulus [ms]
        stim_amplitude = -20.0  # Stimulus amplitude [μA/cm²]
        stim_period = 500.0    # Period for periodic stimuli [ms]
    end

    # Define stimulus current as a symbolic expression
    if periodic
        # Periodic stimulus using modulo arithmetic
        # Calculate time relative to the start of the current period
        t_relative = t - stim_start
        t_in_period = t_relative - floor(t_relative / stim_period) * stim_period

        # Stimulus is on if we're within the duration window of the current period
        # and after the initial start time
        I_stim = ifelse(
            (t >= stim_start) & (t_in_period >= 0) & (t_in_period < stim_duration),
            stim_amplitude,
            0.0
        )
    else
        # Single pulse stimulus
        I_stim = ifelse(
            (t >= stim_start) & (t < stim_start + stim_duration),
            stim_amplitude,
            0.0
        )
    end

    # Rate functions for gating variables
    # Sodium activation (m)
    α_m = 0.1 * (V + 47) / (1 - exp(-0.1 * (V + 47)))
    β_m = 4 * exp(-0.056 * (V + 72))

    # Sodium inactivation (h)
    α_h = 0.126 * exp(-0.25 * (V + 77))
    β_h = 1.7 / (1 + exp(-0.082 * (V + 22.5)))

    # Sodium slow inactivation (j)
    α_j = 0.055 * exp(-0.25 * (V + 78)) / (1 + exp(-0.2 * (V + 78)))
    β_j = 0.3 / (1 + exp(-0.1 * (V + 32)))

    # Calcium activation (d)
    α_d = 0.095 * exp(-0.01 * (V - 5)) / (1 + exp(-0.072 * (V - 5)))
    β_d = 0.07 * exp(-0.017 * (V + 44)) / (1 + exp(0.05 * (V + 44)))

    # Calcium inactivation (f)
    α_f = 0.012 * exp(-0.008 * (V + 28)) / (1 + exp(0.15 * (V + 28)))
    β_f = 0.0065 * exp(-0.02 * (V + 30)) / (1 + exp(-0.2 * (V + 30)))

    # Potassium activation (x1)
    α_x1 = 0.0005 * exp(0.083 * (V + 50)) / (1 + exp(0.057 * (V + 50)))
    β_x1 = 0.0013 * exp(-0.06 * (V + 20)) / (1 + exp(-0.04 * (V + 20)))

    # Ionic currents
    I_Na = (g_Na * m^3 * h * j + g_Nac) * (V - E_Na)
    I_s = g_s * d * f * (V - E_s)
    I_K1 = g_K1 * (4 * (exp(0.04 * (V + 85)) - 1)) / (exp(0.08 * (V + 53)) + exp(0.04 * (V + 53)))
    I_x1 = x1 * 0.8 * (exp(0.04 * (V + 77)) - 1) / exp(0.04 * (V + 35))

    # Differential equations with integrated stimulus
    eqs = [
        # Membrane potential with stimulus current
        D(V) ~ -(I_Na + I_s + I_K1 + I_x1) / C_m + I_stim / C_m,

        # Gating variables (exponential integrator)
        D(m) ~ α_m * (1 - m) - β_m * m,
        D(h) ~ α_h * (1 - h) - β_h * h,
        D(j) ~ α_j * (1 - j) - β_j * j,
        D(d) ~ α_d * (1 - d) - β_d * d,
        D(f) ~ α_f * (1 - f) - β_f * f,
        D(x1) ~ α_x1 * (1 - x1) - β_x1 * x1,

        # Calcium dynamics
        D(Ca_i) ~ -10^(-7) * I_s + 0.07 * (10^(-7) - Ca_i)
    ]

    @named sys = ODESystem(eqs, t)
    sys = structural_simplify(sys)

    # Initial conditions (resting state)
    u0 = [
        V => -84.0,      # Resting potential [mV]
        m => 0.011,      # Sodium activation
        h => 0.988,      # Sodium inactivation
        j => 0.975,      # Sodium slow inactivation
        d => 0.003,      # Calcium activation
        f => 0.994,      # Calcium inactivation
        x1 => 0.0001,    # Potassium activation
        Ca_i => 10^(-7)  # Intracellular calcium [mM]
    ]

    # Parameters with default stimulus values
    p = [
        stim_start => 10.0,
        stim_duration => 2.0,
        stim_amplitude => 20.0,
        stim_period => 500.0
    ]

    # Time span for simulation
    tspan = (0.0, 2000.0)  # 2000 ms

    return sys, u0, p, tspan
end

macro gating(vars...)
    exprs = []
    for var in vars
        # Extract the symbol name from expressions like m(t)
        var_sym = var.args[1]
        push!(exprs, :($(esc(var_sym)) = ModelingToolkit.setmetadata($(esc(var_sym)), GatingVariable, true)))
    end
    return Expr(:block, exprs...)
end

# ============================================================================
# Gating Variable Detection
# ============================================================================

"""
    is_gating_variable(var)

Check if a ModelingToolkit variable is marked as a gating variable.

Returns `true` if the variable has `GatingVariable` metadata, `false` otherwise.
"""
function is_gating_variable(var)
    if ModelingToolkit.hasmetadata(var, GatingVariable)
        metadata = ModelingToolkit.getmetadata(var, GatingVariable)
        # Handle both direct boolean values and struct values
        return if isa(metadata, Bool)
            metadata
        else
            metadata.value
        end
    end
    return false
end

"""
    get_gating_variables(sys::ODESystem)

Get all gating variables from a ModelingToolkit system.
"""
function get_gating_variables(sys::ODESystem)
    states = ModelingToolkit.unknowns(sys)
    return filter(is_gating_variable, states)
end

"""
    get_nongating_variables(sys::ODESystem)

Get all non-gating variables from a ModelingToolkit system.
"""
function get_nongating_variables(sys::ODESystem)
    states = ModelingToolkit.unknowns(sys)
    return filter(var -> !is_gating_variable(var), states)
end

# ============================================================================
# Symbolic Rate Constant Extraction
# ============================================================================

"""
    extract_rate_constants(eq::Equation, var, voltage)

Extract α and β rate constants from a gating variable equation.

Assumes the equation is in the standard Rush-Larsen form:
    du/dt = α(V)(1-u) - β(V)u

Returns `(α_expr, β_expr)` where both are symbolic expressions that can be
evaluated to get the rate constants.

# Arguments
- `eq`: The differential equation for the gating variable
- `var`: The gating variable symbol
- `voltage`: The voltage variable symbol (unused in current implementation)
"""
function extract_rate_constants(eq::Equation, var, voltage)
    rhs = eq.rhs
    expanded = ModelingToolkit.expand_derivatives(rhs)

    # For equation: α(V)(1-var) - β(V)var = α(V) - (α(V) + β(V))*var
    # Substitute var = 0 to get α(V)
    α_coeff = ModelingToolkit.substitute(expanded, Dict(var => 0))

    # Substitute var = 1 to get -β(V), so β(V) = -result
    β_coeff = -ModelingToolkit.substitute(expanded, Dict(var => 1))

    return α_coeff, β_coeff
end

"""
    extract_gating_rate_functions(sys::ODESystem)

Extract α and β rate functions for all gating variables in the system.

Returns a dictionary mapping gating variable indices to `(α_func, β_func)` tuples,
where each function has the signature `(u, p, t) -> Float64`.

# Algorithm
1. Find all gating variables in the system
2. For each gating variable, find its corresponding differential equation
3. Extract α and β expressions symbolically
4. Convert to Julia functions that can be evaluated numerically
"""
function extract_gating_rate_functions(sys::ODESystem)
    eqs = ModelingToolkit.equations(sys)
    states = ModelingToolkit.unknowns(sys)
    voltage = states[1]  # Assume first state is voltage

    rate_functions = Dict{Int,Tuple{Function,Function}}()

    for (i, state) in enumerate(states)
        if !is_gating_variable(state)
            continue
        end

        # Find the differential equation for this gating variable
        eq = find_equation_for_variable(eqs, state, sys)
        if eq === nothing
            continue
        end

        # Extract rate constants symbolically
        α_expr, β_expr = extract_rate_constants(eq, state, voltage)

        # Create evaluation functions
        α_func = create_rate_function(α_expr, states, sys)
        β_func = create_rate_function(β_expr, states, sys)

        rate_functions[i] = (α_func, β_func)
    end

    return rate_functions
end

"""
    find_equation_for_variable(eqs, var, sys)

Find the differential equation corresponding to a specific variable.
"""
function find_equation_for_variable(eqs, var, sys)
    iv = ModelingToolkit.get_iv(sys)
    target_lhs = ModelingToolkit.Differential(iv)(var)

    for eq in eqs
        if ModelingToolkit.isequal(eq.lhs, target_lhs)
            return eq
        end
    end
    return nothing
end

"""
    create_rate_function(expr, states, sys)

Create a function that evaluates a symbolic expression.
"""
function create_rate_function(expr, states, sys)
    return (u, p, t) -> begin
        # Create substitution dictionary for states
        subs = Dict(st => u[j] for (j, st) in enumerate(states))

        # Handle parameters if present
        params = ModelingToolkit.parameters(sys)
        if !isempty(params) && p !== nothing && length(p) >= length(params)
            for (j, par) in enumerate(params)
                subs[par] = p[j]
            end
        end

        # Evaluate and return as float
        result = ModelingToolkit.substitute(expr, subs)
        return float(result)
    end
end

struct RushLarsen{F,R}
    f!::F
    rate_functions::R
    dt::Float64

    function RushLarsen(f!::F, rate_functions::R, dt::Float64) where {F,R}
        dt > 0 || throw(ArgumentError("Time step dt must be positive"))
        new{F,R}(f!, rate_functions, dt)
    end
end

"""
    (rl::RushLarsen)(du, u, p, t)

Function call operator that evaluates the ODE system with Rush-Larsen integration.

# Algorithm
1. Compute standard derivatives using the original ODE function
2. For each gating variable, apply exponential integration:
   - Calculate α and β rate constants at current state
   - Compute steady-state value: u_∞ = α/(α+β)
   - Apply exponential integration: u_new = u_∞ + (u_old - u_∞)exp(-λt)
   - Convert back to effective derivative for the solver

# Arguments
- `du`: Derivative vector to be modified in-place
- `u`: Current state vector
- `p`: Parameter vector
- `t`: Current time
"""
function (rl::RushLarsen)(du, u, p, t)
    # Compute standard derivatives
    rl.f!(du, u, p, t)

    # Apply Rush-Larsen exponential integration to gating variables
    for (i, (α_func, β_func)) in rl.rate_functions
        u_i = u[i]

        # Evaluate rate constants at current state
        α = α_func(u, p, t)
        β = β_func(u, p, t)

        # Apply Rush-Larsen method
        λ = α + β
        if λ > 1e-12  # Avoid division by zero
            u_inf = α / λ
            u_new = u_inf + (u_i - u_inf) * exp(-λ * rl.dt)
            # Convert to effective derivative for Euler step
            du[i] = (u_new - u_i) / rl.dt
        end
        # If λ is too small, keep original derivative
    end
end