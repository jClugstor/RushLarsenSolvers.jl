module RushLarsenSolversModelingToolkitExt

using RushLarsenSolvers
import RushLarsenSolvers: RushLarsenFunction
using RushLarsenSolvers: is_gating_variable
using ModelingToolkit
import ModelingToolkit.SymbolicIndexingInterface as SymbolicIndexingInterface

struct AlphaBetaGate <: ModelingToolkit.Symbolics.AbstractVariableMetadata end
struct TauGate <: ModelingToolkit.Symbolics.AbstractVariableMetadata end

ModelingToolkit.Symbolics.option_to_metadata_type(::Val{:tau_gate}) = TauGate
ModelingToolkit.Symbolics.option_to_metadata_type(::Val{:alpha_beta_gate}) = AlphaBetaGate

# Callable struct to avoid type instability in closures
struct NonGatingWrapper{F,B<:AbstractVector,I<:AbstractVector{Int}}
    base_func::F
    buffer::B
    non_gating_idxs::I
end

@inline function (wrapper::NonGatingWrapper)(du, u, p, t)
    wrapper.base_func(wrapper.buffer, u, p, t)
    @inbounds for (i, idx) in enumerate(wrapper.non_gating_idxs)
        du[idx] = wrapper.buffer[i]
    end
    return nothing
end

# Callable struct for gating functions to avoid type instability
struct GatingWrapper{FA,FB}
    α_func::FA
    β_func::FB
end

@inline function (wrapper::GatingWrapper)(gating_vars, u, p, t)
    wrapper.α_func(gating_vars[1], u, p, t)
    wrapper.β_func(gating_vars[2], u, p, t)
    return nothing
end

RushLarsenSolvers.is_tau_variable(var::ModelingToolkit.Symbolics.BasicSymbolic) = ModelingToolkit.Symbolics.hasmetadata(var, TauGate)
RushLarsenSolvers.is_alphabeta_variable(var::ModelingToolkit.Symbolics.BasicSymbolic) = ModelingToolkit.Symbolics.hasmetadata(var, AlphaBetaGate)
RushLarsenSolvers.is_gating_variable(var::ModelingToolkit.Symbolics.BasicSymbolic) = RushLarsenSolvers.is_tau_variable(var) || RushLarsenSolvers.is_alphabeta_variable(var)


# ============================================================================
# RushLarsenFunction Constructor from ModelingToolkit System
# ============================================================================

"""
    RushLarsenFunction(sys::ODESystem; simplify=true, cse=true)

Construct a RushLarsenFunction from a ModelingToolkit ODESystem.

This constructor automatically:
1. Identifies gating variables using the GatingVariable metadata
2. Extracts α and β rate constants for gating variables
3. Builds optimized functions using Symbolics.build_function
4. Separates gating and non-gating variable equations

# Arguments
- `sys::ODESystem`: A ModelingToolkit ODESystem with gating variables marked

# Keyword Arguments
- `simplify::Bool=true`: Whether to simplify symbolic expressions before code generation
- `cse::Bool=true`: Whether to use common subexpression elimination in code generation

# Returns
- `RushLarsenFunction`: A function object suitable for use with RushLarsen algorithm
"""
function RushLarsenSolvers.RushLarsenFunction(sys::ModelingToolkit.System; simplify::Bool=true, cse::Bool=true)
    # Get system information
    eqs = ModelingToolkit.equations(sys)
    states = ModelingToolkit.unknowns(sys)
    params = ModelingToolkit.parameters(sys)
    iv = ModelingToolkit.get_iv(sys)

    # Get observed equations and create substitution dictionary
    observed_eqs = ModelingToolkit.observed(sys)
    obs_subs = Dict(eq.lhs => eq.rhs for eq in observed_eqs)

    # Helper function to recursively substitute observed variables
    function substitute_observed(expr)
        if isempty(obs_subs)
            return expr
        end
        # Keep substituting until no more substitutions are possible
        prev_expr = expr
        max_iterations = 100  # Prevent infinite loops
        for _ in 1:max_iterations
            new_expr = ModelingToolkit.substitute(prev_expr, obs_subs)
            if isequal(new_expr, prev_expr)
                break
            end
            prev_expr = new_expr
        end
        return prev_expr
    end

    # Separate gating and non-gating variables
    gating_idxs = Int[]
    non_gating_idxs = Int[]
    gating_eqs = Tuple{Any,Any}[]  # Type-annotate to avoid Vector{Any}
    non_gating_eqs = Any[]  # Will be homogeneous symbolic expressions

    for (i, state) in enumerate(states)
        if is_gating_variable(state)
            push!(gating_idxs, i)
            # Find equation for this gating variable
            eq = find_equation_for_variable(eqs, state, sys)
            if eq !== nothing
                push!(gating_eqs, (state, eq))
            end
        else
            push!(non_gating_idxs, i)
            # Find equation for this non-gating variable
            eq = find_equation_for_variable(eqs, state, sys)
            if eq !== nothing
                # Substitute observed variables in the RHS
                rhs_substituted = substitute_observed(eq.rhs)
                # Optionally simplify the expression to reduce complexity
                rhs_final = simplify ? ModelingToolkit.Symbolics.simplify(rhs_substituted) : rhs_substituted
                push!(non_gating_eqs, rhs_final)
            end
        end
    end

    # Build gating function that returns vector of (τ, g_inf) tuples
    # For compatibility, we still call them (α, β) in the tuple but they represent (tau, inf)
    gating_rate_exprs = Tuple{Any,Any}[]  # Type-annotate to avoid Vector{Any}
    gate_types = Symbol[]  # Track whether each gate is tau or alpha-beta
    for (state, eq) in gating_eqs
        # Check if this is a tau-type gate or alpha-beta gate
        if RushLarsenSolvers.is_tau_variable(state)
            # Extract tau and inf from equation structure
            tau_expr, inf_expr = extract_tau_inf_from_equation(eq, state, obs_subs)
            push!(gating_rate_exprs, (tau_expr, inf_expr))
            push!(gate_types, :tau)
        else
            # Extract alpha and beta (default behavior)
            α_expr, β_expr = extract_alpha_beta_equations(state, obs_subs)
            α_expr = substitute_observed(α_expr)
            β_expr = substitute_observed(β_expr)
            push!(gating_rate_exprs, (α_expr, β_expr))
            push!(gate_types, :alpha_beta)
        end
    end
    # Create gating function using Symbolics.build_function
    if isempty(gating_rate_exprs)
        gating_f = (gating_vars, u, p, t) -> nothing
    else
        # Build separate functions for tau/alpha and inf/beta, then combine
        # Extract first and second elements of tuples
        α_exprs = [pair[1] for pair in gating_rate_exprs]
        β_exprs = [pair[2] for pair in gating_rate_exprs]

        # Build IN-PLACE functions to avoid allocations
        # Use ModelingToolkit's build_function_wrapper for better code generation
        α_func_tuple = ModelingToolkit.build_function_wrapper(
            sys,
            α_exprs;
            expression=Val{false},
            cse=cse
        )

        β_func_tuple = ModelingToolkit.build_function_wrapper(
            sys,
            β_exprs;
            expression=Val{false},
            cse=cse
        )

        # Extract in-place versions (second element of tuple)
        α_func_iip = α_func_tuple[2]
        β_func_iip = β_func_tuple[2]

        # Use callable struct to avoid type instability
        gating_f = GatingWrapper(α_func_iip, β_func_iip)
    end

    # Build non-gating function using Symbolics.build_function
    if isempty(non_gating_eqs)
        non_gating_f = (du, u, p, t) -> nothing
    else
        # Build the function using ModelingToolkit's wrapper
        non_gating_func = ModelingToolkit.build_function_wrapper(
            sys,
            non_gating_eqs;
            expression=Val{false},
            cse=cse
        )

        # Use IN-PLACE version to avoid allocations
        # Pre-allocate buffer for non-gating results
        n_non_gating = length(non_gating_eqs)
        non_gating_buffer = zeros(n_non_gating)

        if non_gating_func isa Tuple
            # Extract in-place version (second element)
            base_func_iip = non_gating_func[2]
            # Use callable struct to avoid type instability
            non_gating_f = NonGatingWrapper(base_func_iip, non_gating_buffer, non_gating_idxs)
        else
            # Only out-of-place available (shouldn't happen with multiple expressions)
            base_func = non_gating_func
            non_gating_f = (du, u, p, t) -> begin
                result = base_func(u, p, t)
                for (i, idx) in enumerate(non_gating_idxs)
                    du[idx] = result[i]
                end
            end
        end
    end

    return RushLarsenSolvers.RushLarsenFunction(gating_f, non_gating_f, gating_idxs, non_gating_idxs, gate_types)
end

"""
    get_gating_variables(sys::ODESystem)

Get all gating variables from a ModelingToolkit system.
"""
function get_gating_variables(sys::ModelingToolkit.System)
    states = ModelingToolkit.unknowns(sys)
    return filter(is_gating_variable, states)
end

"""
    get_nongating_variables(sys::ODESystem)

Get all non-gating variables from a ModelingToolkit system.
"""
function get_nongating_variables(sys::ModelingToolkit.System)
    states = ModelingToolkit.unknowns(sys)
    return filter(var -> !is_gating_variable(var), states)
end

# ============================================================================
# Symbolic Rate Constant Extraction
# ============================================================================

"""
    extract_alpha_beta_equations(gating_var, obs_subs)

Extract α and β expressions from the observed equations dictionary.

For an AlphaBetaGate component, the observed equations contain:
    α ~ α_expr
    β ~ β_expr

This function looks up the α and β expressions from the obs_subs dictionary.

# Arguments
- `gating_var`: The gating variable (e.g., m.y, h.y)
- `obs_subs`: Dictionary mapping observed variables to their expressions

# Returns
- `(α_expr, β_expr)`: The right-hand side expressions for α and β
"""
function extract_alpha_beta_equations(gating_var, obs_subs)
    # Get the parent component name by parsing the gating variable name
    # e.g., for sod_current₊sodium₊m₊y, we want to find sod_current₊sodium₊m₊α and sod_current₊sodium₊m₊β
    var_name = string(ModelingToolkit.getname(gating_var))

    # Remove the trailing 'y' to get the gate base name
    if endswith(var_name, "₊y")
        base_name = chop(var_name, tail=1)  # Remove the 'y'
    else
        error("Gating variable $var_name doesn't end with ₊y")
    end

    # Find α and β in the obs_subs dictionary by matching variable names
    α_expr = nothing
    β_expr = nothing

    for (lhs, rhs) in obs_subs
        lhs_name = string(ModelingToolkit.getname(lhs))
        if lhs_name == base_name * "α"
            α_expr = rhs
        elseif lhs_name == base_name * "β"
            β_expr = rhs
        end
    end

    if α_expr === nothing || β_expr === nothing
        error("Could not find equations for α and β for gating variable $var_name")
    end

    return α_expr, β_expr
end

"""
    extract_tau_inf_from_rhs(rhs, gating_var, obs_subs)

Helper function to extract tau and inf from an expression of the form (g_inf - g) / tau.
Handles both subtraction and addition with negation patterns.
"""
function extract_tau_inf_from_rhs(rhs, gating_var, obs_subs)
    if !ModelingToolkit.Symbolics.istree(rhs)
        error("Expected a symbolic tree expression")
    end

    op = ModelingToolkit.Symbolics.operation(rhs)
    args = ModelingToolkit.Symbolics.arguments(rhs)

    # Check if it's a division: (numerator) / (denominator)
    if op != (/)
        error("Expected division operation at top level")
    end

    numerator = args[1]
    denominator = args[2]

    if !ModelingToolkit.Symbolics.istree(numerator)
        error("Expected numerator to be a symbolic tree")
    end

    num_op = ModelingToolkit.Symbolics.operation(numerator)
    num_args = ModelingToolkit.Symbolics.arguments(numerator)

    # Check if numerator is a subtraction: (g_inf - g)
    if num_op == (-) && length(num_args) == 2
        if ModelingToolkit.isequal(num_args[2], gating_var)
            g_inf_expr = num_args[1]
            tau_expr = denominator

            # Substitute observed variables
            g_inf_expr = substitute_observed_recursive(g_inf_expr, obs_subs)
            tau_expr = substitute_observed_recursive(tau_expr, obs_subs)

            return tau_expr, g_inf_expr
        end
    # Check if numerator is an addition: could be (g_inf + (-g)) or ((-g) + g_inf)
    elseif num_op == (+) && length(num_args) == 2
        first_arg = num_args[1]
        second_arg = num_args[2]

        # Check if second argument is negative of gating variable: (g_inf + (-g))
        if ModelingToolkit.Symbolics.istree(second_arg)
            op2 = ModelingToolkit.Symbolics.operation(second_arg)
            args2 = ModelingToolkit.Symbolics.arguments(second_arg)

            # Check for multiplication with -1 or unary negation
            if (op2 == (*) && length(args2) == 2 &&
                ((args2[1] == -1 && ModelingToolkit.isequal(args2[2], gating_var)) ||
                 (args2[2] == -1 && ModelingToolkit.isequal(args2[1], gating_var)))) ||
               (op2 == (-) && length(args2) == 1 && ModelingToolkit.isequal(args2[1], gating_var))

                g_inf_expr = first_arg
                tau_expr = denominator

                # Substitute observed variables
                g_inf_expr = substitute_observed_recursive(g_inf_expr, obs_subs)
                tau_expr = substitute_observed_recursive(tau_expr, obs_subs)

                return tau_expr, g_inf_expr
            end
        end

        # Check if first argument is negative of gating variable: ((-g) + g_inf)
        if ModelingToolkit.Symbolics.istree(first_arg)
            op1 = ModelingToolkit.Symbolics.operation(first_arg)
            args1 = ModelingToolkit.Symbolics.arguments(first_arg)

            # Check for multiplication with -1 or unary negation
            if (op1 == (*) && length(args1) == 2 &&
                ((args1[1] == -1 && ModelingToolkit.isequal(args1[2], gating_var)) ||
                 (args1[2] == -1 && ModelingToolkit.isequal(args1[1], gating_var)))) ||
               (op1 == (-) && length(args1) == 1 && ModelingToolkit.isequal(args1[1], gating_var))

                g_inf_expr = second_arg
                tau_expr = denominator

                # Substitute observed variables
                g_inf_expr = substitute_observed_recursive(g_inf_expr, obs_subs)
                tau_expr = substitute_observed_recursive(tau_expr, obs_subs)

                return tau_expr, g_inf_expr
            end
        end
    end

    error("Could not extract tau and inf from expression for variable $(gating_var)")
end

"""
    extract_tau_inf_from_equation(eq, gating_var, obs_subs)

Extract τ and g_inf directly from the structure of a tau-type gate equation.

For a tau-type gate, the equation has the form:
    dg/dt = (g_inf - g) / tau_g

Or with conditional evolution:
    dg/dt = ifelse(condition, (g_inf - g) / tau_g, 0)

This function parses this structure to extract tau_g and g_inf expressions.

# Arguments
- `eq`: The differential equation for the gating variable
- `gating_var`: The gating variable (e.g., m, h, n)
- `obs_subs`: Dictionary mapping observed variables to their expressions

# Returns
- `(tau_expr, inf_expr)`: The expressions for τ and steady-state
"""
function extract_tau_inf_from_equation(eq, gating_var, obs_subs)
    rhs = eq.rhs

    # Check for ifelse pattern: ifelse(condition, (g_inf - g) / tau, 0)
    # This represents a gating variable that only evolves under certain conditions
    if ModelingToolkit.Symbolics.istree(rhs)
        op = ModelingToolkit.Symbolics.operation(rhs)
        args = ModelingToolkit.Symbolics.arguments(rhs)

        # Handle ifelse at the top level
        if op == ifelse && length(args) == 3
            condition = args[1]
            true_branch = args[2]
            false_branch = args[3]

            # Check if false branch is 0 (no evolution)
            if false_branch == 0 || false_branch == 0.0
                # Extract tau and inf from the true branch
                tau_expr, inf_expr = extract_tau_inf_from_rhs(true_branch, gating_var, obs_subs)

                # Substitute observed variables in the condition
                condition_substituted = substitute_observed_recursive(condition, obs_subs)

                # Wrap them in ifelse to handle the condition
                # When condition is false: tau = Inf (no change), inf = current value (g)
                tau_expr_wrapped = ModelingToolkit.Symbolics.term(ifelse, condition_substituted, tau_expr, Inf)
                inf_expr_wrapped = ModelingToolkit.Symbolics.term(ifelse, condition_substituted, inf_expr, gating_var)

                return tau_expr_wrapped, inf_expr_wrapped
            end
        end
    end

    # Standard case: (g_inf - g) / tau
    # Use the helper function to extract
    return extract_tau_inf_from_rhs(rhs, gating_var, obs_subs)
end

"""
    substitute_observed_recursive(expr, obs_subs)

Recursively substitute observed variables until no more substitutions are possible.
"""
function substitute_observed_recursive(expr, obs_subs)
    if isempty(obs_subs)
        return expr
    end
    prev_expr = expr
    max_iterations = 100
    for _ in 1:max_iterations
        new_expr = ModelingToolkit.substitute(prev_expr, obs_subs)
        if isequal(new_expr, prev_expr)
            break
        end
        prev_expr = new_expr
    end
    return prev_expr
end

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
function extract_rate_constants(eq::ModelingToolkit.Symbolics.Equation, var)
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
function extract_gating_rate_functions(sys::ModelingToolkit.System)
    states = ModelingToolkit.unknowns(sys)

    # Get observed equations and create substitution dictionary
    observed = ModelingToolkit.observed(sys)
    obs_subs = Dict(eq.lhs => eq.rhs for eq in observed)

    rate_functions = Dict{Int,Tuple{Function,Function}}()

    for (i, state) in enumerate(states)
        if !is_gating_variable(state)
            continue
        end

        # Extract rate constants symbolically from the α and β equations
        α_expr, β_expr = extract_alpha_beta_equations(state, obs_subs)

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

"""
    RushLarsenSystem

Wrapper that combines a ModelingToolkit ODESystem with a RushLarsenFunction
to enable symbolic indexing for parameters and initial conditions.

# Fields
- `sys::System`: The original ModelingToolkit system
- `rlf::RushLarsenFunction`: The compiled Rush-Larsen function
"""
struct RushLarsenSystem
    sys::ModelingToolkit.System
    rlf::RushLarsenFunction
end

"""
    RushLarsenSystem(sys::System; simplify=true, cse=true)

Create a RushLarsenSystem from a ModelingToolkit ODESystem.
Automatically constructs the RushLarsenFunction internally.

# Keyword Arguments
- `simplify::Bool=true`: Whether to simplify symbolic expressions before code generation
- `cse::Bool=true`: Whether to use common subexpression elimination in code generation
"""
function RushLarsenSolvers.RushLarsenSystem(sys::ModelingToolkit.System; simplify::Bool=true, cse::Bool=true)
    rlf = RushLarsenFunction(sys; simplify=simplify, cse=cse)
    return RushLarsenSystem(sys, rlf)
end

# ============================================================================
# SymbolicIndexingInterface Implementation
# ============================================================================

# Symbol Classification - Variables (States)
function SymbolicIndexingInterface.is_variable(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.is_variable(sys.sys, sym)
end

function SymbolicIndexingInterface.variable_index(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.variable_index(sys.sys, sym)
end

function SymbolicIndexingInterface.variable_symbols(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.variable_symbols(sys.sys)
end

# Symbol Classification - Parameters
function SymbolicIndexingInterface.is_parameter(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.is_parameter(sys.sys, sym)
end

function SymbolicIndexingInterface.parameter_index(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.parameter_index(sys.sys, sym)
end

function SymbolicIndexingInterface.parameter_symbols(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.parameter_symbols(sys.sys)
end

# Symbol Classification - Independent Variables
function SymbolicIndexingInterface.is_independent_variable(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.is_independent_variable(sys.sys, sym)
end

function SymbolicIndexingInterface.independent_variable_symbols(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.independent_variable_symbols(sys.sys)
end

function SymbolicIndexingInterface.is_time_dependent(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.is_time_dependent(sys.sys)
end

# System Metadata
function SymbolicIndexingInterface.constant_structure(sys::RushLarsenSystem)
    return true  # RushLarsen systems have constant structure
end

function SymbolicIndexingInterface.all_variable_symbols(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.all_variable_symbols(sys.sys)
end

function SymbolicIndexingInterface.all_symbols(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.all_symbols(sys.sys)
end

function SymbolicIndexingInterface.default_values(sys::RushLarsenSystem)
    return SymbolicIndexingInterface.default_values(sys.sys)
end

# Observed Equations
function SymbolicIndexingInterface.is_observed(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.is_observed(sys.sys, sym)
end

function SymbolicIndexingInterface.observed(sys::RushLarsenSystem, sym)
    return SymbolicIndexingInterface.observed(sys.sys, sym)
end

# ============================================================================
# Helper Functions for Creating Problems
# ============================================================================

"""
    process_u0(sys::RushLarsenSystem, u0)

Convert symbolic u0 (Vector of Pairs) to numeric vector in correct order.

# Examples
```julia
sys = RushLarsenSystem(hh_model)
u0_symbolic = [hh_model.V => -65.0, hh_model.m => 0.05, hh_model.h => 0.6, hh_model.n => 0.318]
u0_numeric = process_u0(sys, u0_symbolic)
```
"""
function process_u0(sys::RushLarsenSystem, u0)
    # Handle empty array - use all defaults
    if isempty(u0)
        states = ModelingToolkit.unknowns(sys.sys)
        defaults = SymbolicIndexingInterface.default_values(sys)
        u0_numeric = zeros(length(states))

        for (i, state) in enumerate(states)
            if haskey(defaults, state)
                u0_numeric[i] = defaults[state]
            else
                error("No default value available for state $state")
            end
        end

        return u0_numeric
    elseif u0 isa AbstractVector{<:Pair}
        # Get the state order from the system
        states = ModelingToolkit.unknowns(sys.sys)
        u0_numeric = zeros(length(states))

        # Create a dictionary from the pairs
        u0_dict = Dict(u0)

        # Fill in values in the correct order
        for (i, state) in enumerate(states)
            if haskey(u0_dict, state)
                u0_numeric[i] = u0_dict[state]
            else
                # Try to get default value
                defaults = SymbolicIndexingInterface.default_values(sys)
                if haskey(defaults, state)
                    u0_numeric[i] = defaults[state]
                else
                    error("No initial condition provided for state $state and no default value available")
                end
            end
        end

        return u0_numeric
    else
        # Already numeric
        return u0
    end
end

"""
    process_p(sys::RushLarsenSystem, p)

Convert symbolic parameters (Vector of Pairs) to numeric vector in correct order.

# Examples
```julia
sys = RushLarsenSystem(hh_model)
p_symbolic = [hh_model.C_m => 1.0, hh_model.g_Na => 120.0, ...]
p_numeric = process_p(sys, p_symbolic)
```
"""
function process_p(sys::RushLarsenSystem, p)
    # Handle empty array - use all defaults
    if isempty(p)
        params = ModelingToolkit.parameters(sys.sys)
        defaults = SymbolicIndexingInterface.default_values(sys)
        p_numeric = zeros(length(params))

        for (i, param) in enumerate(params)
            if haskey(defaults, param)
                p_numeric[i] = defaults[param]
            else
                error("No default value available for parameter $param")
            end
        end

        return p_numeric
    elseif p isa AbstractVector{<:Pair}
        # Get the parameter order from the system
        params = ModelingToolkit.parameters(sys.sys)
        p_numeric = zeros(length(params))

        # Create a dictionary from the pairs
        p_dict = Dict(p)

        # Fill in values in the correct order
        for (i, param) in enumerate(params)
            if haskey(p_dict, param)
                p_numeric[i] = p_dict[param]
            else
                # Try to get default value
                defaults = SymbolicIndexingInterface.default_values(sys)
                if haskey(defaults, param)
                    p_numeric[i] = defaults[param]
                else
                    error("No value provided for parameter $param and no default value available")
                end
            end
        end

        return p_numeric
    else
        # Already numeric
        return p
    end
end

"""
    RushLarsenProblem(sys::RushLarsenSystem, u0, tspan, p=nothing)

Create an ODEProblem with the RushLarsenFunction, from a RushLarsenSystem. Supports symbolic u0 and p.

# Examples
```julia
sys = RushLarsenSystem(hh_model)

# Symbolic specification
u0 = [hh_model.V => -65.0, hh_model.m => 0.05, hh_model.h => 0.6, hh_model.n => 0.318]
p = [hh_model.C_m => 1.0, hh_model.g_Na => 120.0, ...]
prob = RushLarsenProblem(sys, u0, (0.0, 100.0), p)

# Or use defaults for missing values
u0 = [hh_model.V => -65.0]  # Other states use defaults
prob = RushLarsenProblem(sys, u0, (0.0, 100.0))
```
"""
function RushLarsenSolvers.RushLarsenProblem(sys::RushLarsenSystem, u0, tspan, p=nothing)
    u0_numeric = process_u0(sys, u0)

    if p === nothing
        # Use all default values
        defaults = SymbolicIndexingInterface.default_values(sys)
        params = ModelingToolkit.parameters(sys.sys)
        p_numeric = [defaults[param] for param in params]
    else
        p_numeric = process_p(sys, p)
    end

    return ODEProblem(sys.rlf, u0_numeric, tspan, p_numeric)
end

end # module