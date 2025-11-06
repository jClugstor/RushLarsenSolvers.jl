module RushLarsenSolversModelingToolkitExt

using RushLarsenSolvers
import RushLarsenSolvers: RushLarsenFunction
using ModelingToolkit

struct GatingVariable <: ModelingToolkit.Symbolics.AbstractVariableMetadata end

is_gating_variable(var) = ModelingToolkit.Symbolics.hasmetadata(var, GatingVariable) ? ModelingToolkit.Symbolics.getmetadata(var, GatingVariable) : false

# ============================================================================
# RushLarsenFunction Constructor from MTK System
# ============================================================================

"""
    RushLarsenFunction(sys::ODESystem)

Construct a RushLarsenFunction from a ModelingToolkit ODESystem.

This constructor automatically:
1. Identifies gating variables using the GatingVariable metadata
2. Extracts α and β rate constants for gating variables
3. Builds optimized functions using Symbolics.build_function
4. Separates gating and non-gating variable equations

# Arguments
- `sys::ODESystem`: A ModelingToolkit ODESystem with gating variables marked

# Returns
- `RushLarsenFunction`: A function object suitable for use with RushLarsen algorithm
"""
function RushLarsenSolvers.RushLarsenFunction(sys::ModelingToolkit.System)
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
    gating_eqs = []
    non_gating_eqs = []

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
                push!(non_gating_eqs, rhs_substituted)
            end
        end
    end

    # Build gating function that returns vector of (α, β) tuples
    gating_rate_exprs = []
    for (state, eq) in gating_eqs
        α_expr, β_expr = extract_alpha_beta_equations(state, obs_subs)
        # Substitute observed variables in α and β expressions
        α_expr = substitute_observed(α_expr)
        β_expr = substitute_observed(β_expr)
        push!(gating_rate_exprs, (α_expr, β_expr))
    end
    # Create gating function using Symbolics.build_function
    if isempty(gating_rate_exprs)
        gating_f = (gating_vars, u, p, t) -> nothing
    else
        # Flatten the (α, β) pairs into a single vector for build_function
        α_exprs = [pair[1] for pair in gating_rate_exprs]
        β_exprs = [pair[2] for pair in gating_rate_exprs]
        all_rate_exprs = [[exprs[1], exprs[2]] for exprs in zip(α_exprs, β_exprs)]
        # Build the function
        gating_func = Symbolics.build_function(
            all_rate_exprs,
            states,
            params,
            iv,
            expression=Val{false}
        )

        # Wrap to produce vector of (α, β) tuples
        n_gating = length(gating_rate_exprs)
        if gating_func isa Tuple
            # Use out-of-place version
            #base_func = gating_func[1]
            gating_f = (gating_vars, u, p, t) -> begin
                result = gating_func[1](u, p, t)
                for i in 1:n_gating
                    gating_vars[i] = (result[i][1], result[i][2])
                end
            end
        else
            gating_f = (gating_vars, u, p, t) -> begin
                result = gating_func[1](u, p, t)
                for i in 1:n_gating
                    gating_vars[i] = (result[i][1], result[i][2])
                end
            end
        end
    end

    # Build non-gating function using Symbolics.build_function
    if isempty(non_gating_eqs)
        non_gating_f = (du, u, p, t) -> nothing
    else
        non_gating_func = Symbolics.build_function(
            non_gating_eqs,
            states,
            params,
            iv,
            expression=Val{false}
        )

        # Wrap to write results to the correct indices in du
        if non_gating_func isa Tuple
            # Use out-of-place version and map to indices
            non_gating_f = (du, u, p, t) -> begin
                result = non_gating_func[1](u, p, t)
                for (i, idx) in enumerate(non_gating_idxs)
                    du[idx] = result[i]
                end
            end
        else
            # Out-of-place only version
            non_gating_f = (du, u, p, t) -> begin
                result = non_gating_func(u, p, t)
                for (i, idx) in enumerate(non_gating_idxs)
                    du[idx] = result[i]
                end
            end
        end
    end

    return RushLarsenSolvers.RushLarsenFunction(gating_f, non_gating_f, gating_idxs, non_gating_idxs)
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
function extract_rate_constants(eq::Equation, var)
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

end # module
