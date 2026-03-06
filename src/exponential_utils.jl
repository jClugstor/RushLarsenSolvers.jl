# ==================== Clean RL Phi Dispatch ====================

"""
    rl_phi(a, dt::Number, expmethod=nothing)

Compute phi(a*dt) with appropriate dispatch based on type of `a`.

This is the primary interface for computing phi in Rush-Larsen methods.

# Arguments
- `a`: Coefficient (can be Number, AbstractVector, AbstractMatrix, or BlockDiagonal)
- `dt`: Time step
- `expmethod`: ExponentialUtilities algorithm (e.g., ExpMethodHigham2005(), optional)

# Returns
Phi matrix/vector/scalar of appropriate type

# Dispatch behavior
- Scalar: Uses scalar phi function
- Vector: Element-wise phi on each component
- Diagonal matrix: Element-wise phi on diagonal elements
- General matrix: Uses ExponentialUtilities.jl's phiv_dense
- BlockDiagonal: Applies phi to each block independently (optimal for mixed HH/Markov models)
"""
# Scalar dispatch
rl_phi(a::Number, dt::Number, expmethod=nothing) = phi(a * dt, 1)

# Vector dispatch - element-wise phi
function rl_phi(a::AbstractVector, dt::Number, expmethod=nothing)
    result = similar(a)
    @inbounds @simd for i in eachindex(a)
        result[i] = phi(a[i] * dt, 1)
    end
    return result
end

# Matrix dispatch
function rl_phi(a::AbstractMatrix, dt::Number, expmethod=nothing)
    n = size(a, 1)
    T = eltype(a)

    # For diagonal matrices, compute element-wise
    if isdiag(a)
        result = zeros(T, n, n)
        @inbounds for i in 1:n
            result[i, i] = phi(a[i, i] * dt, 1)
        end
        return result
    else
        # General matrix case
        return _phi_matrix_general(a, dt, expmethod)
    end
end

# BlockDiagonal dispatch - applies phi to each block independently
function rl_phi(a::BlockDiagonal{T}, dt::Number, expmethod=nothing) where {T}
    phi_blocks = Matrix{T}[]
    for block in blocks(a)
        phi_block = _phi_single_block(block, dt, expmethod)
        push!(phi_blocks, phi_block)
    end
    return BlockDiagonal(phi_blocks)
end

# ==================== In-Place RL Phi Dispatch ====================

"""
    rl_phi!(result, a, dt::Number, expmethod=nothing)

In-place version of `rl_phi`. Computes phi(a*dt) and stores result in `result`.

# Arguments
- `result`: Pre-allocated storage for the result (must be same size/structure as `a`)
- `a`: Coefficient (can be AbstractVector, AbstractMatrix, or BlockDiagonal)
- `dt`: Time step
- `expmethod`: ExponentialUtilities algorithm (optional)

# Notes
- For scalars, use the out-of-place version `rl_phi(a, dt)`
- `result` must be pre-allocated and have the same structure as the output of `rl_phi(a, dt)`
"""
# Vector dispatch - in-place element-wise phi
function rl_phi!(result::AbstractVector, a::AbstractVector, dt::Number, expmethod=nothing)
    @assert length(result) == length(a) "result and a must have the same length"
    @inbounds @simd for i in eachindex(a, result)
        result[i] = phi(a[i] * dt, 1)
    end
    return nothing
end

# Matrix dispatch - in-place
function rl_phi!(result::AbstractMatrix, a::AbstractMatrix, dt::Number, expmethod=nothing)
    @assert size(result) == size(a) "result and a must have the same size"
    n = size(a, 1)

    # For diagonal matrices, compute element-wise
    if isdiag(a)
        fill!(result, zero(eltype(result)))
        @inbounds for i in 1:n
            result[i, i] = phi(a[i, i] * dt, 1)
        end
    else
        # General matrix case - compute then copy
        phi_mat = _phi_matrix_general(a, dt, expmethod)
        copyto!(result, phi_mat)
    end
    return nothing
end

# BlockDiagonal dispatch - in-place
function rl_phi!(result::BlockDiagonal, a::BlockDiagonal, dt::Number, expmethod=nothing)
    @assert length(blocks(result)) == length(blocks(a)) "result and a must have same number of blocks"

    for (i, block) in enumerate(blocks(a))
        result_block = blocks(result)[i]
        @assert size(result_block) == size(block) "Block $i size mismatch"

        n = size(block, 1)

        # Handle 1×1 blocks specially
        if n == 1
            result_block[1, 1] = phi(block[1, 1] * dt, 1)[2]
        else
            # For larger blocks, compute and copy
            phi_block = _phi_matrix_general(block, dt, expmethod)
            copyto!(result_block, phi_block)
        end
    end
    return nothing
end

# ==================== Helper Functions ====================

"""
    _phi_single_block(A::AbstractMatrix, dt::Number, expmethod)

Compute phi for a single block matrix.
Uses scalar phi for 1×1, diagonal phi for diagonal, ExponentialUtilities for general.
"""
function _phi_single_block(A::AbstractMatrix{T}, dt::Number, expmethod) where {T}
    n = size(A, 1)

    # Scalar case - phi returns [phi_0, phi_1], extract phi_1
    if n == 1
        phi_val = phi(A[1, 1] * dt, 1)[2]
        return reshape([phi_val], 1, 1)
    end

    # General case - use ExponentialUtilities (handles diagonal automatically)
    return _phi_matrix_general(A, dt, expmethod)
end

"""
    _phi_matrix_general(A::AbstractMatrix, dt::Number, expmethod)

Compute phi_1(A*dt) for a general matrix using ExponentialUtilities.jl.

Uses the phi function to compute phi_1(A*dt) = (exp(A*dt) - I) / (A*dt).
"""
function _phi_matrix_general(A::AbstractMatrix{T}, dt::Number, _expmethod) where {T}
    Adt = A * dt

    # Use phi function from ExponentialUtilities to compute phi_1(A*dt)
    # phi(A, k) returns [phi_0(A), phi_1(A), ..., phi_k(A)]
    # where phi_0(A) = exp(A) and phi_1(A) = (exp(A) - I) / A

    phi_matrices = ExponentialUtilities.phi(Adt, 1)

    # phi_matrices is a Vector of length 2: [phi_0(Adt), phi_1(Adt)]
    # We want phi_1(Adt) which is the second element
    return phi_matrices[2]
end

