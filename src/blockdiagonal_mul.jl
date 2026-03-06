# Optimized block-wise multiplication for BlockDiagonal matrices
# These methods exploit the sparsity structure for efficient computation

using BlockDiagonals
using LinearAlgebra

# TYPE PIRACY
function LinearAlgebra.mul!(y::AbstractVector, A::BlockDiagonal, x::AbstractVector)
    @assert length(y) == size(A, 1) "Output vector size mismatch"
    @assert length(x) == size(A, 2) "Input vector size mismatch"

    idx = 1
    for block in blocks(A)
        n = size(block, 1)

        # Extract subvector for this block
        x_block = view(x, idx:idx+n-1)
        y_block = view(y, idx:idx+n-1)

        # Multiply this block only
        mul!(y_block, block, x_block)

        idx += n
    end

    return y
end

