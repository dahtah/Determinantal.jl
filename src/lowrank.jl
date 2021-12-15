struct LowRank{T} <: AbstractMatrix{T}
    A::Matrix{T}
    n::Int
    p::Int
end

function LowRank(A)
    @assert size(A, 2) <= size(A, 1)
    return LowRank(A, size(A, 1), size(A, 2))
end

function Base.getindex(K::LowRank, i::Int, j::Int)
    return dot(K.A[i, :], (K.A')[:, j])
end

function Base.getindex(K::LowRank, i::AbstractVector, j::AbstractVector)
    return K.A[i, :] * (K.A')[:, j]
end

function Base.getindex(K::LowRank, i::Int, j::AbstractVector)
    return vec(K.A[i, :]' * (K.A')[:, j])
end

function Base.getindex(K::LowRank, i::AbstractVector, j::Int)
    return K.A[i, :]' * (K.A')[:, j]
end

function Base.Matrix(K::LowRank)
    return K.A * K.A'
end

function LinearAlgebra.eigen(K::LowRank)
    eg = eigen(K.A' * K.A)
    U = K.A * eg.vectors
    U = U ./ sqrt.(sum(U .^ 2; dims=1))
    return (vectors=U, values=eg.values)
end

function Base.size(K::LowRank)
    return (K.n, K.n)
end
