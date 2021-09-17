struct LowRank{T} <: AbstractMatrix{T}
    A::Matrix{T}
    n::Int
    p::Int
end

function LowRank(A)
    @assert size(A, 2) <= size(A, 1)
    LowRank(A, size(A, 1), size(A, 2))
end

function Base.getindex(K::LowRank, i::Int, j::Int)
    dot(K.A[i, :], (K.A')[:, j])
end

function Base.getindex(K::LowRank, i::AbstractVector, j::AbstractVector)
    K.A[i, :] * (K.A')[:, j]
end

function Base.getindex(K::LowRank, i::Int, j::AbstractVector)
    vec(K.A[i, :]' * (K.A')[:, j])
end

function Base.getindex(K::LowRank, i::AbstractVector, j::Int)
    K.A[i, :]' * (K.A')[:, j]
end

function Base.Matrix(K::LowRank)
    K.A * K.A'
end



function LinearAlgebra.eigen(K::LowRank)
    eg = eigen(K.A' * K.A)
    U = K.A * eg.vectors
    U = U ./ sqrt.(sum(U .^ 2, dims = 1))
    (vectors = U, values = eg.values)
end

function Base.size(K::LowRank)
    (K.n, K.n)
end
