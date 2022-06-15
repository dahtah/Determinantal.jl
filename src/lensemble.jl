abstract type AbstractLEnsemble end


"""
   EllEnsemble{T}

This type represents an L-ensemble, a (broad) special case of Determinantal Point Process.

The type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats).


"""
mutable struct EllEnsemble{T} <: AbstractLEnsemble
    L::AbstractMatrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
    α::T

    function EllEnsemble{T}(V::AbstractMatrix{T}) where {T}
        L = V
        @assert size(L, 1) == size(L, 2)
        eg = eigen(L)
        U = eg.vectors
        λ = max.(eg.values, eps(T))
        n = size(L, 1)
        new(L, U, λ, n, length(λ), T(1.0))
    end
    # function FullRankEnsemble{T}(X::AbstractVector{T},k :: Kernel) where T
    #     V = kernelmatrix(Val(:col),k,X)
    #     FullRankEnsemble{T}(V)
    # end

end

"""
    EllEnsemble(L :: AbstractMatrix{T})

Construct an L-ensemble from a (symmetric, non-negative definite) matrix L.

```@example
    Z = randn(5,2)
    EllEnsemble(Z*Z') #not very useful, presumably
```

Note that eigen(L) will be called at construction, which may be computationally costly.

An L-Ensemble can also be constructed based on lazy matrix types, i.e. types that leverage a low-rank representation. In the example above we could also use the LowRank type:

```@example
    Z = randn(5,2)
    EllEnsemble(LowRank(Z)) #more efficient than Z*Z'
```

"""
function EllEnsemble(L::AbstractMatrix{T}) where {T}
    EllEnsemble{T}(L)
end

"""
   EllEnsemble(X::Matrix{T},k :: Kernel)

Construct a full-rank ensemble from a set of points and a kernel function.

X is vector of column vectors (ColVecs) or a vector of row vectors (RowVecs)
k is a kernel (see doc for package KernelFunctions.jl)

Example: points in 2d along the circle, and an exponential kernel
```
t = LinRange(-pi,pi,10)'
X = vcat(cos.(t),sin.(t))
using KernelFunctions
L=EllEnsemble(ColVecs(X),ExponentialKernel())
```

"""
function EllEnsemble(X::AbstractVector, k::Kernel)
    V = kernelmatrix(k, X)
    EllEnsemble(V)
end

function show(io::IO, e::EllEnsemble)
    println(io, "L-Ensemble.")
    println(
        io,
        "Number of items in ground set : $(nitems(e)). Max. rank : $(maxrank(e)). Rescaling constant α=$(round(e.α,digits=3))",
    )
end







mutable struct ProjectionEnsemble{T} <: AbstractLEnsemble
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
    α::T
    function ProjectionEnsemble{T}(M::Matrix{T}, orth = true) where {T}
        n, m = size(M)
        @assert size(M, 1) >= size(M, 2)
        if (orth)
            U = Matrix(qr(M).Q)
        else
            U = M
        end
        λ = ones(m)
        new(U, λ, n, m, T(1.0))
    end
end


function show(io::IO, e::ProjectionEnsemble)
    println(io, "Projection DPP.")
    println(io, "Number of items in ground set : $(nitems(e)). Max. rank : $(maxrank(e)).")
end









@doc raw"""
   ProjectionEnsemble(V::Matrix{T},orth=true)

Construct a projection ensemble from a matrix of features. Here we assume
``\mathbf{L} = \mathbf{V}\mathbf{V}^t``, so that V must be n \times r, where n is the number of items and r is the rank.
V needs not be orthogonal. If orth is set to true (default), then a QR decomposition is performed. If V is orthogonal already, then this computation may be skipped, and you can set orth to false.
"""
ProjectionEnsemble(V::Matrix{T}, orth = true) where {T} = ProjectionEnsemble{T}(V, orth)

@doc raw"""
   rescale!(L,k)

``\DeclareMathOperator{\Tr}{Tr}``

Rescale the L-ensemble such that the expected number of samples equals k.
The expected number of samples of a DPP equals ``\Tr \mathbf{L}\left( \mathbf{L} + \mathbf{I} \right)``. The function rescales ``\mathbf{L}`` to ``\alpha \mathbf{L}`` such that ``\Tr \alpha \mathbf{L}\left( \alpha \mathbf{L} + \mathbf{I} \right) = k``
"""
function rescale! end

function rescale!(L::AbstractLEnsemble, k)
    @assert 0 < k <= L.m
    L.α = solve_sp(L.λ, k)
end

function rescale!(L::ProjectionEnsemble, k)
    throw(
        ArgumentError(
            "Cannot rescale a projection DPP: the set size is fixed and equals the rank.",
        ),
    )
end

"""
   inclusion_prob(L::AbstractLEnsemble)

Compute first-order inclusion probabilities, i.e. the probability that each item in 1..n is included in the DPP.

See also: marginal_kernel
"""
function inclusion_prob(L::AbstractLEnsemble)
    val = L.α * L.λ ./ (1 .+ L.α * L.λ)
    return vec(sum((L.U * Diagonal(sqrt.(val))) .^ 2, dims = 2))
end

function inclusion_prob(L::ProjectionEnsemble)
    return sum((L.U) .^ 2, dims = 2)
end

"""
     marginal_kernel(L::AbstractLEnsemble)

Compute and return the marginal kernel of a DPP, K. The marginal kernel of a DPP is a (n x n) matrix which can be used to find the inclusion probabilities. For any fixed set of indices ind, the probability that ind is included in a sample from the DPP equals det(K[ind,ind]).
"""
function marginal_kernel(L::AbstractLEnsemble)
    val = L.α * L.λ ./ (1 .+ L.α * L.λ)
    L.U * Diagonal(val) * L.U'
end

function marginal_kernel(L::ProjectionEnsemble)
    L.U * L.U'
end

"""
     sample(L::AbstractEnsemble)

Sample from a DPP with L-ensemble L. The return type is a BitSet (indicating which indices are sampled), use collect to get a vector of indices instead.
"""
function sample(L::ProjectionEnsemble)
    sample_pdpp(L.U)
end

function sample(L::AbstractLEnsemble)
    val = L.α * L.λ ./ (1 .+ L.α * L.λ)
    incl = rand(L.m) .< val
    sample_pdpp(L.U[:, incl])
end


"""
    cardinal(L::AbstractLEnsemble)

The size of the set sampled by a DPP is a random variable. This function returns its mean and standard deviation. See also: rescale!, which changes the mean set size.
"""
function cardinal(L::AbstractLEnsemble)
    p = L.α * L.λ ./ (1 .+ L.α * L.λ)
    (mean = sum(p), std = sqrt(sum(p .* (1 .- p))))
end

function diag(L::EllEnsemble)
    L.α * diag(L.L)
end

function diag(L::ProjectionEnsemble)
    vec(sum(L.U .^ 2, dims = 2))
end


function getindex(L::EllEnsemble, I...)
    L.α * getindex(L.L, I...)
end



function getindex(L::ProjectionEnsemble, i1, i2)
    A = getindex(L.U, i1, :)
    B = getindex(L.U, i2, :)
    L.α * A * B'
end

function getindex(L::ProjectionEnsemble, i1, i2::Int)
    A = getindex(L.U, i1, :)
    B = getindex(L.U, i2, :)
    L.α * A * B
end

function getindex(L::ProjectionEnsemble, i1::Int, i2::Int)
    A = getindex(L.U, i1, :)
    B = getindex(L.U, i2, :)
    L.α * dot(A, B)
end

function getindex(L::ProjectionEnsemble, i1::Int, i2)
    A = getindex(L.U, i1, :)
    B = getindex(L.U, i2, :)
    L.α * Matrix(A' * B')
end

function log_prob(L::ProjectionEnsemble, ind)
    if (length(ind) == maxrank(L))
        logdet(L[ind, ind]) - logz(L)
    else
        return -Inf
    end
end


function log_prob(L::AbstractLEnsemble, ind)
    if (length(ind) <= maxrank(L))
        logdet(L[ind, ind]) - logz(L)
    else
        return -Inf
    end
end

function log_prob(L::AbstractLEnsemble, ind, k::Int)
    if (length(ind) == k && k <= maxrank(L))
        logdet(L[ind, ind]) - logz(L, k)
    else
        return -Inf
    end
end

function logz(L::AbstractLEnsemble)
    sum(log1p.(eigenvalues(L)))
end

function logz(L::AbstractLEnsemble, k)
    log_esp_sp(eigenvalues(L), k)[end]
end

function logz(L::ProjectionEnsemble)
    0
end


for type in [:ProjectionEnsemble, :EllEnsemble, :MarginalDPP]
    eval(:(nitems(L::$type) = L.n))
end

for type in [:ProjectionEnsemble, :EllEnsemble, :MarginalDPP]
    eval(:(maxrank(L::$type) = L.m))
end

for type in [:ProjectionEnsemble, :EllEnsemble]
    eval(:(eigenvalues(L::$type) = L.α * L.λ))
end
