abstract type AbstractLEnsemble end

"""
   FullRankEnsemble{T}

This type represents an L-ensemble where the matrix L is full rank. This is the most general representation of an L-ensemble, but also the least efficient, both in terms of memory and computation.

At construction, an eigenvalue decomposition of L will be performed, at O(n^3) cost.

The type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats). 
"""
mutable struct FullRankEnsemble{T} <: AbstractLEnsemble
    L::Matrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
    α::T

    function FullRankEnsemble{T}(V::Matrix{T}) where T
        L = V
        @assert size(L,1) == size(L,2)
        eg = eigen(L)
        U = eg.vectors
        λ = max.(eg.values,eps(T))
        n = size(L,1);
        new(L,U,λ,n,n,T(1.0))
    end
    function FullRankEnsemble{T}(X::Matrix{T},k :: Kernel) where T
        V = kernelmatrix(Val(:col),k,X)
        FullRankEnsemble{T}(V)
    end
end

@doc raw"""
This type represents an L-ensemble where the matrix L is low rank. This enables faster computation. 

The type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats)
"""
mutable struct LowRankEnsemble{T} <: AbstractLEnsemble
    M::Matrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
    α::T

    function LowRankEnsemble{T}(M::Matrix{T}) where T
        @assert size(M,1) >= size(M,2)
        eg = eigen(M'*M)
        U = M*eg.vectors
        U = U ./ sqrt.(sum(U .^ 2,dims=1))
        λ = eg.values
        n,m = size(M);
        new(M,U,λ,n,m,T(1.0))
    end
end

mutable struct ProjectionEnsemble{T} <: AbstractLEnsemble
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
    α::T
    function ProjectionEnsemble{T}(M::Matrix{T},orth=true) where T
        n,m = size(M);
        @assert size(M,1) >= size(M,2)
        if (orth)
            U=M
        else
            U=Matrix(qr(M).Q)
        end
        λ = ones(m)
        new(U,λ,n,m,T(1.0))
    end
end

function show(io::IO, e::FullRankEnsemble)
    println(io, "L-Ensemble with full-rank representation.")
    println(io,"Number of points in ground set : $(e.n). Rescaling constant α=$(round(e.α,digits=3))")
end

function show(io::IO, e::LowRankEnsemble)
    println(io, "L-Ensemble with low-rank representation.")
    println(io,"Number of points in ground set : $(e.n). Rank : $(e.m). Rescaling constant α=$(round(e.α,digits=3))")
end

function show(io::IO, e::ProjectionEnsemble)
    println(io, "Projection DPP.")
    println(io,"Number of points in ground set : $(e.n). Rank : $(e.m).")
end

"""
   FullRankEnsemble(V::Matrix{T})

Construct a full-rank ensemble from a matrix. Here the matrix must be square. 
"""
FullRankEnsemble(V::Matrix{T}) where T = FullRankEnsemble{T}(V)


"""
   FullRankEnsemble(X::Matrix{T},k :: Kernel)

Construct a full-rank ensemble from a set of points and a kernel function.

X (the set of points) is assumed to have dimension d x n, where d is the dimension and n is the number of points.
k is a kernel (see doc for package MLKernels)

Example: points in 2d along the circle, and an exponential kernel
```
t = LinRange(-pi,pi,10)'
X = vcat(cos.(t),sin.(t))
L=FullRankEnsemble(X,ExponentialKernel(.1))
```

"""
FullRankEnsemble(X::Matrix{T},k :: Kernel) where T = FullRankEnsemble{T}(X,k)


@doc raw"""
   LowRankEnsemble(V::Matrix{T})

Construct a low-rank ensemble from a matrix of features. Here we assume 
``\mathbf{L} = \mathbf{V}\mathbf{V}^t``, so that V must be n \times r, where n is the number of items and r is the rank of the L-ensemble.

You will not be able to sample a number of items greater than the rank. At construction, an eigenvalue decomposition of V'*V will be perfomed, with cost nr^2. 

"""
LowRankEnsemble(V::Matrix{T}) where T = LowRankEnsemble{T}(V)

@doc raw"""
   ProjectionEnsemble(V::Matrix{T},orth=true)

Construct a projection ensemble from a matrix of features. Here we assume 
``\mathbf{L} = \mathbf{V}\mathbf{V}^t``, so that V must be n \times r, where n is the number of items and r is the rank.
V needs not be orthogonal. If orth is set to true (default), then a QR decomposition is performed. If V is orthogonal already, then this computation may be skipped, and you can set orth to false. 
"""
ProjectionEnsemble(V::Matrix{T},orth=true) where T = ProjectionEnsemble{T}(V,orth)

@doc raw"""
   rescale!(L,k)

``\DeclareMathOperator{\Tr}{Tr}``

Rescale the L-ensemble such that the expected number of samples equals k.
The expected number of samples of a DPP equals ``\Tr \mathbf{L}\left( \mathbf{L} + \mathbf{I} \right)``. The function rescales ``\mathbf{L}`` to ``\alpha \mathbf{L}`` such that ``\Tr \alpha \mathbf{L}\left( \alpha \mathbf{L} + \mathbf{I} \right) = k``
"""
function rescale! end

function rescale!(L::AbstractLEnsemble,k)
    @assert 0 < k <= L.m
    L.α = solve_sp(L.λ,k);
end

function rescale!(L::ProjectionEnsemble,k)
    throw(ArgumentError("Cannot rescale a projection DPP: the set size is fixed and equals the rank."))
end


function marginal_kernel(L::AbstractLEnsemble)
    val = L.α*L.λ ./ (1 .+ L.α*L.λ)
    L.U*Diagonal(val)*L.U'
end

function marginal_kernel(L::ProjectionEnsemble)
    L.U*L.U'
end

"""
     sample(L::AbstractEnsemble)

Sample from a DPP with L-ensemble L. The return type is a BitSet (indicating which indices are sampled), use collect to get a vector of indices instead.
"""
function sample(L::ProjectionEnsemble)
    sample_pdpp(L.U)
end

function sample(L::AbstractLEnsemble)
    val = L.α*L.λ ./ (1 .+ L.α*L.λ)
    incl = rand(L.m) .< val
    sample_pdpp(L.U[:,incl])
end
