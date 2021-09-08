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
        n,m = size(M);
        eg = eigen(M'*M)
        U = M*eg.vectors
        U = U ./ sqrt.(sum(U .^ 2,dims=1))
        λ = eg.values
        keep = (abs.(λ)  .> 10*eps(T)) .& (λ .> 0)
        m_num = sum(keep)
        if (m_num < m)
            @warn "Numerical rank is lower than number of matrix columns"
            U = U[:,keep]
            λ = λ[keep]
            m = m_num
        end
        new(M,U,λ,n,m,T(1.0))
    end
end

@doc raw"""
   LowRankEnsemble(V::Matrix{T})

Construct a low-rank ensemble from a matrix of features. Here we assume 
``\mathbf{L} = \mathbf{V}\mathbf{V}^t``, so that V must be n \times r, where n is the number of items and r is the rank of the L-ensemble.

You will not be able to sample a number of items greater than the rank. At construction, an eigenvalue decomposition of V'*V will be perfomed, with cost nr^2. 

"""
LowRankEnsemble(V::Matrix{T}) where T = LowRankEnsemble{T}(V)

function show(io::IO, e::LowRankEnsemble)
    println(io, "L-Ensemble with low-rank representation.")
    println(io,"Number of items in ground set : $(nitems(e)). Max. rank : $(maxrank(e)). Rescaling constant α=$(round(e.α,digits=3))")
end

function diag(L::LowRankEnsemble)
    vec(sum(L.M.^2,dims=2))
end
function getindex(L::LowRankEnsemble,i1,i2)
    A=getindex(L.M,i1,:)
    B=getindex(L.M,i2,:)
    L.α*(A*B')
end
function getindex(L::LowRankEnsemble,i1,i2 :: Int)
    A=getindex(L.M,i1,:)
    B=getindex(L.M,i2,:)
    L.α*(A*B)
end

function getindex(L::LowRankEnsemble,i1 :: Int,i2 :: Int)
    A=getindex(L.M,i1,:)
    B=getindex(L.M,i2,:)
    L.α*dot(A,B)
end


function getindex(L::LowRankEnsemble,i1 :: Int,i2)
    A=getindex(L.M,i1,:)
    B=getindex(L.M,i2,:)
    L.α*Matrix(A'*B')
end
