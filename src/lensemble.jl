abstract type AbstractLEnsemble end
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


FullRankEnsemble(V::Matrix{T}) where T = FullRankEnsemble{T}(V)
FullRankEnsemble(X::Matrix{T},k :: Kernel) where T = FullRankEnsemble{T}(X,k)
LowRankEnsemble(V::Matrix{T}) where T = LowRankEnsemble{T}(V)
ProjectionEnsemble(V::Matrix{T},orth=true) where T = ProjectionEnsemble{T}(V,orth)

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


function sample(L::ProjectionEnsemble)
    sample_pdpp(L.U)
end

function sample(L::AbstractLEnsemble)
    val = L.α*L.λ ./ (1 .+ L.α*L.λ)
    incl = rand(L.m) .< val
#    vec = L*eg.vectors[:,incl]

    sample_pdpp(L.U[:,incl])
end
