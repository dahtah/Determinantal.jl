#defining a DPP using a marginal kernel

mutable struct FullRankDPP{T} 
    K::Matrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64

    function FullRankDPP{T}(V::Matrix{T}) where T
        K = V
        @assert size(K,1) == size(K,2) "Kernel must be square"
        eg = eigen(K)
        U = eg.vectors
        λ = max.(eg.values,eps(T))
        @assert maximum(λ) <= 1.0 "Eigenvalues need to be less than or equal to 1"
        n = size(K,1);
        new(K,U,λ,n,n)
    end

end


function show(io::IO, e::FullRankDPP)
    println(io, "DPP with full-rank marginal kernel representation.")
    println(io,"Number of items in ground set : $(nitems(e)).")
end

"""
   FullRankDPP(V::Matrix{T})

Construct a full-rank DPP from a matrix defining the marginal kernel. Here the matrix must be square and its eigenvalues must be between 0 and 1.
"""
FullRankDPP(V::Matrix{T}) where T = FullRankDPP{T}(V)

function inclusion_prob(M::FullRankDPP)
    diag(M.K)
end

function marginal_kernel(M::FullRankDPP)
    M.K
end

function sample(M::FullRankDPP)
    incl = rand(M.m) .< M.λ
    sample_pdpp(M.U[:,incl])
end

function cardinal(M::FullRankDPP)
    p = M.λ
    (mean=sum(p),std=sqrt(sum(p.*(1 .- p))))
end
