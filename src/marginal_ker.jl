#defining a DPP using a marginal kernel

mutable struct MarginalDPP{T}
    K::AbstractMatrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64

    function MarginalDPP{T}(V::AbstractMatrix{T}) where {T}
        K = V
        @assert size(K, 1) == size(K, 2) "Kernel must be square"
        eg = eigen(K)
        U = eg.vectors
        λ = max.(eg.values, eps(T))
        @assert maximum(λ) <= 1.0 "Eigenvalues need to be less than or equal to 1"
        n = size(K, 1)
        return new(K, U, λ, n, length(λ))
    end
end

function show(io::IO, e::MarginalDPP)
    println(io, "DPP with marginal kernel representation.")
    return println(io, "Number of items in ground set : $(nitems(e)).")
end

"""
MarginalDPP(V::Matrix{T})

Construct a DPP from a matrix defining the marginal kernel. Here the matrix must be square and its eigenvalues must be between 0 and 1.
"""
MarginalDPP(V::AbstractMatrix{T}) where {T} = MarginalDPP{T}(V)

function inclusion_prob(M::MarginalDPP)
    return diag(M.K)
end

function marginal_kernel(M::MarginalDPP)
    return M.K
end

function sample(M::MarginalDPP)
    incl = rand(M.m) .< M.λ
    if (sum(incl) > 0)
        sample_pdpp(M.U[:, incl])
    else
        Vector{Int64}()
    end
end

function cardinal(M::MarginalDPP)
    p = M.λ
    return (mean=sum(p), std=sqrt(sum(p .* (1 .- p))))
end
