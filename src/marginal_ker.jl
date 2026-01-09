#defining a DPP using a marginal kernel

mutable struct MarginalDPP{T}
    K::AbstractMatrix{T}
    U::Matrix{T}
    λ::Vector{T}
    n::Int64
    m::Int64
end

"""
MarginalDPP(V::Matrix{T})

Construct a DPP from a matrix defining the marginal kernel. Here the matrix must be square and its eigenvalues must be between 0 and 1.
"""
function MarginalDPP(V::AbstractMatrix{T}) where {T}
    K = V
    @assert size(K, 1) == size(K, 2) "Kernel must be square"
    eg = eigen(K)
    U = eg.vectors
    λ = max.(eg.values, eps(T))
    @assert maximum(λ) <= 1.0 "Eigenvalues need to be less than or equal to 1"
    n = size(K, 1)
    return MarginalDPP{T}(K, U, λ, n, length(λ))
end



function show(io::IO, e::MarginalDPP)
    println(io, "DPP with marginal kernel representation.")
    return println(io, "Number of items in ground set : $(nitems(e)).")
end

#MarginalDPP(V::AbstractMatrix{T}) where {T} = MarginalDPP{T}(V)



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

#Compute the log-likelihood of an outcome
#Implementation is not particularly efficient and has cubic cost.
function log_prob(M::MarginalDPP,ind)
    length(ind) > 0 && @assert all([i ∈ 1:nitems(M) for i in ind])
    if length(ind) == 0
        return sum(log.(1 .- M.λ))
    elseif length(ind) > M.m
        return -Inf
    elseif length(ind) == nitems(M)
        return sum(log.(M.λ))
    else
        B = setdiff(1:nitems(M),ind)
        A = ind
        K = M.K
        C = cholesky(Symmetric(K[A, A]),check=false)
        !issuccess(C) && return -Inf
        S = Symmetric(I - (K[B,B] - K[B,A]*(C \ K[A,B] )))
        CS = cholesky(S,check=false)
        !issuccess(CS) && return -Inf
        return logdet(C) + logdet(CS)
    end
end

#Moment-generating function - mostly of theoretical interest
function mgf(M::MarginalDPP,t :: Vector)
    b = exp.(t) .- 1
    det(I - Diagonal(b)*M)
end
