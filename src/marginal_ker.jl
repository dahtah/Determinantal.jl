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

Construct a DPP from a matrix defining the marginal kernel. Here the matrix must be square and its eigenvalues must be between 0 and 1. All eigenvalues above 1 will be clipped and a warning issued.


"""
function MarginalDPP(V::AbstractMatrix{T}) where {T}
    K = V
    @assert size(K, 1) == size(K, 2) "Kernel must be square"
    eg = eigen(K)
    U = eg.vectors

    λ = max.(eg.values, eps(T))
    if maximum(λ) > 1.0
        @warn "Clipping eigenvalues > 1 to 1"
        λ[λ .> 1] .= 1.0
        if isa(K, LowRank)
            K.A .= U*Diagonal(sqrt.(λ))
        else
            K .= Symmetric(U*Diagonal(λ)*U')
        end
    end
    n = size(K, 1)
    return MarginalDPP{T}(K, U, λ, n, length(λ))
end


function show(io::IO, e::MarginalDPP)
    println(io, "DPP with marginal kernel representation.")
    return println(io, "Number of items in ground set : $(nitems(e)).")
end

#MarginalDPP(V::AbstractMatrix{T}) where {T} = MarginalDPP{T}(V)


#The void probability is the probability that none of the items in ind
#are selected.
function void_prob(M::MarginalDPP, ind)
    det(I - M.K[ind, ind])
end

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

function size_statistics(M::MarginalDPP)
    p = M.λ
    return (mean = sum(p), std = sqrt(sum(p .* (1 .- p))))
end

@doc raw"""
    log_prob(M::MarginalDPP, ind)

Compute the log-likelihood of a subset `ind` under the marginal DPP `M`, i.e.
$\log p(\X = ind)$, where $\X$ is the DPP with marginal kernel L.

# Arguments
- `M::MarginalDPP`: A marginal Determinantal Point Process.
- `ind`: A subset of item indices (1-based) to evaluate.

# Returns
- The log-probability of the subset `ind` under `M`. Returns `-Inf` for invalid subsets.

# Notes
- The implementation has cubic cost in the size of `ind`.
- Returns `-Inf` if `ind` is larger than `M.m` or if the Cholesky decomposition fails.
"""
function log_prob(M::MarginalDPP, ind)
    length(ind) > 0 && @assert all(0 .< ind .<= nitems(M))
    if length(ind) == 0
        return sum(log.(1 .- M.λ))
    elseif length(ind) > maxrank(M)
        return -Inf
    elseif length(ind) == nitems(M)
        return sum(log.(M.λ))
    else
        B = setdiff(1:nitems(M), ind)
        A = ind
        K = M.K
        C = cholesky(Symmetric(K[A, A]), check = false)
        !issuccess(C) && return -Inf
        S = Symmetric(I - (K[B, B] - K[B, A] * (C \ K[A, B])))
        CS = cholesky(S, check = false)
        !issuccess(CS) && return -Inf
        return logdet(C) + logdet(CS)
    end
end

#Moment-generating function - mostly of theoretical interest


raw"""
    mgf(DPP, t)

Compute the moment generating function of a DPP evaluated at $t$, defined as
$ m(\mathbf{t}) = \mathrm{E}\left( \exp( \mathbf{t}^\top \mathbf{z}) \right) $
where $\mathbf{z}$ is the indicator vector of the DPP ($z_i=1$ if item $i$ is included, $0$ otherwise).
If $t$ is a scalar, then:
$ m(t) = \mathrm{E}\left( \exp( t \sum z_i \right) $

# Arguments
- `DPP`: A DPP (either an L-ensemble, a projection DPP, or a marginal DPP)
- `t`: Either a vector of length equal to the number of items in `M`, or a scalar

"""
function mgf(M::MarginalDPP, t::Vector)
    @assert length(t) == nitems(M)
    b = 1 .- exp.(t)
    det(I - Diagonal(b) * M.K)
end

function mgf(M::MarginalDPP, t::Number)
    b = 1-exp(t)
    prod(1 .- b * M.λ)
end
