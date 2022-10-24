#Some code for partial projection ensembles, of mostly theoretical interest for
#now.

mutable struct ExtEnsemble <: AbstractLEnsemble
    Lopt::AbstractLEnsemble
    Lproj::ProjectionEnsemble
    M::Matrix
    V::Matrix
    α::Float64
end

function ExtEnsemble(M::Matrix, V::Matrix)
    @assert (size(M, 1) == size(V, 1))
    @assert (size(M, 1) > size(V, 2))
    Lproj = ProjectionEnsemble(V)
    #Orthogonalise (inefficient)
    Morth = (I - Lproj.U * Lproj.U') * (M * (I - Lproj.U * Lproj.U'))
    Lopt = EllEnsemble((Morth + Morth') / 2)
    return ExtEnsemble(Lopt, Lproj, M, V, 1.0)
end

nitems(L::ExtEnsemble) = nitems(L.Lopt)
maxrank(L::ExtEnsemble) = min(nitems(L), maxrank(L.Lopt) + maxrank(L.Lproj))
min_items(L::ExtEnsemble) = maxrank(L.Lproj)
logz(L::ExtEnsemble) = logz(L.Lopt)
logz(L::ExtEnsemble, k) = (k > min_items(L) ? logz(L.Lopt, k - min_items(L)) : 0.0)
function log_prob(L::ExtEnsemble, ind, k::Int)
    m = min_items(L)
    @assert (length(ind) == k)
    if (k < m || m > maxrank(L))
        return -Inf
    else
        Laug = [
            L.α*L.M[ind, ind] L.Lproj.U[ind, :]
            L.Lproj.U[ind, :]' zeros(m, m)
        ]
        logabsdet(Laug)[1] - logz(L, k)
    end
end

function log_prob(L::ExtEnsemble, ind)
    m = min_items(L)
    k = length(ind)
    if (k < m || k > maxrank(L))
        return -Inf
    else
        Laug = [
            L.α*L.M[ind, ind] L.Lproj.U[ind, :]
            L.Lproj.U[ind, :]' zeros(m, m)
        ]
        logabsdet(Laug)[1] - logz(L)
    end
end

function show(io::IO, e::ExtEnsemble)
    println(io, "Partial projection DPP.")
    println(
        io,
        "Number of items in ground set : $(nitems(e)). Max. rank :
$(maxrank(e))",
    )
    return println(io, "Rank of projective part : $(maxrank(e.Lproj))")
end

function rescale!(L::ExtEnsemble, k)
    @assert min_items(L) < k <= maxrank(L)
    return L.α = rescale!(L.Lopt, k - min_items(L))
end

function sample(L::ExtEnsemble)
    r = maxrank(L.Lproj)
    λ = eigenvalues(L.Lopt)
    val = @. λ / (1 + λ)
    ii = rand(length(val)) .< val
    return sample_pdpp([L.Lproj.U L.Lopt.U[:, ii]])
end

function sample(L::ExtEnsemble, k)
    @assert(k >= maxrank(L.Lproj))
    r = maxrank(L.Lproj)
    if (k > r)
        ii = collect(sample_diag_kdpp(L.Lopt, k - r))
        sample_pdpp([L.Lproj.U L.Lopt.U[:, ii]])
    else
        sample_pdpp(L.Lproj.U)
    end
end

function inclusion_prob(L::ExtEnsemble, k)
    λ = eigenvalues(L.Lopt)
    @assert k >= maxrank(L.Lproj)
    m = k - maxrank(L.Lproj)
    if (m == 0)
        sum((L.Lproj.U) .^ 2; dims=2)
    else
        val = inclusion_prob_diag(λ, m)
        val[val .< 0] .= 0
        val[val .> 1] .= 1
        val = (val ./ sum(val)) .* m
        return sum((L.Lopt.U * Diagonal(sqrt.(val))) .^ 2; dims=2) +
               sum((L.Lproj.U) .^ 2; dims=2)
    end
end

function marginal_kernel(L::ExtEnsemble)
    return marginal_kernel(L.Lopt) + marginal_kernel(L.Lproj)
end
