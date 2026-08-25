@doc raw"""
    match_prob(K::MarginalDPP, pr)

Compute a diagonal modification of a marginal DPP such that
the inclusion probabilities match the given target probabilities `pr`.

This simply sets a diagonal matrix ``D`` such that ``(DKD)_{i,i} = \pi_i``. The function returns a DPP with kernel ``DKD``.

```@example ex1
K=MarginalDPP(Diagonal([.5,.5,.5]))
inclusion_prob(match_prob(K,[.6,.7,.8]))
```
"""
function match_prob(K::MarginalDPP, pr)
    d = pr ./ inclusion_prob(K)
    D = Diagonal(sqrt.(d))
    if isa(K.K,LowRank)
        MarginalDPP(LowRank(D*K.K.A))
    else
        MarginalDPP(Symmetric(D * K.K * D))
    end
end


@doc raw"""
    match_prob(L :: AbstractLEnsemble, pr; thr=1e-6)

Compute a diagonal modification of an L-ensemble such that the inclusion probabilities match target probabilities.

Given an L-ensemble `L` and target probabilities `pr`, find a diagonal matrix `D` such that the L-ensemble ``DLD`` has inclusion probabilities `pr`.

Note that this does *not* give the same result as match_prob on the marginal representation of L.

# Arguments
- `L :: AbstractLEnsemble`: The input L-ensemble.
- `pr`: The target inclusion probabilities.

# Returns

If the solver is successful, returns a diagonally rescaled L-ensemble whose inclusion probabilities are as specified (up to tolerance).


"""
function match_prob(L :: AbstractLEnsemble, pr)
    @assert sum(pr) <= maxrank(L) "Sum of inclusion probabilities is larger than rank of L-ensemble, matching is impossible"
    ip = (v) -> compute_inc_prob(L.L, exp.(v))
    f = (v) -> ip(v) - pr
    res = nlsolve(f, ones(nitems(L)))
    if res.f_converged
        D=Diagonal(exp.(res.zero))
        if isa(L.L,LowRank)
            return EllEnsemble(LowRank(D*L.L.A))
        else
            return  EllEnsemble(Symmetric(D*L.L*D))
        end
    else
        #res
        error("Numerical minimisation failed.")
    end
end

#Compute diag(I-(D*L*D+I)^(-1))
function compute_inc_prob(L :: AbstractMatrix,d)
    D=Diagonal(d)
    C = cholesky(Symmetric(D*L*D+I))
    diag_inv = sum(abs2, inv(C.L), dims=1)[:]
    1 .- diag_inv
    # C = cholesky(L+Diagonal(d.^(-2)))
    # diag_inv = sum(abs2, inv(C.L), dims=1)[:]
    # 1 .- (d.^(-2)) .* diag_inv
 end

function compute_inc_prob(L :: LowRank,d)
    diag_inv = diag_inv_AAt_plus_D(L.A, d.^(-2))
    1 .- (d.^(-2)) .* diag_inv
end


#Compute the diagonal entries of the inverse of (A*A' + diag(d))
function diag_inv_AAt_plus_D(A, d)
    L_C = cholesky(Symmetric(I+A'*Diagonal(1 ./ d)*A)).L
    W = L_C \ A'          # BLAS triangular solve: W = L_C⁻¹ A'
    (d - sum(abs2, W, dims=1)[:]) ./ (d.^2)
end
