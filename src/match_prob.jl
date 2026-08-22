
function match_prob(K::MarginalDPP,pr)
    d = pr ./ inclusion_prob(K)
    D = Diagonal(sqrt.(d))
    MarginalDPP(D*K.K*D)
end


#Compute a diagonal modification of an L-ensemble such that
#the inclusion probabilities match target probabilities
#Given an L-ensemble L, and target probabilities pr
#find a diagonal matrix D, such that the L-ensemble D*L*D
#has inclusion probabilities pr
#Right now this uses NLsolve and is not particularly efficient
#Should work out the Jacobian analytically
function match_prob_nl(L :: AbstractLEnsemble,pr;thr=1e-6)
    @assert sum(pr) < maxrank(L) "Sum of inclusion probabilities is larger than rank of L-ensemble, matching is impossible"
    ip = (v)-> compute_inc_prob(L.L,exp.(v))
    f = (v)->ip(v) - pr
    res=nlsolve(f,ones(nitems(L)))
    if res.f_converged
        return Diagonal(exp.(res.zero))
    else
        error("Numerical minimisation failed.")
    end
end

#Compute diag(I-(D*L*D+I)^(-1))
function compute_inc_prob(L :: Matrix,d)
    C = cholesky(L+Diagonal(d.^(-2)))
    diag_inv = sum(abs2, inv(C.L), dims=1)[:]
    1 .- (d.^(-2)) .* diag_inv
 end

function compute_inc_prob(L :: LowRank,d)
    diag_inv = diag_inv_AtA_plus_D(L.A, d.^(-2))
    1 .- (d.^(-2)) .* diag_inv
end


#Compute the diagonal entries of the inverse of (A'*A + diag(d))
function diag_inv_AtA_plus_D(A, d)
    L_C = cholesky(Symmetric(I+A'*Diagonal(1 ./ d)*A)).L
    W = L_C \ A'          # BLAS triangular solve: W = L_C⁻¹ A'
    (d - sum(abs2, W, dims=1)[:]) ./ (d.^2)
end
