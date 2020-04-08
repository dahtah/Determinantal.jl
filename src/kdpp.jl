#some functions for handling k-DPPs
"""
    inclusion_prob(L::AbstractLEnsemble,k)

First-order inclusion probabilities in a k-DPP with L-ensemble L. Uses a (typically very accurate) saddlepoint approximation from Barthelmé, Amblard, Tremblay (2019).
"""
function inclusion_prob(L::AbstractLEnsemble,k)
    val = inclusion_prob_diag(L.λ,k)
    val[val.<0] .= 0
    val[val.>1] .= 1
    val = (val ./ sum(val)) .* k
    return sum( (L.U*Diagonal(sqrt.(val))).^2,dims=2)
end


function inclusion_prob_diag(λ,k)
    α=solve_sp(λ,k)
    h=(1 .+ α*λ)
    r=[sum(λ .* (h.^-i)) for i in 1:3]
    @. (λ/h)*(r[3]/(h*r[2]^2) - 1/(r[2]*h^2) + α )
end


"""
    sample(L::AbstractLEnsemble,k)

Sample a k-DPP, i.e. a DPP with fixed size. k needs to be strictly smaller than the rank of L (if it equals the rank of L, use a ProjectionEnsemble).

The algorithm uses a saddle-point approximation adapted from Barthelmé, Amblard, Tremblay (2019).
"""
function sample(L::AbstractLEnsemble,k)
    incl = sample_diag_kdpp(L,k)
    sample_pdpp(L.U[:,collect(incl)])
end



function sample_diag_kdpp(L::AbstractLEnsemble,k)
    set = BitSet()
    (k==0) && return set
    (k==L.m) && return 1:L.m
    s = 0
    α = solve_sp(L.λ,k)

    λ = L.λ
    lp = zeros(length(λ))
    lt = zeros(length(λ))
    for t in 1:L.m
        if (k-s == L.m-t+1) #prob accepting next == 1
            p = 1
        else
            λ_sub = @view λ[t:end]
            α=solve_sp(λ_sub,k-s;nu0=log(α))
            #compute cond. inclusion probability
            r = zeros(3)
            for i in t:L.m
                lp[i] = (1 .+ α*λ[i])
                lt[i] = λ[i]
                for j in 1:3
                    lt[i] /= lp[i]
                    r[j] += lt[i]
                end
            end
            #@assert all(r .≈ r2)
            p = (λ[t]/lp[t])*(r[3]/(lp[t]*r[2]^2) - 1/(r[2]*lp[t]^2) + α )
            #@assert p ≈ p2
        end

        if (rand()<=p)
            push!(set,t)
            s+=1
        end
        (length(set)==k) && return set
    end
    throw(ErrorException("Algorithm failed, did not reach the required number of samples"))
end

@doc raw"""

    esp(L::AbstractLEnsemble,approx=false)

Compute the elementary symmetric polynomials of the L-ensemble L, e₁(L) ...
eₙ(L). e₁(L) is the trace and eₙ(L) is the determinant. The ESPs determine the
distribution of the sample size of a DPP: 

``p(|X| = k) = \frac{e_k}{\sum_{i=1}^n e_i}``

The default algorithm uses the Newton equations, but may be unstable numerically
for n large. If approx=true, a stable saddle-point approximation (as in
Barthelmé et al. (2019)) is used instead for all eₖ with k>5. 

"""
function esp(L::AbstractLEnsemble,approx=false)
    esp(L.λ,length(L.λ),approx)
end

function esp(ls,k=length(ls),approx = false)
    if (!approx)
        esp_newton(ls,k)
    else
        esp_sp(ls,k)
    end
end

function esp_newton(ls,k=length(ls))
    N = length(ls)
    eprev = zeros(N+1,k+1)
    eprev[:,1] .= 1.
#    eprev[1,2:end] .= 0.
    for l in 1:k
        e = zeros(N+1)
        for n in 1:N
            e[n+1] = e[n]+ls[n]*eprev[n,l]
        end
        eprev[:,l+1] = e
    end
    eprev[N+1,2:end]
end

function esp_sp(ls,kmax=length(ls))
    exp.(log_esp_sp(ls,kmax))
end




#compute (log) ESPs using saddlepoint approximation
function log_esp_sp(ls,kmax=length(ls))
    n = length(ls)
    nu=-log(sum(ls[ls .> 0]))
    l2pi=log(2*π)
    logesp = zeros(kmax);
    N_EXACT = 15
    logesp[1:min(kmax,N_EXACT)] = log.(esp_newton(ls,min(kmax,N_EXACT)))
    (kmax <= N_EXACT) && return logesp

    for i in 1:(kmax)
        if (i == n)
            logesp[i] = sum(log.(ls))
        else
            nu=log(solve_sp(ls,i;nu0=nu))
            p = ls*exp(nu)
            sig = sum(@. p/((1+p)^2))
            tmp = sum(log1p.(p))
            if (i>N_EXACT)
                logesp[i] = -i*nu + tmp  - .5*(l2pi+log(sig))
            end
        end
    end
    return logesp;
end

