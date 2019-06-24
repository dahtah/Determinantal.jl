#some functions for handling k-DPPs
"""
    inclusion_prob(L::AbstractLEnsemble,k)

First-order inclusion probabilities in a k-DPP with L-ensemble L. Uses a (typically very accurate) saddlepoint approximation from Barthelmé, Amblard, Tremblay (2019). 
"""
function inclusion_prob(L::AbstractLEnsemble,k)
    val = inclusion_prob_diag(L.λ,k)
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
    α_0 = solve_sp(L.λ,k)
    λ = L.λ
    for t in 1:L.m
        α=solve_sp(λ[t:L.m],k-s)
        #compute cond. inclusion probability
        h=(1 .+ α*λ[t:L.m])
        r=[sum(λ[t:L.m] .* (h.^-i)) for i in 1:3]
        p = (λ[t]/h[1])*(r[3]/(h[1]*r[2]^2) - 1/(r[2]*h[1]^2) + α )
        #@show p
        #@assert p ≈ inclusion_prob_diag(λ[t:L.m],k-s)[1]
        if (rand()<=p)
            push!(set,t)
            s+=1
        end
        (length(set)==k) && return set
    end
    throw(ErrorException("Algorithm failed, did not reach the required number of samples"))
end
