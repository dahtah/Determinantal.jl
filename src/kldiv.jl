# Estimate the KL divergence between two m-DPPs with L-ensembles L1 and L2

@doc raw"""
    kl_divergence(L1::AbstractLEnsemble,L2::AbstractLEnsemble,k::Int;nsamples=100)

Estimate the KL divergence between two (k-)DPPs. The KL divergence is estimated
by sampling from the k-DPP with L-ensemble L1 and computing the mean log-ratio
of the probabilities. nsamples controls how many samples are taken. 

"""
function kl_divergence(L1::AbstractLEnsemble,L2::AbstractLEnsemble,k::Int;nsamples=100)
    div=0
    for ii in 1:nsamples
        ind = sample(L1,k) |> collect
        div+= log_prob(L1,ind,k) - log_prob(L2,ind,k)
    end
    div /= nsamples
end

function kl_divergence(L1::AbstractLEnsemble,L2::AbstractLEnsemble;nsamples=100)
    div=0
    for ii in 1:nsamples
        ind = sample(L1) |> collect
        div+= log_prob(L1,ind) - log_prob(L2,ind)
    end
    div /= nsamples
end

function total_variation(L1::AbstractLEnsemble,L2::AbstractLEnsemble,k::Int;nsamples=100)
    div=0
    for ii in 1:nsamples
        ind = sample(L1,k) |> collect
        p1 = exp(log_prob(L1,ind,k))
        p2 = exp(log_prob(L2,ind,k))
        div+= abs(p2 - p1)/p1
    end
    div /= nsamples
end

function total_variation(L1::AbstractLEnsemble,L2::AbstractLEnsemble;nsamples=100)
    div=0
    for ii in 1:nsamples
        ind = sample(L1) |> collect
        p1 = exp(log_prob(L1,ind))
        p2 = exp(log_prob(L2,ind))
        div+= abs(p2 - p1)/p1
    end
    div /= nsamples
end


