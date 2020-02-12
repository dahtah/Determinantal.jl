# Estimate the KL divergence between two m-DPPs with L-ensembles L1 and L2

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

