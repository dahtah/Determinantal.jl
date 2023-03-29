import Combinatorics.combinations

@testset "pmf" begin
    n = 6
    X = randn(2, n)
    Ls = [
        EllEnsemble(gaussker(ColVecs(X))),
        EllEnsemble(LowRank(randn(n, 3))),
        ProjectionEnsemble(randn(n, 3)),
    ]
    for L in Ls
        pr = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        @assert sum(pr) + exp(log_prob(L, [])) ≈ 1
    end
    Ls = [EllEnsemble(gaussker(ColVecs(X))), EllEnsemble(LowRank(randn(n, 3)))]
    #Test invariance to rescaling
    for L in Ls
        rescale!(L, 2)
        pr = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        @assert sum(pr) + exp(log_prob(L, [])) ≈ 1
    end
    for L in Ls
        for k in 1:maxrank(L)
            pr = map((ind) -> exp(log_prob(L, ind, k)), combinations(1:nitems(L), k))
            @show k, sum(pr)
            @assert sum(pr) ≈ 1
        end
    end
    n = 6
    #Test extended L-ensembles
    X = randn(2, n)
    U = randn(n, 3)
    Ls = [ExtEnsemble(gaussker(ColVecs(X)), U[:, 1:2]), ExtEnsemble(gaussker(ColVecs(X)), U)]

    for L in Ls
        tmp = sum(map((i) -> exp(Determinantal.logz(L, i)), Determinantal.min_items(L):n))
        @assert abs(log(tmp) - Determinantal.logz(L)) < 1e-4
    end

    for L in Ls
        pr = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        @assert sum(pr) ≈ 1
    end

    for L in Ls
        for k in Determinantal.min_items(L):n
            pr = map((ind) -> exp(log_prob(L, ind, k)), combinations(1:nitems(L), k))
            @show k, sum(pr)
            @assert sum(pr) ≈ 1
        end
    end
end
