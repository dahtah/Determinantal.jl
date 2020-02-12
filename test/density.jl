import Combinatorics.combinations

@testset "pmf" begin
    n = 6
    X = randn(2,n)
    Ls = [FullRankEnsemble(gaussker(X)),LowRankEnsemble(randn(n,3)),
          ProjectionEnsemble(randn(n,3))]
    for L in Ls
        pr = map((ind) -> log_prob(L,ind) |> exp,combinations(1:nitems(L)));
        @assert sum(pr) + exp(log_prob(L,[])) ≈ 1
    end
    Ls = [FullRankEnsemble(gaussker(X)),LowRankEnsemble(randn(n,3))]
    #Test invariance to rescaling
    for L in Ls
        rescale!(L,2)
        pr = map((ind) -> log_prob(L,ind) |> exp,combinations(1:nitems(L)));
        @assert sum(pr) + exp(log_prob(L,[])) ≈ 1
    end
    for L in Ls
        for k in 1:maxrank(L)
            pr = map((ind) -> log_prob(L,ind,k) |>
                     exp,combinations(1:nitems(L),k))
            @show k,sum(pr)
            @assert sum(pr)  ≈ 1
        end
    end
    n =6
    #Test partial projection ensembles
    X = randn(2,n)
    U = randn(n,3)
    Ls=[PPEnsemble(gaussker(X),U[:,1:2]), PPEnsemble(gaussker(X),U)]

    for L in Ls
        tmp = sum(map((i) -> exp(DPP.logz(L,i)),DPP.min_items(L):n))
        @assert abs(log(tmp) - DPP.logz(L)) < 1e-4
    end


    for L in Ls
        pr = map((ind) -> log_prob(L,ind) |> exp,combinations(1:nitems(L)));
        @assert sum(pr)  ≈ 1
    end

    for L in Ls
        for k in DPP.min_items(L):n
            pr = map((ind) -> log_prob(L,ind,k) |>
                     exp,combinations(1:nitems(L),k));
            @show k,sum(pr)
            @assert sum(pr)  ≈ 1
        end
    end


end
