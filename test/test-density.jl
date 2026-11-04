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
        @test sum(pr) ≈ 1
    end
    Ls = [EllEnsemble(gaussker(ColVecs(X))), EllEnsemble(LowRank(randn(n, 3)))]
    #Test invariance to rescaling
    for L in Ls
        rescale!(L, 2)
        pr = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        @test sum(pr) ≈ 1
    end

    #Test that we get correct expressions using the marginal representations
    for L in Ls
        M = MarginalDPP(L)
        @test nitems(L) == nitems(M)
        prL = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        prM = map((ind) -> exp(log_prob(M, ind)), combinations(1:nitems(L)))
        @test prL ≈ prM
    end


    #Test k-DPPs
    for L in Ls
        for k = 1:maxrank(L)
            pr = map((ind) -> exp(log_prob(L, ind, k)), combinations(1:nitems(L), k))
            @show k, sum(pr)
            @test sum(pr) ≈ 1
        end
    end
    n = 6
    #Test extended L-ensembles
    X = randn(2, n)
    U = randn(n, 3)
    Ls =
        [ExtEnsemble(gaussker(ColVecs(X)), U[:, 1:2]), ExtEnsemble(gaussker(ColVecs(X)), U)]

    for L in Ls
        tmp = sum(map((i) -> exp(Determinantal.logz(L, i)), Determinantal.min_items(L):n))
        @test abs(log(tmp) - Determinantal.logz(L)) < 1e-4
    end

    for L in Ls
        pr = map((ind) -> exp(log_prob(L, ind)), combinations(1:nitems(L)))
        @test sum(pr) ≈ 1
    end

    for L in Ls
        for k = Determinantal.min_items(L):n
            pr = map((ind) -> exp(log_prob(L, ind, k)), combinations(1:nitems(L), k))
            @show k, sum(pr)
            @test sum(pr) ≈ 1
        end
    end
end

@testset "condition" begin
    n = 6
    L = EllEnsemble(LowRank(randn(n, n)))
    M = MarginalDPP(L)
    #Test product of conditional probabilities
    incl=[2, 3, 4]
    excl=[5, 6]
    M_incl_excl = condition_on(M, incl = incl, excl = excl)
    p_1_incl_excl = inclusion_prob(M_incl_excl)[1]
    p_incl=det(M.K[incl, incl])
    M_incl = condition_on(M, incl = incl)
    p_excl_given_incl = void_prob(M_incl, excl)
    p_event = p_incl*p_excl_given_incl*p_1_incl_excl
    p_check = exp(log_prob(M, [incl; 1]))
    @test p_event ≈ p_check
end
