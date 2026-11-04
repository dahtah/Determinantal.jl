@testset "sample_dpp" begin
    Lr = EllEnsemble(LowRank(randn(10, 4)))
    @test length(sample(Lr)) <= 4
    @test length(sample(Lr, 3)) <= 3

    Lp = ProjectionEnsemble(randn(10, 4))
    @test length(sample(Lp)) == 4

    #Test behaviour under stratified sampling
    #Construct projection kernel so that DPP
    #should sample exactly one item per stratum
    idx = [1, 1, 2, 2, 3, 3] #strata
    nstrata = maximum(idx)
    n = length(idx)
    U = float.([idx[i] == j for i = 1:n, j = 1:nstrata])
    Lp = ProjectionEnsemble(U)
    for _ = 1:100
        ind = sample(Lp)
        @test length(unique(idx[ind])) == nstrata
    end

    #Some cases should lead to deterministic samples, check that
    ind=sample(MarginalDPP(Diagonal([1.0, 0.0, 1.0])))
    @test sort(ind) == [1, 3]

end
