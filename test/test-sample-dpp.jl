@testset "sample_dpp" begin
    Lr = EllEnsemble(LowRank(randn(10, 4)))
    @test length(sample(Lr)) <= 4
    @test length(sample(Lr, 3)) <= 3

    Lp = ProjectionEnsemble(randn(10, 4))
    @test length(sample(Lp)) == 4

    #Test behaviour under stratified sampling
    #Construct projection kernel so that DPP
    #should sample exactly one item per stratum
    idx = [1,1,2,2,3,3] #strata
    nstrata = maximum(idx)
    n = length(idx)
    U = float.([idx[i] == j for i in 1:n,  j in 1:nstrata])
    Lp = ProjectionEnsemble(U)
    for _ in 1:100
        ind = sample(Lp)
        @test length(unique(idx[ind])) == nstrata
    end

end
