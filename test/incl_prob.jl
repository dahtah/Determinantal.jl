@testset "sample_dpp" begin
    Lr = LowRankEnsemble(randn(10,4))
    @test all(0 .<= inclusion_prob(Lr) .<=1)
    rescale!(Lr,3)
    @test all(0 .<=inclusion_prob(Lr) .<=1)
    K = marginal_kernel(Lr)
    @test all(diag(K).≈inclusion_prob(Lr))

    Lp = ProjectionEnsemble(randn(10,4))
    @test all(0 .<=inclusion_prob(Lp) .<=1)
    K = marginal_kernel(Lp)
    @test all(diag(K).≈inclusion_prob(Lp))
end
