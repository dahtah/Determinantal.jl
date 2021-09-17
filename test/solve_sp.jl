@testset "solve_sp" begin
    f = (ls) -> sum(ls ./ (1 .+ ls))
    ls = collect(1:10)
    for k = 1:9
        α = DPP.solve_sp(ls, k)
        @test abs(f(α * ls) - k) < 1e-5
    end
    X = randn(3, 10)
    L = gaussker(X, 0.5) |> FullRankEnsemble
    rescale!(L, 4)
    @test abs(f(L.α * L.λ) - 4) < 1e-5
    rescale!(L, 2)
    @test abs(cardinal(L)[1] - 2) < 1e-5
end
