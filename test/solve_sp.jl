@testset "solve_sp" begin
    f = (ls) -> sum(ls ./ (1 .+ ls))
    ls = collect(1:10)
    for k in 1:9
        α = Determinantal.solve_sp(ls, k)
        @test abs(f(α * ls) - k) < 1e-5
    end
    X = randn(3, 10)
    L = EllEnsemble(gaussker(ColVecs(X), 0.5))
    rescale!(L, 4)
    @test abs(f(L.α * L.λ) - 4) < 1e-5
    rescale!(L, 2)
    @test abs(cardinal(L)[1] - 2) < 1e-5
end
