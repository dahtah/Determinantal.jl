using LinearAlgebra


@testset "incl_prob" begin
    Lr = EllEnsemble(LowRank(randn(10, 4)))
    @test all(0 .<= inclusion_prob(Lr) .<= 1)
    rescale!(Lr, 3)
    @test all(0 .<= inclusion_prob(Lr) .<= 1)
    K = marginal_kernel(Lr)
    @test all(diag(K) .≈ inclusion_prob(Lr))
    Lp = ProjectionEnsemble(randn(10, 4))
    @test all(0 .<= inclusion_prob(Lp) .<= 1)
    K = marginal_kernel(Lp)
    @test all(diag(K) .≈ inclusion_prob(Lp))
end

#Test that the moment generating functions are correct
@testset "mgf" begin
    n=5
    p=0.3
    q=1-p
    M=MarginalDPP(p*I(n)) #This is just a Poisson sample
    m = (v)->Determinantal.mgf(M, v)
    mt = (v) -> (q+p*exp(v))^n #Theoretical value
    vs = range(-1, 1, 20)
    @assert m.(vs) ≈ mt.(vs)

    n = 6
    X = randn(2, n)
    Ls = [
        EllEnsemble(gaussker(ColVecs(X))),
        EllEnsemble(LowRank(randn(n, 3))),
        ProjectionEnsemble(randn(n, 3)),
    ]
    for L in Ls
        M = MarginalDPP(L)
        #Test that both representations give the same result
        @assert Determinantal.mgf(M, 0.5) ≈ Determinantal.mgf(L, 0.5)
        @assert Determinantal.mgf(M, 0.15*ones(nitems(M))) ≈ Determinantal.mgf(M, 0.15)
        @assert Determinantal.mgf(L, 0.15*ones(nitems(M))) ≈ Determinantal.mgf(L, 0.15)
        t = rand(nitems(M))
        @assert Determinantal.mgf(M, t) ≈ Determinantal.mgf(L, t)
    end
end

#Test that the void probabilities are correct
@testset "voidprob" begin
    @test void_prob(MarginalDPP(Diagonal([1.0, 0.0, 1.0])), [1, 3]) ≈ 0.0
    @test void_prob(MarginalDPP(Diagonal([0.2, 0.4, 0.3])), [1, 3]) ≈ (0.8*0.7)
    L=EllEnsemble(LowRank(randn(5, 2)))
    ii = [4, 2]
    vp=void_prob(L, ii)
    @assert void_prob(MarginalDPP(L), ii) ≈ vp
end
