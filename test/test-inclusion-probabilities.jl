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
    p=.3
    q=1-p
    M=MarginalDPP(p*I(n)) #This is just a Poisson sample
    m = (v)->Determinantal.mgf(M,v)
    mt = (v)-> (q+p*exp(v))^n #Theoretical value
    vs = range(-1,1,20)
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
        @assert Determinantal.mgf(M,.5)  ≈ Determinantal.mgf(L,.5)
        @assert Determinantal.mgf(M,.15*ones(nitems(M))) ≈ Determinantal.mgf(M,.15)
        @assert Determinantal.mgf(L,.15*ones(nitems(M))) ≈ Determinantal.mgf(L,.15)
        t = rand(nitems(M))
        @assert Determinantal.mgf(M,t)  ≈ Determinantal.mgf(L,t)
    end
end
