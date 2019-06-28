@testset "sample_dpp" begin
    Lr = LowRankEnsemble(randn(10,4))
    @test length(sample(Lr)) <= 4
    @test length(sample(Lr,3)) <= 3

    Lp = ProjectionEnsemble(randn(10,4))
    @test length(sample(Lp)) == 4


    #Test a strong property: for a certain L-ensemble based on the incidence matrix of graph, the corresponding DPP samples spanning trees of that graph
    using LightGraphs
    G = Grid([5,5])
    n = nv(G)
    M=incidence_matrix(G,oriented=true)
    Lr=LowRankEnsemble(float.(Matrix(M')))
    set = sample(Lr,n-1) |> collect
    T = SimpleGraphFromIterator(collect(edges(G))[set])
    @test vertices(T) == vertices(G)
    @test is_connected(T)
    @test ne(T) == n-1
end
