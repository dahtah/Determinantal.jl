# Tests for greedy_subset function

using Determinantal
using LinearAlgebra
using Test
using Random

Random.seed!(42)

@testset "greedy_subset" begin
    # Test with simple matrix - use a well-conditioned matrix
    L_mat = [1.0 0.5 0.1; 0.5 1.0 0.2; 0.1 0.2 1.0]

    # Test basic functionality with matrix
    k = 2
    ind = greedy_subset(L_mat, k)
    @test isa(ind, Vector{Int})
    @test length(ind) == k
    @test all(1 .<= ind .<= 3)
    @test length(unique(ind)) == k  # All unique

    # Test with EllEnsemble
    L_ens = EllEnsemble(L_mat)
    ind2 = greedy_subset(L_ens, k)
    @test isa(ind2, Vector{Int})
    @test length(ind2) == k

    # Test with k=1
    ind1 = greedy_subset(L_mat, 1)
    @test length(ind1) == 1

    # Test with k equal to full size
    ind_full = greedy_subset(L_mat, 3)
    @test length(ind_full) == 3

    # Test error handling
    @test_throws AssertionError greedy_subset(L_mat, 0)  # k >= 1

    # Test with diagonal matrix (easy case)
    D = Diagonal([1.0, 0.5, 0.1, 0.01])
    ind_diag = greedy_subset(D, 2)
    @test sort(ind_diag) == [1, 2]
end
