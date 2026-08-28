# Tests for distance_sampling and LazyDist

using Determinantal
using LinearAlgebra
using Test
using Random
using StatsBase

Random.seed!(42)

@testset "LazyDist" begin
    # Test basic construction
    x = randn(2, 5)
    points = [x[:, i] for i in 1:size(x, 2)]
    D = LazyDist(points)
    
    # Test size
    @test size(D) == (5, 5)
    
    # Test indexing
    @test D[1, 1] ≈ 0.0  # Distance to self
    @test D[2, 1] == D[1, 2]  # Symmetric
    @test D[1, 2] ≈ norm(points[1] - points[2])
    
    # Test column indexing
    col1 = D[:, 1]
    @test length(col1) == 5
    @test col1[1] ≈ 0.0
    @test col1[2] ≈ D[2, 1]
    
    # Test row indexing
    row1 = D[1, :]
    @test length(row1) == 5
    @test row1[1] ≈ 0.0
    @test row1[2] ≈ D[1, 2]
    
    # Test full matrix indexing
    full = D[:, :]
    @test size(full) == (5, 5)
    @test full[1, 1] ≈ 0.0
    @test full[1, 2] ≈ D[1, 2]
    
    # Test with custom distance function
    D_custom = LazyDist(points, (a, b) -> sum(abs.(a - b)))
    @test D_custom[1, 1] ≈ 0.0
    @test D_custom[1, 2] ≈ sum(abs.(points[1] - points[2]))
    
    # Test that diagonal is zero
    for i in 1:5
        @test D[i, i] ≈ 0.0
    end
    
    # Test symmetry
    for i in 1:5, j in 1:5
        @test D[i, j] ≈ D[j, i]
    end
end

@testset "distance_sampling" begin
    # Test basic functionality with :d2 sampling
    x = randn(2, 20)
    points = [x[:, i] for i in 1:size(x, 2)]
    D = LazyDist(points)
    
    # Test with :d2 sampling
    ind_d2 = distance_sampling(D, 5, :d2)
    @test isa(ind_d2, Vector{Int})
    @test length(ind_d2) == 5
    @test all(1 .<= ind_d2 .<= 20)
    @test length(unique(ind_d2)) == 5
    
    # Test with :farthest sampling
    ind_farthest = distance_sampling(D, 5, :farthest)
    @test isa(ind_farthest, Vector{Int})
    @test length(ind_farthest) == 5
    @test all(1 .<= ind_farthest .<= 20)
    @test length(unique(ind_farthest)) == 5
    
    # Test that farthest sampling spreads points out
    # Farthest sampling should give points that are well-separated
    # We can check that the minimum distance between selected points is reasonable
    selected_points = [points[i] for i in ind_farthest]
    n_sel = length(selected_points)
    # Compute pairwise distances
    pairwise_dists = Float64[]
    for i_idx in 1:n_sel
        for j_idx in (i_idx+1):n_sel
            push!(pairwise_dists, norm(selected_points[i_idx] - selected_points[j_idx]))
        end
    end
    min_dist = minimum(pairwise_dists)
    # The minimum distance should be non-zero (points are distinct)
    @test min_dist > 0
    
    # Test with custom function - use abs to ensure positive weights
    custom_fun = (d) -> abs(d)  # Absolute value of distances
    ind_custom = distance_sampling(D, 5, custom_fun)
    @test length(ind_custom) == 5
    
    # Test edge case: m = 1
    ind1 = distance_sampling(D, 1, :d2)
    @test length(ind1) == 1
    
    # Test edge case: m = n
    ind_all = distance_sampling(D, 20, :d2)
    @test length(ind_all) == 20
    @test Set(ind_all) == Set(1:20)
    
    # Test error handling
    @test_throws AssertionError distance_sampling(D, 25, :d2)  # m > n
    
    # Test that custom function with all zeros returns early
    zero_fun = (d) -> 0.0  # Return scalar zero for each distance
    ind_zero = distance_sampling(D, 5, zero_fun)
    # Should return early with whatever it has (at least 1 point)
    @test 1 <= length(ind_zero) <= 5
    
    # Test determinism with seed
    Random.seed!(42)
    ind_a = distance_sampling(D, 5, :d2)
    Random.seed!(42)
    ind_b = distance_sampling(D, 5, :d2)
    @test ind_a == ind_b
end

@testset "distance_sampling with full matrix" begin
    # Test with precomputed distance matrix
    n = 10
    full_D = [norm(randn(2)) for i in 1:n, j in 1:n]
    full_D = full_D + full_D'  # Make symmetric
    
    ind = distance_sampling(full_D, 4, :farthest)
    @test length(ind) == 4
    
    # Test that farthest point sampling selects diverse points
    # The first point should be random
    # Subsequent points should be far from all previous
    # This is harder to test precisely, but we can check it runs
    
    # Test with identity matrix (all points at distance 1 from each other except self)
    I_dist = [i == j ? 0.0 : 1.0 for i in 1:5, j in 1:5]
    ind_identity = distance_sampling(I_dist, 3, :farthest)
    @test length(ind_identity) == 3
end
