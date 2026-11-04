# Tests for feature-related functions: polyfeatures, rff, gaussker, nystrom_approx, estmediandist

using Determinantal
using LinearAlgebra
using KernelFunctions
using Test
using Random

Random.seed!(42)  # For reproducibility

@testset "polyfeatures" begin
    # Test 2D input with ColVecs - this is the primary use case
    # x is d x n matrix, ColVecs(x) gives n points in d dimensions
    x2d = randn(2, 4)  # 4 points in 2D (2x4 matrix)
    X_2d = ColVecs(x2d)

    # Degree 0: should have 1 feature (just 1)
    F0 = polyfeatures(X_2d, 0)
    @test size(F0) == (4, 1)
    @test F0[:, 1] ≈ ones(4)

    # Degree 1: should have 3 features (1, x1, x2)
    F1 = polyfeatures(X_2d, 1)
    @test size(F1) == (4, 3)
    @test F1[:, 1] ≈ ones(4)


    # Degree 2: should have 6 features (1, x1, x2, x1^2, x1x2, x2^2)
    F2 = polyfeatures(X_2d, 2)
    @test size(F2) == (4, 6)
    @test F2[:, 1] ≈ ones(4)

    # Test with 3D points
    x3d = randn(3, 5)  # 5 points in 3D
    X_3d = ColVecs(x3d)
    F1_3d = polyfeatures(X_3d, 1)
    # Degree 1 in 3D: 1 + 3 = 4 features
    @test size(F1_3d) == (5, 4)
    @test F1_3d[:, 1] ≈ ones(5)

    # Degree 2 in 3D: binomial(3+2, 2) = 10 features
    F2_3d = polyfeatures(X_3d, 2)
    @test size(F2_3d) == (5, binomial(3 + 2, 2))

    # Degree 3 in 3D: binomial(3+3, 3) = 20 features
    F3_3d = polyfeatures(X_3d, 3)
    @test size(F3_3d) == (5, binomial(3 + 3, 3))

    # Test with RowVecs - for a matrix with 3 rows and 4 columns,
    # RowVecs treats each row as a point, so we have 3 points in 4D
    x_row = randn(3, 4)
    X_row = RowVecs(x_row)
    F1_row = polyfeatures(X_row, 1)
    # Should have 3 points (rows), each in 4D, so output is 3 x (1+4) = 3 x 5
    @test size(F1_row) == (3, 5)
    @test F1_row[:, 1] ≈ ones(3)
end

@testset "gaussker" begin
    # Test basic functionality
    x = randn(2, 5)
    X = ColVecs(x)

    # With explicit sigma
    K1 = gaussker(X, 1.0)
    @test size(K1) == (5, 5)
    @test K1 ≈ K1'  # Should be symmetric
    @test all(diag(K1) .≈ 1.0)  # Diagonal should be 1 for Gaussian kernel

    # With automatic sigma (median heuristic)
    K2 = gaussker(X)
    @test size(K2) == (5, 5)
    @test K2 ≈ K2'
    @test all(diag(K2) .≈ 1.0)

    # Test that different sigmas give different results
    K_small = gaussker(X, 0.1)
    K_large = gaussker(X, 10.0)
    # Larger sigma should give more uniform kernel
    @test mean(abs.(K_large .- mean(K_large))) < mean(abs.(K_small .- mean(K_small)))

    # Test with RowVecs
    x_row = randn(5, 2)
    X_row = RowVecs(x_row)
    K_row = gaussker(X_row, 1.0)
    @test size(K_row) == (5, 5)
end

@testset "rff" begin
    # Test basic functionality
    x = randn(2, 10)
    X = ColVecs(x)

    # Test with explicit m and sigma
    m = 5
    σ = 0.5
    R = rff(X, m, σ)

    # R should be a LowRank matrix
    @test isa(R, LowRank)
    # Should have 2*m columns (cos and sin for each frequency)
    @test size(R.A) == (10, 2 * m)

    # Test that the kernel approximation is positive semi-definite
    K_approx = Matrix(R)
    @test issymmetric(K_approx)
    # Check PSD: all eigenvalues should be non-negative
    eigvals = eigen(K_approx).values
    @test all(eigvals .>= -1e-10)  # Allow small numerical errors

    # Test with automatic sigma
    R2 = rff(X, m)
    @test isa(R2, LowRank)

    # Test with RowVecs
    x_row = randn(10, 2)
    X_row = RowVecs(x_row)
    R_row = rff(X_row, m, σ)
    @test isa(R_row, LowRank)
    @test size(R_row.A) == (10, 2 * m)

    # Test dimensionality
    x3d = randn(3, 8)
    X3d = ColVecs(x3d)
    R3d = rff(X3d, 4, 1.0)
    @test size(R3d.A) == (8, 8)  # 2*4 columns
end

@testset "nystrom_approx" begin
    # Test with matrix input - use a proper PSD matrix
    n = 10
    A = randn(n, n)
    K = A * A'  # Make it PSD
    # Add small diagonal for numerical stability
    K = K + 1e-6 * I

    # Test with explicit indices (more reliable than random)
    ind = 1:5
    N1 = nystrom_approx(K, ind)
    @test isa(N1, LowRank)
    @test size(N1) == (n, n)

    # Test with explicit indices (smaller subset)
    ind = 1:3
    N2 = nystrom_approx(K, ind)
    @test isa(N2, LowRank)
    @test size(N2) == (n, n)

    # Test with kernel and points
    x = randn(2, 15)
    X = ColVecs(x)
    ker = SqExponentialKernel()

    # With explicit indices
    ind = 1:4
    N3 = nystrom_approx(X, ker, ind)
    @test isa(N3, LowRank)

    # Check that Nyström is exact in the limit where we include all points
    K_full = kernelmatrix(ker, X)
    N4 = nystrom_approx(X, ker, 1:length(X))
    @test K_full ≈ Matrix(N4)
end

@testset "estmediandist" begin
    # Test with simple data
    x = [0.0, 1.0, 2.0, 3.0]
    X = ColVecs([x zeros(4)]')  # 4 points in 1D

    d1 = Determinantal.estmediandist(X, m = 10)
    @test isa(d1, Real)
    @test d1 > 0

    # Test with 2D data
    x2d = randn(2, 10)
    X2d = ColVecs(x2d)
    d2 = Determinantal.estmediandist(X2d)
    @test isa(d2, Real)
    @test d2 > 0

    # Test that result is reasonable
    # For random 2D points, median distance should be in a reasonable range
    @test 0 < d2 < 10  # Very loose bounds

    # Test with m parameter (subsampling)
    d3 = Determinantal.estmediandist(X2d, m = 5)
    @test isa(d3, Real)
end
