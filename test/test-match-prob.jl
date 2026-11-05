using Determinantal
using LinearAlgebra
using Test
using Random

Random.seed!(1234) # deterministic test data (match_prob involves numerical optimisation)

# Test for match_prob on MarginalDPP
@testset "match_prob MarginalDPP" begin
    # Test 1: Diagonal kernel - simple case
    # Create a MarginalDPP with diagonal kernel
    K = MarginalDPP(Diagonal([0.5, 0.5, 0.5]))
    original_ip = inclusion_prob(K)

    # Target probabilities
    target_pr = [0.6, 0.7, 0.8]

    # Apply match_prob
    K_modified = match_prob(K, target_pr)
    new_ip = inclusion_prob(K_modified)

    # Check that inclusion probabilities match target
    @test new_ip ≈ target_pr rtol=1e-6

    A = randn(10, 4)
    A = A/(1.1*maximum(svd(A).S))
    K_lr = MarginalDPP(LowRank(A))
    target_pr_lr = inclusion_prob(K_lr) .* rand(nitems(K_lr))
    K_modified_lr = match_prob(K_lr, target_pr_lr)


    new_ip_lr = inclusion_prob(K_modified_lr)

    @test new_ip_lr ≈ target_pr_lr rtol=1e-6

    # Test 3: Full matrix kernel
    # Create a valid kernel with eigenvalues in [0, 1]
    K_full = MarginalDPP(Symmetric(Matrix(K_lr.K)))
    target_pr_full = target_pr_lr
    K_modified_full = match_prob(K_full, target_pr_full)
    new_ip_full = inclusion_prob(K_modified_full)

    @test new_ip_full ≈ target_pr_full rtol=1e-6
end

# Test for match_prob on L-ensemble
@testset "match_prob L-ensemble" begin
    # Test 1: EllEnsemble with LowRank
    A = randn(10, 4)
    L = EllEnsemble(LowRank(A))

    target_pr = rand(nitems(L))
    target_pr *= 0.9*maxrank(L)/sum(target_pr)

    L_modified = match_prob(L, target_pr)

    new_ip = inclusion_prob(L_modified)

    @test new_ip ≈ target_pr rtol=1e-4  # Slightly looser tolerance for numerical optimization

    # Test 2: EllEnsemble with full matrix
    L_full = EllEnsemble(Symmetric(Matrix(L.L)))
    target_pr_full = target_pr

    L_modified_full = match_prob(L_full, target_pr_full)
    new_ip_full = inclusion_prob(L_modified_full)

    @test new_ip_full ≈ target_pr_full rtol=1e-4
end


# Test for diag_inv_AAt_plus_D
@testset "diag_inv_AAt_plus_D" begin
    # Test 2: Verify against direct computation for small matrix
    A_small = randn(5, 3)
    d_small = rand(5)

    # Direct computation: diagonal of inv(A'*A + diag(d))
    M = A_small * A_small' + Diagonal(d_small)
    expected = diag(inv(M))

    result_small = Determinantal.diag_inv_AAt_plus_D(A_small, d_small)

    @test result_small ≈ expected rtol=1e-6
end
