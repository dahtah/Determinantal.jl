#Representative workload for Determinantal.jl
#Used for analysis by JET.jl
#FIXME: use later for precompilation?
function exercise_determinantal()
    X = randn(5, 3)
    Le = X*X'
    L=EllEnsemble(Le)
    L_lr=EllEnsemble(LowRank(X))
    K=MarginalDPP(L)
    K_lr=MarginalDPP(L_lr)
    sample(K)
    sample(L)
    sample(L_lr)
    inclusion_prob(L)
end
