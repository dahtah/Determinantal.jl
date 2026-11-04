
@doc raw"""
        condition_on(K::MarginalDPP; incl=Int[], excl=Int[])

        Condition a Determinantal Point Process (DPP) on the event that all items in `incl` appear and none of the items in `excl` appear. Returns a new `MarginalDPP` with a modified kernel.

        In the current implementation, this involves an $\mathcal{O}(n^3)$ operation and is mostly of pedagogical interest.

    # Arguments
    - `K::MarginalDPP`: The original DPP.
    - `incl::Vector{Int}`: Items that must be included in the conditioned DPP.
    - `excl::Vector{Int}`: Items that must be excluded from the conditioned DPP.

    # Returns
    - `MarginalDPP`: A new DPP conditioned on the inclusion and exclusion constraints.

    ```@example ex1
     K=MarginalDPP([ .5 .2;.2 .5 ])
     inclusion_prob(condition_on(K,incl=[1])) #1 should appear with prob. 1
     inclusion_prob(condition_on(K,excl=[1])) #1 should appear with prob. 0
    ```
    """
function condition_on(K::MarginalDPP; incl = Int[], excl = Int[])
    n = nitems(K)
    @assert length(intersect(incl, excl)) == 0
    @assert all(0 .< incl .<= n) "Values must be between 1 and $(n)"
    @assert all(0 .< excl .<= n) "Values must be between 1 and $(n)"

    ie = [incl; excl]
    rem = setdiff(1:nitems(K), ie)
    Kn = copy(K.K)
    @inbounds for i in excl
        Kn[i, i] -= 1.0
    end
    Kn[rem, rem] .-= Kn[rem, ie] * (Kn[ie, ie] \ Kn[ie, rem])
    @inbounds for i in excl
        for j = 1:n
            Kn[i, j] = 0
            Kn[j, i] = 0
        end
    end
    @inbounds for i in incl
        for j = 1:n
            Kn[i, j] = 0
            Kn[j, i] = 0
        end
        Kn[i, i] = 1
    end
    MarginalDPP(Kn)
end
