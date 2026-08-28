# struct EllEnsIE <: AbstractLEnsemble
#     n :: Int
#     m :: Int
#     incl :: Vector{Int64}
#     excl :: Vector{Int64}
#     compl :: Vector{Int64}
#     ell :: EllEnsemble
#     α :: Float64
# end

# function EllEnsIE(L :: EllEnsemble,incl,excl)
#     compl = setdiff(setdiff(1:L.n,excl),incl)
#     S = L.L[compl,compl]-L.L[compl,incl]*(L.L[incl,incl]\L.L[incl,compl])
#     EllEnsIE(L.n,L.m,incl,excl,compl,EllEnsemble(S),L.α)
# end

# function inclusion_prob(L :: EllEnsIE)
#     pr = zeros(L.n)
#     pr[L.incl] .= 1
#     pri = inclusion_prob(L.ell)
#     for j in 1:length(pri)
#         pr[L.compl[j]] = pri[j]
#     end
#     pr
# end

# function sample(L :: EllEnsIE)
#     [collect(L.incl);L.compl[sample(L.ell)]]
# end




# function check_incl_excl(Q, incl, excl)
#     compl = setdiff(setdiff(1:size(Q,1),excl),incl)
#     M = copy(Q)
#     M[incl,:] .= 0
#     M[:,incl] .= 0
#     for i in incl
#         M[i,i] = 1
#     end
#     for j in compl
#         M[j,j] += 1
#     end
#     display(M)
#     inv(M)[compl,compl]
# end


# function condition_inclusion(L :: EllEnsemble,inc)
#     compl = setdiff(1:L.n,inc)
#     S = L.L[compl,compl]-L.L[compl,inc]*(L.L[inc,inc]\L.L[inc,compl])
#     EllEnsemble(S)
# end

# function isample(L::EllEnsemble)
#     pr = inclusion_prob(L)
#     incl = Vector{Int64}()
#     Lc = EllEnsIE(L,[],[])
#     for i in 1:L.n
#         smpl = sample(Lc)
#         (length(smpl) == length(incl)) && break
#         push!(incl,setdiff(smpl,incl)[1])
#         Lc = EllEnsIE(L,incl,[])
#         # pr_stop = 1/det(Lc.α*Lc.ell.L+I)
#         # (rand()<=pr_stop) && break
#         # pri = inclusion_prob(Lc)
#         # pri[incl] .= 0
#         # ind =wsample(pri,sum(pri))
#         # push!(incl,ind)
#         # Lc = EllEnsIE(L,incl,[])
#     end
#     incl
# end




# function iesample(L::EllEnsemble)
#     z = zeros(Bool,L.n)
#     pr = inclusion_prob(L)
#     incl = Vector{Int64}()
#     excl = Vector{Int64}()
#     for i in 1:L.n
#         z[i] = rand() < pr[i]
#         if (z[i])
#             push!(incl,i)
#         else
#             push!(excl,i)
#         end
#         ie = EllEnsIE(L,incl,excl)
#         pr = inclusion_prob(ie)
#     end
#     incl
# end


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
function condition_on(K::MarginalDPP; incl=Int[], excl=Int[])
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
        for j in 1:n
            Kn[i, j] = 0
            Kn[j, i] = 0
        end
    end
    @inbounds for i in incl
        for j in 1:n
            Kn[i, j] = 0
            Kn[j, i] = 0
        end
        Kn[i, i] = 1
    end
    MarginalDPP(Kn)
end
