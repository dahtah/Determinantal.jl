@doc raw"""
    greedy_subset(L :: AbstractLEnsemble,k)

Perform greedy subset selection: find a subset ``\mathcal{X}`` of size ``k``, such that
``\det{L}_{\mathcal{X}}`` is high, by building the set item-by-item and picking at
each step the item that maximises the determinant. You can view this as finding
an (approximate) mode of the DPP with L-ensemble L.

If ``k`` is too large relative to the (numerical) rank of L, the problem is not well-defined
as so the algorithm will stop prematurely.

The implementation runs in $\O(nk^2)$ but is not particularly optimised. If the end result looks screwy, it is probably due to numerical instability: try improving the conditioning of $\bL$.

# Example

```@example
x = randn(2,100) #10 points in dim 2
greedy_subset(EllEnsemble(gaussker(x)),12)
#same thing but faster:
greedy_subset(gaussker(x),12)
```

"""
function greedy_subset(L::AbstractLEnsemble, k)
    greedy_subset(L.L, k)
end

function greedy_subset(L::AbstractMatrix, k)
    @assert k >= 1
    n = size(L, 1)
    ind = BitSet()
    dets = diag(L)
    mx = maximum(dets)
    i = rand(findall(diag(L) .== mx))
    push!(ind, i)
    Q = ones(1, 1) ./ L[i, i]
    Lv = L[:, i]
    dL = diag(L)
    for iter = 2:k
        #ld = diag(L) - Q*Lv
        opti = 0
        optv = 0
        @inbounds for i = 1:n
            if i ∉ ind
                dets[i] = L[i, i] - dot(Lv[i, :], Q * Lv[i, :])
                if (optv < dets[i])
                    optv = dets[i]
                    opti = i
                end
            else
                dets[i] = 0.0
            end
        end
        #dets_tst = map(g,1:L.n)
        #@show extrema(dets)
        #optv,opti = findmin(dets)



        if (opti <= 0) && iter < k
            break
        end
        if (any(dets .< -.01)) #Q is probably bad
            #Recompute matrix inverse
            indc = [collect(ind); opti]
            Q = inv(L[indc, indc])
            Lv = L[:, indc]
        else
            Q = update_inverse(Q, Lv[opti, :], L[opti, opti])
            Lv = [Lv L[:, opti]]
        end
        push!(ind, opti)
    end
    collect(ind)
end

# function greedy_subset_k3(L :: AbstractLEnsemble,k)
#     @assert k >= 1
#     ind = BitSet()
#     dets = diag(L)
#     mx = maximum(dets)
#     i = rand(findall(diag(L) .== mx ))
#     push!(ind,i)
#     #Q = ones(1,1) ./ L[i,i]
#     Lv = L[:,i]
#     Li = L[i,i]*ones(1,1)
#     dL = diag(L)
#     for iter in 2:k
#         #ld = diag(L) - Q*Lv
#         opti=0
#         optv=0

#         C = cholesky(Symmetric(Li))
#         @inbounds for i in 1:L.n
#             if i ∉ ind
#                 dets[i] = L[i,i] - dot(Lv[i,:],C\Lv[i,:])
#                 if (optv < dets[i])
#                     optv = dets[i]
#                     opti = i
#                 end
#             else
#                 dets[i] = 0.0
#             end
#         end
#         #dets_tst = map(g,1:L.n)
# #        @show extrema(dets)
# #        @show sum(dets .< 0)
#         #optv,opti = findmin(dets)



#         if (opti <= 0) && iter < k
#             break
#         end
#         tmp = L[:,opti]
#         Li = [Li Lv[opti,:]
#               Lv[opti,:]' L[opti,opti]]
#         Lv = [Lv tmp]

#         push!(ind,opti)
#     end
#     collect(ind)
# end



#Let K' = [Q^-1 r; r' α], find inverse of K' given Q
function update_inverse(Q, r, α)
    #  @show Q,r,α
    v = Q * r
    C2inv = 1 / (α - dot(r, v))
    C1inv = Q + (r * r') * C2inv
    a = -v * C2inv
    [C1inv a; a' C2inv]
end
