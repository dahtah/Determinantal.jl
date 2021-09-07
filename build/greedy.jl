@doc raw"""
    greedy_subset(L :: AbstractLEnsemble,k)

Perform greedy subset selection: find a subset ``\mathcal{X}`` of size ``k``, such that
``\det{L}_{\mathcal{X}}`` is high, by building the set item-by-item and picking at
each step the item that maximises the determinant. You can view this as finding
an (approximate) mode of the DPP with L-ensemble L.

If ``k`` is too large relative to the (numerical) rank of L, the problem is not well-defined
as so the algorithm will stop prematurely. 

# Example

```
X = randn(2,100) #10 points in dim 2
L = LowRankEnsemble(polyfeatures(X,2)) 
greedy_subset(L,6)
```

"""
function greedy_subset(L :: AbstractLEnsemble,k)
    @assert k >= 1
    ind = BitSet()
    i = argmax(diag(L))
    push!(ind,i)
    Q = ones(1,1) ./ L[i,i]
    Lv = L[:,i]
    for iter in 2:k
        optv = 0
        opti= 0
        for i in 1:L.n
            if i ∉ ind
                fv =  L[i,i] - dot(Lv[i,:],(Q*Lv[i,:]))
#                @show fv
                if (fv > optv)
                    opti = i
                    optv= fv
                end
            end
        end
        if (opti == 0)
            break
        end
        Q = update_inverse(Q,Lv[opti,:],L[opti,opti])
        Lv = [Lv L[:,opti]]
        push!(ind,opti)
    end
    ind
end

#Let K' = [Q^-1 r; r' α], find inverse of K' given Q
function update_inverse(Q,r,α)
  #  @show Q,r,α
    v = Q*r
    C2inv = 1/(α-dot(r,v))
    C1inv = Q+(r*r')*C2inv
    a = -v*C2inv
    [C1inv a; a' C2inv]
end
