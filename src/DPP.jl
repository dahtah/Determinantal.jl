module DPP
using LightGraphs,SparseArrays,LinearAlgebra,StatsBase,Clustering,Combinatorics,Distances,NearestNeighbors
export sample_pdpp,polyfeatures,kmeans_coreset,sample_dsquared,random_forest_direct,smooth_wilson

function vdm(x :: Array{T,1}, order :: Int) where T <: Real
    [u^k for u in x, k in 0:order-1]
end

#Empirical volume of Voronoi cell, used for weights in D^2 sampling
function empirical_volume(X :: Matrix,ind :: Vector{Int})
    tr=KDTree(X[:,ind])
    idx,dst = knn(tr,X,1)
    counts(reduce(vcat,idx))
end

function polyfeatures(X :: Array{T,2},order :: Int) where T <: Real
    m = size(X,2)
    if (m==1)
        vdm(vec(X),order)
    else
        g = (z) -> reduce(vcat,map((u) -> [ [u v] for v in 0:(order-1) if  sum(u) + v < order],z))
        dd = g(0:(order-1))
        for i in 1:(m-2)
            dd = g(dd)
        end
        reduce(hcat,[prod(X .^ d,dims=2) for d in dd])
        #[prod(X .^ d') for d in dd ]
        #dd
    end
end

function kmeans_d2(X :: Array{T,2},set_size :: Int, k :: Int; nrep = 10) where T <: Real
    ind = sample_dsquared(X,set_size)
    w = float.(empirical_volume(X,ind))
    @assert sum(w) ≈ size(X,2)
    C = kmeans_restart(Matrix(X[:,ind]),k,nrep,w ./ sum(w)).centers
    (centers=C,cluster=nnclass(X,C))
end

function kmeans_coreset(X :: Array{T,2},order :: Int, k :: Int; nrep = 10) where T <: Real
    U=Matrix(qr(polyfeatures(Matrix(X'),order)).Q)
    #@show size(U,2)
    w = 1 ./ vec(sum(U.^2;dims=(2)))
    pp = collect(sample_pdpp(U))
    C = kmeans_restart(Matrix(X[:,pp]),k,nrep,w[pp]).centers
    (centers=C,cluster=nnclass(X,C))
end

function kmeans_restart(X :: Array{T1,2},k :: Int,nrep,weights :: Array{Float64,1}) where T1 <: Real
    f = () -> kmeans(X,k,init=:kmpp)
    res = f()
    for i in 2:nrep
        nres = f()
        if (dot(nres.costs,weights) < dot(res.costs,weights))
            res = nres
        end
    end
    res
end



function kmeans_restart(X :: Array{T,2},k :: Int,nrep :: Int) where T <: Real
    f = () -> kmeans(X,k,init=:kmpp)
    res = f()
    for i in 2:nrep
        nres = f()
        if (sum(nres.costs) < sum(res.costs))
            @show sum(nres.costs)
            res = nres
        end
    end
    res
end

function clustdist(A :: Array{T,2}, B :: Array{T,2}) where T <: Real
    k = size(A,1)
    map((p) -> sum((A[:,p]-B).^2),permutations(1:k)) |> minimum
end

function sample_pdpp(U :: Array{T,2}) where T <: Real
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p = vec(sum(U.^2;dims=(2)))
    F = zeros(Float64,m,m)
    inds = BitSet()
    for i = 1:m
        itm = sample(Weights(p))
        push!(inds,itm)
        v = U[itm,:]
        f = v
        if (i > 1)
            Fv = @view F[:,1:(i-1)]
            Z = v'*Fv
            f -= Fv*Z'
        end
        F[:,i] = f / sqrt(dot(f,v))
        p = p .- vec(U*F[:,i]).^2
        #Some clean up necessary
        p[p .< 0] .= 0
        for j in inds
            p[j] = 0
        end
    end
    inds
end

function nnclass(X::Matrix,C::Matrix)
    if (size(C,1) < 10)
        tr=KDTree(C)
    else
        tr=BruteTree(C)
    end
    idx,dst = knn(tr,X,1)
    reduce(vcat,idx)
end


function sample_dsquared(X::Matrix,nsamples :: Int)
    m = size(X,1)
    n = size(X,2)
    set = zeros(Int,nsamples)
    #initial point is sampled unif.
    set[1] = rand(1:n)
    d2 = colwise(SqEuclidean(),X,X[:,set[1]])
    for i in 2:nsamples
        set[i] = sample(Weights(d2))
        dd = colwise(SqEuclidean(),X,X[:,set[i]])
        d2 = min.(dd,d2)
    end
    set
end

function random_forest_direct(G::Graph,q::AbstractFloat)
    roots = Set{Int64}()
    root = zeros(Int64,nv(G))
    nroots = Int(0)
    
    n = nv(G)
    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = i
        
        while !in_tree[u]
            if (rand() < q/(q+degree(G,u)))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]
        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    (next=next,roots=roots,nroots=nroots,root=root)
end

function smooth_over_partition(G :: SimpleGraph,root :: Array{Int64,1},y :: Array{Float64,1})
    xhat = zeros(Float64,nv(G))
    #    ysum = weighted_sum_by(y,deg,state.root)
    ysum = sum_by(y,root)
    for v in vertices(G)
        xhat[v] = ysum[root[v]]
    end
    xhat
end


function sure(y,xhat,nroots,s2)
    err = sum((y .- xhat).^2)
    @show err
    -length(y)*s2+err+2*s2*nroots
end

function random_successor(G::SimpleGraph{T},i :: T) where T <: Int
    nbrs = neighbors(G, i)
    rand(nbrs)
end


function smooth_wilson_adapt(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed") where T
    nr = 0;
    rf = random_forest_direct(G,q)
    xhat = smooth_over_partition(G,rf.root,y)
#    @show xhat
    L = lap(G)
    A = (L+q*I)/q
    res = A*xhat - y
    gamma =alpha
#    @show res
    #    @show norm(res)


    for indr in 2:nrep
        rf = random_forest_direct(G,q)
        dir = smooth_over_partition(G,rf.root,res)
        if (step == "optimal")
            u = A*dir
            gamma = ( xhat'*A*u - dot(y,u)   )/dot(u,u)
 #           @show gamma
        elseif (step=="backtrack")
            curr = norm(res)
            gamma = alpha
            while (norm(A*(xhat-gamma*dir) - y) > curr)
                gamma = gamma/2
            end
#            @show gamma
        end

        xhat -= gamma*dir
        res = A*xhat - y
#        @show res
 #       @show norm(res)
        nr += rf.nroots
    end
    (xhat=xhat,nroots=nr/nrep)
end


function smooth_wilson(G :: SimpleGraph{T},q,y :: Vector;nrep=10,variant=1) where T
    xhat = zeros(Float64,length(y));
    nr = 0;
    for indr in Base.OneTo(nrep)
        rf = random_forest_direct(G,q)
        nr += rf.nroots
        if variant==1
            xhat += y[rf.root]
        elseif variant==2
            xhat += smooth_over_partition(G,rf.root,y)
        end
    end
    (xhat=xhat ./ nrep,nroots=nr/nrep)
end
function sum_by(v :: Array{T,1}, g :: Array{Int64,1}) where T
    cc = spzeros(Int64,length(v))
    vv = spzeros(Float64,length(v))
    for i in 1:length(v)
        vv[g[i]] += v[i]
        cc[g[i]] += 1
    end
    nz = findnz(vv)
    for i in nz[1]
        vv[i] /= cc[i]
    end
    vv
end


function lap(G :: SimpleGraph)
    i = collect(1:nv(G))
    j = collect(1:nv(G))
    x = degree(G)
    for u in vertices(G)
        for v in neighbors(G,u)
            push!(i,u)
            push!(j,v)
            push!(x,-1)
        end
    end
    sparse(i,j,x)
end

function sample_ball(n :: Int,d :: Int)
    X = randn(d,n)
    X = X ./ sqrt.(sum( X .^ 2,dims=1))
    r = rand(n).^(1/d)
    r' .* X
end

end # module
