module DPP
using LightGraphs,SparseArrays,LinearAlgebra,StatsBase,Clustering,Combinatorics,Distances,NearestNeighbors,MLKernels,Optim
# export sample_pdpp,polyfeatures,kmeans_coreset,sample_dsquared,random_forest_direct,smooth_wilson,rff,gaussker,sample_dpp
export sample,marginal_kernel,gaussker,rescale!,FullRankEnsemble,LowRankEnsemble,ProjectionEnsemble,show, polyfeatures, rff

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

include("lensemble.jl")
include("laplacians.jl")
include("features.jl")
include("sampling.jl")

end # module

