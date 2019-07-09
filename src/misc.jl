import StatsBase
#using RCall
#misc functions, to clean up

#nearest-neighbour classifier
function nnclass(X::Matrix,C::Matrix)
    if (size(C,1) < 10)
        tr=KDTree(C)
    else
        tr=BruteTree(C)
    end
    idx,dst = knn(tr,X,1)
    reduce(vcat,idx)
end

function kmcost(X::Matrix,C::Matrix)
    if (size(C,1) < 10)
        tr=KDTree(C)
    else
        tr=BruteTree(C)
    end
    idx,dst = knn(tr,X,1)
    s = 0
    for i in 1:length(dst)
        s += dst[i][1]^2
    end
    s
end


#D^2 sampling
function sample_dsquared(X::Matrix,nsamples :: Int)
    m = size(X,1)
    n = size(X,2)
    set = zeros(Int,nsamples)
    #initial point is sampled unif.
    set[1] = rand(1:n)
    d2 = colwise(SqEuclidean(),X,X[:,set[1]])
    for i in 2:nsamples
        set[i] = StatsBase.sample(Weights(d2))
        dd = colwise(SqEuclidean(),X,X[:,set[i]])
        d2 = min.(dd,d2)
    end
    (set=set,min_dist_sq=d2)
end

"""
    kmeans_coreset_iid(X,k,m;kmpp=10)

Construct a coreset for k-means, using the algorithm of Bachem et al. (2017).
Returns a set of indices and a set of weights.
"""
function kmeans_coreset_iid(X,k,m;kmpp=10)
    f,_=kmeans_coreset_heuristic(X,k,m;kmpp=kmpp)
    f = f/sum(f)
    set = StatsBase.sample(1:size(X,2),StatsBase.Weights(f),m)
    (set = set,weights=(1/m)*(1 ./ f[set]))
end

function kmeans_coreset_heuristic(X,k,m;kmpp=10)
    d,n = size(X)
    α=16*(log(k)+2)
    # run km++ several times, keep best set
    Bs = [sample_dsquared(X,k) for _ in 1:kmpp]
    costs = [sum(b.min_dist_sq)/n for b in Bs]
    best = argmin(costs)
    B = Bs[best]

    # build kd-tree for fast nn lookup
    if (d < 10)
        tr=KDTree(X[:,B.set])
    else
        tr=BruteTree(X[:,B.set])
    end

    idx,dst = knn(tr,X,1)
    idx = reduce(vcat,idx)
    dst = reduce(vcat,dst).^2
    s = zeros(n)
    cost = costs[best]
    mdist  =  group_mean(dst,idx) #returns dict. of tuples (group count,mean dist)
    #compute upper bound on sensitivity
    for i in 1:n
        tmp = (1/cost)*(α*dst[i] + 2*α*mdist[idx[i]][2])
        s[i] = tmp + 4*n/mdist[idx[i]][1]
    end
    (s,X[:,B.set])
end


#sample uniformly from the ball in R^d
function sample_ball(n :: Int,d :: Int)
    X = randn(d,n)
    X = X ./ sqrt.(sum( X .^ 2,dims=1))
    r = rand(n).^(1/d)
    r' .* X
end

function group_mean(v :: Array{T,1}, g :: Array{Int64,1}) where T
    dd = Dict{T,Tuple{Int64,T}}()
    vv = spzeros(T,length(v))
    for i in 1:length(v)
        if (haskey(dd,g[i]))
            dd[g[i]] = (dd[g[i]][1]+1,dd[g[i]][2]+v[i])
        else
            dd[g[i]] = (1,v[i])
        end
    end
#    @show keys(dd)
    for k in keys(dd)
        dd[k] = (dd[k][1],dd[k][2]/dd[k][1])
    end
    # nz = findnz(cc)[1]
    # @show nz
    # for i in nz[1]
    #     vv[i] /= cc[i]
    # end
    # (mean=vv,counts=cc)
    dd

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

