module DPP
using LightGraphs,SparseArrays,LinearAlgebra,StatsBase,Clustering,Combinatorics,Distances,NearestNeighbors
export sample_pdpp,polyfeatures,kmeans_coreset,sample_dsquared

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

end # module
