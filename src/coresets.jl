#Functions for building coresets 

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

#Empirical volume of Voronoi cell, used for weights in D^2 sampling
function empirical_volume(X :: Matrix,ind :: Vector{Int})
    tr=KDTree(X[:,ind])
    idx,dst = knn(tr,X,1)
    counts(reduce(vcat,idx))
end
