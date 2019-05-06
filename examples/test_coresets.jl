using Revise
using DPP
using DataFrames
using Gadfly
using Statistics
using StatsBase
using Clustering
function gen_clust(M,csize;sigma2=1)
    d = size(M,1)
    s = sqrt(sigma2)
    reduce(hcat,map((i) -> M[:,i] .+ s*randn(d,csize[i]),1:length(csize)))
end

M = [-1 0 1; 0 -1 -1; 1 -1 1]
d = 3

csize = [10000,10000,50]
cluster= reduce(vcat,map(i-> repeat([i],csize[i]),1:length(csize)))


# X = [randn(9500,3);randn(9500,3).+5;randn(50,3).-4];
# V = float.([5 5 5; 0 0 0; -4 -4 -4])
# V = Matrix(V')
# A = DPP.kmeans_restart(Matrix(X'),3,100).centers
# d = size(X,2)
dim_ord = (ord) -> binomial(ord-1+d,d)

function kmeans_random(X :: Array{Float64,2},ns :: Int, k :: Int; nrep = 10)
    ind = sample(1:size(X,2),ns,replace=false)
    C= DPP.kmeans_restart(X[:,ind],k,nrep).centers
    (centers=C,cluster=DPP.nnclass(X,C))
end

function run_tests(orders,nrep=30)

    test1 = (ord) -> randindex(DPP.kmeans_coreset(X,ord,3).cluster,cluster)
    test_rep1 = (ord) -> [test1(ord)[2] for i in 1:nrep]
    df1 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep1(ord),order=ord),orders))
    df1[:type] = "DPP"
    test2 = (ord) -> randindex(kmeans_random(X,dim_ord(ord),3).cluster,cluster)
    test_rep2 = (ord) -> [test2(ord)[2] for i in 1:nrep]
    df2 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep2(ord),order=ord),orders))
    df2[:type] = "unif"
    test3 = (ord) -> randindex(DPP.kmeans_d2(X,dim_ord(ord),3).cluster,cluster)
    test_rep3 = (ord) -> [test3(ord)[2] for i in 1:nrep]
    df3 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep3(ord),order=ord),orders))
    df3[:type] = "D2"



    #    test2 = (ord) -> DPP.clustdist(kmeans_random(Matrix(X'),dim_ord(ord),3).centers,V)
    # test_rep2 = (ord) -> [test2(ord) for i in 1:30]
    # df2 = reduce(vcat,map((ord) -> DataFrame(err=test_rep2(ord),order=ord),orders))
    # df2[:type] = "unif"
    # test3 = (ord) -> DPP.clustdist(DPP.kmeans_d2(Matrix(X'),dim_ord(ord),3).centers,V)
    # test_rep3 = (ord) -> [test3(ord) for i in 1:30]
    # df3 = reduce(vcat,map((ord) -> DataFrame(err=test_rep3(ord),order=ord),orders))
    # df3[:type] = "d2"
    df = vcat(df1,df2,df3)
    df[:k] = dim_ord.(df[:order])
    df
end
