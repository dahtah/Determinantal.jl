using DPP
using DataFrames
using Gadfly
using Statistics
using StatsBase
X = [randn(3500,3);randn(3500,3).+5;randn(50,3).-4];
V = float.([5 5 5; 0 0 0; -4 -4 -4])
V = Matrix(V')
A = DPP.kmeans_restart(Matrix(X'),3,100).centers
d = size(X,2)
dim_ord = (ord) -> binomial(ord-1+d,d)

function kmeans_random(X :: Array{Float64,2},order :: Int, k :: Int; nrep = 10)
    ns = dim_ord(order)
    ind = sample(1:size(X,2),ns,replace=false)
    DPP.kmeans_restart(X[:,ind],k,nrep)
end

function run_tests(orders)
    test1 = (ord) -> DPP.clustdist(DPP.kmeans_coreset(Matrix(X'),ord,3).centers,V)
    test_rep1 = (ord) -> [test1(ord) for i in 1:30]
    df1 = reduce(vcat,map((ord) -> DataFrame(err=test_rep(ord),order=ord),orders))
    df1[:type] = "DPP"
    test2 = (ord) -> DPP.clustdist(kmeans_random(Matrix(X'),ord,3).centers,V)
    test_rep2 = (ord) -> [test2(ord) for i in 1:30]
    df2 = reduce(vcat,map((ord) -> DataFrame(err=test_rep2(ord),order=ord),orders))
    df2[:type] = "unif"
    df = vcat(df1,df2)
    df[:k] = dim_ord.(df[:order])
    df
end
