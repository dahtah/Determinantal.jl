using Revise
using DPP
using LightGraphs,SparseArrays,LinearAlgebra,StatsBase,Clustering,Combinatorics,Distances,NearestNeighbors
using MLDatasets
train_x, train_y = MNIST.traindata()
X = Matrix(MNIST.convert2features(train_x));

function mnist_coreset(k=10,nrep=3)
    U = Matrix(qr(hcat(X',ones(size(X,2)))).Q)
    #@show size(U,2)
    w = 1 ./ vec(sum(U.^2;dims=(2)))
    pp = collect(sample_pdpp(U))
    C = DPP.kmeans_restart(X[:,pp],k,nrep,w[pp]).centers
    (centers=C,cluster=DPP.nnclass(X,C))
end



