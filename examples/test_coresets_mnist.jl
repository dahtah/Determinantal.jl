using Revise
using DPP
using LightGraphs,SparseArrays,LinearAlgebra,StatsBase,Clustering,Combinatorics,Distances,NearestNeighbors
using MLDatasets
train_x, train_y = MNIST.traindata()
X = float.(Matrix(MNIST.convert2features(train_x)));
R=Matrix(qr(randn(size(X,1),10)).Q);
Z=R'*X;

function standardise!(X :: Matrix,along=1)
    X .-= mean(X,dims=along)
    X ./= std(X,dims=along)
    X
end

function mnist_coreset(X :: Matrix,k=10,nrep=3)
    U = Matrix(qr(hcat(X',ones(size(X,2)))).Q)
    #@show size(U,2)
    w = 1 ./ vec(sum(U.^2;dims=(2)))
    pp = collect(sample_pdpp(U))
    C = DPP.kmeans_restart(X[:,pp],k,nrep,w[pp]).centers
    (centers=C,cluster=DPP.nnclass(X,C))
end

function guess_labels(train_y,cluster)
    M =  counts(train_y .+ 1,cluster,(1:10,1:10))
    [findmax(M[i,:])[2] for i in 1:10]
end

function mnist_random_coreset(X :: Matrix,setsize,k=10,nrep=3)
    pp = sample(1:size(X,2),setsize,replace=false)
    C = DPP.kmeans_restart(X[:,pp],k,nrep).centers
    (centers=C,cluster=DPP.nnclass(X,C))
end



