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

#D^2 sampling
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

#sample uniformly from the ball in R^d
function sample_ball(n :: Int,d :: Int)
    X = randn(d,n)
    X = X ./ sqrt.(sum( X .^ 2,dims=1))
    r = rand(n).^(1/d)
    r' .* X
end

