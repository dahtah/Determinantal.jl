var documenterSearchIndex = {"docs":
[{"location":"#DPP.jl:-a-Julia-package-for-sampling-Determinantal-Point-Processes-1","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"","category":"section"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"DPP.jl provides some types and functions for sampling from DPPs (and related models). ","category":"page"},{"location":"#Quick-start-1","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Quick start","text":"","category":"section"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"To define a DPP, first define an L-ensemble. The L-ensemble can either be defined as:","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"full-rank, in which case it's represented as a n times n matrix mathbfL\nlow-rank, in which case it's represented as mathbfL = mathbfMmathbfM^t where mathbfM is n times m, m leq n. Low-rank ensembles are always faster to sample from. \n\"projection\", which is just like low-rank, expect you're restricted to sampling exactly m points (i.e., the rank of the matrix)","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"An example for full-rank L-ensembles:","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using DPP\nX = randn(2,1000) #1,000 points in dim 2\nL = DPP.gaussker(X,.5) |> FullRankEnsemble\nrescale!(L,40)\nind = sample(L) |> collect #sample returns a BitSet, we collect all indices\n\n# On this plot the original points are in grey, the sampled ones in red\nusing Plots\nPlots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5)\nPlots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1)\nsavefig(\"test.svg\"); # hide","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"(Image: )","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"For low-rank ensembles, we can use an RFF approximation:","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Lr = rff(X,150,.5) |> LowRankEnsemble\nrescale!(Lr,40)\nind = sample(Lr) |> collect\nPlots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide \nPlots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide \nsavefig(\"test2.svg\") # hide ","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"(Image: )","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Example using polynomial features and a projection ensemble: ","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Lp = polyfeatures(X,10) |> LowRankEnsemble\nind = sample(Lr) |> collect\nPlots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide \nPlots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide \nsavefig(\"test3.svg\"); # hide ","category":"page"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"(Image: )","category":"page"},{"location":"#Functions-and-types-1","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Functions and types","text":"","category":"section"},{"location":"#","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Modules = [DPP]\nOrder   = [:function, :type]","category":"page"},{"location":"#DPP.gaussker-Tuple{Array{T,2} where T,Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.gaussker","text":"gaussker(X,σ)\n\nCompute the Gaussian kernel matrix for X and parameter σ, ie. a matrix with entry i,j equal to exp(-frac(x_i-x_j)^22σ^2)\n\nSee also: rff, kernelmatrix \n\n\n\n\n\n","category":"method"},{"location":"#DPP.inclusion_prob-Tuple{DPP.AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.inclusion_prob","text":"inclusion_prob(L::AbstractLEnsemble)\n\nCompute first-order inclusion probabilities, i.e. the probability that each item in 1..n is included in the DPP.\n\nSee also: marginal_kernel\n\n\n\n\n\n","category":"method"},{"location":"#DPP.marginal_kernel-Tuple{DPP.AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.marginal_kernel","text":" marginal_kernel(L::AbstractLEnsemble)\n\nCompute and return the marginal kernel of a DPP, K. The marginal kernel of a DPP is a (n x n) matrix which can be used to find the inclusion probabilities. For any fixed set of indices ind, the probability that ind is included in a sample from the DPP equals det(K[ind,ind]). \n\n\n\n\n\n","category":"method"},{"location":"#DPP.polyfeatures-Union{Tuple{T}, Tuple{Array{T,2},Int64}} where T<:Real","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.polyfeatures","text":"polyfeatures(X,order)\n\nCompute monomial features up to a certain degree. For instance, if X is a 2 x n matrix and the degree argument equals 2, it will return a matrix with columns 1,X[1,:],X[2,:],X[1,:].^2,X[2,:].^2,X[1,:]*X[2,:] Note that the number of monomials of degree r in dimension d equals  d+r choose r\n\nX is assumed to be of dimension d times n where d is the dimension and n is the number of points.\n\nExamples\n\nX = randn(2,10) #10 points in dim 2\npolyfeatures(X,2) #Output has three columns\n\n\n\n\n\n","category":"method"},{"location":"#DPP.rescale!","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.rescale!","text":"rescale!(L,k)\n\nDeclareMathOperatorTrTr\n\nRescale the L-ensemble such that the expected number of samples equals k. The expected number of samples of a DPP equals Tr mathbfLleft( mathbfL + mathbfI right). The function rescales mathbfL to alpha mathbfL such that Tr alpha mathbfLleft( alpha mathbfL + mathbfI right) = k\n\n\n\n\n\n","category":"function"},{"location":"#DPP.rff-Tuple{Array{T,2} where T,Any,Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.rff","text":"rff(X,m,σ)\n\nCompute Random Fourier Features for the Gaussian kernel matrix with input points X and parameter σ. Returns a random matrix M such that, in expectation mathbfMM^t = mathbfK, the Gaussian kernel matrix.  M has 2*m columns. The higher m, the better the approximation. \n\nExamples\n\nX = randn(2,10) #10 points in dim 2\nrff(X,4,1.0)\n\nSee also: gaussker, kernelmatrix \n\n\n\n\n\n","category":"method"},{"location":"#DPP.sample-Tuple{ProjectionEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.sample","text":" sample(L::AbstractEnsemble)\n\nSample from a DPP with L-ensemble L. The return type is a BitSet (indicating which indices are sampled), use collect to get a vector of indices instead.\n\n\n\n\n\n","category":"method"},{"location":"#DPP.FullRankEnsemble","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.FullRankEnsemble","text":"FullRankEnsemble{T}\n\nThis type represents an L-ensemble where the matrix L is full rank. This is the most general representation of an L-ensemble, but also the least efficient, both in terms of memory and computation.\n\nAt construction, an eigenvalue decomposition of L will be performed, at O(n^3) cost.\n\nThe type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats). \n\n\n\n\n\n","category":"type"},{"location":"#DPP.FullRankEnsemble-Union{Tuple{Array{T,2}}, Tuple{T}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.FullRankEnsemble","text":"FullRankEnsemble(V::Matrix{T})\n\nConstruct a full-rank ensemble from a matrix. Here the matrix must be square. \n\n\n\n\n\n","category":"method"},{"location":"#DPP.FullRankEnsemble-Union{Tuple{T}, Tuple{Array{T,2},Kernel}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.FullRankEnsemble","text":"FullRankEnsemble(X::Matrix{T},k :: Kernel)\n\nConstruct a full-rank ensemble from a set of points and a kernel function.\n\nX (the set of points) is assumed to have dimension d x n, where d is the dimension and n is the number of points. k is a kernel (see doc for package MLKernels)\n\nExample: points in 2d along the circle, and an exponential kernel\n\nt = LinRange(-pi,pi,10)'\nX = vcat(cos.(t),sin.(t))\nusing MLKernels\nL=FullRankEnsemble(X,ExponentialKernel(.1))\n\n\n\n\n\n","category":"method"},{"location":"#DPP.LowRankEnsemble","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.LowRankEnsemble","text":"This type represents an L-ensemble where the matrix L is low rank. This enables faster computation. \n\nThe type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats)\n\n\n\n\n\n","category":"type"},{"location":"#DPP.LowRankEnsemble-Union{Tuple{Array{T,2}}, Tuple{T}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.LowRankEnsemble","text":"LowRankEnsemble(V::Matrix{T})\n\nConstruct a low-rank ensemble from a matrix of features. Here we assume  mathbfL = mathbfVmathbfV^t, so that V must be n \\times r, where n is the number of items and r is the rank of the L-ensemble.\n\nYou will not be able to sample a number of items greater than the rank. At construction, an eigenvalue decomposition of V'*V will be perfomed, with cost nr^2. \n\n\n\n\n\n","category":"method"},{"location":"#DPP.ProjectionEnsemble-Union{Tuple{Array{T,2}}, Tuple{T}, Tuple{Array{T,2},Any}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.ProjectionEnsemble","text":"ProjectionEnsemble(V::Matrix{T},orth=true)\n\nConstruct a projection ensemble from a matrix of features. Here we assume  mathbfL = mathbfVmathbfV^t, so that V must be n \\times r, where n is the number of items and r is the rank. V needs not be orthogonal. If orth is set to true (default), then a QR decomposition is performed. If V is orthogonal already, then this computation may be skipped, and you can set orth to false. \n\n\n\n\n\n","category":"method"}]
}