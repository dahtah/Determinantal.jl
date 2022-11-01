var documenterSearchIndex = {"docs":
[{"location":"#DPP.jl:-a-Julia-package-for-sampling-Determinantal-Point-Processes","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"DPP.jl provides some types and functions for sampling from DPPs (and related mod els).","category":"page"},{"location":"#Brief-background","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Brief background","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"A Determinantal Point Process is a random subset X of a \"ground set\" Omega. Here we think of Omega as a set of items or points, and X is a random subset selected in a way that preserves some of the \"diversity\" in Omega.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The traditional definition of a DPP you will see in most of the literature is based on inclusion probabilities. We say X is a DPP if for all fixed subsets  S subseteq Omega there exists a matrix bK (called the marginal kernel) such that:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"p(S subseteq X) = det bK_S","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"where bK_S is the sub-matrix of bK indexed by S. For instance, if S=i, a single item, then this says that","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"p(i in X) = K_ii","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"so the inclusion probabilities for single items can be read out of the diagonal of bK. If S = ij, then we have","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"p(S subseteq X) = K_ii K_jj - K_ij^2","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Because K_ij^2 is non-negative, this means that the probability of getting both i and j is less than the product of the invididual probabilities. This tells us that a DPP is in general repulsive (compared to a Poisson process).","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Here is an example of defining a very simple DPP in DPP.jl, over a set of size 2, using the matrix","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"bK = beginpmatrix\n34  14 \n14  34\nendpmatrix","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using DPP\n#A 2x2 matrix, so here Ω={1,2}\nK = [3 1; 1 3]/4\ndpp = MarginalDPP(K)\nsample(dpp) #𝒳, a random subset\n#estimate prob. that item 1 is in 𝒳\nsum([1 ∈ sample(dpp) for _ in 1:1000])/1000\n#should be approx. equal to K[1,1]","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The name \"MarginalDPP\" refers to the fact that the DPP is defined based on its marginal kernel. There is one important constraint when defining DPPs based on the marginal kernel bK; the eigenvalues of bK need to be in 01. Because of this, and for reasons of interpretability, it is often more convenient to work with L-ensembles or extended L-ensembles.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"L-ensembles are a (large) subset of DPPs with the following property. We say X is a L-ensemble if there exists bL such that:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"p(X=X) propto det bL_X","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"This equation relates the likelihood (probability mass function) of the DPP to the determinant of a submatrix. Subsets s.t. bL_X is large have a high probability of being selected. It is quite natural to define a L-ensemble based on kernel matrix that measures similarity (i.e. where L_ij measures the similarity of points i and j). Submatrices of bL with points that are unlike one another (diverse) will be closer to the identity matrix and therefore have a higher determinant. The next example shows a more realistic use of DPP.jl","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using LinearAlgebra\nx = randn(2,500) #some points in dim 2\n\n#compute a kernel matrix for the points in x\nL = [ exp(-norm(a-b)^2) for a in eachcol(x), b in eachcol(x) ]\ndpp = EllEnsemble(L) #form an L-ensemble based on the 𝐋 matrix\nrescale!(dpp,50) #scale so that the expected size is 50\nind = sample(dpp) #a sample from the DPP (indices)\n\nusing Plots\n\ngr() #hide\nscatter(x[1,:],x[2,:],marker_z = map((v) -> v ∈ ind, 1:size(x,2)),legend=:none,alpha=.75) #hide","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The line rescale!(dpp,50) sets the expected size of the L-ensemble to 50. One could also decide to use a fixed-size DPP, and call instead sample(dpp,50), which always returns a subset of size 50. For more on DPPs, L-ensembles, etc. see  Nicolas Tremblay, Simon Barthelm{\\'e}, Konstantin Usevich, Pierre-Olivier Amblard (2021).","category":"page"},{"location":"#Using-other-kernels","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Using other kernels","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"EllEnsemble requires a SPD matrix as input. For more exotic kernels than the Gaussian, you can either do things by hand or use KernelFunctions.jl, which DPP.jl supports.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using KernelFunctions\nx = randn(2,100)\nL = EllEnsemble(ColVecs(x),ExponentialKernel())","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Here we need to specify whether the 2times 100 matrix  'x' should be considered to represent 100 points in dimension 2, or 2 points in dimension 100. ColVecs specifies the former, RowVecs the latter. This mechanism is borrowed from KernelFunctions and used in other places as well.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"See the documentation of KernelFunctions.jl for a list of available kernels.","category":"page"},{"location":"#Using-low-rank-matrices-for-speed","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Using low-rank matrices for speed","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"L-ensembles defined using a full-rank matrix are expensive in large n, because they require an eigendecomposition of the bL matrix (at cost O(n^3)). For practical applications in large n it is preferable to use a low-rank ensemble, i.e. one such that bL = bM bM^t with bM a n times m matrix with m ll n.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"DPP.jl provides a type called \"LowRank\" that represents a symmetric low-rank matrix efficiently:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Z = randn(5,2)\nK = Z*Z' #a rank 2 matrix of size 5x5\nK_lr = LowRank(Z)\nall(K .≈ K_lr)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"It provides a specialised implementation of eigen that only returns the non-null eigenvalues. Here is an example:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"eigen(K_lr).values","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Because K has rank 2, there are only two eigenvalues.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The constructor for EllEnsemble accepts matrices of LowRank type as arguments:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"EllEnsemble(K_lr)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The advantage of using LowRank is that the cost of forming the L-ensemble drops from O(n^3) to O(nm^2), where m is the rank. Note that the maximum size of the DPP cannot exceed m in this case.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The rff functions computes a low-rank approximation to a Gaussian kernel matrix using Random Fourier Features, Ali Rahimi, Benjamin Recht (2007)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = randn(2,1000) #some points in dim 2\nL_rff = rff(ColVecs(x),150,.5) #second argument determines rank, third is standard deviation of Gaussian kernel\nL_exact = gaussker(ColVecs(x),.5)\nlr=EllEnsemble(L_rff)\nlex = EllEnsemble(L_exact)\nplot(sort(lex.λ,rev=true)); plot!(sort(lr.λ,rev=true),legend=:none) # hide","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The plot shows the approximation of the spectrum of the kernel matrix by the low-rank approximation.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Using low-rank representations DPP.jl can scale up to millions of points. Keep in mind that DPPs have good scaling in the size of Omega (n) but poor scaling in the rank (m, number of columns of bM). The overall cost scales as O(nm^2), so m should be kept in the hundreds at most.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"As an alternative to Random Fourier Features, we also provide an implementation of the Nyström approximation Christopher Williams, Matthias Seeger (2001). The function again returns a matrix bM such that bL approx bM bM^t, but the approximation is formed using a subset of points.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = randn(3,1000)\nM_n =  nystrom_approx(ColVecs(x),SqExponentialKernel(),50) #use 50 points\nL = EllEnsemble(ColVecs(x),SqExponentialKernel())\nL_n = EllEnsemble(M_n)\n\nplot(sort(L.λ,rev=:true)) # hide\nplot!(sort(L_n.λ,rev=:true),legend=:false) # hide","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Not all subsets give good Nyström approximations. You can indicate a specific subset to use:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"nystrom_approx(ColVecs(x),SqExponentialKernel(),1:50); #use first 50 points\nnothing; # hide","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Instead of using low-rank approximations to kernel matrices, you can also design your own features","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = randn(3,1000)\n#some non-linear features we've just made up\nfeat = (xi,xj,a) -> @. exp(-a*xi*xj)*cos(xi+xj)\nftrs = [ feat(vec(x[i,:]),vec(x[j,:]),a) for i in 1:3, j in 1:3, a in [0,1,2] if i >= j ]\nM = reduce(hcat,ftrs)\nll = EllEnsemble(LowRank(M))\nrescale!(ll,14)\nsample(ll)\nnothing; # hide","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"A sensible set of features to use are multivariate polynomial features, here used to set up a ProjectionEnsemble (a special case of a low-rank DPP that has fixed sample size, equal to rank of bM)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = randn(2,1000)\nLp = polyfeatures(ColVecs(x),10) |> ProjectionEnsemble\nind = sample(Lp)\nPlots.scatter(x[1,:],x[2,:],color=:gray,alpha=.5,legend=:none) # hide\nPlots.scatter!(x[1,ind],x[2,ind],color=:red,alpha=1) # hide","category":"page"},{"location":"#Inclusion-probabilities","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Inclusion probabilities","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"An attractive aspect of DPPs is that inclusion probabilities are easy to compute. An inclusion probability is the probability that a certain item (or items) is included in the random set produced by a DPP.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using StatsBase,Statistics\n#sample 1,000 times and compute empirical inclusion frequencies\nreps = [StatsBase.counts(sample(Lp),1:Lp.n) for _ in 1:1000];\n#compare to theoretical values\nscatter(inclusion_prob(Lp),mean(reps),legend=:none)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"So far these are just first-order inclusion probabilities. More generally, you can obtain higher-order probabilities (ie prob that items i,j,k,... are in the set jointly) from the marginal kernel of the DPP, given by \"marginal_kernel\"","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"In the next example we compute the empirical inclusion probability of a set of items:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"using Statistics\nx = randn(2,10)\nL = DPP.gaussker(ColVecs(x),.5) |> EllEnsemble\nrescale!(L,4)\nset = [3,5]\n\nincl = [ length(intersect(set,sample(L)))==length(set) for _ in 1:10000];\n#empirical inclusion prob.\nemp = mean(incl)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"The theoretical value is given by","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"th = det(marginal_kernel(L)[set,set])","category":"page"},{"location":"#Other-sampling-methods","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Other sampling methods","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"DPP.jl offers other sampling methods that are based on inter-point distances.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"In these algorithms, the initial point is selected uniformly. At all subsequent steps, the next point is selected based on its distance to the current set X(t), meaning d_t(xX(t)) = min   d(xx_i)  x_i in X(t) . The sampling probability depends on the method. In farthest-point sampling, which is deterministic, at each step, the point selected is one that is farthest from all currently selected points. In D²-sampling Sergei Vassilvitskii, David Arthur (2006), which is a relaxed stochastic version of farthest-point sampling,  points are selected with prob. proportional to squared distance to the current set.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = rand(2,1000);\nind = distance_sampling(ColVecs(x),40,:farthest)\nscatter(x[1,ind],x[2,ind],title=\"Farthest-point sample\",legend=:none)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"ind = distance_sampling(ColVecs(x),40,:d2)\nscatter(x[1,ind],x[2,ind],title=\"D² sample\",legend=:none)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"You can obtain other methods by changing how the prob. of selection depends on the distance. For instance, selecting points uniformly as long as they are more than distance r away from the other points gives a so-called \"hard-sphere\" sample.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"ind = distance_sampling(ColVecs(x),40,(v)-> v > .1) #may get fewer than 40 points\nscatter(x[1,ind],x[2,ind],title=\"Hard-sphere sample\",legend=:none)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Distance-based sampling is quite general, all it needs is a (pseudo-)distance matrix.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"D = [sum(abs.(a-b)) for a in eachcol(x), b in eachcol(x)] #L1 distance\nind = distance_sampling(D,40,(v)-> v > .1) #may get fewer than 40 points\nscatter(x[1,ind],x[2,ind],title=\"Hard-sphere sample in L1 dist\",legend=:none)","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"For datasets that are large, it may not be wise (or even possible) to pre-compute and hold a full distance matrix in memory. You can use the LazyDist type, which behaves like a standard matrix, but whose entries are computed on-the-fly and not stored:","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"D = LazyDist(ColVecs(x),(a,b) -> sum(abs.(a-b)))\nD[3,1] #not pre-computed!\nind = distance_sampling(D,40,(v)-> v >.5);","category":"page"},{"location":"#Extended-L-ensembles","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Extended L-ensembles","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Extended L-ensembles are representations of DPPs that extend L-ensembles, introduced in Nicolas Tremblay, Simon Barthelm{\\'e}, Konstantin Usevich, Pierre-Olivier Amblard (2021). They are defined by a pair of matrices bL and bV, such that","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"p(X=X) propto det beginpmatrix\nbL_X   bV_X \n bV_X^t  mathbf0\nendpmatrix","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"bV","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"should be thought of as defining \"mandatory\" features of the DPP, while bL can be interpreted more or less as a regular kernel, see  Nicolas Tremblay, Simon Barthelm{\\'e}, Konstantin Usevich, Pierre-Olivier Amblard (2021).","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"DPP.jl provides some basic support for defining extended L-ensembles. The following is the \"default\" DPP described in  Nicolas Tremblay, Simon Barthelm{\\'e}, Konstantin Usevich, Pierre-Olivier Amblard (2021), at order 3.","category":"page"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"x = randn(2,1000)\nL = [norm(a-b).^3 for a in eachcol(x), b in eachcol(x)]\nV = polyfeatures(ColVecs(x),1)\nele = ExtEnsemble(L,V)\nrescale!(ele,50)\nind = sample(ele)\nPlots.scatter(x[1,:],x[2,:],color=:gray,alpha=.5) # hide\nPlots.scatter!(x[1,ind],x[2,ind],color=:red,alpha=1,legend=:none) # hide","category":"page"},{"location":"#Functions-and-types","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"Functions and types","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"Modules = [DPP]\nOrder   = [:function, :type]","category":"page"},{"location":"#DPP.cardinal-Tuple{AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.cardinal","text":"cardinal(L::AbstractLEnsemble)\n\nThe size of the set sampled by a DPP is a random variable. This function returns its mean and standard deviation. See also: rescale!, which changes the mean set size.\n\n\n\n\n\n","category":"method"},{"location":"#DPP.distance_sampling","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.distance_sampling","text":"distance_sampling(x,m,sampling)\n\nSelect a random subset of size m from x based on a greedy distance criterion. The initial point is selected uniformly. Then, if sampling == :farthest, at each step, the point selected is one that is farthest from all currently selected points. If sampling == :d2, the algorithm is D²-sampling [vassilvitskii2006k](@vassilvitskii2006k), which is a relaxed stochastic version of farthest-point sampling (selecting points with prob. proportional to squared distance).\n\nx = rand(2, 200);\nind = distance_sampling(ColVecs(x), 40, :farthest)\nscatter(x[1, :], x[2, :]; marker_z=map((v) -> v ∈ ind, 1:size(x, 2)), legend=:none)\n\n\n\n\n\n","category":"function"},{"location":"#DPP.esp","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.esp","text":"esp(L::AbstractLEnsemble,approx=false)\n\nCompute the elementary symmetric polynomials of the L-ensemble L, e₁(L) ... eₙ(L). e₁(L) is the trace and eₙ(L) is the determinant. The ESPs determine the distribution of the sample size of a DPP:\n\np(X = k) = frace_ksum_i=1^n e_i\n\nThe default algorithm uses the Newton equations, but may be unstable numerically for n large. If approx=true, a stable saddle-point approximation (as in Barthelmé et al. (2019)) is used instead for all eₖ with k>5.\n\n\n\n\n\n","category":"function"},{"location":"#DPP.gaussker-Tuple{AbstractVector{T} where T, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.gaussker","text":"gaussker(X,σ)\n\nCompute the Gaussian kernel matrix for points X and parameter σ, ie. a matrix with entry i,j equal to exp(-frac(x_i-x_j)^22σ^2)\n\nIf σ is missing, it is set using the median heuristic. If the number of points is very large, the median is estimated on a random subset.\n\nx = randn(2, 6)\ngaussker(ColVecs(x), 0.1)\n\nSee also: rff, KernelMatrix:kernelmatrix\n\n\n\n\n\n","category":"method"},{"location":"#DPP.greedy_subset-Tuple{AbstractLEnsemble, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.greedy_subset","text":"greedy_subset(L :: AbstractLEnsemble,k)\n\nPerform greedy subset selection: find a subset mathcalX of size k, such that detL_mathcalX is high, by building the set item-by-item and picking at each step the item that maximises the determinant. You can view this as finding an (approximate) mode of the DPP with L-ensemble L.\n\nIf k is too large relative to the (numerical) rank of L, the problem is not well-defined as so the algorithm will stop prematurely.\n\nThe implementation runs in O(nk^2) but is not particularly optimised. If the end result looks screwy, it is probably due to numerical instability: try improving the conditioning of bL.\n\nExample\n\nx = randn(2,100) #10 points in dim 2\ngreedy_subset(EllEnsemble(gaussker(x)),12)\n#same thing but faster:\ngreedy_subset(gaussker(x),12)\n\n\n\n\n\n","category":"method"},{"location":"#DPP.inclusion_prob-Tuple{AbstractLEnsemble, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.inclusion_prob","text":"inclusion_prob(L::AbstractLEnsemble,k)\n\nFirst-order inclusion probabilities in a k-DPP with L-ensemble L. Uses a (typically very accurate) saddlepoint approximation from Barthelmé, Amblard, Tremblay (2019).\n\n\n\n\n\n","category":"method"},{"location":"#DPP.inclusion_prob-Tuple{AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.inclusion_prob","text":"inclusion_prob(L::AbstractLEnsemble)\n\nCompute first-order inclusion probabilities, i.e. the probability that each item in 1..n is included in the DPP.\n\nSee also: marginal_kernel\n\n\n\n\n\n","category":"method"},{"location":"#DPP.kl_divergence-Tuple{AbstractLEnsemble, AbstractLEnsemble, Int64}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.kl_divergence","text":"kl_divergence(L1::AbstractLEnsemble,L2::AbstractLEnsemble,k::Int;nsamples=100)\n\nEstimate the KL divergence between two (k-)DPPs. The KL divergence is estimated by sampling from the k-DPP with L-ensemble L1 and computing the mean log-ratio of the probabilities. nsamples controls how many samples are taken.\n\n\n\n\n\n","category":"method"},{"location":"#DPP.marginal_kernel-Tuple{AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.marginal_kernel","text":" marginal_kernel(L::AbstractLEnsemble)\n\nCompute and return the marginal kernel of a DPP, K. The marginal kernel of a DPP is a (n x n) matrix which can be used to find the inclusion probabilities. For any fixed set of indices ind, the probability that ind is included in a sample from the DPP equals det(K[ind,ind]).\n\n\n\n\n\n","category":"method"},{"location":"#DPP.nystrom_approx-Tuple{AbstractVector{T} where T, KernelFunctions.Kernel, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.nystrom_approx","text":"nystrom_approx(x :: AbstractVector,ker :: Kernel,ind)\n\nCompute a low-rank approximation of a kernel matrix (with kernel \"ker\") using the rows and columns indexed by \"ind\".\n\nusing KernelFunctions\nx = rand(2, 100)\nK = kernelmatrix(SqExponentialKernel(), ColVecs(x))\n#build a rank 30 approx. to K\nV = nystrom_approx(ColVecs(x), SqExponentialKernel(), 1:30)\nnorm(K - V * V') #should be small\n\n\n\n\n\n","category":"method"},{"location":"#DPP.polyfeatures-Tuple{AbstractVector{T} where T, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.polyfeatures","text":"polyfeatures(x,order)\n\nCompute monomial features up to a certain degree. For instance, if the input consists in vectors in ℝ², then the monomials of degree ≤ 2 are 1,x₁,x₂,x₁x₂,x₁²,x₂² Note that the number of monomials of degree r in dimension d equals  d+r choose r\n\nIf the input points are in d > 1, indicate whether the input x is d * n or n * d using ColVecs or RowVecs.\n\nThe output has points along rows and monomials along columns.\n\nExamples\n\nx = randn(2,10) #10 points in dim 2\npolyfeatures(ColVecs(x),2) #Output has 10 rows and 6 column\npolyfeatures(RowVecs(x),2) #Output has 2 rows and 66 columns\n\n\n\n\n\n\n","category":"method"},{"location":"#DPP.rescale!","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.rescale!","text":"rescale!(L,k)\n\nDeclareMathOperatorTrTr\n\nRescale the L-ensemble such that the expected number of samples equals k. The expected number of samples of a DPP equals Tr mathbfLleft( mathbfL + mathbfI right). The function rescales mathbfL to alpha mathbfL such that Tr alpha mathbfLleft( alpha mathbfL + mathbfI right) = k\n\n\n\n\n\n","category":"function"},{"location":"#DPP.rff-Tuple{AbstractMatrix{T} where T, Any, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.rff","text":"rff(X,m,σ)\n\nCompute Random Fourier Features Ali Rahimi, Benjamin Recht (2007) for the Gaussian kernel matrix with input points X and parameter σ. Returns a random matrix M such that, in expectation mathbfMM^t = mathbfK, the Gaussian kernel matrix. M has 2*m columns. The higher m, the better the approximation.\n\nExamples\n\nx = randn(2, 10) #10 points in dim 2\nrff(ColVecs(x), 4, 1.0)\n\nSee also: gaussker, kernelmatrix\n\n\n\n\n\n","category":"method"},{"location":"#DPP.sample-Tuple{AbstractLEnsemble, Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.sample","text":"sample(L::AbstractLEnsemble,k)\n\nSample a k-DPP, i.e. a DPP with fixed size. k needs to be strictly smaller than the rank of L (if it equals the rank of L, use a ProjectionEnsemble).\n\nThe algorithm uses a saddle-point approximation adapted from Barthelmé, Amblard, Tremblay (2019).\n\n\n\n\n\n","category":"method"},{"location":"#DPP.sample-Tuple{AbstractLEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.sample","text":" sample(L::AbstractEnsemble)\n\nSample from a DPP with L-ensemble L. The return type is a BitSet (indicating which indices are sampled), use collect to get a vector of indices instead.\n\n\n\n\n\n","category":"method"},{"location":"#DPP.sample-Tuple{ProjectionEnsemble}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.sample","text":" sample(L::ProjectionEnsemble;nsamples=1)\n\nSample from a projection DPP. If nsamples > 1, return a vector of nsamples realisations from the process.\n\nIf n is much larger than m, this calls the optimised accept/reject sampler instead of the regular sampler. In addition, the leverage scores are precomputed if nsamples > 1.\n\nThe optimised A/R sampler is described in  Barthelme, S, Tremblay, N, Amblard, P-O, (2022)  A Faster Sampler for Discrete Determinantal Point Processes. \n\n    Z = randn(150,10) #random feature matrix\n    Pp = ProjectionEnsemble(Z)\n    sample(Pp) #should output a vector of length 10\n    sample(Pp,nsamples=5) #should output a vector of 5 realisations\n\n\n\n\n\n","category":"method"},{"location":"#DPP.EllEnsemble","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.EllEnsemble","text":"EllEnsemble{T}\n\nThis type represents an L-ensemble, a (broad) special case of Determinantal Point Process.\n\nThe type parameter corresponds to the type of the entries in the matrix given as input (most likely, double precision floats).\n\n\n\n\n\n","category":"type"},{"location":"#DPP.EllEnsemble-Tuple{AbstractVector{T} where T, KernelFunctions.Kernel}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.EllEnsemble","text":"EllEnsemble(X::Matrix{T},k :: Kernel)\n\nConstruct a full-rank ensemble from a set of points and a kernel function.\n\nX is vector of column vectors (ColVecs) or a vector of row vectors (RowVecs) k is a kernel (see doc for package KernelFunctions.jl)\n\nExample: points in 2d along the circle, and an exponential kernel\n\nt = LinRange(-pi,pi,10)'\nX = vcat(cos.(t),sin.(t))\nusing KernelFunctions\nL=EllEnsemble(ColVecs(X),ExponentialKernel())\n\n\n\n\n\n","category":"method"},{"location":"#DPP.EllEnsemble-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.EllEnsemble","text":"EllEnsemble(L :: AbstractMatrix{T})\n\nConstruct an L-ensemble from a (symmetric, non-negative definite) matrix L.\n\nZ = randn(5, 2)\nEllEnsemble(Z * Z') #not very useful, presumably\n\nNote that eigen(L) will be called at construction, which may be computationally costly.\n\nAn L-Ensemble can also be constructed based on lazy matrix types, i.e. types that leverage a low-rank representation. In the example above we could also use the LowRank type:\n\nZ = randn(5, 2)\nEllEnsemble(LowRank(Z)) #more efficient than Z*Z'\n\n\n\n\n\n","category":"method"},{"location":"#DPP.LazyDist-Tuple{Any}","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.LazyDist","text":"LazyDist(x)\n\nLazy representation of a distance matrix, meaning that the object can be indexed like a matrix, but the values are computed on-the-fly. This is useful whenever an algorithm only requires some of the entries, or when the dataset is very large.\n\n\nx = randn(2, 100)\nD = LazyDist(x) #Euclidean dist. by default\nD[3, 2] #the getindex method is defined\nD = LazyDist(x, (a, b) -> sum(abs.(a - b))) #L1 distance\nD[3, 2]\n\n\n\n\n\n","category":"method"},{"location":"#DPP.MarginalDPP-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.MarginalDPP","text":"MarginalDPP(V::Matrix{T})\n\nConstruct a DPP from a matrix defining the marginal kernel. Here the matrix must be square and its eigenvalues must be between 0 and 1.\n\n\n\n\n\n","category":"method"},{"location":"#DPP.ProjectionEnsemble-Union{Tuple{Matrix{T}}, Tuple{T}, Tuple{Matrix{T}, Any}} where T","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.ProjectionEnsemble","text":"ProjectionEnsemble(V::Matrix{T},orth=true)\n\nConstruct a projection ensemble from a matrix of features. Here we assume mathbfL = mathbfVmathbfV^t, so that V must be n \\times r, where n is the number of items and r is the rank. V needs not be orthogonal. If orth is set to true (default), then a QR decomposition is performed. If V is orthogonal already, then this computation may be skipped, and you can set orth to false.\n\n\n\n\n\n","category":"method"},{"location":"#References","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"References","text":"","category":"section"},{"location":"","page":"DPP.jl: a Julia package for sampling Determinantal Point Processes","title":"DPP.jl: a Julia package for sampling Determinantal Point Processes","text":"","category":"page"}]
}
