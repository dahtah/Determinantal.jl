# DPP.jl: a Julia package for sampling Determinantal Point Processes

DPP.jl provides some types and functions for sampling from DPPs (and related mod
els). 

# Brief background 

A Determinantal Point Process is a random subset $\X$ of a "ground set" $\Omega$. Here we think of $\Omega$ as a set of items or points, and $\X$ is a random subset selected in a way that preserves some of the "diversity" in $\Omega$.

The traditional definition of a DPP you will see in most of the literature is based on *inclusion probabilities*. We say $\X$ is a DPP if for all fixed subsets  $S \subseteq \Omega$ there exists a matrix $\bK$ (called the marginal kernel) such that:
```math
p(S \subseteq \X) = \det \bK_{S} 
```

where $\bK_{S}$ is the sub-matrix of $\bK$ indexed by $S$. For instance, if $S=\{i\}$, a single item, then this says that 
```math
p(i \in \X) = K_{i,i} 
```
so the inclusion probabilities for single items can be read out of the diagonal of $\bK$. If $S =\{ i,j\}$, then we have 
```math
p(S \subseteq \X) = K_{ii} K_{jj} - K_{ij}^2 
```

Because $K_{ij}^2$ is non-negative, this means that the probability of getting both $i$ and $j$ is less than the product of the invididual probabilities. This tells us that a DPP is in general repulsive (compared to a Poisson process).

Here is an example of defining a very simple DPP in DPP.jl, over a set of size 2, using the matrix 
```math
\bK = \begin{pmatrix}
3/4 & 1/4 \\
1/4 & 3/4 
\end{pmatrix}
```


```{julia}
#A 2x2 matrix, so here Ω={1,2}
K = [3 1; 1 3]/4
dpp = FullRankDPP(K)
sample(dpp) #𝒳, a random subset
#estimate prob. that item 1 is in 𝒳 
sum([1 ∈ sample(dpp) for _ in 1:1000])/1000
#should be approx. equal to K[1,1]
```

There is one important constraint when defining DPPs based on the marginal kernel $\bK$; the eigenvalues of $\bK$ need to be in $[0,1]$. Because of this, and for reasons of interpretability, it is often more convenient to work with L-ensembles or extended L-ensembles. 

L-ensembles are a (large) subset of DPPs with the following property. We say $\X$ is a L-ensemble if there exists $\bL$ such that:
```math
p(\X=X) \propto \det \bL_{X}
```
This equation relates the *likelihood* (probability mass function) of the DPP to the determinant of a submatrix. Subsets s.t. $\bL_{X}$ is large have a high probability of being selected. It is quite natural to define a L-ensemble based on kernel matrix that measures similarity (i.e. where $L_{ij}$ measures the similarity of points $i$ and $j$). Submatrices of $\bL$ with points that are unlike one another (diverse) will be closer to the identity matrix and therefore have a higher determinant. 
The next example shows a more realistic use of DPP.jl
```{julia}
x = randn(2,500) #some points in dim 2

#compute a kernel matrix for the points in x 
L = [ exp(-norm(a-b)^2) for a in eachcol(x), b in eachcol(x) ]
dpp = FullRankEnsemble(L) #form an L-ensemble based on the 𝐋 matrix 
rescale!(dpp,50) #scale so that the expected size is 50
ind = sample(dpp) #a sample from the DPP (indices)

using Plots
scatter(x[1,:],x[2,:],marker_z = map((v) -> v ∈ ind, 1:size(x,2)),legend=:none,alpha=.75) #show the selected points in white
savefig("example1.svg"); #hide
```
![](example1.svg)

The line `rescale!(dpp,50)` sets the expected size of the L-ensemble to 50. One could also decide to use a fixed-size DPP, and call instead `sample(dpp,50)`, which always returns a subset of size 50. For more on DPPs, L-ensembles, etc. see [ref].

## Using low-rank matrices for speed

L-ensembles defined via 'FullRankEnsemble' are the most general kind but not very efficient. They require an eigendecomposition of the $\bL$ matrix, which comes at cost $\O(n^3)$. For practical applications in large $n$ it is preferable to use a low-rank ensemble, i.e. one such that $\bL = \bM \bM^t$ with $\bM$ a $n$ times $m$ matrix with $m \ll n$.

The function rff computes a low-rank approximation (Random Fourier Features) to a Gaussian kernel matrix:
```{julia}
x = randn(2,1000) #some points in dim 2
M = rff(x,150,.5) #second argument determines rank, third is standard deviation of Gaussian kernel
lr=LowRankEnsemble(M)
rescale!(lr,50)
ind = sample(lr) #a sample from the DPP (indices)

scatter(x[1,:],x[2,:],marker_z = map((v) -> v ∈ ind, 1:size(x,2)),legend=:none,alpha=.75) #show the selected points in white
```
Using low-rank representations DPP.jl can scale up to millions of points. Keep in mind that DPPs have good scaling in the size of $\Omega$ (n) but poor scaling in the rank ($m$, number of columns of $\bM$). The overall cost scales as $\O(nm^2)$, so $m$ should be kept in the hundreds at most. 

Instead of using low-rank approximations to kernel matrices, you can also design your own features 
```{julia}
x = randn(3,1000)
#some non-linear features we've just made up
feat = (xi,xj,a) -> @. exp(-a*xi*xj)*cos(xi+xj)
ftrs = [ feat(vec(x[i,:]),vec(x[j,:]),a) for i in 1:3, j in 1:3, a in [0,1,2] if i >= j ]
M = reduce(hcat,ftrs)
ll = LowRankEnsemble(M)
rescale!(ll,14)
sample(ll)
```

A sensible set of features to use are multivariate polynomial features, here used to set up a ProjectionEnsemble (a special case of a low-rank DPP that has fixed sample size, equal to rank of $\bM$)
```@example 1
x = randn(2,1000)
Lp = polyfeatures(x,10) |> ProjectionEnsemble
ind = sample(Lp) 
Plots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide 
Plots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide 
savefig("test3.svg"); # hide 
```
![](test3.svg)


# Inclusion probabilities 

An attractive aspect of DPPs is that inclusion probabilities are easy to compute. An inclusion probability is the probability that a certain item (or items) is included in the random set produced by a DPP. 

```@example 1
using StatsBase,Statistics
#sample 1,000 times and compute empirical inclusion frequencies 
reps = [StatsBase.counts(sample(Lr),1:Lr.n) for _ in 1:1000];
#compare to theoretical values
scatter(inclusion_prob(Lr),mean(reps))
savefig("example_incl.svg"); nothing # hide 
```
![](example_incl.svg)

So far these are just first-order inclusion probabilities. More generally, you can obtain higher-order probabilities (ie prob that items i,j,k,... are in the set *jointly*) from the marginal kernel of the DPP, given by "marginal_kernel"

In the next example we compute the empirical inclusion probability of a set of items:
```@example 1
using LinearAlgebra,Statistics
X = randn(2,10)
L = DPP.gaussker(X,.5) |> FullRankEnsemble
rescale!(L,4)
set = [3,5]

incl = [ length(intersect(set,sample(L)))==length(set) for _ in 1:10000];
#empirical inclusion prob.
emp = mean(incl)
```

The theoretical value is given by 
```@example 1
th = det(marginal_kernel(L)[[3,5],[3,5]])
```

## Functions and types

```@autodocs
Modules = [DPP]
Order   = [:function, :type]
```

