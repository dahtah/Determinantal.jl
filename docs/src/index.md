# DPP.jl: a Julia package for sampling Determinantal Point Processes

DPP.jl provides some types and functions for sampling from DPPs (and related models). 

## Quick start

To define a DPP, first define an L-ensemble. The L-ensemble can either be defined as:

- full-rank, in which case it's represented as a $n \times n$ matrix $\mathbf{L}$
- low-rank, in which case it's represented as $\mathbf{L} = \mathbf{M}\mathbf{M}^t$ where $\mathbf{M}$ is $n \times m$, $m \leq n$. Low-rank ensembles are always faster to sample from. 
- "projection", which is just like low-rank, expect you're restricted to sampling exactly $m$ points (i.e., the rank of the matrix)

An example for full-rank L-ensembles:
```@example 1
using DPP
X = randn(2,1000) #1,000 points in dim 2
L = DPP.gaussker(X,.5) |> FullRankEnsemble
rescale!(L,40)
ind = sample(L) |> collect #sample returns a BitSet, we collect all indices

# On this plot the original points are in grey, the sampled ones in red
using Plots
Plots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5)
Plots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1)
savefig("test.svg"); # hide
```

![](test.svg)

For low-rank ensembles, we can use an RFF approximation:
```@example 1
Lr = rff(X,150,.5) |> LowRankEnsemble
rescale!(Lr,40)
ind = sample(Lr) |> collect
Plots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide 
Plots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide 
savefig("test2.svg") # hide 
```
![](test2.svg)


Example using polynomial features and a projection ensemble: 
```@example 1
Lp = polyfeatures(X,10) |> ProjectionEnsemble
ind = sample(Lp) |> collect
Plots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide 
Plots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide 
savefig("test3.svg"); # hide 
```
![](test3.svg)


# Inclusion probabilities 

An attractive aspect of DPPs is that inclusion probabilities are easy to compute. An inclusion probability is the probability that a certain item (or items) is included in the random set produced by a DPP. 

```@example 1
using StatsBase
#sample 1,000 times and compute empirical inclusion frequencies 
reps = [StatsBase.counts(collect(sample(Lr)),1:Lr.n) for _ in 1:1000];
#compare to theoretical values
scatter(inclusion_prob(Lr),mean(reps))
savefig("example_incl.svg"); # hide 
```
![](example_incl.svg)

So far these are just first-order inclusion probabilities. More generally, you can obtain higher-order probabilities (ie prob that items i,j,k,... are in the set *jointly*) from the marginal kernel of the DPP, given by "marginal_kernel"

In the next example we compute the empirical inclusion probability of a set of items:
```@example 1
using LinearAlgebra
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
th = det(marginal_kernel(L)[set,set])
```


# k-DPPs

Generally, a DPP will generate a subset of random size (except for projection DPPs). If you'd like to sample a subset of fixed size, use a k-DPP. Specifying a size argument in the "sample" function will do the trick: 
```@example 1
sample(Lr,20) |> length
```

Be careful, k-DPPs do not have a marginal kernel. Although the inclusion probabilities are nominally intractable, there exist good approximations that can be computed quickly. Use *inclusion_prob* and specify *k* to get approximate first-order inclusion probabilities. 



## Functions and types

```@autodocs
Modules = [DPP]
Order   = [:function, :type]
```

