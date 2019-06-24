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
Lp = polyfeatures(X,10) |> LowRankEnsemble
ind = sample(Lr) |> collect
Plots.scatter(X[1,:],X[2,:],color=:gray,alpha=.5) # hide 
Plots.scatter!(X[1,ind],X[2,ind],color=:red,alpha=1) # hide 
savefig("test3.svg"); # hide 
```
![](test3.svg)


## Functions and types

```@autodocs
Modules = [DPP]
Order   = [:function, :type]
```
