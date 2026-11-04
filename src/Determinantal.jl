module Determinantal
import Combinatorics
import Distances
import Distances: Euclidean
import KernelFunctions
import KernelFunctions:
    ColVecs, Kernel, RowVecs, SqExponentialKernel, kernelmatrix, with_lengthscale
import LinearAlgebra
import LinearAlgebra:
    Diagonal,
    I,
    Symmetric,
    cholesky,
    det,
    diag,
    dot,
    eigen,
    issuccess,
    logabsdet,
    logdet,
    mul!,
    norm,
    qr,
    svd
import NLsolve
import NLsolve: nlsolve
import Random
import Random: AbstractRNG, default_rng
import StatsBase
import StatsBase: Weights
import Base: getindex, show
export LowRank,
    nitems,
    maxrank,
    sample,
    marginal_kernel,
    gaussker,
    rescale!,
    AbstractLEnsemble,
    EllEnsemble,
    ProjectionEnsemble,
    ExtEnsemble,
    MarginalDPP,
    log_prob,
    polyfeatures,
    rff,
    inclusion_prob,
    void_prob,
    size_statistics,
    greedy_subset,
    esp,
    kl_divergence,
    total_variation,
    RowVecs,
    ColVecs,
    LazyDist,
    distance_sampling,
    nystrom_approx,
    match_prob,
    condition_on

include("lowrank.jl")
include("marginal_ker.jl")
include("lensemble.jl")
include("kdpp.jl")
include("saddlepoint.jl")
include("features.jl")
include("sampling.jl")
include("match_prob.jl")
include("conditioning.jl")

include("greedy.jl")
include("ppensembles.jl")
include("kldiv.jl")
#include("nystrom.jl")
include("other.jl")
#include("pivotal_sampling.jl")
end # module
