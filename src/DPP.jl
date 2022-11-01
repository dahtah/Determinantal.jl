module DPP
using LinearAlgebra, StatsBase, Combinatorics, Distances, KernelFunctions
import Base.show, Base.getindex, LinearAlgebra.diag
export LowRank,
    nitems,
    maxrank,
    sample,
    marginal_kernel,
    gaussker,
    rescale!,
    AbstractLEnsemble,
    EllEnsemble,
    LowRankEnsemble,
    ProjectionEnsemble,
    ExtEnsemble,
    MarginalDPP,
    log_prob,
    polyfeatures,
    rff,
    inclusion_prob,
    cardinal,
    greedy_subset,
    esp,
    kl_divergence,
    total_variation,
    RowVecs,
    ColVecs,
    LazyDist,
    distance_sampling,
    nystrom_approx

include("lowrank.jl")
include("marginal_ker.jl")
include("lensemble.jl")
include("kdpp.jl")
include("saddlepoint.jl")
include("features.jl")
include("sampling.jl")
include("greedy.jl")
include("ppensembles.jl")
include("kldiv.jl")
include("nystrom.jl")
include("other.jl")
#include("pivotal_sampling.jl")
end # module
