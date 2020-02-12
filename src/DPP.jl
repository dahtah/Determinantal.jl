module DPP
using LinearAlgebra,StatsBase,Combinatorics,Distances,MLKernels
import Base.show,Base.getindex,LinearAlgebra.diag
export nitems,maxrank,
    sample,marginal_kernel,gaussker,rescale!,FullRankEnsemble,LowRankEnsemble,ProjectionEnsemble,PPEnsemble,log_prob,
    polyfeatures, rff, inclusion_prob, cardinal, greedy_subset,
    esp, kl_divergence

include("lensemble.jl")
include("kdpp.jl")
include("saddlepoint.jl")
include("features.jl")
include("sampling.jl")
include("subset_kernels.jl")
include("greedy.jl")
include("ppensembles.jl")
include("kldiv.jl")


end # module

