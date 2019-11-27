module DPP
using LinearAlgebra,StatsBase,Combinatorics,Distances,MLKernels
import Base.show,Base.getindex
export sample,marginal_kernel,gaussker,rescale!,FullRankEnsemble,LowRankEnsemble,ProjectionEnsemble, polyfeatures, rff, inclusion_prob, cardinal

include("lensemble.jl")
include("kdpp.jl")
include("saddlepoint.jl")
include("features.jl")
include("sampling.jl")
include("subset_kernels.jl")
end # module

