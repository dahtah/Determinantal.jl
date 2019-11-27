module DPP
using LinearAlgebra,StatsBase,Combinatorics,Distances,MLKernels
import Base.show,Base.getindex,LinearAlgebra.diag
export    sample,marginal_kernel,gaussker,rescale!,FullRankEnsemble,LowRankEnsemble,ProjectionEnsemble,
    polyfeatures, rff, inclusion_prob, cardinal, greedy_subset

include("lensemble.jl")
include("kdpp.jl")
include("saddlepoint.jl")
include("features.jl")
include("sampling.jl")
include("subset_kernels.jl")
include("greedy.jl")
end # module

