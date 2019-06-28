#basic structure copied from LightGraphs.jl
using DPP
using Test
using LinearAlgebra

const testdir = dirname(@__FILE__)
tests = [
    "solve_sp","sample_dpp","incl_prob"
]

@testset "DPP" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
