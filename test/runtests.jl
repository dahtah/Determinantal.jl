#basic structure copied from LightGraphs.jl
using Determinantal
using Test
using LinearAlgebra

const testdir = dirname(@__FILE__)
tests = ["solve_sp", "sample_dpp", "incl_prob", "density"]

@testset "Determinantal" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
