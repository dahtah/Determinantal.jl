using Determinantal
using Test
using Aqua
using JET
using Statistics, LinearAlgebra

Aqua.test_all(Determinantal)
import CompatHelperLocal as CHL
CHL.@check()

# JET static analysis: possible errors are reported as warnings only, never as
# test failures (on Julia 1.10 JET false-positives on Adjoint-of-view matvecs
# in sample_pdpp, see src/sampling.jl).
@testset "JET static analysis (warnings only)" begin
    include(joinpath(@__DIR__, "..", "src", "workload.jl"))
    result = JET.report_call(exercise_determinantal)
    # `report_call` returns a JETCallResult on recent JET versions, a plain
    # vector of reports on older ones.
    reports = result isa JET.JETCallResult ? JET.get_reports(result) : result
    if isempty(reports)
        @info "JET static analysis: no issues found"
    else
        @warn "JET static analysis found $(length(reports)) possible error(s) (treated as warnings, not failures)" reports
    end
    @test true # informational only: JET findings never fail the suite
end


#=
Don't add your tests to runtests.jl. Instead, create files named

    test-title-for-my-test.jl

The file will be automatically included inside a `@testset` with title "Title For My Test".
=#
for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if isnothing(match(r"^test-.*\.jl$", file))
            continue
        end
        title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
        @testset "$title" begin
            include(file)
        end
    end
end
