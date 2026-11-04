using Determinantal, LinearAlgebra
using Documenter, DocumenterCitations
using Documenter.Remotes: GitHub
using Literate
using Pkg

bib = CitationBibliography(joinpath(@__DIR__, "biblio.bib"))
const NOTEBOOKS_SRC_DIR = joinpath(@__DIR__, "src", "notebooks")

# Convert Literate.jl notebooks to markdown
function generate_notebooks()
    notebooks = ["recommendation"]
    cur_dir = pwd()
    for nb in notebooks
        nb_path = joinpath(NOTEBOOKS_SRC_DIR, nb)
        if isdir(nb_path)
            cd(nb_path)
            Pkg.activate(".")
            Pkg.instantiate()
            fl = "$(nb).jl"
            Literate.markdown(fl, NOTEBOOKS_SRC_DIR; documenter = true, execute = true)
            cd(cur_dir)
            Pkg.activate(".")
        else
            @warn "Notebook source file not found: $(nb_path)"
        end
    end
end

# Generate notebooks before makedocs
generate_notebooks()

makedocs(;
    modules = [Determinantal],
    authors = "Simon Barthelme <simon.barthelme@gipsa-lab.fr>,Nicolas Tremblay <nicolas.tremblay@gipsa-lab.fr>",
    repo = GitHub("dahtah/Determinantal.jl"),
    plugins = [bib],
    sitename = "Determinantal.jl",
    format = Documenter.HTML(;
        canonical = "https://dahtah.github.io/Determinantal.jl",
        mathengine = MathJax(
            Dict(
                :TeX => Dict(
                    :equationNumbers => Dict(:autoNumber => "AMS"),
                    :Macros => Dict(
                        :X => ["\\mathcal{X}", 0],
                        :bK => ["\\mathbf{K}", 0],
                        :bL => ["\\mathbf{L}", 0],
                        :bM => ["\\mathbf{M}", 0],
                        :bV => ["\\mathbf{V}", 0],
                        :O => ["\\mathcal{O}", 0],
                        :defd => "≝",
                        :ket => ["|#1\\rangle", 1],
                        :bra => ["\\langle#1|", 1],
                    ),
                ),
            ),
        ),
    ),
    pages = [
        "Basic usage" => "index.md"
        "Reference" => "reference.md"
        "Examples" => ["Building a recommender system" => "notebooks/recommendation.md"]
    ],
)



deploydocs(; repo = "github.com/dahtah/Determinantal.jl", devbranch = "main")
