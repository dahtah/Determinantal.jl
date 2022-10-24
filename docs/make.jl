push!(LOAD_PATH, "../src/")
using Documenter, DPP
using DocumenterCitations
# makedocs(
#     bib,
#     sitename = "DPP.jl")

bib = CitationBibliography("biblio.bib")
makedocs(
    bib;
    sitename="DPP.jl",
    format=Documenter.HTML(;
        mathengine=MathJax(
            Dict(
                :TeX => Dict(
                    :equationNumbers => Dict(:autoNumber => "AMS"),
                    # Custom Tex commands:
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
)

deploydocs(; repo="github.com/dahtah/DPP.jl.git", devbranch="main")
