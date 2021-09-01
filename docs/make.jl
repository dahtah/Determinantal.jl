push!(LOAD_PATH,"../src/")
using Documenter, DPP
makedocs(sitename="DPP.jl",
         format = Documenter.HTML(
                                      mathengine = MathJax(Dict(
                                          :TeX => Dict(
                                              :equationNumbers => Dict(:autoNumber => "AMS"),
                                              # Custom Tex commands:
                                              :Macros => Dict(
                                                  :X => ["\\mathcal{X}",0],
                                                  :bK => ["\\mathbf{K}",0],
                                                  :bL => ["\\mathbf{L}",0],
                                                  :defd => "≝",
                                                  :ket => ["|#1\\rangle", 1],
                                                  :bra => ["\\langle#1|", 1],
                                              )
                                          )
    ),
)))
