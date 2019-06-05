function marginal_kernel(L :: Union{Symmetric,Hermitian},α=1)
    (α*L+I)\(α*L)
end

function marginal_kernel(M :: AbstractMatrix,α=1)
    α*M'*((α*M*M' + I)\M)
end


function sample_dpp(L :: Union{Symmetric,Hermitian},k :: Integer)
    n = size(L,1)
    eg = eigen(L)
    eg.values .= abs.(eg.values)
    α = solve_sp(eg.values,k)
    val = α*eg.values ./ (1 .+ α*eg.values)
    incl = rand(n) .< val
    (sample_pdpp(eg.vectors[:,incl]),α)
end


function sample_dpp(L :: Union{Symmetric,Hermitian})
    n = size(L,1)
    eg = eigen(L)
    val = eg.values ./ (1 .+ eg.values)
    incl = rand(n) .< val
    (sample_pdpp(eg.vectors[:,incl]),1)
end

function sample_dpp(L :: Matrix)
    eg = eigen(L'*L)
    val = eg.values ./ (1 .+ eg.values)
    incl = rand(size(L,2)) .< val
    vec = L*eg.vectors[:,incl]
    vec = vec ./ sqrt.(sum(vec .^ 2,dims=1))
    (sample_pdpp(vec),1)
end

function sample_dpp(L :: Matrix,k::Integer)
    eg = eigen(L'*L)
    eg.values .= abs.(eg.values)
    α = solve_sp(eg.values,k)
    val = α*eg.values ./ (1 .+ α*eg.values)
    incl = rand(size(L,2)) .< val
    vec = L*eg.vectors[:,incl]
    vec = vec ./ sqrt.(sum(vec .^ 2,dims=1))
    (sample_pdpp(vec),α)
end


function sample_pdpp(U :: Array{T,2}) where T <: Real
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p = vec(sum(U.^2;dims=(2)))
    F = zeros(Float64,m,m)
    inds = BitSet()
    for i = 1:m
        itm = StatsBase.sample(Weights(p))
        push!(inds,itm)
        v = U[itm,:]
        f = v
        if (i > 1)
            Fv = @view F[:,1:(i-1)]
            Z = v'*Fv
            f -= Fv*Z'
        end
        F[:,i] = f / sqrt(dot(f,v))
        p = p .- vec(U*F[:,i]).^2
        #Some clean up necessary
        p[p .< 0] .= 0
        for j in inds
            p[j] = 0
        end
    end
    inds
end

#Find a rescaling, typically α s.t. Tr(αL(αL + I)^-1) = k
function solve_sp(ls :: Vector,k :: Int)
    v = log.(ls)
    f = (nu) -> (sum(exp.( nu .+ v)./(1 .+ exp.(nu .+ v))) .- k)^2
    res = Optim.optimize(f,-10,10)
    (Optim.minimum(res) > 1e-5) && throw(ArgumentError("Could not find an appropriate rescaling"))
    exp(Optim.minimizer(res))
end
