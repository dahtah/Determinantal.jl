

function sample_dpp(L :: Matrix,feat=false)
    n = size(L,1)
    if (!feat) #We got an L-ensemble
        ev = eigen(L)
        val = ev.values ./ (1 .+ ev.values)
        incl = rand(n) .< val
        sample_pdpp(eg.vectors[:,incl])
    else
        eg = eigen(L'*L)
        val = eg.values ./ (1 .+ eg.values)
        incl = rand(size(L,2)) .< val
        vec = L*eg.vectors[:,incl]
        vec = vec ./ sqrt.(sum(vec .^ 2,dims=1))
        sample_pdpp(vec)
    end
end

function sample_pdpp(U :: Array{T,2}) where T <: Real
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p = vec(sum(U.^2;dims=(2)))
    F = zeros(Float64,m,m)
    inds = BitSet()
    for i = 1:m
        itm = sample(Weights(p))
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
    f = (nu) -> (sum(exp.(nu+v)./(1+exp.(nu+v)))-k)^2
    Optim.optimize(f,-10,10)
end
