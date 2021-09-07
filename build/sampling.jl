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
    collect(inds)
end
