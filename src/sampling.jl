using LoopVectorization

function sample_pdpp(U)
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

function sample_pdpp2(U) 
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p = vec(sum(U.^2;dims=(2)))
    F = zeros(Float64,m,m)
    f = zeros(m)
    tmp = zeros(n)
    inds = BitSet()
    @inbounds for i = 1:m
        itm = StatsBase.sample(Weights(p))
        push!(inds,itm)
        v = @view U[itm,:]
        if i==1
            f .= v
        else
            Fv = @view F[:,1:(i-1)]
            f = v - Fv*(Fv'*v)
        end
        F[:,i] = f / sqrt(dot(f,v))
        mul!(tmp,U,@view F[:,i])
        @inbounds for j in 1:n
            s = p[j] - tmp[j]^2
            p[j] = (s > 0 ? s : 0)
        end
        @inbounds for j in inds
            p[j] = 0
        end
    end
    collect(inds)
end

function sample_pdppc(U) 
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p = (norm.(eachrow(U))).^2
    F = zeros(eltype(U),m,m)
    f = zeros(eltype(U),m)
    tmp = zeros(eltype(U),n)
    inds = BitSet()
    @inbounds for i = 1:m
        itm = StatsBase.sample(Weights(p))
        push!(inds,itm)
        v = @view U[itm,:]
        if i==1
            f .= v
        else
            Fv = @view F[:,1:(i-1)]
            f = v - Fv*(Fv'*v)
        end
        F[:,i] = f / sqrt(dot(f,v))
        mul!(tmp,U,@view F[:,i])
        @inbounds for j in 1:n
            s = p[j] - abs2(tmp[j])
            p[j] = (s > 0 ? s : 0)
        end
        @inbounds for j in inds
            p[j] = 0
        end
    end
    collect(inds)
end

function sample_pdpp3(U) 
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    #uut = U*U'
    #p = [((uut')[i,i]*ones(1))[1] for i in 1:n]
    Um = Matrix(U)
    p = vec(sum(Um.^2;dims=(2)))
    F = zeros(Float64,m,m)
    f = zeros(m)
    inds = BitSet()
    @inbounds for i = 1:m
        itm = StatsBase.sample(Weights(p))
        push!(inds,itm)
        v = vec(Matrix(U[itm,:]))
        if i==1
            f = vec(v)
        else
            Fv = @view F[:,1:(i-1)]
            f = vec(v) - Fv*(Fv'*v)
        end
        F[:,i] = f / sqrt(dot(v,f))
        tmp = U*F[:,i]
        @inbounds for j in 1:n
            s = p[j] - tmp[j]^2
            p[j] = (s > 0 ? s : 0)
        end
        @inbounds for j in inds
            p[j] = 0
        end
    end
    collect(inds)
end

