using LoopVectorization,Random

function lvg(U)
    p = U[:,1].^2
    if (size(U,2) > 1)
        @turbo for j in 2:size(U,2)
            for i in 1:size(U,1)
                p[i] += U[i,j]^2
            end
        end
    end
    p
end





function wsample(w::AbstractVector, ttl)
    u = rand() * ttl
    i = 0
    s = 0
    while s < u
        i += 1
        s += w[i]
    end
    return i
end

function sample_pdpp(U::AbstractMatrix)
    return sample_pdpp(U, lvg(U))
end

function sample_pdpp(U::AbstractMatrix, lvg::AbstractVector)
    n = size(U, 1)
    m = size(U, 2)
    #Initial distribution
    #Um = Matrix(U)
    p = copy(lvg)
    F = zeros(Float64, m, m)
    f = zeros(m)
    v = zeros(m)
    tmp = zeros(n)
    inds = BitSet()
    ss = sum(lvg)
    @inbounds for i in 1:m
        itm = wsample(p, ss)
        push!(inds, itm)
        #v = vec(Matrix(U[itm,:]))
        #v = @view U[itm,:]
        #v = U[itm,:]
        copyto!(v, U[itm, :])
        if i == 1
            copyto!(f, v)
        else
            Fv = @view F[:, 1:(i - 1)]
            copyto!(f, v - Fv * (Fv' * v))
        end
        F[:, i] = f / sqrt(dot(v, f))
        mul!(tmp, U, @view F[:, i])
        ss = 0.0
        @turbo for j in 1:n
            s = p[j] - tmp[j]^2
            p[j] = (s > 0 ? s : 0)
            ss += p[j]
        end
        @inbounds for j in inds
            ss -= p[j]
            p[j] = 0
        end
    end
    return collect(inds)
end


function sample_pdpp_mixed(U;cutoff=2*size(U,2))
    lv = lvg(U)
    perm = sortperm(lv,rev=true)
    cutoff = min(cutoff,size(U,1))
    indp = sample_pdpp_mixed(U[perm,:],lv[perm],cutoff)
    perm[indp]
end


#assume items are sorted by lvg score! 
function sample_pdpp_mixed(U,lv,cutoff)
    n = size(U,1)
    m = size(U,2)
    c = cutoff
    p0 = lv
    pc = p0[1:c]
    al = setup_alias(p0[c+1:end])
    rng = MersenneTwister()
    Q = zeros(Float64,m,m)
    inds = BitSet()
    for ind in 1:m
        wc = sum(pc)
        rat = (wc)/(m-(ind-1))
        if (rand() < rat) #sample below
            itm = wsample(pc,wc)
#            @info "Below!"
        else
#            @info "Above!"
            accept = false
            itm =  c+sample_alias(rng,al)
            nattempts = 1
            Qv = @view Q[:,1:(ind-1)]
            while !accept
                pp = p0[itm] - sum((Qv'*U[itm,:]).^2)
                if (rand() < pp/p0[itm])
                    accept = true
                else
                itm = c+sample_alias(rng,al)
                nattempts+=1
                end
                (nattempts > 1e4) && error("Too many attempts at iteration $(ind)")
            end
        end
        push!(inds,itm)
        f = vec(U[itm,:])
        #Gram-Schmidt
        f -= Q*Q'*f
        Q[:,ind] = f/norm(f)
        #update prob. below threshold
        pc = pc .- vec(U[1:c,:]*Q[:,ind]).^2
    end
    collect(inds)
end

function sample_pdpp_ar(U)
    sample_pdpp_ar(U,lvg(U))
end

function sample_pdpp_ar(U,lvg)
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p0 = copy(lvg)
    al = setup_alias(p0)
    rng = MersenneTwister()
    Q = zeros(eltype(U),m,m)
    inds = BitSet()
    f = zeros(eltype(U),m)
    max_attempts = 150*m
    for ind in 1:m
        accept = false
        itm =  sample_alias(rng,al)
        nattempts = 1
        Qv = @view Q[:,1:(ind-1)]
        while !accept
            while (itm ∈ inds)
                itm = sample_alias(rng,al)
            end
            pp = p0[itm] - sum(abs2.(Qv'*U[itm,:]))
            if (rand() < pp/p0[itm])
                accept = true
            else
                itm = sample_alias(rng,al)
                nattempts+=1
            end
            if (nattempts > max_attempts)
                error("Too many A/R attempts, $(nattempts) at index $(ind). Make sure that U is actually an orthonormal matrix")
            end
        end
        push!(inds,itm)
        f .= U[itm,:]
        #Gram-Schmidt
        f -= Qv*(Qv'*f)
        Q[:,ind] = f/norm(f)
    end
    #collect(inds),Q
    collect(inds)
end




#slow!!! reimplement in block-wise manner
function early_reject(itm,Qv,Ui,ind,p0)
    u = rand()
    pp = p0[itm]
    reject = false
    for j in 1:(ind-1)
        pp -= dot(Ui,Qv[:,j])^2
        if (u > pp/p0[itm])
            reject = true
            break
        end
    end
    reject
end   

#ar with early reject
function sample_pdpp_er(U)
    n = size(U,1)
    m = size(U,2)
    #Initial distribution
    p0 = lvg(U)
    al = setup_alias(p0)
    rng = MersenneTwister()
    Q = zeros(Float64,m,m)
    inds = BitSet()
    for ind in 1:m
        accept = false
        itm =  sample_alias(rng,al)
        nattempts = 1
        if (ind > 1)
            reject = true
            Ui = @view U[itm,:]
            Qv = @view Q[:,1:(ind-1)]
            while reject
                nattempts += 1
                (nattempts > 1e4) && error("Too many attempts")
                
                reject = early_reject(itm,Qv,Ui,ind,p0)
                if reject
                    itm =  sample_alias(rng,al)
                    Ui = @view U[itm,:]
                end
            end
        end
        push!(inds,itm)
        f = vec(U[itm,:])
        #Gram-Schmidt
        f -= Q*Q'*f
        Q[:,ind] = f/norm(f)
    end
    collect(inds)
end




function sample_alias(rng::AbstractRNG,al)
    #s = Sampler(rng, 1:n)
    a = 1:length(al.ap)
    # for i = 1:length(x)
    j = rand(rng, a)
    rand(rng) < al.ap[j] ? a[j] : a[al.alias[j]]
end

function setup_alias(lv)
    n = length(lv)
    ap = Vector{Float64}(undef, n)
    alias = Vector{Int}(undef, n)
    make_alias_table!(lv, sum(lv), ap, alias)
    (ap=ap,alias=alias)
end



#taken from StatsBase
function make_alias_table!(w::AbstractVector, wsum,
                           a::AbstractVector{Float64},
                           alias::AbstractVector{Int})
    # Arguments:
    #
    #   w [in]:         input weights
    #   wsum [in]:      pre-computed sum(w)
    #
    #   a [out]:        acceptance probabilities
    #   alias [out]:    alias table
    #
    # Note: a and w can be the same array, then that array will be
    #       overwritten inplace by acceptance probabilities
    #
    # Returns nothing
    #

    n = length(w)
    length(a) == length(alias) == n ||
        throw(DimensionMismatch("Inconsistent array lengths."))

    ac = n / wsum
    for i = 1:n
        @inbounds a[i] = w[i] * ac
    end

    larges = Vector{Int}(undef, n)
    smalls = Vector{Int}(undef, n)
    kl = 0  # actual number of larges
    ks = 0  # actual number of smalls

    for i = 1:n
        @inbounds ai = a[i]
        if ai > 1.0
            larges[kl+=1] = i  # push to larges
        elseif ai < 1.0
            smalls[ks+=1] = i  # push to smalls
        end
    end

    while kl > 0 && ks > 0
        s = smalls[ks]; ks -= 1  # pop from smalls
        l = larges[kl]; kl -= 1  # pop from larges
        @inbounds alias[s] = l
        @inbounds al = a[l] = (a[l] - 1.0) + a[s]
        if al > 1.0
            larges[kl+=1] = l  # push to larges
        else
            smalls[ks+=1] = l  # push to smalls
        end
    end

    # this loop should be redundant, except for rounding
    for i = 1:ks
        @inbounds a[smalls[i]] = 1.0
    end
    nothing
end
