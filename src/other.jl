struct LazyDist <: AbstractMatrix{Float64}
    x :: Vector
    dfun :: Function
end

"""
    LazyDist(x)

Lazy representation of a distance matrix, meaning that the object can be indexed like a matrix, but the values are computed on-the-fly. This is useful whenever an algorithm only requires some of the entries, or when the dataset is very large.

```@example

x = randn(2,100)
D= LazyDist(x) #Euclidean dist. by default
D[3,2] #the getindex method is defined
D = LazyDist(x,(a,b) -> sum(abs.(a-b))) #L1 distance
D[3,2]
```

"""
function LazyDist(x)
    LazyDist(x,(x,y)->norm(x-y))
end

function Base.getindex(D :: LazyDist,i,j)
    D.dfun(D.x[i],D.x[j])
end

function Base.getindex(D :: LazyDist,i,j :: Colon)
    xi = D.x[i]
    Matrix([D.dfun(xi,xj) for xj in D.x]')
end

function Base.getindex(D :: LazyDist,i  :: Colon,j)
    xj = D.x[j]
    [D.dfun(xi,xj) for xi in D.x]
end

function Base.getindex(D :: LazyDist,i :: Colon,j :: Colon)
    [D.dfun(xi,xj) for xi in D.x,xj in D.x]
end



function Base.size(D :: LazyDist)
    (length(D.x),length(D.x))
end



"""
    distance_sampling(x,m,sampling)

    Select a random subset of size m from x based on a greedy distance criterion. The initial point is selected uniformly. Then, if sampling == :farthest, at each step, the point selected is one that is farthest from all currently selected points. If sampling == :d2, the algorithm is D²-sampling [vassilvitskii2006k](@vassilvitskii2006k), which is a relaxed stochastic version of farthest-point sampling (selecting points with prob. proportional to squared distance).

   ```@example
    x = rand(2,200);
    ind = distance_sampling(x,40,:farthest)
    scatter(x[1,:],x[2,:],marker_z = map((v) -> v ∈ ind, 1:size(x,2)),legend=:none)
   ```
"""
function distance_sampling(x :: AbstractVector,m,sampling=:d2)
    distance_sampling(LazyDist(x),m,sampling)
end

function distance_sampling(D :: AbstractMatrix,m,sampling :: Union{Symbol,Function})
    @assert size(D,1) == size(D,2)
    if (isa(sampling,Symbol))
        @assert sampling ∈ [:d2,:farthest]
    end
    n = size(D,1)
    ind = BitSet()
    i = rand(1:n)
    push!(ind,i)
    dd = D[:,i]
    while (length(ind) < m)
        if (sampling == :d2)
            i = StatsBase.sample(Weights(dd.^2))
        elseif (sampling == :farthest)
            i = argmax(dd)
        elseif (isa(sampling,Function))
            w = sampling.(dd)
            if (all(w .== 0))
                return collect(ind)
            else
                i = StatsBase.sample(Weights(w))
            end
        end
        push!(ind,i)
        dd[i] = 0
        dd = min.(dd,D[:,i])
    end
    collect(ind)
end


function bernoulli_process(p,nrep=1)
    if (nrep>1)
        [ findall(rand(length(p)) .<= p) for _ in 1:nrep]
    else
        findall(rand(length(p)) .<= p)
    end
end

function matched_poisson(L :: AbstractLEnsemble,nrep)
    p = inclusion_prob(L)
    bernoulli_process(p)
end


