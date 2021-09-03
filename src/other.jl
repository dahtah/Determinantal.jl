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




function distance_sampling(x :: Union{ColVecs,RowVecs},m,sampling=:d2)
    distance_sampling(LazyDist(x),m,sampling)
end

function distance_sampling(D :: AbstractMatrix,m,sampling=:d2)
    @assert sampling ∈ [:d2,:farthest]
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
        end
        push!(ind,i)
        dd[i] = 0
        dd = min.(dd,D[:,i])
    end
    collect(ind)
end
