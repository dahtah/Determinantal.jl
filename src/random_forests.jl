# random forest stuff

function random_forest_direct(G::Graph,q::AbstractFloat)
    roots = Set{Int64}()
    root = zeros(Int64,nv(G))
    nroots = Int(0)
    
    n = nv(G)
    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = i
        
        while !in_tree[u]
            if (rand() < q/(q+degree(G,u)))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]
        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    (next=next,roots=roots,nroots=nroots,root=root)
end

function smooth_over_partition(G :: SimpleGraph,root :: Array{Int64,1},y :: Array{Float64,1})
    xhat = zeros(Float64,nv(G))
    #    ysum = weighted_sum_by(y,deg,state.root)
    ysum = sum_by(y,root)
    for v in vertices(G)
        xhat[v] = ysum[root[v]]
    end
    xhat
end


function sure(y,xhat,nroots,s2)
    err = sum((y .- xhat).^2)
    @show err
    -length(y)*s2+err+2*s2*nroots
end

function random_successor(G::SimpleGraph{T},i :: T) where T <: Int
    nbrs = neighbors(G, i)
    rand(nbrs)
end


function smooth_wilson_adapt(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed") where T
    nr = 0;
    rf = random_forest_direct(G,q)
    xhat = smooth_over_partition(G,rf.root,y)
#    @show xhat
    L = lap(G)
    A = (L+q*I)/q
    res = A*xhat - y
    gamma =alpha
#    @show res
    #    @show norm(res)


    for indr in 2:nrep
        rf = random_forest_direct(G,q)
        dir = smooth_over_partition(G,rf.root,res)
        if (step == "optimal")
            u = A*dir
            gamma = ( xhat'*A*u - dot(y,u)   )/dot(u,u)
 #           @show gamma
        elseif (step=="backtrack")
            curr = norm(res)
            gamma = alpha
            while (norm(A*(xhat-gamma*dir) - y) > curr)
                gamma = gamma/2
            end
#            @show gamma
        end

        xhat -= gamma*dir
        res = A*xhat - y
#        @show res
 #       @show norm(res)
        nr += rf.nroots
    end
    (xhat=xhat,nroots=nr/nrep)
end


function smooth_wilson(G :: SimpleGraph{T},q,y :: Vector;nrep=10,variant=1) where T
    xhat = zeros(Float64,length(y));
    nr = 0;
    for indr in Base.OneTo(nrep)
        rf = random_forest_direct(G,q)
        nr += rf.nroots
        if variant==1
            xhat += y[rf.root]
        elseif variant==2
            xhat += smooth_over_partition(G,rf.root,y)
        end
    end
    (xhat=xhat ./ nrep,nroots=nr/nrep)
end


function lap(G :: SimpleGraph)
    i = collect(1:nv(G))
    j = collect(1:nv(G))
    x = degree(G)
    for u in vertices(G)
        for v in neighbors(G,u)
            push!(i,u)
            push!(j,v)
            push!(x,-1)
        end
    end
    sparse(i,j,x)
end
