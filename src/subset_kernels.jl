export subset_kernel, dist_subsets

function subset_kernel(sub, normalise=false)
    if (normalise)
        f = (i, j) -> length(intersect(sub[i], sub[j])) / √(length(sub[i]) * length(sub[j]))
    else
        f = (i, j) -> length(intersect(sub[i], sub[j]))
    end
    return float.([f(i, j) for i in 1:length(sub), j in 1:length(sub)])
end

#warning: naive implementation
function dist_subsets(X, ϵ)
    sub = Vector{Set{Int}}()
    for i in 1:size(X, 2)
        s = Set{Int}()
        push!(s, i)
        for j in 1:size(X, 2)
            if norm(X[:, i] - X[:, j]) <= ϵ
                push!(s, j)
            end
        end
        push!(sub, s)
    end
    return sub
end
