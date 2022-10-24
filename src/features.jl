using LoopVectorization

function vdm(x::Array{T,1}, order::Int) where {T<:Real}
    [u^k for u in x, k = 0:order]
end

"""
    polyfeatures(x,order)

Compute monomial features up to a certain degree. For instance, if the input consists in vectors in ℝ², then the monomials of degree ≤ 2 are 1,x₁,x₂,x₁x₂,x₁²,x₂²
Note that the number of monomials of degree r in dimension d equals ``{ d+r \\choose r}``

If the input points are in d > 1, indicate whether the input x is d * n or n * d using ColVecs or RowVecs.

The output has points along rows and monomials along columns.

## Examples

```
x = randn(2,10) #10 points in dim 2
polyfeatures(ColVecs(x),2) #Output has 10 rows and 6 column
polyfeatures(RowVecs(x),2) #Output has 2 rows and 66 columns

```

"""
function polyfeatures(x::AbstractVector, degree)
    n = length(x)
    d = length(x[1])
    #   total number of features
    k = binomial(d + degree, degree)
    F = zeros(n, k)
    tdeg = zeros(Int64, k)
    if (d == 1)
        F = vdm(x, degree)
    else
        F[:, 1:(degree+1)] = vdm([x[1] for x in x], degree)
        tdeg[1:(degree+1)] = 0:degree
        tot = degree + 1
        for dd = 2:d
            for i = 1:tot
                delta = degree - tdeg[i]
                if (delta > 0)
                    idx = i
                    for j = 1:delta
                        F[:, tot+1] = F[:, idx] .* [x[dd] for x in x]
                        tot += 1
                        tdeg[tot] = tdeg[idx] + 1
                        idx = tot
                    end
                end
            end
        end
    end
    F
end



"""
    rff(X,m,σ)

Compute Random Fourier Features [rahimi2007random](@cite) for the Gaussian kernel matrix with input points X and parameter σ.
Returns a random matrix M such that, in expectation `` \\mathbf{MM}^t = \\mathbf{K}``, the Gaussian kernel matrix.
M has 2*m columns. The higher m, the better the approximation.

## Examples

```@example
x = randn(2,10) #10 points in dim 2
rff(ColVecs(x),4,1.0)
```
See also: gaussker, kernelmatrix
"""
function rff(X::AbstractMatrix, m, σ) #assume x is d × n
    d = size(X, 1)
    n = size(X, 2)
    Ω = randn(d, m) / sqrt(σ^2)
    T = X' * Ω
    s = sqrt(m)
    f = (x) -> cos(x) / s
    g = (x) -> sin(x) / s
    [f.(T) g.(T)] |> LowRank
end

# function rff_opt(X::AbstractMatrix, m, σ) #assume x is d × n
#     d = size(X, 1)
#     n = size(X, 2)
#     Ω = randn(d, m) / sqrt(σ^2)
#     T = X' * Ω
#     s = sqrt(m)
#     Z = Matrix{eltype(X)}(undef,n,2m)
#     f = (x) -> cos(x) / s
#     g = (x) -> sin(x) / s
    
#     @turbo for i in 1:m
#         for j in 1:n
#             Z[j,i] = cos(T[j,i])/s
#             Z[j,m+i] = sin(T[j,i])/s
#         end
#     end
#     LowRank(Z)
# end

#faster RFF implementation using LoopVectorization.jl
function rff_fused(X::AbstractMatrix, m, σ) #assume x is d × n
    d = size(X, 1)
    n = size(X, 2)
    s = sqrt(m)
    Z = Matrix{eltype(X)}(undef,n,2m)
    Ω = randn(d, m)
    @turbo for i in 1:m
        for j in 1:n
            tij = zero(eltype(X))
            for k in 1:d
                tij += X[k,j]*Ω[k,i] / σ
                #tij += X[k,j]*ω[k]
            end
            Z[j,i] = cos(tij)/s
            Z[j,m+i] = sin(tij)/s
        end
    end
    LowRank(Z)
end




function rff(x::ColVecs, m, σ)
    rff(x.X, m, σ)
end

function rff(x::RowVecs, m, σ)
    rff(x.X', m, σ)
end




"""
    nystrom_approx(x :: AbstractVector,ker :: Kernel,ind)

Compute a low-rank approximation of a kernel matrix (with kernel "ker") using the rows and columns indexed by "ind".

```@example
using KernelFunctions
x = rand(2,100)
K = kernelmatrix(SqExponentialKernel(),ColVecs(x))
#build a rank 30 approx. to K
V = nystrom_approx(ColVecs(x),SqExponentialKernel(),1:30)
norm(K-V*V') #should be small
```

"""
function nystrom_approx(x::AbstractVector, ker::Kernel, ind)
    #K_a = [kfun(x[:,i],x[:,j]) for i in 1:size(x,2), j in ind]
    K_a = kernelmatrix(ker, x, x[ind])
    U = cholesky(K_a[ind, :]).L
    #K[:,ind] * inv(U')
    LowRank(K_a / U')
end

function nystrom_approx(x::AbstractVector, ker::Kernel, m::Integer)
    ind = sortperm(rand(length(x)))[1:m]
    @show length(ind)
    nystrom_approx(x, ker, ind)
end


function nystrom_approx(K::Matrix, ind)
    Kaa = K[ind, ind]
    U = cholesky(Kaa).L
    #K[:,ind] * inv(U')
    LowRank(K[:, ind] / U')
end

function nystrom_approx(K, m::Integer)
    ind = rand(1:size(K, 1), m)
    nystrom_approx(K, ind)
end



function rff(X, m)
    rff(X, m, estmediandist(X))
end


"""
    gaussker(X,σ)

Compute the Gaussian kernel matrix for points X and parameter σ, ie. a matrix with entry i,j
equal to ``\\exp(-\\frac{(x_i-x_j)^2}{2σ^2})``

If σ is missing, it is set using the median heuristic. If the number of points is very large, the median is estimated on a random subset.

```@example
x = randn(2,6)
gaussker(ColVecs(x),.1)
```

See also: rff, KernelMatrix:kernelmatrix
"""
function gaussker(X::AbstractVector, σ)
    kernelmatrix(with_lengthscale(SqExponentialKernel(), σ), X)
end

function gaussker(X::AbstractVector)
    gaussker(X, estmediandist(X))
end

#Quick estimate for median distance
function estmediandist(X::AbstractVector; m = 1000)
    n = length(X)
    if (n > m)
        sel = rand(1:n, m)
    else
        sel = 1:n
    end
    StatsBase.median(KernelFunctions.pairwise(Euclidean(), X[sel], X[sel]))
end
