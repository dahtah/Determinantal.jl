function vdm(x :: Array{T,1}, order :: Int) where T <: Real
    [u^k for u in x, k in 0:order]
end

"""
    polyfeatures(X,order)

Compute monomial features up to a certain degree. For instance, if X is a 2 x n matrix and the degree argument equals 2, it will
return a matrix with columns 1,X[1,:],X[2,:],X[1,:].^2,X[2,:].^2,X[1,:]*X[2,:]
Note that the number of monomials of degree r in dimension d equals ``{ d+r \\choose r}``

X is assumed to be of dimension ``d \\times n`` where d is the dimension and n is the number of points.

## Examples

```
X = randn(2,10) #10 points in dim 2
polyfeatures(X,2) #Output has 6 columns
```

"""
function polyfeatures(X,degree)
    d,n = size(X)
    #   total number of features
    k = binomial(d+degree,degree)
    F = zeros(n,k)
    tdeg = zeros(Int64,k)
    if (d==1)
        F = vdm(vec(X),degree)
    else
        F[:,1:(degree+1)] = vdm(X[1,:],degree)
        tdeg[1:(degree+1)] = 0:degree
        tot = degree+1
        for dd in 2:d
            for i in 1:tot
                delta = degree-tdeg[i]
                if (delta > 0)
                    idx = i
                    for j in 1:delta
                        F[:,tot+1] = F[:,idx] .* vec(X[dd,:])
                        tot += 1
                        tdeg[tot] = tdeg[idx]+1
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

Compute Random Fourier Features for the Gaussian kernel matrix with input points X and parameter σ.
Returns a random matrix M such that, in expectation `` \\mathbf{MM}^t = \\mathbf{K}``, the Gaussian kernel matrix. 
M has 2*m columns. The higher m, the better the approximation. 

## Examples

```
X = randn(2,10) #10 points in dim 2
rff(X,4,1.0)
```
See also: gaussker, kernelmatrix 
"""
function rff(X :: Matrix, m, σ)
    d = size(X,1)
    n = size(X,2)
    Ω = randn(d,m) / sqrt(σ^2)
    T = X'*Ω
    s = sqrt(m)
    f = (x) -> cos(x)/s
    g = (x) -> sin(x)/s
    [f.(T) g.(T)]
end

function nystrom_approx(K,ind)
    Kaa = K[ind,ind]
    U = cholesky(Kaa).L
    #K[:,ind] * inv(U')
    K[:,ind] / U'
end


function rff(X :: Matrix, m)
    rff(X,m,estmediandist(X))
end


"""
    gaussker(X,σ)

Compute the Gaussian kernel matrix for X and parameter σ, ie. a matrix with entry i,j
equal to ``\\exp(-\\frac{(x_i-x_j)^2}{2σ^2})``

See also: rff, kernelmatrix 
"""
function gaussker(X::Matrix,σ)
    tau = 1/(2*σ^2)
    kernelmatrix(Val(:col),SquaredExponentialKernel(tau),X)
end

function gaussker(X::Matrix)
    gaussker(X,estmediandist(X))
end

#Quick estimate for median distance
function estmediandist(X::Matrix;m=1000)
    n = size(X,2)
    if (n > m)
        sel = sample(1:n,m)
    else
        sel = 1:n
    end
    StatsBase.median(pairwise(Euclidean(),X[:,sel],X[:,sel];dims=2))
end

