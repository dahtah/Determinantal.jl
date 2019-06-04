function vdm(x :: Array{T,1}, order :: Int) where T <: Real
    [u^k for u in x, k in 0:order]
end

"""
    polyfeatures(X,order)

Compute monomial features up to a certain degree. For instance, if X is a 2 x n matrix and the degree argument equals 2, it will
return a matrix with columns 1,X[1,:],X[2,:],X[1,:].^2,X[2,:].^2,X[1,:]*X[2,:]
Note that the number of monomials of degree r in dimension d equals ``{ d+r \\choose r}``

X is assumed to be of dimension ``d \\times n`` where d is the dimension and n is the number of points.
"""
function polyfeatures(X :: Array{T,2},degree :: Int) where T <: Real
    m = size(X,2)
    if (m==1)
        vdm(vec(X),degree)
    else
        g = (z) -> reduce(vcat,map((u) -> [ [u v] for v in 0:(degree-1) if  sum(u) + v < degree],z))
        dd = g(0:(degree))
        for i in 1:(m-2)
            dd = g(dd)
        end
        reduce(hcat,[prod(X .^ d,dims=2) for d in dd])
        #[prod(X .^ d') for d in dd ]
        #dd
    end
end
"""
    rff(X,m,σ)

Compute Random Fourier Features for the Gaussian kernel matrix with input points X and parameter σ.
Returns a random matrix M such that, in expectation `` \\mathbf{MM}^t = \\mathbf{K}``, the Gaussian kernel matrix. 
M has 2*m columns. The higher m, the better the approximation. 

See also: gaussker, kernelmatrix 
"""
function rff(X :: Matrix, m, σ)
    d = size(X,1)
    n = size(X,2)
    Ω = randn(d,m) / sqrt(σ^2)
    T = X'*Ω
    [cos.(T) sin.(T)]/sqrt(m)
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
    median(pairwise(Euclidean(),X[:,sel],X[:,sel];dims=2))
end

