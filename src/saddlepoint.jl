#Functions for solving the saddlepoint equation of DPPs
#Equivalent to finding a rescaling  α s.t. Tr(αL(αL + I)^-1) = k

#solve the saddlepoint equation using a fixed-point iteration
function solve_sp_fp(ls :: Vector, k :: Int; nu0=0.0, tol=0.001 ,maxit=100)
    (k >= length(ls)) && throw(ArgumentError("Error: k larger or equal to total number of eigenvalues: cannot find an appropriate rescaling"))
    g=exp(nu0);
    for i in 1:maxit
        fv = sum(ls ./ (1 .+ g .*ls ));
        nv = k/fv;
        ((abs(k-g*fv)/k) < tol) && return nv;
	      g = nv;
    end
    g
end

#find initial value for ν=log α
function guess_nu(ls :: Vector, k :: Int)
    -StatsBase.quantile(log.(ls),1-k/length(ls))
end

#solve the saddlepoint eq. using Newton's method
function solve_sp(ls :: AbstractVector, k :: Int; nu0=nothing, tol=0.001 ,maxit=100)
    (k >= length(ls)) && throw(ArgumentError("Error: k larger or equal to total number of eigenvalues: cannot find an appropriate rescaling"))
    if (nu0 == nothing)
        nu0 = guess_nu(ls,k)
    end
    nu = nu0;
    converged = false;
    n = length(ls)
    nIter=0
    els = zeros(n)
    elsp1 = zeros(n)
    rat = zeros(n)
    while (!converged && nIter < maxit)
      els = exp(nu).*ls
      elsp1 = els .+ 1
      rat = els ./ elsp1
      g = sum(rat) - k
      h = sum(rat ./ elsp1)
      (abs(g/h) < tol) && (converged = true)
      nu -= g/h
      nIter+=1
    end
    (nIter == maxit) && @warn "Max. number of iterations reached in solve_sp_newton, check that eigenvalues are within reasonable range"
    return exp(nu)
end

