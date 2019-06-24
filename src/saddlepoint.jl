#Find a rescaling, typically α s.t. Tr(αL(αL + I)^-1) = k
function solve_sp(ls :: Vector,k :: Int)
    v = log.(ls)
    f = (nu) -> (sum(exp.( nu .+ v)./(1 .+ exp.(nu .+ v))) .- k)^2
    res = Optim.optimize(f,-10,10)
    (Optim.minimum(res) > 1e-5) && throw(ArgumentError("Could not find an appropriate rescaling"))
    exp(Optim.minimizer(res))
end
