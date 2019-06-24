# using JuMP
# using OSQP
# using SparseArrays
# using Statistics

# function circmean(X :: Matrix)
#     Xt = X ./ sqrt.( sum( X .^ 2,dims=1))
#     mean(Xt,dims=2)
# end

# function insert_greedy(X :: Matrix,kmin=3,kmax=size(X,2);threshold=.1)
#     # Z = Array{Array{eltype(X),1},1}()
#     keep = collect(1:kmin)
#     cf = (set) -> norm(circmean(X[:,set]))
#     cv = cf(keep)
#     for i in (kmin+1):kmax
#         δ=cf(keep)-cf([keep;i])
#         if δ > threshold
#             push!(keep,i)
#         end
#     end
#     keep
# end

# function select_neighbours(X :: Matrix,kmin=3,kmax=7;threshold=.1)
#     nb = Vector{Vector{Int}}()
#     ds = Vector{Vector{Float64}}()
#     n = size(X,2)
#     d = size(X,1)
#     tree = BallTree(float.(X))
#     idx,dst = knn(tree, X, kmax,true)
#     for i in 1:n
#         nn = (idx[i])[2:end]
#         dd = (dst[i])[2:end]

#         Y = X[:,nn] .- X[:,i]
#         keep = insert_greedy(Y,kmin;threshold=threshold)
#         push!(nb,nn[keep])
#         push!(ds,dd[keep])

#     end
#     nb,ds
# end

# function discrete_lap_dual(X :: Matrix,Y :: Matrix;reweight=true,kmax=min(30,size(Y,2)))
#     tree = BallTree(float.(X))
#     idx,dst = knn(tree, Y, kmax,true)
#     ii = Vector{Int}()
#     jj = Vector{Int}()
#     xx = Vector{Float64}()
#     for i in 1:size(Y,2)
#         nn = (idx[i])
#         @show nn
#         dd = (dst[i])
#         @show dd
#         if reweight
#             Z = X[:,nn] .- Y[:,i]
#             w = weighted_circ_dist(Z) ./ (dd .^ 2)
#         else
#             w = 1 ./ (length(dd)*(dd .^ 2))
#         end
#         for j in 1:length(w)
#             push!(ii,i)
#             push!(jj,nn[j])
#             push!(xx,w[j])
#         end
#         push!(ii,i)
#         push!(jj,i)
#         push!(xx,-sum(w))
#     end
#     sparse(ii,jj,xx)
# end


# function discrete_lap(X :: Matrix,kmin=5,kmax=Int(ceil(1.5*kmin));reweight=true)
#     n = size(X,2)
#     d = size(X,1)
#     idx,dst = select_neighbours(X,kmin,kmax)
#     ii = Vector{Int}()
#     jj = Vector{Int}()
#     xx = Vector{Float64}()
#     for i in 1:n
#         id = idx[i]
#         dd = dst[i]
#         Y = X[:,id] .- X[:,i]
#         if reweight
#             w = weighted_circ_dist(Y) ./ (dd .^ 2)
#         else
#             w = 1 ./ (length(dd)*(dd .^ 2))
#         end
#         for j in 1:length(w)
#             push!(ii,i)
#             push!(jj,id[j])
#             push!(xx,w[j])
#         end
#         push!(ii,i)
#         push!(jj,i)
#         push!(xx,-sum(w))
#     end
#     sparse(ii,jj,xx)
# end

# function biweighted_circ_dist(X :: Matrix)
#     d = size(X,1)
#     n = size(X,2)
#     r = sqrt.(sum(X .^ 2,dims=1));

#     Xt = X ./ r;
#     inds_up = tril(trues(d,d))
#     npairs = n^2
#     scov = length(inds_up)
#     #Collect all pairwise statistics
#     pw_r = zeros(1,n,n)
#     pw_m = zeros(d,n,n)
#     pw_cv = zeros(scov,n,n)
#     for i in 1:n
#         for j in 1:n
#             pw_r[1,i,j] = 1/(r[i]^2) - 1/(r[j]^2)
#             pw_m[:,i,j] = vec(X[:,i]/(r[i]^2) - X[:,j]/(r[j]^2))
#             pw_cv[:,i,j] = vec((Xt[:,i]*Xt[:,i]'- Xt[:,j]*Xt[:,j]')[inds_up])
#             @show Xt[:,i]*Xt[:,i]'
#             @show Xt[:,j]*Xt[:,j]'
#         end
#     end
#     rs = (v) -> reshape(v,(size(v,1),n^2))
#     pw_r =rs(pw_r)
#     pw_m =rs(pw_m)
#     pw_cv =rs(pw_cv)

# #    @show pw_cv
#     model = Model(with_optimizer(OSQP.Optimizer))
#      @variable(model,0 <= w[1:(n^2)] <= 1)
#     # #First order obj:
#     # O1 = X ./ sum(X .^ 2,dims=1)
#     # A = [O1;Cv]
#     idd = Matrix{Float64}(I,d,d)[inds_up];
#     # b = [zeros(d);idd];
#     @show idd
#     @objective(model,Min,sum((pw_m*w).^2)+sum((pw_cv*w - idd).^2))
#     # @constraint(model,sum(w)==1)
#     # TT = stdout # save original STDOUT stream
#     # redirect_stdout()
#     # optimize!(model)
#     # redirect_stdout(TT) # restore STDOUT
#     # value.(w)
# end


# function weighted_circ_dist(X :: Matrix)
#     d = size(X,1)
#     n = size(X,2)
#     Xt = X ./ sqrt.(sum(X .^ 2,dims=1))

#     inds = tril(trues(d,d))
#     Cv = zeros(sum(inds),n)
#     for i in 1:n
#         Cv[:,i] = (Xt[:,i]*Xt[:,i]')[inds]
#     end
#     model = Model(with_optimizer(OSQP.Optimizer))
#     @variable(model,0 <= w[1:n] <= 1)
#     #First order obj:
#     O1 = X ./ sum(X .^ 2,dims=1)
#     A = [O1;Cv]
#     idd = Matrix(I,d,d)[inds];
#     b = [zeros(d);idd];
#     @objective(model,Min,sum((A*w-b).^2))
#     @constraint(model,sum(w)==1)
#     TT = stdout # save original STDOUT stream
#     redirect_stdout()
#     optimize!(model)
#     redirect_stdout(TT) # restore STDOUT
#     value.(w)
# end
