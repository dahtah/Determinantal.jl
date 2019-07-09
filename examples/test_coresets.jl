using Revise
using DPP
using DataFrames
using Gadfly
using Statistics
using StatsBase
using Clustering
#using RCall
using LinearAlgebra

function gen_clust(M,csize;sigma2=1)
    d = size(M,1)
    s = sqrt(sigma2)
    reduce(hcat,map((i) -> M[:,i] .+ s*randn(d,csize[i]),1:length(csize)))
end

M = [-1 0 1; 0 -1 -1; 1 -1 1]
d = 3

csize = [10000,10000,50]
cluster= reduce(vcat,map(i-> repeat([i],csize[i]),1:length(csize)))


# X = [randn(9500,3);randn(9500,3).+5;randn(50,3).-4];
# V = float.([5 5 5; 0 0 0; -4 -4 -4])
# V = Matrix(V')
# A = DPP.kmeans_restart(Matrix(X'),3,100).centers
# d = size(X,2)
dim_ord = (ord) -> binomial(ord-1+d,d)

function kmeans_random(X :: Array{Float64,2},ns :: Int, k :: Int; nrep = 10)
    ind = sample(1:size(X,2),ns,replace=false)
    C= DPP.kmeans_restart(X[:,ind],k,nrep).centers
    (centers=C,cluster=DPP.nnclass(X,C))
end

function run_tests(orders,nrep=30)

    test1 = (ord) -> randindex(DPP.kmeans_coreset(X,ord,3).cluster,cluster)
    test_rep1 = (ord) -> [test1(ord)[2] for i in 1:nrep]
    df1 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep1(ord),order=ord),orders))
    df1[:type] = "DPP"
    test2 = (ord) -> randindex(kmeans_random(X,dim_ord(ord),3).cluster,cluster)
    test_rep2 = (ord) -> [test2(ord)[2] for i in 1:nrep]
    df2 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep2(ord),order=ord),orders))
    df2[:type] = "unif"
    test3 = (ord) -> randindex(DPP.kmeans_d2(X,dim_ord(ord),3).cluster,cluster)
    test_rep3 = (ord) -> [test3(ord)[2] for i in 1:nrep]
    df3 = reduce(vcat,map((ord) -> DataFrame(ri=test_rep3(ord),order=ord),orders))
    df3[:type] = "D2"



    #    test2 = (ord) -> DPP.clustdist(kmeans_random(Matrix(X'),dim_ord(ord),3).centers,V)
    # test_rep2 = (ord) -> [test2(ord) for i in 1:30]
    # df2 = reduce(vcat,map((ord) -> DataFrame(err=test_rep2(ord),order=ord),orders))
    # df2[:type] = "unif"
    # test3 = (ord) -> DPP.clustdist(DPP.kmeans_d2(Matrix(X'),dim_ord(ord),3).centers,V)
    # test_rep3 = (ord) -> [test3(ord) for i in 1:30]
    # df3 = reduce(vcat,map((ord) -> DataFrame(err=test_rep3(ord),order=ord),orders))
    # df3[:type] = "d2"
    df = vcat(df1,df2,df3)
    df[:k] = dim_ord.(df[:order])
    df
end

function asymptotic_indep()
    k = 10
    x = logis.(randn(10000))*2 .- 1
    x2 = logis.(1.5*randn(10000).+2)*2 .- 1
    Plots.plot(ash(x))
    Plots.plot!(ash(x2))
    U=Matrix(qr(DPP.vdm(x,k)).Q);
    U2=Matrix(qr(DPP.vdm(x2,k)).Q);
    Z2 = reduce(vcat,[x2[collect(sample_pdpp(U2))] for _ in 1:1000])
    Z = reduce(vcat,[x[collect(sample_pdpp(U))] for _ in 1:1000])
    Plots.plot(ash(Z))
    Plots.plot!(ash(Z2))
end

function sample_ball(n :: Int,d :: Int,gamma=1)
    X = randn(d,n)
    X = X ./ sqrt.(sum( X .^ 2,dims=1))
    r = rand(n).^gamma
    r' .* X
end


function sample_trunc(m :: Vector,n :: Int)
    d = length(m)
    x = zeros(d,n)
    for i in 1:n
        accepted = false
        while !accepted
            z = randn(d) .+ m
            if (sum(z.^2) < 1)
                accepted=true
                x[:,i] = z
            end
        end
    end
    x
end

function illus_coreset_dpp()
    M = [-1 0.4 0; .8 1.2 -1 ]
    d = 2
    csize = [10000,10000,50]
    cluster= reduce(vcat,map(i-> repeat([i],csize[i]),1:length(csize)))
    order = 6
    U=Matrix(qr(polyfeatures(Matrix(xc'),order)).Q)
    sel = collect(sample_pdpp(U))
    sel2 = collect(sample_pdpp(U))
    R"""
    pdf("~/Repos/dpp_at_gipsa/marginal_approximations/figures_slides/unbalanced_cluster.pdf",w=6,h=6)
    plot(t($xc),col=rgb(.5,.5,.5,.2),pch=19,xlab="x",ylab="y")
    dev.off()
    pdf("~/Repos/dpp_at_gipsa/marginal_approximations/figures_slides/unbalanced_cluster_thinned.pdf",w=6,h=6)
    plot(t($xc),col=rgb(.5,.5,.5,.2),pch=19,xlab="x",ylab="y")
    pointsb(t($xc)[$sel,])
    dev.off()
    pdf("~/Repos/dpp_at_gipsa/marginal_approximations/figures_slides/unbalanced_cluster_thinned_2.pdf",w=6,h=6)
    plot(t($xc),col=rgb(.5,.5,.5,.2),pch=19,xlab="x",ylab="y")
    pointsb(t($xc)[$sel2,])
    dev.off()
    """
end

function asymptotic_indep_2D()
    k = 10
    

    x = sample_ball(100000,2,1/2)
    x2 = sample_trunc([2,2],100000)
    # Plots.plot(ash(x))
    # Plots.plot!(ash(x2))
    order=12
    U=Matrix(qr(polyfeatures(Matrix(x'),order)).Q)
    U2=Matrix(qr(polyfeatures(Matrix(x2'),order)).Q)
    Z = reduce(hcat,[x[:,collect(sample_pdpp(U))] for _ in 1:100])    
    Z2 = reduce(hcat,[x2[:,collect(sample_pdpp(U2))] for _ in 1:100])
    k = size(U,2)

    #Plot results in R
    R"
    df=data.frame(x=c($x[1,],$x2[1,]),y=c($x[2,],$x2[2,]),distr=rep(1:2,each=ncol($x)))
    pa=ggplot(df,aes(x,y))+geom_point(alpha=.1)+facet_wrap(~ distr)
    df2=data.frame(x=c($Z[1,],$Z2[1,]),y=c($Z[2,],$Z2[2,]),distr=rep(1:2,each=ncol($Z)),index=rep(1:100,each=$k))
    pb = ggplot(df2,aes(x,y))+geom_point(alpha=.1)+facet_wrap(~ distr);
    print(pa)
    "



    #Plots.plot(ash(Z))
    #Plots.plot!(ash(Z2))
end

