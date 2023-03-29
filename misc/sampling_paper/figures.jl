#this code is meant to replicate the timing results obtained
#in the tech report
#Barthelme, S, Tremblay, N, Amblard, P-O, (2022)  A Faster Sampler for Discrete Determinantal Point Processes
#To run, start Julia, then activate local project
#] activate .
#] instantiate
#load figures.jl, then run
#plot_runtime()
using Determinantal,Plots,BenchmarkTools,KernelFunctions,LowRankApprox

function sample_ar(L::AbstractLEnsemble)
    val = L.α * L.λ ./ (1 .+ L.α * L.λ)
    incl = rand(L.m) .< val
    Determinantal.sample_pdpp_ar(L.U[:, incl])
end

function sample_pqr(x,p,m,ar=false)
    y = x[:,rand(1:size(x,2),p)]
    t1 = @elapsed Kf=kernelmatrix(SqExponentialKernel(),x,y);
    t2 = @elapsed F=pqrfact(Kf,rank=m);
    if (ar)
        t3 = @elapsed  Determinantal.sample_pdpp_ar(F.Q)
    else
        t3 = @elapsed Determinantal.sample_pdpp(F.Q)
    end
    [t1,t2,t3]
    #F
end

function runtime_pqr(n,m)
    p = 5m
    x = randn(2,n)
    m_old = map(median,eachrow(reduce(hcat,[sample_pqr(x,p,m,false) for _ in 1:300])))
    m_new = map(median,eachrow(reduce(hcat,[sample_pqr(x,p,m,true) for _ in 1:300])))
    [m_old m_new]
end

function sample_rff(x,p,m,ar=false)
    ell = EllEnsemble(Determinantal.rff_fused(x,Int(floor(p/2)),1.0))
    rescale!(ell,m)
    if (ar)
        sample_ar(ell)
    else
        Determinantal.sample(ell)
    end
end

function runtime(n,m)
    Qt=Matrix(Matrix(qr(randn(n,m)).Q)');
    Q = Qt';
    lv = Determinantal.lvg(Q);
    m_o=median([@elapsed Determinantal.sample_pdpp(Q,lv) for _ in 1:100])
    m_n=median([@elapsed Determinantal.sample_pdpp_ar(Q,lv) for _ in 1:100])
    [m_o,m_n]
end

function plot_runtime()
    ns = [10,100,500,1000,5000,10000,30000,100000]
    ns2 = [1000,5000,10000,50000]
    resa = runtime.(ns,30)
    resb = runtime.(ns,60)
    res = runtime_pqr.(ns2,100)

    tt = (s) -> text(s,10,:right)
    f = (v) -> (v ./ sum.(eachcol(v))')[end,:]
    x0 = ns2[end]
    pa=plot(ns,reduce(hcat,resa)',xlab="n",ylab="Runtime (sec.)",legend=:none,title="m=30",labels=["Standard" "A/R"],linecolor=:black,linestyle = [:solid :dash])
    xx = 1e5
    annotate!([(xx,.028,tt("Classical")),(xx,0.003,tt("Accept/reject")) ])
    pb=plot(ns,reduce(hcat,resb)',xlab="n",ylab="",title="m=60",legend=:none,linecolor=:black,linestyle = [:solid :dash])
    annotate!([(xx,.145,tt("Classical")),(xx,0.01,tt("Accept/reject")) ])
    plot(pa,pb) |> display
    pc=plot(ns2,reduce(hcat,vec.(res))',linecolor=[:red :green :blue :red :green :blue],linestyle = [:solid :solid :solid :dash :dash :dash],legend=:none,xlabel="n",ylabel="Time (sec.)")
    annotate!([(x0,.4,tt("RRQR")),(x0,.35,tt("Kernel")),(x0,.16,tt("Sampling")),(x0,0.03,tt("Sampling with A/R")) ])
    pd =plot(ns2,reduce(hcat,f.(res))',xlabel="n",ylabel="Sampling time rel. to total time",linecolor=:black,linestyle=[:solid :dash],legend=:none)
    annotate!([(x0,.2,tt("Classical")),(x0,0.02,tt("Accept/reject")) ])
    p= plot(pa,pb,pc,pd,layout=@layout([a b c d]),size=(1400,300),title=["a." "b." "c." "d."],titlelocation=:left)
    p
    #savefig(p,"figures/runtime_aistats.pdf")
end
