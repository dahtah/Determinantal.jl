struct EllEnsIE <: AbstractLEnsemble
    n :: Int
    m :: Int
    incl :: Vector{Int64}
    excl :: Vector{Int64}
    compl :: Vector{Int64}
    ell :: EllEnsemble
    α :: Float64
end

function EllEnsIE(L :: EllEnsemble,incl,excl)
    compl = setdiff(setdiff(1:L.n,excl),incl)
    S = L.L[compl,compl]-L.L[compl,incl]*(L.L[incl,incl]\L.L[incl,compl])
    EllEnsIE(L.n,L.m,incl,excl,compl,EllEnsemble(S),L.α)
end

function inclusion_prob(L :: EllEnsIE)
    pr = zeros(L.n)
    pr[L.incl] .= 1
    pri = inclusion_prob(L.ell)
    for j in 1:length(pri)
        pr[L.compl[j]] = pri[j]
    end
    pr
end

function sample(L :: EllEnsIE)
    [collect(L.incl);L.compl[sample(L.ell)]]
end




function check_incl_excl(Q, incl, excl)
    compl = setdiff(setdiff(1:size(Q,1),excl),incl)
    M = copy(Q)
    M[incl,:] .= 0
    M[:,incl] .= 0
    for i in incl
        M[i,i] = 1
    end
    for j in compl
        M[j,j] += 1
    end
    display(M)
    inv(M)[compl,compl]
end


function condition_inclusion(L :: EllEnsemble,inc)
    compl = setdiff(1:L.n,inc)
    S = L.L[compl,compl]-L.L[compl,inc]*(L.L[inc,inc]\L.L[inc,compl])
    EllEnsemble(S)
end

function isample(L::EllEnsemble)
    pr = inclusion_prob(L)
    incl = Vector{Int64}()
    Lc = EllEnsIE(L,[],[])
    for i in 1:L.n
        smpl = sample(Lc)
        (length(smpl) == length(incl)) && break
        push!(incl,setdiff(smpl,incl)[1])
        Lc = EllEnsIE(L,incl,[])
        # pr_stop = 1/det(Lc.α*Lc.ell.L+I)
        # (rand()<=pr_stop) && break
        # pri = inclusion_prob(Lc)
        # pri[incl] .= 0
        # ind =wsample(pri,sum(pri))
        # push!(incl,ind)
        # Lc = EllEnsIE(L,incl,[])
    end
    incl
end




function iesample(L::EllEnsemble)
    z = zeros(Bool,L.n)
    pr = inclusion_prob(L)
    incl = Vector{Int64}()
    excl = Vector{Int64}()
    for i in 1:L.n
        z[i] = rand() < pr[i]
        if (z[i])
            push!(incl,i)
        else
            push!(excl,i)
        end
        ie = EllEnsIE(L,incl,excl)
        pr = inclusion_prob(ie)
    end
    incl
end
