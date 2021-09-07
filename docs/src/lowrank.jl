struct LowRank <: AbstractMatrix
    A :: Matrix
    n :: Int
    p :: Int

end

function LowRank(A)
    @assert size(A,1) <= size(A,2)
    LowRank(A,size(A,1),size(A,2))
end

function Base.getindex(K :: LowRank,i,j)
    K.A[i,:]*K.A[:,j]
end

function LinearAlgebra.eigen(K :: LowRank)
    eg = eigen(K.A'*K.A)
end
