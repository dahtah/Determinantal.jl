struct Nystrom
    A :: Matrix
    B :: Matrix
    BABinv :: Matrix
end

function Nystrom(A,B)
    Nystrom(A,B,inv(B'*A*B))
end

function Base.:*(N :: Nystrom,u :: Union{Matrix,Vector})
    N.A*(N.B*(N.BABinv*(N.B'*(N.A*u))))
end

function Base.:convert(::Type{Matrix},N :: Nystrom)
    M = N.A*N.B*N.BABinv*N.B'*N.A
    .5*(M+M')
end
