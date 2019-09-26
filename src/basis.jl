

import JuLIP
using JuLIP: JVec, JMat, Atoms
using JuLIP.MLIPs: IPBasis
using LinearAlgebra: norm


export PolyPairBasis

struct PolyPairBasis{TJ} <: IPBasis
   J::TJ
end

PolyPairBasis(maxdeg::Integer, trans::DistanceTransform, fcut::PolyCutoff) =
      PolyPairBasis(TransformedJacobi(maxdeg, trans, fcut))

==(B1::PolyPairBasis, B2::PolyPairBasis) = (B1.J == B2.J)

Base.length(pB::PolyPairBasis) = length(pB.J)

JuLIP.cutoff(pB::PolyPairBasis) = cutoff(pB.J)

Dict(pB::PolyPairBasis) = Dict(
      "__id__" => "PolyPairPots_PolyPairBasis",
      "J" => Dict(pB.J) )

PolyPairBasis(D::Dict) = PolyPairBasis(TransformedJacobi(D["J"]))

convert(::Val{:PolyPairPots_PolyPairBasis}, D::Dict) = PolyPairBasis(D)


# TODO: REMOVE???
# alloc_B(pB::PolyPairBasis, args...) = zeros(Float64, length(pB))
# alloc_dB(pB::PolyPairBasis, N::Integer) = zeros(JVec{Float64}, N, length_B(pB))
# alloc_dB(pB::PolyPairBasis, Rs::AbstractVector) = alloc_dB(pB, length(Rs))

alloc_temp(pB::PolyPairBasis) = (J = alloc_B(pB.J),)
alloc_temp_d(pB::PolyPairBasis, args...) = ( J = alloc_B( pB.J),
                                            dJ = alloc_dB(pB.J) )

function energy(pB::PolyPairBasis, at::Atoms{T}) where {T}
   E = zeros(T, length(pB))
   stor = alloc_temp(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate!(stor.J, nothing, pB.J, r)
      E[:] .+= stor.J[:]
   end
   return E
end

function forces(pB::PolyPairBasis, at::Atoms{T}) where {T}
   F = zeros(JVec{T}, length(at), length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(stor.J, stor.dJ, nothing, pB.J, r)
      for iB = 1:length(pB)
         F[i, iB] += stor.dJ[iB] * (R/r)
         F[j, iB] -= stor.dJ[iB] * (R/r)
      end
   end
   return [ F[:, iB] for iB = 1:length(pB) ]
end

function virial(pB::PolyPairBasis, at::Atoms{T}) where {T}
   V = zeros(JMat{T}, length(pB))
   stor = alloc_temp_d(pB)
   for (i, j, R) in pairs(at, cutoff(pB))
      r = norm(R)
      evaluate_d!(stor.J, stor.dJ, nothing, pB.J, r)
      for iB = 1:length(pB)
         V[iB] -= (stor.dJ[iB]/r) * R * R'
      end
   end
   return V
end
