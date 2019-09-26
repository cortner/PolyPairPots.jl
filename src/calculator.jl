

import JuLIP
using JuLIP: JVec, Atoms, PairPotential
using JuLIP.Potentials: @pot
using LinearAlgebra: dot

# ----------------------------------------------------

export PolyPairPot

struct PolyPairPot{T,TJ} <: PairPotential
   J::TJ
   coeffs::Vector{T}
end

@pot PolyPairPot

PolyPairPot(pB::PolyPairBasis, coeffs::Vector) =
            PolyPairPot(pB.J, collect(coeffs))

JuLIP.MLIPs.combine(pB::PolyPairBasis, coeffs::AbstractVector) =
            PolyPairPot(pB, collect(coeffs))

JuLIP.cutoff(V::PolyPairPot) = cutoff(V.J)

==(V1::PolyPairPot, V2::PolyPairPot) =
            ( (V1.J == V2.J) && (V1.coeffs == V2.coeffs) )

Dict(V::PolyPairPot) = Dict(
      "__id__" => "PolyPairPots_PolyPairPot",
      "J" => Dict(V.J),
      "coeffs" => V.coeffs)

PolyPairPot(D::Dict) = PolyPairPot(
      TransformedJacobi(D["J"]),
      Vector{Float64}(D["coeffs"]))

convert(::Val{:PolyPairPots_PolyPairPot}, D::Dict) = PolyPairPot(D)


alloc_temp(V::PolyPairPot{T}, N::Integer) where {T} =
      ( J = alloc_B(V.J), R = zeros(JVec{T}, N) )

alloc_temp_d(V::PolyPairPot{T}, N::Integer) where {T} =
      ( J = alloc_B(V.J), dJ = alloc_dB(V.J),
        dV = zeros(JVec{T}, N), R = zeros(JVec{T}, N) )

evaluate!(tmp, V::PolyPairPot, r::Number) =
      2 * dot(V.coeffs, evaluate!(tmp.J, nothing, V.J, r))

evaluate_d!(tmp, V::PolyPairPot, r::Number) =
      2 * dot(V.coeffs, evaluate_d!(tmp.J, tmp.dJ, nothing, V.J, r))
