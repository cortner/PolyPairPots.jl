

module Repulsion

import JuLIP: decode_dict
import JuLIP.Potentials: @pot, evaluate, evaluate_d, PairPotential, @D, cutoff,
                         @analytic, evaluate!, evaluate_d!,
                         alloc_temp, alloc_temp_d
import Base: Dict, convert

struct RepulsiveCore{TV1, TV2} <: PairPotential
   Vout::TV1         # the outer pair potential
   Vin::TV2          # the inner (repulsive) pair potential
   ri::Float64       # the interface between the two
   e0::Float64
end

@pot RepulsiveCore

cutoff(V::RepulsiveCore) = cutoff(V.Vout)

alloc_temp(V::RepulsiveCore, N::Integer) = alloc_temp(V.Vout, N)
alloc_temp_d(V::RepulsiveCore, N::Integer) = alloc_temp_d(V.Vout, N)

evaluate!(tmp, V::RepulsiveCore, r::Number) =  (
   r > V.ri ? evaluate!(tmp, V.Vout, r)
            : evaluate!(nothing, V.Vin, r) )

evaluate_d!(tmp, V::RepulsiveCore, r::Number) =  (
   r > V.ri ? evaluate_d!(tmp, V.Vout, r)
            : evaluate_d!(nothing, V.Vin, r) )



function RepulsiveCore(Vout, ri, e0=0.0; verbose=false)
   v = Vout(ri)
   dv = @D Vout(ri)
   if dv >= 0.0
      @error("The slope `Vout'(ri)` is required to be negative")
   end
   if dv > -1.0
      @warn("""The slope `Vout'(ri) = $dv` may not be steep enough to attach a
               repulsive core. Proceed at your own risk.""")
   end
   if v-e0 <= 0.0
      @warn("it is recommended that `Vout(ri) > 0`.")
   end
   if v-e0 <= 1.0
      @warn("""Ideally the repulsive core should not be attached at small
               values of `Vout(ri) = $v`. Proceed at your own risk.""")
   end
   # e0 + B e^{-A (r/ri-1)} * ri/r
   #    => e0 + B = Vout(ri) => = Vout(ri) - e0 = v - e0
   # dv = - A*B/ri e^{-A (r/ri-1)} * ri/r - B*ri*e^{...} / r^2
   #    = - A/ri * (v - 1/ri * (v = - (1+A)/ri * (v-e0)
   #    => -(1+A)/ri * (v-e0) = dv
   #    => 1+A = - ri dv / (v-e0)
   #    => A = -1 - ri dv / (v-e0)
   Vin = let A = -1 - ri * dv / (v-e0), B = v-e0, e0=e0, ri = ri
      if verbose
         @show A, B
      end
      @analytic r -> e0 + B * exp( - A * (r/ri-1) ) * ri/r
   end
   if verbose
      @show ri
      @show Vout(ri), (@D Vout(ri))
      @show Vin(ri), (@D Vin(ri))
   end
   # construct the piecewise potential
   return RepulsiveCore(Vout, Vin, ri, e0)
end

# ----------------------------------------------------
#  File IO
# ----------------------------------------------------

Dict(V::RepulsiveCore) = Dict("__id__" => "PolyPairPots_RepulsiveCore",
                              "Vout" => Dict(V.Vout),
                              "e0" => V.e0,
                              "ri" => V.ri)

function RepulsiveCore(D::Dict)
   if haskey(D, "e0")
      return RepulsiveCore(decode_dict(D["Vout"]), D["ri"], D["e0"])
   else
      return RepulsiveCore(decode_dict(D["Vout"]), D["ri"])
   end
end

convert(::Val{:PolyPairPots_RepulsiveCore}, D::Dict) = RepulsiveCore(D)

end
