using Test
using PolyPairPots, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, JuLIP.MLIPs

randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

@testset "PolyPairPots.jl" begin


@testset "PolyPairBasis" begin

@info("-------- Test PolyPairBasis Implementation ---------")

##

using PolyPairPots, JuLIP, LinearAlgebra, Test
using JuLIP.Testing

##

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
X = copy(positions(at))

trans = PolyTransform(2, r0)
fcut = PolyCutoff1s(2, 0.5*r0, 2.5*r0)
pB = PolyPairBasis(:W, 10, trans, fcut)

##

@info("test (de-)dictionisation of PairBasis")
println(@test decode_dict(Dict(pB)) == pB)

E = energy(pB, at)
DE = - forces(pB, at)

@info("Finite-difference test on PolyPairBasis forces")
for ntest = 1:20
   U = rand(JVecF, length(at)) .- 0.5
   DExU = dot.(DE, Ref(U))
   errs = Float64[]
   for p = 2:10
      h = 0.1^p
      Eh = energy(pB, set_positions!(at, X+h*U))
      DEhxU = (Eh-E) / h
      push!(errs, norm(DExU - DEhxU, Inf))
   end
   success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
   print_tf(@test success)
end
println()
##

end


@testset "PolyPairPot" begin

@info("--------------- PolyPairPot Implementation ---------------")

##

using PolyPairPots, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, LinearAlgebra
using JuLIP.MLIPs

##

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)

trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis(:W, 10, trans, fcut)

##


@info("Testing correctness of `PolyPairPot` against `PolyPairBasis`")

@info("    test `combine`")
coeffs = randcoeffs(B)
pot = combine(B, coeffs)
println(@test pot == PolyPairPot(B, coeffs))


##
@info("   test (de-)dictionisation")
println(@test decode_dict(Dict(pot)) == pot)

@info("      check that PolyPairBasis ≈ PolyPairPot")
for ntest = 1:30
   rattle!(at, 0.01)

   E_pot = energy(pot, at)
   E_b = dot(energy(B, at), coeffs)
   print_tf(@test E_pot ≈ E_b)

   F_pot = forces(pot, at)
   F_b = sum(coeffs .* forces(B, at))
   print_tf(@test F_pot ≈ F_b)

   V_pot = virial(pot, at)
   V_b = sum(coeffs .* virial(B, at))
   print_tf(@test V_pot ≈ V_b)
end
println()

@info("      Standard JuLIP Consistency Test")
variablecell!(at)
rattle!(at, 0.03)
JuLIP.Testing.fdtest(pot, at)

##
end



@testset "RepulsiveCore" begin

@info("--------------- Testing RepulsiveCore Implementation ---------------")

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis(:W, 10, trans, fcut)
coeffs = randcoeffs(B)
pot = combine(B, coeffs)

## try out the repulsive potential
Vfit = pot

ri = 2.1
@show @D Vfit(ri)
if (@D Vfit(ri)) > 0
   Vfit = PolyPairPot(- Vfit.coeffs, Vfit.J, Vfit.zlist, Vfit.bidx0)
end
@show @D Vfit(ri)
e0 = Vfit(ri) - 1.0
Vrep = PolyPairPots.Repulsion.RepulsiveCore(Vfit, ri)


rout = range(ri, 4.0, length=100)
println(@test all(Vfit(r) == Vrep(r) for r in rout))
rin = range(0.5, ri, length=100)
println(@test all(Vrep.Vin(r) == Vrep(r) for r in rin))

@info("JuLIP FD test")
println(@test JuLIP.Testing.fdtest(Vrep, at))

@info("check scaling")
println(@test energy(Vfit, at) ≈ energy(Vrep, at))

##

end


end
