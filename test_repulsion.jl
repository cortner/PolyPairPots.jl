

@testset "RepulsiveCore" begin

@info("--------------- Testing RepulsiveCore Implementation ---------------")

at = bulk(:W, cubic=true) * 3
rattle!(at, 0.03)
r0 = rnn(:W)
z = atomic_number(:W)
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
println(@test all(Vfit(r) == Vrep(r,z,z) for r in rout))
rin = range(0.5, ri, length=100)
println(@test all(Vrep.Vin[1](r) == Vrep(r,z,z) for r in rin))

@info("JuLIP FD test")
println(@test JuLIP.Testing.fdtest(Vrep, at))

@info("check scaling")
println(@test energy(Vfit, at) ≈ energy(Vrep, at))

##


@info("--------------- Multi-Species RepulsiveCore ---------------")

at = bulk(:W, cubic=true) * 3
at.Z[2:3:end] .= atomic_number(:Fe)
rattle!(at, 0.03)
r0 = rnn(:W)
trans = PolyTransform(2, r0)
fcut = PolyCutoff2s(2, 0.5*r0, 1.95*r0)
B = PolyPairBasis([:W, :Fe], 10, trans, fcut)
coeffs = randcoeffs(B)
pot = combine(B, coeffs)

## try out the repulsive potential
Vfit = pot

ri = 2.1
z1 = 26
z2 = 74
@show @D Vfit(ri, 26, 74)
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
