using Test
using PolyPairPots, JuLIP, LinearAlgebra, Test
using JuLIP.Testing, JuLIP.MLIPs

randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

@testset "PolyPairPots.jl" begin

include("test_basics.jl")
include("test_repulsion.jl")


end
