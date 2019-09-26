module PolyPairPots

import JuLIP: energy, forces, virial, alloc_temp, alloc_temp_d, cutoff

import JuLIP.Potentials:  evaluate, evaluate_d, evaluate!, evaluate_d!
import Base: Dict, convert, ==

function alloc_B end
function alloc_dB end


include("jacobi.jl")

include("transforms.jl")

include("basis.jl")

include("calculator.jl")

include("repulsion.jl") 

end # module
