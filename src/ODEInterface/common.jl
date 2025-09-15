using Reexport
@reexport using DiffEqBase
import DiffeqBase: solve

abstract type RushLarsenAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct RushLarsen <: RushLarsenAlgorithm end


function DiffEqBase.__solve()

end