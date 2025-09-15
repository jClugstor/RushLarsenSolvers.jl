@cache struct EulerCache{uType,rateType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
end

function alg_cache(alg::RushLarsen, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{true}) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    EulerCache(u, uprev, zero(u), zero(rate_prototype), zero(rate_prototype))
end

struct RushLarsenConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(alg::RushLarsen, u, rate_prototype, ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
    dt, reltol, p, calck,
    ::Val{false}) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    RushLarsenConstantCache()
end



@inline function perform_step!(integrator, cache::RushLarsenConstantCache, f=integrator.f)

end