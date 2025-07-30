mutable struct Particle{T<:AbstractFloat,U<:Integer}
    x::T
    y::T
    z::T
    type::U
    sigma::T
end

"""
    ImageLists

Holds the two image lists used in FBMC:
- `self`: reduced self‑image list (for step (ii) in the paper)
- `pair`: pair‑image list (self list + one positive layer in each vi direction; step (iii))
- `N`: tuple `(N1,N2,N3)` of bounds used to build the lists
"""
mutable struct ImageLists
    self::Vector{NTuple{3,Int}}
    pair::Vector{NTuple{3,Int}}
    N::NTuple{3,Int}
end

"""
    Enhanced AdaptiveParameters to include deformation move tracking with bounds
"""
mutable struct EnhancedAdaptiveParameters{T<:AbstractFloat}
    max_displacement::T
    max_ln_vol_change::T
    max_deformation::T  # New field for deformation moves
    target_trans_acceptance::T
    target_vol_acceptance::T
    target_deform_acceptance::T  # New field
    adaptation_interval::Int
    adaptation_factor::T

    # Minimum bounds to prevent parameters from becoming too small
    min_displacement::T
    min_ln_vol_change::T
    min_deformation::T

    # Counters for current interval
    trans_accepted::Int
    trans_attempts::Int
    vol_accepted::Int
    vol_attempts::Int
    deform_accepted::Int  # New counters
    deform_attempts::Int
end

"""
    EnhancedAdaptiveParameters constructor with minimum bounds
"""
function EnhancedAdaptiveParameters(
    max_displacement::T,
    max_ln_vol_change::T,
    max_deformation::T,
    target_trans_acc::T=0.25,
    target_vol_acc::T=0.15,
    target_deform_acc::T=0.20,
    interval::Int=1000,
    factor::T=1.05,
    min_displacement::T=1e-4,
    min_ln_vol_change::T=1e-4,
    min_deformation::T=1e-4,
) where {T}
    return EnhancedAdaptiveParameters{T}(
        max_displacement,
        max_ln_vol_change,
        max_deformation,
        target_trans_acc,
        target_vol_acc,
        target_deform_acc,
        interval,
        factor,
        min_displacement,
        min_ln_vol_change,
        min_deformation,
        0,
        0,
        0,
        0,
        0,
        0,  # Initialize all counters to zero
    )
end

"""
    adapt_enhanced_parameters!(params::EnhancedAdaptiveParameters)

Adapt all move parameters with bounds and smart adaptation logic.
"""
function adapt_enhanced_parameters!(params::EnhancedAdaptiveParameters, box)
    # Tolerance for acceptance rate comparison
    tolerance = 0.0

    # Adapt translational moves
    if params.trans_attempts > 0
        current_trans_acc = params.trans_accepted / params.trans_attempts
        if current_trans_acc > params.target_trans_acceptance + tolerance
            params.max_displacement = min(
                params.max_displacement * params.adaptation_factor, 0.5 * minimum(compute_box_heights(box))
            )
            # params.max_displacement = params.max_displacement * params.adaptation_factor
        elseif current_trans_acc < params.target_trans_acceptance - tolerance
            params.max_displacement = max(
                params.max_displacement / params.adaptation_factor, params.min_displacement
            )
            # params.max_displacement = params.max_displacement / params.adaptation_factor
        end
    end

    # Adapt volume moves
    if params.vol_attempts > 0
        # max_ln_vol_change_limit = 1.0
        p_acc = params.vol_accepted / params.vol_attempts
        p_target = params.target_vol_acceptance

        if p_acc > p_target
            params.max_ln_vol_change *= params.adaptation_factor
        else
            params.max_ln_vol_change /= params.adaptation_factor
        end

        # params.max_ln_vol_change = clamp(
        #     params.max_ln_vol_change, params.min_ln_vol_change, max_ln_vol_change_limit
        # )
        params.max_ln_vol_change = max(params.max_ln_vol_change, params.min_ln_vol_change)
    end

    # Adapt deformation moves
    if params.deform_attempts > 0
        current_deform_acc = params.deform_accepted / params.deform_attempts
        if current_deform_acc > params.target_deform_acceptance + tolerance
            # params.max_deformation = min(
            #     params.max_deformation * params.adaptation_factor, 0.5
            # )
            params.max_deformation = params.max_deformation * params.adaptation_factor
        elseif current_deform_acc < params.target_deform_acceptance - tolerance
            params.max_deformation = max(
                params.max_deformation / params.adaptation_factor, params.min_deformation
            )
            # params.max_deformation = params.max_deformation / params.adaptation_factor
        end
    end

    # Reset counters for next interval
    params.trans_accepted = 0
    params.trans_attempts = 0
    params.vol_accepted = 0
    params.vol_attempts = 0
    params.deform_accepted = 0
    params.deform_attempts = 0

    return nothing
end
