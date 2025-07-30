mutable struct Box{T<:AbstractFloat}
    h::Matrix{T}      # 3x3 box matrix (column vectors are box vectors)
    h_inv::Matrix{T}  # inverse of box matrix
end

"""
    Box(Lx::T, Ly::T, Lz::T) where T

Create a Box with orthogonal box vectors of lengths Lx, Ly, Lz.
"""
function Box(Lx::T, Ly::T, Lz::T) where {T}
    h = diagm([Lx, Ly, Lz])
    h_inv = diagm([1 / Lx, 1 / Ly, 1 / Lz])
    return Box{T}(h, h_inv)
end

"""
    Box(L::T) where T

Create a cubic Box with side length L.
"""
function Box(L::T) where {T}
    return Box(L, L, L)
end

"""
    minimum_image_distance(r1::Vector{T}, r2::Vector{T}, box::Box{T}) where T

Compute the minimum image distance vector between two positions using arbitrary box shape.

Algorithm:
1. dr = r2 - r1
2. fractional displacement: s = h⁻¹ * dr
3. apply MIC in fractional space: s_c = s - round.(s)
4. back to Cartesian: dr_c = h * s_c
"""
function minimum_image_distance(r1::Vector{T}, r2::Vector{T}, box::Box{T}) where {T}
    s = box.h_inv * (r2 .- r1)            # fractional displacement
    s0 = round.(s)                        # nearest integer vector in Z^3
    best_d = typemax(T)
    best_dr = zeros(T, 3)
    @inbounds for di in -1:1, dj in -1:1, dk in -1:1
        m = s0 .+ (di, dj, dk)           # 27 candidates around s0
        dr = box.h * (s .- m)
        d = norm(dr)
        if d < best_d
            best_d = d
            best_dr .= dr
        end
    end
    return best_dr, best_d
end

"""
    minimum_image_distance(p1::Particle{T}, p2::Particle{T}, box::Box{T})

Convenience wrapper for particles.
"""
function minimum_image_distance(p1::Particle{T}, p2::Particle{T}, box::Box{T}) where {T}
    r1 = [p1.x, p1.y, p1.z]
    r2 = [p2.x, p2.y, p2.z]
    return minimum_image_distance(r1, r2, box)
end

"""
    apply_pbc!(particle::Particle{T}, box::Box{T})

Wrap a particle back into the primary cell [0,1)³ in fractional space.
"""
function apply_pbc!(particle::Particle{T}, box::Box{T}) where {T}
    r = [particle.x, particle.y, particle.z]
    s = box.h_inv * r
    # positions live in [0,1)
    s_wrapped = @. s - floor(s)
    r_wrapped = box.h * s_wrapped
    particle.x = r_wrapped[1]
    particle.y = r_wrapped[2]
    particle.z = r_wrapped[3]

    return nothing
end

function calculate_box_size(
    n_particles::Int, packing_fraction::T, frac_B::T, sigma_A::T, size_ratio::T
) where {T<:AbstractFloat}
    vol_A = π * sigma_A^3 / 6.0
    vol_B = π * (sigma_A * size_ratio)^3 / 6.0
    n_A = round(Int, n_particles * (1 - frac_B))
    n_B = n_particles - n_A
    total_particle_volume = n_A * vol_A + n_B * vol_B
    box_volume = total_particle_volume / packing_fraction
    return T(box_volume^(1 / 3))
end

"""
    calculate_current_packing_fraction(particles, box, sigma_A, size_ratio)
"""
function calculate_current_packing_fraction(particles, box, sigma_A, size_ratio)
    n_A = count(p -> p.type == 1, particles)
    n_B = count(p -> p.type == 2, particles)
    vol_A = π * sigma_A^3 / 6.0
    vol_B = π * (sigma_A * size_ratio)^3 / 6.0
    total_particle_vol = n_A * vol_A + n_B * vol_B
    box_vol = det(box.h)
    return total_particle_vol / box_vol
end

"""
    compute_box_heights(box::Box{T}) where T

Return the distances between opposite faces along v1,v2,v3.
"""
function compute_box_heights(box::Box{T}) where {T}
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    volume = abs(det(box.h))
    area_23 = norm(cross(v2, v3))
    area_13 = norm(cross(v1, v3))
    area_12 = norm(cross(v1, v2))
    h1 = volume / area_23
    h2 = volume / area_13
    h3 = volume / area_12
    return (h1, h2, h3)
end
