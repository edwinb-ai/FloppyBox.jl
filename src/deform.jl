"""
    check_box_validity(M::Matrix{T}, particles) -> Bool

Simple geometric guard: face‑to‑face distances must exceed the max diameter.
"""
function check_box_validity(M::Matrix{T}, particles) where {T<:AbstractFloat}
    v1 = @views M[:, 1]
    v2 = @views M[:, 2]
    v3 = @views M[:, 3]
    volume = abs(det(M))
    area_23 = norm(cross(v2, v3))
    area_13 = norm(cross(v1, v3))
    area_12 = norm(cross(v1, v2))
    h1 = volume / area_23
    h2 = volume / area_13
    h3 = volume / area_12
    max_sigma = maximum(p.sigma for p in particles)
    min_required = max_sigma + 1e-6
    return (h1 >= min_required) && (h2 >= min_required) && (h3 >= min_required)
end

"""
    deformation_move!(particles, box, pressure, max_deformation, images, cutoff_buffer;
                      auto_reduce=true, reduction_threshold=2.0) -> (Int, Bool)
"""
function deformation_move!(
    particles::Vector{Particle{T,Int}},
    box::Box{T},
    pressure::T,
    max_deformation::T,
    images::ImageLists,
    auto_reduce::Bool=true,
    reduction_threshold::T=2.0,
    sigma_A=1.0,
    size_ratio=0.0,
) where {T<:AbstractFloat}
    N = length(particles)

    # Store old configuration
    old_box = deepcopy(box)
    old_particles = deepcopy(particles)
    old_images = deepcopy(images)

    # Random element perturbation
    a = rand(1:3)
    b = rand(1:3)
    Δ = (2.0 * rand() - 1.0) * max_deformation

    # Proposed matrix and quick validity checks
    M_new = copy(box.h)
    M_new[a, b] += Δ
    V_new = det(M_new)
    if V_new <= 0
        return 0, false
    elseif !check_box_validity(M_new, particles)
        return 0, false
    end

    # Apply box
    box.h .= M_new
    try
        box.h_inv .= inv(M_new)
    catch
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        return 0, false
    end

    # Recompute particle positions from old scaled coords
    scaled = Vector{Vector{T}}(undef, N)
    for (i, p) in enumerate(old_particles)
        r_old = [p.x, p.y, p.z]
        scaled[i] = old_box.h_inv * r_old
    end
    for (i, p) in enumerate(particles)
        r_new = box.h * scaled[i]
        p.x = r_new[1]
        p.y = r_new[2]
        p.z = r_new[3]
    end

    current_phi = calculate_current_packing_fraction(particles, box, sigma_A, size_ratio)
    if current_phi >= 1.0
        # Restore
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        for (i, p) in enumerate(particles)
            p.x = old_particles[i].x
            p.y = old_particles[i].y
            p.z = old_particles[i].z
        end
        return 0, false
    end

    # New image lists
    new_images = compute_image_lists(box, particles)

    # Overlap check
    overlaps, _ = check_overlaps_with_images(particles, box, new_images)
    if overlaps
        # Restore
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        for (i, p) in enumerate(particles)
            p.x = old_particles[i].x
            p.y = old_particles[i].y
            p.z = old_particles[i].z
        end
        images.self = deepcopy(old_images.self)
        images.pair = deepcopy(old_images.pair)
        images.N = old_images.N
        return 0, false
    end

    # NPT acceptance for deformation (Eq. (2) in the paper)
    V_old = det(old_box.h)
    ΔE = pressure * (V_new - V_old)
    ln_jac = N * log(V_new / V_old)
    ln_acc = -ΔE + ln_jac

    if ln_acc >= 0 || rand() < exp(ln_acc)
        # Accept: update images
        images.self = deepcopy(new_images.self)
        images.pair = deepcopy(new_images.pair)
        images.N = new_images.N

        # Optional lattice reduction
        red_performed = false
        if auto_reduce
            should_reduce, _ = should_reduce_lattice(box, reduction_threshold)
            if should_reduce
                red_performed, _ = lattice_reduce!(
                    box, particles, images; threshold=reduction_threshold
                )
            end
        end
        return 1, red_performed
    else
        # Reject: restore everything
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        for (i, p) in enumerate(particles)
            p.x = old_particles[i].x
            p.y = old_particles[i].y
            p.z = old_particles[i].z
        end
        images.self = deepcopy(old_images.self)
        images.pair = deepcopy(old_images.pair)
        images.N = old_images.N
        return 0, false
    end
end

"""
    compute_box_strain(box_current, box_reference) -> strain
"""
function compute_box_strain(box_current::Box{T}, box_reference::Box{T}) where {T}
    F = box_current.h * box_reference.h_inv
    return 0.5 * (F + F') - I
end
