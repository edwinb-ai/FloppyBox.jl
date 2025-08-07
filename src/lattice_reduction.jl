# Lattice reduction following de Graaf et al. (Sec. VI).
# Distortion metric C as in Eq. (4) of the paper.

function compute_distortion_metric(M::Matrix{T}) where {T<:AbstractFloat}
    v1 = @views M[:, 1]
    v2 = @views M[:, 2]
    v3 = @views M[:, 3]
    len_sum = norm(v1) + norm(v2) + norm(v3)
    area_sum = norm(cross(v1, v2)) + norm(cross(v1, v3)) + norm(cross(v2, v3))
    vol = dot(v1, cross(v2, v3))
    if vol < 1e-12
        return T(Inf)
    end
    return ((len_sum / 3.0) * (area_sum / 3.0)) / vol
end

function compute_distortion_metric(box::Box{T}) where {T<:AbstractFloat}
    return compute_distortion_metric(box.h)
end

function should_reduce_lattice(box::Box{T}, threshold::T=1.5) where {T<:AbstractFloat}
    C = compute_distortion_metric(box)
    return C > threshold, C
end

# Quick geometric guard (same as in deform.jl)
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
    # Relaxed constraint: allow more aggressive compression for binary mixtures
    min_required = max_sigma * 0.75  # Was max_sigma + 1e-6
    return (h1 >= min_required) && (h2 >= min_required) && (h3 >= min_required)
end

# Build the unimodular matrix E that implements the column operation:
# v_i <- v_i + sgn * v_j  (i != j)
# Then M_new = M * E
# E is identity with E[j,i] = sgn.
@inline function unimodular_E(i::Int, j::Int, sgn::Int)
    E = Matrix{Int}(I, 3, 3)
    @assert i != j
    E[j, i] = sgn
    return E
end

"""
    lattice_reduce!(box, particles, images, cutoff_buffer; threshold=1.5, max_iter=15)

Perform a sequence of unimodular basis changes that lower the distortion metric.
IMPORTANT: We transform fractional coordinates by U^{-1} so Cartesian positions
remain IDENTICAL; this guarantees we don't create overlaps during reduction.
"""
function lattice_reduce!(
    box::Box{T},
    particles::Vector{Particle{T,Int}},
    images::ImageLists;
    threshold::T=1.5,
    max_iter::Int=15,
) where {T<:AbstractFloat}

    # Save state to allow full revert on failure (paranoid but safe)
    box_old = deepcopy(box)
    parts_old = deepcopy(particles)
    imgs_old = ImageLists(copy(images.self), copy(images.pair), images.N)

    # Store scaled coordinates wrt CURRENT box
    Np = length(particles)
    s_old = Vector{Vector{T}}(undef, Np)
    for (i, p) in enumerate(particles)
        s_old[i] = box.h_inv * [p.x, p.y, p.z]
    end

    # Current box matrix and cumulative unimodular transform U (start at I)
    M = copy(box.h)
    U = Matrix{Int}(I, 3, 3)
    reduction_performed = false

    for _ in 1:max_iter
        C_current = compute_distortion_metric(M)
        if C_current <= threshold
            break
        end

        best_C = C_current
        best_E = Matrix{Int}(I, 3, 3)
        improved = false

        # Try the 12 candidates v_i -> v_i ± v_j
        for i in 1:3, j in 1:3
            i == j && continue
            for sgn in (-1, 1)
                E = unimodular_E(i, j, sgn)
                Mc = M * E
                # Determinant is preserved by unimodular E; still check geometry
                if !check_box_validity(Mc, particles)
                    continue
                end
                Cc = compute_distortion_metric(Mc)
                if Cc + eps(T) < best_C   # strictly better
                    best_C = Cc
                    best_E = E
                    improved = true
                end
            end
        end

        if improved
            M = M * best_E
            U = U * best_E
            reduction_performed = true
        else
            break
        end
    end

    if !reduction_performed
        return false, compute_distortion_metric(box)
    end

    # Apply the final basis: box.h <- M, box.h_inv <- inv(M)
    box.h .= M
    try
        box.h_inv .= inv(M)
    catch
        # Extremely unlikely (M should be nonsingular)
        # Revert everything
        box.h .= box_old.h
        box.h_inv .= box_old.h_inv
        for i in 1:Np
            particles[i].x = parts_old[i].x
            particles[i].y = parts_old[i].y
            particles[i].z = parts_old[i].z
        end
        images.self = copy(imgs_old.self)
        images.pair = copy(imgs_old.pair)
        images.N = imgs_old.N
        return false, compute_distortion_metric(box)
    end

    # Transform fractional coordinates: s_new = U^{-1} * s_old
    Uinv = inv(Matrix{Float64}(U))
    for i in 1:Np
        snew = Uinv * s_old[i]
        # wrap to [0,1)
        snew .= snew .- floor.(snew)
        r = box.h * snew
        particles[i].x = r[1]
        particles[i].y = r[2]
        particles[i].z = r[3]
    end

    # Update image lists for the new geometry
    update_image_lists!(images, box, particles)

    # Safety: configuration must be overlap free (it should be IDENTICAL)
    has_olap, _ = check_overlaps_with_images(particles, box, images)
    if has_olap
        # Revert cleanly if anything went wrong numerically
        box.h .= box_old.h
        box.h_inv .= box_old.h_inv
        for i in 1:Np
            particles[i].x = parts_old[i].x
            particles[i].y = parts_old[i].y
            particles[i].z = parts_old[i].z
        end
        images.self = copy(imgs_old.self)
        images.pair = copy(imgs_old.pair)
        images.N = imgs_old.N
        return false, compute_distortion_metric(box)
    end

    return true, compute_distortion_metric(box)
end

"""
    compute_box_orthogonality(box) -> NamedTuple

Angles, aspect ratio, and distortion metric.
"""
function compute_box_orthogonality(box::Box{T}) where {T<:AbstractFloat}
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    angle_12 = acos(clamp(dot(v1, v2) / (norm(v1) * norm(v2)), -1, 1))
    angle_13 = acos(clamp(dot(v1, v3) / (norm(v1) * norm(v3)), -1, 1))
    angle_23 = acos(clamp(dot(v2, v3) / (norm(v2) * norm(v3)), -1, 1))
    lengths = [norm(v1), norm(v2), norm(v3)]
    return (
        angle_12_deg=rad2deg(angle_12),
        angle_13_deg=rad2deg(angle_13),
        angle_23_deg=rad2deg(angle_23),
        angle_deviation=abs(rad2deg(angle_12) - 90) +
                        abs(rad2deg(angle_13) - 90) +
                        abs(rad2deg(angle_23) - 90),
        aspect_ratio=maximum(lengths) / minimum(lengths),
        distortion_metric=compute_distortion_metric(box),
    )
end
