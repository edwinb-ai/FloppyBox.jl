# Image‑list construction and overlap checks following de Graaf et al. (2012)
# Steps (ii) and (iii) in Sec. IV.  See text and Eq. (3) therein. 
# Self‑images from mapped cube; pair‑images add one positive layer + the origin. 
# (J. Chem. Phys. 137, 214101 (2012))

# === Core builders ==========================================================

"""
    compute_self_image_list(box::Box{T}, RO::T; tol=eps(T)*10) -> (list, (N1,N2,N3))

Build reduced self‑image list P̃_im as in the paper:
1) Map the 8 cube corners c_n = 2RO(±x̂ ±ŷ ±ẑ) via p_n = M⁻¹ c_n.
2) N_i = ceil( max_n |(p_n)_i| ).
3) Enumerate i∈[-N1,N1], j∈[-N2,N2], k∈[-N3,N3], exclude (0,0,0).
4) Keep only the half‑space dot( i v1 + j v2 + k v3 , v1+v2+v3 ) ≥ 0
   and || i v1 + j v2 + k v3 || ≤ 2RO + tol.
5) Sort concentrically by |i|+|j|+|k|.
"""
function compute_self_image_list(
    box::Box{T}, RO::T; tol::T=eps(T) * 10
) where {T<:AbstractFloat}
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    M_inv = box.h_inv

    # 8 corners in Cartesian
    corners = Vector{Vector{T}}()
    for sx in (-one(T), one(T)), sy in (-one(T), one(T)), sz in (-one(T), one(T))
        push!(corners, [2RO * sx, 2RO * sy, 2RO * sz])
    end

    # p_n = M^{-1} c_n  -> bounds
    maxabs = zeros(T, 3)
    for c in corners
        p = M_inv * c
        maxabs[1] = max(maxabs[1], abs(p[1]))
        maxabs[2] = max(maxabs[2], abs(p[2]))
        maxabs[3] = max(maxabs[3], abs(p[3]))
    end
    N1 = ceil(Int, maxabs[1])
    N2 = ceil(Int, maxabs[2])
    N3 = ceil(Int, maxabs[3])

    n_hat = v1 + v2 + v3
    Rcut = T(2) * RO + tol

    lst = NTuple{3,Int}[]
    for i in (-N1):N1, j in (-N2):N2, k in (-N3):N3
        if i == 0 && j == 0 && k == 0
            continue
        end
        t = i * v1 + j * v2 + k * v3
        if dot(t, n_hat) < 0
            continue
        end      # half‑space
        if norm(t) > Rcut
            continue
        end      # radius ≤ 2RO
        push!(lst, (i, j, k))
    end
    sort!(lst; by=s -> (abs(s[1]) + abs(s[2]) + abs(s[3]), s[1]^2 + s[2]^2 + s[3]^2))
    return lst, (N1, N2, N3)
end

"""
    compute_pair_image_list(box::Box{T}, N::NTuple{3,Int}, self_list) -> list

Extend the reduced self list by adding ONE additional layer in the POSITIVE
vi directions: i=N1+1 and/or j=N2+1 and/or k=N3+1, and **include (0,0,0)** so
in-cell pairs are tested. We reuse the same half-space filter.
"""
function compute_pair_image_list(
    box::Box{T}, N::NTuple{3,Int}, self_list::Vector{NTuple{3,Int}}
) where {T<:AbstractFloat}
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    n_hat = v1 + v2 + v3
    N1, N2, N3 = N

    pairset = Set{NTuple{3,Int}}(self_list)
    push!(pairset, (0, 0, 0))  # include origin

    for i in (-N1 - 1):(N1 + 1), j in (-N2 - 1):(N2 + 1), k in (-N3 - 1):(N3 + 1)
        # add only the shell outside the self-list bounding box
        if (abs(i) == N1 + 1) || (abs(j) == N2 + 1) || (abs(k) == N3 + 1)
            t = i * v1 + j * v2 + k * v3
            if dot(t, n_hat) >= 0
                push!(pairset, (i, j, k))
            end
        end
    end
    lst = collect(pairset)
    sort!(lst; by=s -> (abs(s[1]) + abs(s[2]) + abs(s[3]), s[1]^2 + s[2]^2 + s[3]^2))
    return lst
end

"""
    compute_image_lists(box, particles) -> ImageLists

Compute RO = max(sigma)/2 and build both lists.
"""
function compute_image_lists(box, particles)
    RO = maximum(p.sigma for p in particles) / 2
    self_list, N = compute_self_image_list(box, RO)
    pair_list = compute_pair_image_list(box, N, self_list)
    return ImageLists(self_list, pair_list, N)
end

"""
    update_image_lists!(images, box, particles)

In‑place update of an existing ImageLists instance.
"""
function update_image_lists!(
    images::ImageLists, box::Box{T}, particles
) where {T<:AbstractFloat}
    tmp = compute_image_lists(box, particles)
    empty!(images.self)
    append!(images.self, tmp.self)
    empty!(images.pair)
    append!(images.pair, tmp.pair)
    images.N = tmp.N
    return images
end

# === Geometry utilities on lists ============================================

"""
    minimum_image_distance(r1, r2, box, list)

Return (vector, distance) to the closest image of r2 among `list`.
"""
function minimum_image_distance(
    r1::Vector{T}, r2::Vector{T}, box::Box{T}, list
) where {T<:AbstractFloat}
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    min_d = Inf
    min_v = zeros(T, 3)
    for (i, j, k) in list
        dr = r1 .- (r2 .+ i * v1 .+ j * v2 .+ k * v3)
        d = norm(dr)
        if d < min_d
            min_d = d
            min_v .= dr
        end
    end
    return min_v, min_d
end

function minimum_image_distance(
    p1::Particle{T}, p2::Particle{T}, box, list
) where {T<:AbstractFloat}
    r1 = [p1.x, p1.y, p1.z]
    r2 = [p2.x, p2.y, p2.z]
    return minimum_image_distance(r1, r2, box, list)
end

# === Overlap checks ==========================================================

"""
    check_overlaps_with_images(particles, box, images::ImageLists)

Return (has_overlap::Bool, details). Self–image overlaps use `images.self`.
Pair overlaps use `images.pair` **and** a central‑cell MIC fallback.
Both orderings (j relative to i, and i relative to j) are tested.
"""
function check_overlaps_with_images(
    particles::Vector{Particle{T,U}}, box::Box{T}, images::ImageLists
) where {T<:AbstractFloat,U<:Integer}

    # 1) Self–image overlaps
    v1 = @views box.h[:, 1]
    v2 = @views box.h[:, 2]
    v3 = @views box.h[:, 3]
    for (idx, p) in enumerate(particles)
        for (i, j, k) in images.self
            Δr = i * v1 + j * v2 + k * v3
            if norm(Δr) < p.sigma - 1e-12
                return true, ("self-image", idx, (i, j, k), norm(Δr), p.sigma)
            end
        end
    end

    # 2) Pairwise overlaps — use pair list + central MIC fallback
    n = length(particles)
    for a in 1:(n - 1), b in (a + 1):n
        pa = particles[a]
        pb = particles[b]
        # via pair list (both orderings, b/c list is not point symmetric)
        _, d_ab = minimum_image_distance(pa, pb, box, images.pair)
        _, d_ba = minimum_image_distance(pb, pa, box, images.pair)
        # central MIC fallback (from box.jl)
        _, d_c = minimum_image_distance(pa, pb, box)   # no list => MIC in fractional space

        d = min(d_c, min(d_ab, d_ba))
        thresh = (pa.sigma + pb.sigma) / 2
        if d < thresh - 1e-12
            return true, ("pair", a, b, d, thresh)
        end
    end
    return false, nothing
end
