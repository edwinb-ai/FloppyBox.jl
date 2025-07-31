module fbmc

using Random
using Printf: @sprintf
using LinearAlgebra

include("types.jl")
include("box.jl")
include("images.jl")
include("lattice_reduction.jl")
include("deform.jl")

"""
    initialize_cubic_particles(...)

Unchanged from your version; creates a dilute, overlap‑free start.
"""
function initialize_cubic_particles(
    n_particles::Int,
    frac_B::T,
    size_ratio::T,
    sigma_A::T,
    sigma_B::T,
    packing_fraction::T
) where {T<:AbstractFloat}

    # Composition & types
    n_B = round(Int, n_particles * frac_B)
    n_A = n_particles - n_B
    types = vcat(fill(1, n_A), fill(2, n_B))
    # random assignment on lattice
    Random.shuffle!(types)

    # Grid size
    n_per_dim = ceil(Int, n_particles^(1 / 3))

    # Box from packing fraction
    Lφ = calculate_box_size(n_particles, packing_fraction, frac_B, sigma_A, size_ratio)

    # Spacing ≥ σA
    sφ = Lφ / n_per_dim
    s_req = sigma_A
    spacing = max(sφ, s_req)
    L = spacing * n_per_dim
    box = Box(L)

    # Offset to center simple‑cubic lattice
    offset = spacing / 2.0

    # Place particles deterministically
    particles = Vector{Particle{T,Int}}()
    idx = 1
    for i in 0:(n_per_dim - 1), j in 0:(n_per_dim - 1), k in 0:(n_per_dim - 1)
        if idx > n_particles
            break
        end
        typ = types[idx]
        sigma = typ == 1 ? sigma_A : sigma_B

        x = offset + i * spacing
        y = offset + j * spacing
        z = offset + k * spacing

        push!(particles, Particle{T,Int}(x, y, z, typ, sigma))
        idx += 1
    end

    # Sanity check (should never fire)
    for a in 1:(length(particles) - 1), b in (a + 1):length(particles)
        _, d = minimum_image_distance(particles[a], particles[b], box)
        req = (particles[a].sigma + particles[b].sigma) / 2.0
        @assert d ≥ req - 1e-12 "Overlap: d=$d < req=$req"
    end

    return particles, box
end

@inline function wrap_all_particles!(particles, box)
    for p in particles
        apply_pbc!(p, box)
    end
    return nothing
end

function write_xyz(particles, box, filename, images::ImageLists)
    (overlap, _) = check_overlaps_with_images(particles, box, images)
    if overlap
        @warn "Configuration overlapping!!"
    end

    wrap_all_particles!(particles, box)

    open(filename, "w") do io
        println(io, length(particles))
        box_info = @sprintf(
            "Lattice=\"%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\" Properties=type:I:1:pos:R:3:radius:R:1 ",
            box.h[1, 1],
            box.h[2, 1],
            box.h[3, 1],
            box.h[1, 2],
            box.h[2, 2],
            box.h[3, 2],
            box.h[1, 3],
            box.h[2, 3],
            box.h[3, 3],
        )
        println(io, box_info)
        for p in particles
            radius = p.sigma / 2.0
            formatted = @sprintf("%d %.4f %.4f %.4f %.4f", p.type, p.x, p.y, p.z, radius)
            println(io, formatted)
        end
    end
    return nothing
end

"""
    mc_cycle!(particles, box, pressure, params; move_frequencies=(0.1,0.1),
              images=ImageLists([],[],(0,0,0)), cutoff_buffer=[1.0],
              auto_reduce=true, reduction_threshold=2.0)

Perform one MC cycle. Image lists are refreshed at the start of the cycle
and after any volume/deformation move that changes the box.
"""
function mc_cycle!(
    particles,
    box,
    pressure,
    params::EnhancedAdaptiveParameters;
    move_frequencies=(0.1, 0.1),
    images=ImageLists(NTuple{3,Int}[], NTuple{3,Int}[], (0, 0, 0)),
    auto_reduce=true,
    reduction_threshold=2.0,
    sigma_A=1.0,
    size_ratio=0.0,
)
    N = length(particles)
    volume_freq, deform_freq = move_frequencies

    # Refresh image lists for current geometry
    update_image_lists!(images, box, particles)

    # Translation attempts
    accepted_translations = 0
    total_translations = 0
    for _ in 1:N
        result = translation_with_images!(particles, box, params.max_displacement, images)
        accepted_translations += result
        total_translations += 1
        params.trans_accepted += result
        params.trans_attempts += 1
    end

    # One box move per cycle
    accepted_volumes = 0
    total_volumes = 0
    accepted_deformations = 0
    total_deformations = 0
    reductions_performed = 0

    move_type = rand()
    if move_type < volume_freq
        result = volume_move_with_images!(
            particles, box, pressure, params.max_ln_vol_change, images, sigma_A, size_ratio
        )
        accepted_volumes += result
        total_volumes += 1
        params.vol_accepted += result
        params.vol_attempts += 1
    elseif move_type < deform_freq + volume_freq
        result, reduced = deformation_move!(
            particles,
            box,
            pressure,
            params.max_deformation,
            images,
            auto_reduce,
            reduction_threshold,
            sigma_A,
            size_ratio,
        )
        accepted_deformations += result
        total_deformations += 1
        reductions_performed += reduced ? 1 : 0
        params.deform_accepted += result
        params.deform_attempts += 1
    end

    return accepted_translations,
    total_translations,
    accepted_volumes,
    total_volumes,
    accepted_deformations,
    total_deformations,
    reductions_performed
end

function translation_with_images!(particles, box, max_displacement, images::ImageLists)
    random_particle = rand(particles)
    oldp = deepcopy(random_particle)

    random_particle.x += (2.0 * rand() - 1.0) * max_displacement
    random_particle.y += (2.0 * rand() - 1.0) * max_displacement
    random_particle.z += (2.0 * rand() - 1.0) * max_displacement
    apply_pbc!(random_particle, box)

    overlap, _ = check_overlaps_with_images(particles, box, images)
    if overlap
        random_particle.x = oldp.x
        random_particle.y = oldp.y
        random_particle.z = oldp.z
        return 0
    end
    return 1
end

function volume_move_with_images!(
    particles, box, pressure, max_ln_vol_change, images::ImageLists, sigma_A, size_ratio
)
    N = length(particles)

    old_box = deepcopy(box)
    old_particles = deepcopy(particles)
    old_images = deepcopy(images)

    V_old = det(box.h)
    δlnV = (2.0 * rand() - 1.0) * max_ln_vol_change
    V_new = V_old * exp(δlnV)
    s = cbrt(V_new / V_old)

    # Scale box and positions
    box.h .*= s
    box.h_inv ./= s

    if !check_box_validity(box.h, particles)
        # Restore
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        return 0
    end

    for p in particles
        p.x *= s
        p.y *= s
        p.z *= s
    end

    current_phi = calculate_current_packing_fraction(particles, box, sigma_A, size_ratio)
    if current_phi >= 1.0
        for p in particles
            p.x /= s
            p.y /= s
            p.z /= s
        end
        # Restore
        box.h .= old_box.h
        box.h_inv .= old_box.h_inv
        return 0
    end

    # New lists for the new geometry
    new_images = compute_image_lists(box, particles)

    # Overlap check
    overlaps, _ = check_overlaps_with_images(particles, box, new_images)
    if overlaps
        # Revert
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
        return 0
    else
        # Metropolis (Eq. (1) in the paper)
        ΔV = V_new - V_old
        ln_acc = -(pressure * ΔV - (N + 1.0) * δlnV)
        if ln_acc >= 0 || rand() < exp(ln_acc)
            images.self = deepcopy(new_images.self)
            images.pair = deepcopy(new_images.pair)
            images.N = new_images.N
            return 1
        else
            # Reject and restore
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
            return 0
        end
    end
end

end # module fbmc

# === Utilities outside the module ===========================================

using .fbmc
using LinearAlgebra
using ArgParse

function parse_commandline()
    s = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table s begin
        "--num-particles", "-n"
        help = "Total number of particles"
        arg_type = Int
        default = 1
        "--size-ratio", "-q"
        help = "Size ratio defined as sigma_B / sigma_A, where sigma_B < sigma_A."
        arg_type = Float64
        default = 1.0
        "--composition", "-x"
        help = "The composition or stoichiometry defined as x = N_B / N, where N_B is the type B of particle with diameter sigma_B. In this case, N is the total number of particles. If 0.0 (the default) the simulation will do a monodisperse hard-sphere simulation."
        arg_type = Float64
        default = 0.0
        "--sigma-A", "-s"
        help = "The diameter of type A. Along with the size ratio this will define the diameter of type B particles, only if the composition is larger than zero. By default this is always one, and considered the unit of length in the simulation."
        arg_type = Float64
        default = 1.0
        "--initial-packing-fraction", "-e"
        help = "The packing fraction of the system, ranging from 0 to 1. The code already computes the correct packing fraction for mixtures. This will be used to initialize the system."
        arg_type = Float64
        default = 0.05
        "--initial-pressure"
        help = "The value of the initial pressure. Must be something reasonable related to the initial packing fraction, otherwise the simulation will get stuck."
        arg_type = Float64
        default = 0.1
        "--final-pressure"
        help = "The value of the final pressure. This is the target pressure to obtain and this will determine when the simulation will stop."
        arg_type = Float64
        default = 10_000.0
        "--init-equilibration-cycles"
        help = "The number of Monte Carlo cycles to equilibrate the initial configuration. This is to remove all correlations with the initial placement of the particles."
        arg_type = Int
        default = 100_000
        "--equilibration-cycles"
        help = "The number of Monte Carlo cycles to reach the final pressure."
        arg_type = Int
        default = 1_000_000
        "--final-equilibration-cycles"
        help = "The number of Monte Carlo cycles to equilibrate the configuration at the end of the simulation."
        arg_type = Int
        default = 100_000
        "--move-acceptance", "-d"
        help = "The acceptance rate of translational moves. It is important to note that due to how the simulation prints the values taking into account the global accumulants, the resulting acceptance rate might be higher."
        arg_type = Float64
        default = 0.2
        "--volume-acceptance", "-v"
        help = "The acceptance rate of volume (scaling) moves. It is important to note that due to how the simulation prints the values taking into account the global accumulants, the resulting acceptance rate might be higher."
        arg_type = Float64
        default = 0.15
        "--deform-acceptance", "-f"
        help = "The acceptance rate of deformation (shearing) moves. It is important to note that due to how the simulation prints the values taking into account the global accumulants, the resulting acceptance rate might be higher."
        arg_type = Float64
        default = 0.15
    end

    return ArgParse.parse_args(s)
end

function main()
    # Parse the necessary arguments from the CLI
    parsed_args = parse_commandline()
    # Physical parameters (keeping as requested)
    num_particles = parsed_args["num-particles"]
    size_ratio = parsed_args["size-ratio"]
    frac_B = parsed_args["composition"]
    sigma_A = parsed_args["sigma-A"]
    sigma_B = size_ratio * sigma_A
    packing_fraction = parsed_args["sigma-A"]

    # Pressure ramp parameters
    initial_pressure = parsed_args["initial-pressure"]
    final_pressure = parsed_args["final-pressure"]

    # Simulation parameters
    equilibration_cycles = parsed_args["init-equilibration-cycles"]
    ramp_cycles = parsed_args["equilibration-cycles"]
    final_cycles = parsed_args["final-equilibration-cycles"]

    # Move frequency chooses randomly which move will be sampled
    volume_move_freq = 0.15
    deform_move_freq = 0.15

    # Lattice reduction parameters
    auto_reduce = true
    reduction_threshold = 1.5  # Trigger reduction when distortion > 1.5

    # Enhanced adaptive parameters with bounds
    initial_max_displacement = 1.0
    initial_max_ln_vol_change = 1.0
    initial_max_deformation = 1.0
    adaptation_interval = 100  # Adapt every 50 cycles for stability
    printing_interval = 10000   # Print progress every 1000 cycles
    target_translation = parsed_args["move-acceptance"]
    target_volume = parsed_args["volume-acceptance"]
    target_deformation = parsed_args["deform-acceptance"]

    # Minimum bounds to prevent freeze
    min_displacement = 1e-4
    min_ln_vol_change = 1e-4
    min_deformation = 1e-4
    adaptation_factor = 1.05

    params = fbmc.EnhancedAdaptiveParameters(
        initial_max_displacement,
        initial_max_ln_vol_change,
        initial_max_deformation,
        target_translation,
        target_volume,
        target_deformation,
        adaptation_interval,
        adaptation_factor,
        min_displacement,
        min_ln_vol_change,
        min_deformation,
    )

    # Initialize the configuration
    particles, box = fbmc.initialize_cubic_particles(
        num_particles, frac_B, size_ratio, sigma_A, sigma_B, packing_fraction
    )

    # Initialize image list for overlap checking
    images = fbmc.compute_image_lists(box, particles)

    println("Initialized $(length(particles)) particles")
    println("Pressure ramp: $initial_pressure → $final_pressure")
    println("Equilibration cycles: $equilibration_cycles")
    println("Ramp cycles: $ramp_cycles")
    println("Final cycles: $final_cycles")
    println("Total cycles: $(equilibration_cycles + ramp_cycles + final_cycles)")
    println("Lattice reduction: $auto_reduce (threshold: $reduction_threshold)")
    println(
        "Initial distortion metric: $(round(fbmc.compute_distortion_metric(box), digits=3))"
    )
    println("Adaptation interval: $adaptation_interval cycles")
    println("Printing interval: $printing_interval cycles")
    println(
        "Parameter bounds: disp ≥ $min_displacement, vol ≥ $min_ln_vol_change, deform ≥ $min_deformation",
    )

    # Write initial configuration
    fbmc.write_xyz(particles, box, "initial.xyz", images)
    println("\nInitial configuration written to initial.xyz")

    # Monte Carlo simulation tracking
    total_accepted_translations = 0
    total_translation_attempts = 0
    total_accepted_volumes = 0
    total_volume_attempts = 0
    total_accepted_deformations = 0
    total_deformation_attempts = 0
    total_reductions = 0

    # Calculate geometric ramp factor
    # P(cycle) = P_initial * (P_final/P_initial)^(cycle/ramp_cycles)
    ramp_factor = (final_pressure / initial_pressure)^(1.0 / ramp_cycles)

    println(
        "\nStarting Enhanced Floppy Box Monte Carlo with Pressure Ramp and Lattice Reduction...",
    )
    println("Geometric ramp factor: $(round(ramp_factor, digits=6)) per cycle")

    # PHASE 1: EQUILIBRATION
    current_pressure = initial_pressure
    println("\n=== PHASE 1: EQUILIBRATION ===")
    println(
        "Equilibrating at pressure = $current_pressure for $equilibration_cycles cycles"
    )

    for cycle in 1:equilibration_cycles
        # Perform enhanced MC cycle with lattice reduction
        accepted_trans, total_trans, accepted_vol, total_vol, accepted_deform, total_deform, reductions = fbmc.mc_cycle!(
            particles,
            box,
            current_pressure,
            params;
            move_frequencies=(volume_move_freq, deform_move_freq),
            images=images,
            auto_reduce=auto_reduce,
            reduction_threshold=reduction_threshold,
            size_ratio=size_ratio,
        )

        # Accumulate statistics
        total_accepted_translations += accepted_trans
        total_translation_attempts += total_trans
        total_accepted_volumes += accepted_vol
        total_volume_attempts += total_vol
        total_accepted_deformations += accepted_deform
        total_deformation_attempts += total_deform
        total_reductions += reductions

        # Adapt parameters (with bounds checking)
        if cycle % adaptation_interval == 0
            fbmc.adapt_enhanced_parameters!(params, box)
        end

        # Print progress
        if cycle % printing_interval == 0
            current_volume = det(box.h)
            current_packing = fbmc.calculate_current_packing_fraction(
                particles, box, sigma_A, size_ratio
            )

            # Compute box heights
            heights = fbmc.compute_box_heights(box)
            max_particle_diameter = maximum(p.sigma for p in particles)

            trans_acc = if total_translation_attempts > 0
                total_accepted_translations / total_translation_attempts
            else
                0.0
            end
            vol_acc = if total_volume_attempts > 0
                total_accepted_volumes / total_volume_attempts
            else
                0.0
            end
            deform_acc = if total_deformation_attempts > 0
                total_accepted_deformations / total_deformation_attempts
            else
                0.0
            end

            println(
                "Equil. Cycle $cycle: P = $(round(current_pressure, digits=3)), V = $(round(current_volume, digits=3)), η = $(round(current_packing, digits=8)), " *
                "T_acc = $(round(trans_acc, digits=3)), V_acc = $(round(vol_acc, digits=3)), " *
                "D_acc = $(round(deform_acc, digits=3)), ",
            )
        end
    end

    # Reset counters
    total_accepted_translations = 0
    total_translation_attempts = 0
    total_accepted_volumes = 0
    total_volume_attempts = 0
    total_accepted_deformations = 0
    total_deformation_attempts = 0
    total_reductions = 0

    # PHASE 2: PRESSURE RAMP
    println("\n=== PHASE 2: PRESSURE RAMP ===")
    println(
        "Ramping pressure from $initial_pressure to $final_pressure over $ramp_cycles cycles",
    )

    for cycle in 1:ramp_cycles
        # Update pressure geometrically
        current_pressure = initial_pressure * (ramp_factor^cycle)

        # Perform enhanced MC cycle with lattice reduction
        accepted_trans, total_trans, accepted_vol, total_vol, accepted_deform, total_deform, reductions = fbmc.mc_cycle!(
            particles,
            box,
            current_pressure,
            params;
            move_frequencies=(volume_move_freq, deform_move_freq),
            images=images,
            auto_reduce=auto_reduce,
            reduction_threshold=reduction_threshold,
            size_ratio=size_ratio,
        )

        # Accumulate statistics
        total_accepted_translations += accepted_trans
        total_translation_attempts += total_trans
        total_accepted_volumes += accepted_vol
        total_volume_attempts += total_vol
        total_accepted_deformations += accepted_deform
        total_deformation_attempts += total_deform
        total_reductions += reductions

        # Adapt parameters (with bounds checking)
        if cycle % adaptation_interval == 0
            fbmc.adapt_enhanced_parameters!(params, box)
        end

        # Print progress
        if cycle % printing_interval == 0
            current_volume = det(box.h)
            current_packing = fbmc.calculate_current_packing_fraction(
                particles, box, sigma_A, size_ratio
            )

            # Compute box heights
            heights = fbmc.compute_box_heights(box)
            max_particle_diameter = maximum(p.sigma for p in particles)

            trans_acc = if total_translation_attempts > 0
                total_accepted_translations / total_translation_attempts
            else
                0.0
            end
            vol_acc = if total_volume_attempts > 0
                total_accepted_volumes / total_volume_attempts
            else
                0.0
            end
            deform_acc = if total_deformation_attempts > 0
                total_accepted_deformations / total_deformation_attempts
            else
                0.0
            end

            println(
                "Equil. Cycle $cycle: P = $(round(current_pressure, digits=3)), V = $(round(current_volume, digits=3)), η = $(round(current_packing, digits=8)), " *
                "T_acc = $(round(trans_acc, digits=3)), V_acc = $(round(vol_acc, digits=3)), " *
                "D_acc = $(round(deform_acc, digits=3)), ",
            )
        end
    end

    # Reset counters
    total_accepted_translations = 0
    total_translation_attempts = 0
    total_accepted_volumes = 0
    total_volume_attempts = 0
    total_accepted_deformations = 0
    total_deformation_attempts = 0
    total_reductions = 0

    # PHASE 3: FINAL EQUILIBRATION
    current_pressure = final_pressure
    println("\n=== PHASE 3: FINAL EQUILIBRATION ===")
    println("Final equilibration at pressure = $current_pressure for $final_cycles cycles")

    for cycle in 1:final_cycles
        # Perform enhanced MC cycle with lattice reduction
        accepted_trans, total_trans, accepted_vol, total_vol, accepted_deform, total_deform, reductions = fbmc.mc_cycle!(
            particles,
            box,
            current_pressure,
            params;
            move_frequencies=(volume_move_freq, deform_move_freq),
            images=images,
            auto_reduce=auto_reduce,
            reduction_threshold=reduction_threshold,
            size_ratio=size_ratio,
        )

        # Accumulate statistics
        total_accepted_translations += accepted_trans
        total_translation_attempts += total_trans
        total_accepted_volumes += accepted_vol
        total_volume_attempts += total_vol
        total_accepted_deformations += accepted_deform
        total_deformation_attempts += total_deform
        total_reductions += reductions

        # Adapt parameters (with bounds checking)
        if cycle % adaptation_interval == 0
            fbmc.adapt_enhanced_parameters!(params, box)
        end

        # Print progress
        if cycle % printing_interval == 0
            current_volume = det(box.h)
            current_packing = fbmc.calculate_current_packing_fraction(
                particles, box, sigma_A, size_ratio
            )

            # Compute box heights
            heights = fbmc.compute_box_heights(box)
            max_particle_diameter = maximum(p.sigma for p in particles)

            trans_acc = if total_translation_attempts > 0
                total_accepted_translations / total_translation_attempts
            else
                0.0
            end
            vol_acc = if total_volume_attempts > 0
                total_accepted_volumes / total_volume_attempts
            else
                0.0
            end
            deform_acc = if total_deformation_attempts > 0
                total_accepted_deformations / total_deformation_attempts
            else
                0.0
            end
        end
    end

    # Final statistics and output
    total_cycles = equilibration_cycles + ramp_cycles + final_cycles
    translation_acceptance = if total_translation_attempts > 0
        total_accepted_translations / total_translation_attempts
    else
        0.0
    end
    volume_acceptance =
        total_volume_attempts > 0 ? total_accepted_volumes / total_volume_attempts : 0.0
    deformation_acceptance = if total_deformation_attempts > 0
        total_accepted_deformations / total_deformation_attempts
    else
        0.0
    end

    println("\n=== FINAL STATISTICS ===")
    println(
        "Total cycles: $total_cycles (Equil: $equilibration_cycles, Ramp: $ramp_cycles, Final: $final_cycles)",
    )
    println("Pressure ramp: $initial_pressure → $final_pressure")
    println(
        "Translation acceptance: $(round(translation_acceptance, digits=4)) (target: $(params.target_trans_acceptance))",
    )
    println(
        "Volume acceptance: $(round(volume_acceptance, digits=4)) (target: $(params.target_vol_acceptance))",
    )
    println(
        "Deformation acceptance: $(round(deformation_acceptance, digits=4)) (target: $(params.target_deform_acceptance))",
    )
    println("Total lattice reductions: $total_reductions")
    println("Total adaptations performed: $(div(total_cycles, adaptation_interval))")

    # Final box analysis
    final_orthogonality = fbmc.compute_box_orthogonality(box)

    println("\nFinal Box Analysis:")
    println("Box matrix:")
    for i in 1:3
        println(
            "  $(round(box.h[i,1], digits=4))  $(round(box.h[i,2], digits=4))  $(round(box.h[i,3], digits=4))",
        )
    end
    println(
        "Angle deviations from 90°: $(round(final_orthogonality.angle_deviation, digits=2))°",
    )
    println("Aspect ratio: $(round(final_orthogonality.aspect_ratio, digits=3))")

    # Attempt a final lattice reduction
    println("Attempting a final lattice reduction...")
    (reduced, distortion) = fbmc.lattice_reduce!(
        box, particles, images; threshold=1.05, max_iter=100
    )

    if reduced
        println("Lattice reduced! Final distortion: $distortion")
    else
        println("Lattice could not be reduced.")
    end

    println("\nBox Analysis after lattice reduction attempt:")
    println("Box matrix:")
    for i in 1:3
        println(
            "  $(round(box.h[i,1], digits=4))  $(round(box.h[i,2], digits=4))  $(round(box.h[i,3], digits=4))",
        )
    end
    println(
        "Angle deviations from 90°: $(round(final_orthogonality.angle_deviation, digits=2))°",
    )
    println("Aspect ratio: $(round(final_orthogonality.aspect_ratio, digits=3))")

    # Final adaptive parameters
    println("\nFinal Adaptive Parameters:")
    println(
        "Max displacement: $(round(params.max_displacement, digits=4)) (min bound: $(params.min_displacement))",
    )
    println(
        "Max ln(V) change: $(round(params.max_ln_vol_change, digits=4)) (min bound: $(params.min_ln_vol_change))",
    )
    println(
        "Max deformation: $(round(params.max_deformation, digits=4)) (min bound: $(params.min_deformation))",
    )

    # Final configuration and packing analysis
    final_volume = det(box.h)
    final_packing = fbmc.calculate_current_packing_fraction(
        particles, box, sigma_A, size_ratio
    )

    println("\nFinal State:")
    println("Final pressure: $(round(current_pressure, digits=3))")
    println("Final volume: $(round(final_volume, digits=3))")
    println("Final packing fraction: $(round(final_packing, digits=8))")

    # Write final configuration
    images = fbmc.compute_image_lists(box, particles)
    fbmc.check_configuration_validity(particles, box, images)
    fbmc.write_xyz(particles, box, "final.xyz", images)

    println("\nFinal configuration written to final.xyz")

    return nothing
end

main()
