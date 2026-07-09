# HDP Bernoulli mixture validation on the full synthetic cohort.
#
# Milestone 5: confirms the HDP pipeline runs to completion within a
# practical wall time on the full multi-item cohort (~N=5-20K records,
# D=41 items, J=2 group groups).
#
# Usage:
#   julia --project=clustering clustering/scripts/run_hdp_demo.jl
#
# Outputs (saved to clustering/scripts/output/):
#   hdp_sticks.png   — global β + per-group π bar chart
#   hdp_profiles.png — cluster × item heatmap
#   hdp_sharing.png  — cluster × group sharing matrix

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis
using DataFrames, Random, Printf, CairoMakie

const DATA_PATH  = joinpath(@__DIR__, "..", "..", "ddata", "synthetic_events.arrow")
const OUTPUT_DIR = joinpath(@__DIR__, "output")

# ── Dataset summary ────────────────────────────────────────────────────────────

function dataset_summary(event_df::DataFrame)
    n_total = length(unique(event_df.id))
    println("  Total records:    $n_total")
    println("  Total event rows:  $(nrow(event_df))")

    for group in ["A", "B"]
        ids = unique(filter(r -> r.Group == group, event_df).id)
        println("  $group records:    $(length(ids))")
    end

    txns = build_transactions(event_df; min_items=2)
    n_multi = nrow(txns)
    n_items = ncol(txns)
    density = sum(Matrix{Bool}(txns)) / (n_multi * n_items)
    mean_items = sum(Matrix{Bool}(txns)) / n_multi
    println("  Multi-item records (≥2): $n_multi ($(round(n_multi/n_total*100, digits=1))%)")
    println("  Items in matrix:     $n_items")
    @printf "  Mean items per record:     %.2f\n" mean_items
    @printf "  Matrix density:             %.2f%%\n" density * 100
end

# ── Main ───────────────────────────────────────────────────────────────────────

function main()
    mkpath(OUTPUT_DIR)

    println("═══ Loading data ═══")
    println("  Path: $DATA_PATH")
    event_df = load_event_data(DATA_PATH)
    dataset_summary(event_df)

    println("\n═══ HDP Bernoulli mixture (group-stratified, K_max=20) ═══")
    println("  Prior: empirical Bayes (concentration=10, floor=1)")
    println("  Restarts: 3  |  max_iter: 300  |  tol: 1e-5")

    t_start = time()
    result = hdp_clustering(event_df;
        group_by       = :Group,
        K_max          = 20,
        prior          = :empirical_bayes,
        concentration  = 10.0,
        floor          = 1.0,
        alpha          = 1.0,
        gamma          = 1.0,
        n_init         = 3,
        max_iter       = 300,
        tol            = 1e-5,
        rng            = MersenneTwister(2026))
    elapsed = time() - t_start

    @printf "\n  Wall time: %.1f seconds\n" elapsed
    elapsed > 300 && @warn "Exceeded 5-minute target ($(round(elapsed, digits=0))s)"

    # ── Textual summary ────────────────────────────────────────────────────
    println()
    clustering_summary(result)

    # ── Categorization table ───────────────────────────────────────────────
    println("\n═══ Cross-strata categorization ═══")
    cat_df = hdp_cluster_categorization(result; beta_threshold=0.01)
    active = filter(r -> r.category != :negligible, cat_df)
    println(active)

    # ── Figures ────────────────────────────────────────────────────────────
    println("\n═══ Saving figures to $OUTPUT_DIR ═══")

    fig1 = plot_hdp_stick_weights(result)
    save(joinpath(OUTPUT_DIR, "hdp_sticks.png"), fig1)
    println("  hdp_sticks.png")

    fig2 = plot_hdp_class_profiles(result)
    save(joinpath(OUTPUT_DIR, "hdp_profiles.png"), fig2)
    println("  hdp_profiles.png")

    fig3 = plot_hdp_sharing_heatmap(result)
    save(joinpath(OUTPUT_DIR, "hdp_sharing.png"), fig3)
    println("  hdp_sharing.png")

    # ── Sanity checks ──────────────────────────────────────────────────────
    println("\n═══ Sanity checks ═══")
    hdp = result.hdp_result

    beta_sum = sum(hdp.beta_mean)
    @printf "  β weights sum:    %.4f  (should be ≈ 1.0)\n" beta_sum

    for j in 1:hdp.n_groups
        pi_sum = sum(hdp.pi_mean[j, :])
        @printf "  π[%s] sum:  %.4f  (should be ≈ 1.0)\n" hdp.group_labels[j] pi_sum
    end

    n_active  = count(hdp.beta_mean .> 0.01)
    n_univ    = count(cat_df.category .== :universal)
    n_group_a    = count(cat_df.category .== :group_a_only)
    n_group_b  = count(cat_df.category .== :group_b_only)
    n_unequal = count(cat_df.category .== :both_present_but_unequal)
    println("  Active clusters:          $n_active / $(hdp.K_max)")
    println("  Universal:                $n_univ")
    println("  Group A-only:                $n_group_a")
    println("  Group B-only:              $n_group_b")
    println("  Both (unequal weights):   $n_unequal")
    println("  Effective K:              $(hdp.effective_K)")
    println("  Converged:                $(hdp.converged)  ($(hdp.n_iter) iters)")
    println("  ELBO:                     $(round(hdp.elbo, digits=2))")
    println("\nDone.")
end

main()
