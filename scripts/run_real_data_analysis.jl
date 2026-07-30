# Full analysis pipeline on real tabular data (odata/real_data.arrow).
#
# Runs:
#   1. Dataset overview statistics
#   2. HDP Bernoulli mixture (group-stratified, K_max=20, empirical-Bayes prior)
#   3. Co-occurrence network analysis with community detection
#   4. All visualizations: prevalence, phi heatmaps, butterfly, HDP profiles,
#      network graph, community heatmap
#
# Usage:
#   julia --project=clustering clustering/scripts/run_real_data_analysis.jl
#
# Output figures → clustering/scripts/output/real/

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis
using DataFrames, Random, Printf, CairoMakie, Graphs

const DATA_PATH  = joinpath(@__DIR__, "..", "..", "odata", "real_data.arrow")
const OUTPUT_DIR = joinpath(@__DIR__, "output", "real")

# ── Dataset summary ────────────────────────────────────────────────────────────

function dataset_summary(event_df::DataFrame)
    n_total = length(unique(event_df.id))
    @printf "  Total records:    %d\n" n_total
    @printf "  Total event rows:  %d\n" nrow(event_df)

    for group in ["A", "B"]
        ids = unique(filter(r -> r.Group == group, event_df).id)
        @printf "  %s records:     %d\n" group length(ids)
    end

    txns = build_transactions(event_df; min_items=2)
    n_multi  = nrow(txns)
    n_items  = ncol(txns)
    mat      = Matrix{Bool}(txns)
    density  = sum(mat) / (n_multi * n_items)
    mean_s   = sum(mat) / n_multi
    @printf "  Multi-item records (≥2): %d (%.1f%%)\n" n_multi (n_multi/n_total*100)
    @printf "  Items in matrix:     %d\n" n_items
    @printf "  Mean items per record:     %.2f\n" mean_s
    @printf "  Matrix density:             %.2f%%\n" (density * 100)
end

# ── Main ───────────────────────────────────────────────────────────────────────

function main()
    mkpath(OUTPUT_DIR)

    println("═══ Loading real tabular data ═══")
    println("  Path: $DATA_PATH")
    event_df = load_event_data(DATA_PATH)
    dataset_summary(event_df)

    # ── Overview visualizations (no model required) ────────────────────────

    println("\n═══ Overview visualizations ═══")

    fig_prev = plot_item_prevalence(event_df; top_n=41)
    save(joinpath(OUTPUT_DIR, "item_prevalence.png"), fig_prev; px_per_unit=2)
    println("  item_prevalence.png")

    fig_phi_all = plot_cooccurrence_heatmap(event_df; group=:both, min_items=2)
    save(joinpath(OUTPUT_DIR, "phi_heatmap_all.png"), fig_phi_all; px_per_unit=2)
    println("  phi_heatmap_all.png")

    fig_phi_f = plot_cooccurrence_heatmap(event_df; group=:group_b, min_items=2)
    save(joinpath(OUTPUT_DIR, "phi_heatmap_group_b.png"), fig_phi_f; px_per_unit=2)
    println("  phi_heatmap_group_b.png")

    fig_phi_m = plot_cooccurrence_heatmap(event_df; group=:group_a, min_items=2)
    save(joinpath(OUTPUT_DIR, "phi_heatmap_group_a.png"), fig_phi_m; px_per_unit=2)
    println("  phi_heatmap_group_a.png")

    # ── Network analysis ──────────────────────────────────────────────────

    println("\n═══ Co-occurrence network ═══")
    t_net = time()
    net = build_cooccurrence_network(event_df;
        weight_metric = :phi,
        min_count     = 30,
        alpha         = 0.05,
        test          = :fisher,
        correction    = :bh)
    comm = detect_communities(net; method=:louvain)
    metrics = compute_network_metrics(net)
    network_summary(net, comm)
    @printf "  Network wall time: %.1f s\n" (time() - t_net)

    fig_net = plot_cooccurrence_network(net; communities=comm)
    save(joinpath(OUTPUT_DIR, "network.png"), fig_net; px_per_unit=2)
    println("  network.png")

    fig_comm = plot_community_heatmap(net, comm)
    save(joinpath(OUTPUT_DIR, "community_heatmap.png"), fig_comm; px_per_unit=2)
    println("  community_heatmap.png")

    fig_cent = plot_centrality_barchart(metrics; centrality=:betweenness, top_n=20)
    save(joinpath(OUTPUT_DIR, "centrality.png"), fig_cent; px_per_unit=2)
    println("  centrality.png")

    # ── Group-stratified networks ───────────────────────────────────────────

    println("\n═══ Group-stratified networks ═══")
    strat_net = stratified_network_analysis(event_df;
        weight_metric    = :phi,
        min_count        = 30,
        alpha            = 0.05,
        test             = :fisher,
        correction       = :bh,
        community_method = :louvain)
    # stratified_network_analysis keys its results by group value (an OrderedDict
    # in sorted group order), not by positional field name.
    labels = collect(keys(strat_net))
    for g in labels
        @printf "  %s network: %d nodes, %d edges\n" g nv(strat_net[g].net.graph) ne(strat_net[g].net.graph)
    end

    ga, gb = labels[1], labels[2]
    fig_groupnet = plot_network_comparison(strat_net[ga].net, strat_net[gb].net;
                                         group_a_comm=strat_net[ga].communities,
                                         group_b_comm=strat_net[gb].communities,
                                         labels=(ga, gb))
    save(joinpath(OUTPUT_DIR, "network_group_stratified.png"), fig_groupnet; px_per_unit=2)
    println("  network_group_stratified.png")

    # ── HDP Bernoulli mixture ─────────────────────────────────────────────

    println("\n═══ HDP Bernoulli mixture (K_max=20, empirical Bayes) ═══")
    t_hdp = time()
    hdp_result = hdp_clustering(event_df;
        group_by      = :Group,
        K_max         = 20,
        prior         = :empirical_bayes,
        concentration = 10.0,
        floor         = 1.0,
        alpha         = 1.0,
        gamma         = 1.0,
        n_init        = 3,
        max_iter      = 300,
        tol           = 1e-5,
        rng           = MersenneTwister(2026))
    @printf "\n  Wall time: %.1f seconds\n" (time() - t_hdp)

    println()
    clustering_summary(hdp_result)

    println("\n═══ Cross-strata categorization ═══")
    cat_df = hdp_cluster_categorization(hdp_result; beta_threshold=0.01)
    active = filter(r -> r.category != :negligible, cat_df)
    println(active)

    hdp = hdp_result.hdp_result
    beta_sum = sum(hdp.beta_mean)
    @printf "\n  β weights sum: %.4f  (should ≈ 1.0)\n" beta_sum
    for j in 1:hdp.n_groups
        @printf "  π[%s] sum: %.4f\n" hdp.group_labels[j] sum(hdp.pi_mean[j, :])
    end
    println("  Effective K:   $(hdp.effective_K)")
    println("  Converged:     $(hdp.converged)  ($(hdp.n_iter) iters)")
    println("  ELBO:          $(round(hdp.elbo, digits=2))")

    # ── HDP figures ───────────────────────────────────────────────────────

    println("\n═══ Saving HDP figures ═══")

    fig_sticks = plot_hdp_stick_weights(hdp_result)
    save(joinpath(OUTPUT_DIR, "hdp_sticks.png"), fig_sticks; px_per_unit=2)
    println("  hdp_sticks.png")

    fig_profiles = plot_hdp_class_profiles(hdp_result)
    save(joinpath(OUTPUT_DIR, "hdp_profiles.png"), fig_profiles; px_per_unit=2)
    println("  hdp_profiles.png")

    fig_sharing = plot_hdp_sharing_heatmap(hdp_result)
    save(joinpath(OUTPUT_DIR, "hdp_sharing.png"), fig_sharing; px_per_unit=2)
    println("  hdp_sharing.png")

    fig_butterfly = plot_hdp_cluster_butterfly(hdp_result; beta_threshold=0.01)
    save(joinpath(OUTPUT_DIR, "hdp_butterfly.png"), fig_butterfly; px_per_unit=2)
    println("  hdp_butterfly.png")

    println("\nAll figures written to $(relpath(OUTPUT_DIR))")
    println("Done.")
end

main()
