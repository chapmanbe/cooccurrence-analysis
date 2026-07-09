#!/usr/bin/env julia
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#   kernelspec:
#     display_name: Julia
#     language: julia
#     name: julia
# ---

# %% [markdown]
# # Item Co-occurrence Network Analysis
#
# Builds weighted co-occurrence networks from the tabular synthetic dataset to
# reveal *global structure* in item associations. Complements the association
# rule mining analysis by identifying communities of items that cluster
# together and quantifying each item's role in the network.
#
# **Approach**: Pairwise association statistics (lift, phi coefficient) define
# edge weights. Edges are filtered by minimum co-occurrence count, statistical
# significance (Fisher's exact test + BH correction), and positive association
# (lift > 1). Community detection uses the Louvain algorithm.

# %%
using Pkg
Pkg.activate(@__DIR__)

# %%
using Revise

# %%
using CooccurrenceAnalysis
using DataFrames

# %% [markdown]
# ## Load data

# %%
data_path = joinpath(@__DIR__, "..", "ddata", "synthetic_events.arrow")
event_df = load_event_data(data_path)
println("Total event rows: $(nrow(event_df))")
println("Total records: $(length(unique(event_df.id)))")

ps = get_record_summary(event_df)
n_multi = count(ps.n_items .>= 2)
println("Multi-item records: $n_multi ($(round(n_multi/nrow(ps)*100, digits=1))%)")

# %% [markdown]
# ## Full population co-occurrence network
#
# Build a network where nodes are items and edges represent statistically
# significant co-occurrence associations. Edge weights are **lift** values
# (observed / expected co-occurrence), which normalize for prevalence differences.

# %%
# Compute all pairwise associations
println("Computing pairwise associations...")
pairwise = compute_pairwise_associations(event_df)
println("Total pairs with co-occurrence: $(nrow(pairwise))")
println("Pairs with lift > 1: $(count(pairwise.lift .> 1))")

# Show top pairs by lift
top_pairs = sort(pairwise, :lift, rev=true)
println("\nTop 20 pairwise associations by lift:")
for (i, row) in enumerate(first(eachrow(top_pairs), 20))
    sig = row.p_adjusted < 0.05 ? significance_stars(row.p_adjusted) : "ns"
    println("  $i. $(row.item_a) + $(row.item_b) | lift=$(row.lift) phi=$(row.phi) N=$(row.observed) p_adj=$(round(row.p_adjusted, sigdigits=3)) $sig")
end

# %%
# Build filtered network (min 30 co-occurrences, p_adj < 0.05, lift > 1)
net = build_cooccurrence_network(event_df;
    weight_metric=:lift, min_count=30, alpha=0.05)
println("Network: $(length(net.items)) nodes, edges from edge_data")

# %%
# Detect communities
comm = detect_communities(net; method=:louvain)
network_summary(net, comm)

# %%
# Per-node metrics
metrics = compute_network_metrics(net)
println("\nNetwork metrics (sorted by strength):")
sorted_metrics = sort(metrics, :strength, rev=true)
for row in eachrow(sorted_metrics)
    println("  $(row.item) | deg=$(row.degree) str=$(row.strength) " *
            "btw=$(row.betweenness) cc=$(row.clustering_coeff) prev=$(row.prevalence)")
end

# %% [markdown]
# ## Network visualization
#
# Node size reflects prevalence (number of records with that item).
# Node color indicates community membership. Edge width is proportional to lift.

# %%
fig_net = plot_cooccurrence_network(net;
    communities=comm,
    title="Item Co-occurrence Network (Full Population)")
save(joinpath(@__DIR__, "..", "ddata", "network_full.png"), fig_net; px_per_unit=2)
fig_net

# %%
# Community heatmap — block-diagonal structure reveals clusters
fig_heat = plot_community_heatmap(net, comm)
save(joinpath(@__DIR__, "..", "ddata", "network_heatmap.png"), fig_heat; px_per_unit=2)
fig_heat

# %% [markdown]
# ## Group-stratified network analysis
#
# Build separate networks for group_a and group_b cohorts to identify group-specific
# vs universal network structure. Compare edge sets and community similarity
# using the Adjusted Rand Index (ARI).

# %%
strat = stratified_network_analysis(event_df;
    min_count=20, alpha=0.05)

# %%
# Group A network summary
println("\n=== Group A Network ===")
if length(strat.group_a_net.items) > 0
    network_summary(strat.group_a_net, strat.group_a_communities)

    println("\nGroup A metrics:")
    group_a_metrics = sort(strat.group_a_metrics, :strength, rev=true)
    for row in eachrow(group_a_metrics)
        println("  $(row.item) | deg=$(row.degree) str=$(row.strength) prev=$(row.prevalence)")
    end
else
    println("  No significant edges in group_a network at current thresholds.")
end

# %%
# Group B network summary
println("\n=== Group B Network ===")
if length(strat.group_b_net.items) > 0
    network_summary(strat.group_b_net, strat.group_b_communities)

    println("\nGroup B metrics:")
    group_b_metrics = sort(strat.group_b_metrics, :strength, rev=true)
    for row in eachrow(group_b_metrics)
        println("  $(row.item) | deg=$(row.degree) str=$(row.strength) prev=$(row.prevalence)")
    end
else
    println("  No significant edges in group_b network at current thresholds.")
end

# %%
# Side-by-side comparison plot
if length(strat.group_a_net.items) > 0 && length(strat.group_b_net.items) > 0
    fig_comp = plot_network_comparison(strat.group_a_net, strat.group_b_net;
        group_a_comm=strat.group_a_communities,
        group_b_comm=strat.group_b_communities)
    save(joinpath(@__DIR__, "..", "ddata", "network_group_comparison.png"), fig_comp; px_per_unit=2)
    fig_comp
end

# %%
# Quantitative comparison
if length(strat.group_a_net.items) > 0 && length(strat.group_b_net.items) > 0
    comp = compare_networks(strat.group_a_net, strat.group_b_net,
                            strat.group_a_communities, strat.group_b_communities)

    println("=== Network Comparison ===")
    println("  Shared edges: $(nrow(comp.shared_edges))")
    if nrow(comp.shared_edges) > 0
        println("  Shared edge details:")
        for row in eachrow(comp.shared_edges)
            println("    $(row.item_a) + $(row.item_b) | group_a_w=$(round(row.group_a_weight, digits=2)) group_b_w=$(round(row.group_b_weight, digits=2))")
        end
    end

    println("\n  Group A-only edges: $(nrow(comp.group_a_only_edges))")
    for row in eachrow(comp.group_a_only_edges)
        println("    $(row.item_a) + $(row.item_b) | w=$(round(row.group_a_weight, digits=2))")
    end

    println("\n  Group B-only edges: $(nrow(comp.group_b_only_edges))")
    for row in eachrow(comp.group_b_only_edges)
        println("    $(row.item_a) + $(row.item_b) | w=$(round(row.group_b_weight, digits=2))")
    end

    println("\n  Community similarity (ARI): $(round(comp.community_ari, digits=4))")
    println("    ARI=1: identical community structure on shared items")
    println("    ARI≈0: random-level agreement")
    println("    ARI<0: less agreement than random")
end

# %% [markdown]
# ## Comparison with association rule mining
#
# The network communities should broadly align with the top ARM associations.
# Network analysis adds:
# - **Global structure**: Which items form interconnected clusters (not just pairwise rules)
# - **Hub identification**: Centrality metrics reveal which items bridge multiple clusters
# - **Community-level patterns**: The heatmap shows block structure that ARM misses
# - **Group comparison**: ARI provides a single metric for structural similarity

# %%
# Run the complete pipeline for reference
println("=== Full Network Pipeline ===\n")
result = run_network_pipeline(data_path;
    min_count=30, stratify_by_group=true)
