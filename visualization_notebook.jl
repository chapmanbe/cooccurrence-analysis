# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:percent,ipynb
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # Item Co-occurrence Visualization Gallery
#
# This notebook demonstrates all visualization functions in the
# CooccurrenceAnalysis module, using synthetic event data.

# %%
using Pkg
Pkg.activate(@__DIR__)

# %%
using Revise

# %%
using CooccurrenceAnalysis
using DataFrames
import Arrow
using CairoMakie
using Graphs: nv, ne

# %% [markdown]
# ## Load Data

# %%
data_path = joinpath(@__DIR__, "..", "ddata", "synthetic_events.arrow")
event_df = load_event_data(data_path)
println("Records: $(length(unique(event_df.id))), Event rows: $(nrow(event_df))")

# %%
txns = build_transactions(event_df; min_items=2)
println("Multi-item records: $(nrow(txns)), Items: $(ncol(txns))")

# %% [markdown]
# ## 1. Data Overview
#
# ### UpSet Plot
# Shows which item combinations occur most frequently.

# %%
plot_upset(txns; min_count=5)

# %% [markdown]
# ### Record × Item Heatmap
# Binary matrix showing which items each record has.

# %%
plot_record_heatmap(txns; max_records=200)

# %% [markdown]
# ## 2. Association Rule Mining Visualizations
#
# ### Mine and Validate Rules

# %%
rules = mine_association_rules(txns; min_support=0.005, min_confidence=0.1)
validated_rules = validate_rules(rules, event_df)
println("Rules: $(nrow(validated_rules)), Significant: $(count(validated_rules.significant))")

# %% [markdown]
# ### Rule Quality Scatter Plot
# Support vs Confidence, colored by Lift.

# %%
plot_arm_scatter(validated_rules)

# %% [markdown]
# ### Pairwise Association Matrix
# Item-by-item heatmap of maximum Lift values.

# %%
plot_arm_matrix(validated_rules)

# %% [markdown]
# ### Group-Stratified ARM Comparison

# %%
strat = stratified_analysis(event_df;
    min_support=0.005, min_confidence=0.1, min_count=nothing)
if nrow(strat.group_a_rules) > 0 && nrow(strat.group_b_rules) > 0
    comp = compare_strata(strat.group_a_rules, strat.group_b_rules)
    plot_arm_comparison(comp)
end

# %% [markdown]
# ## 3. Network Visualizations
#
# ### Build Network and Detect Communities

# %%
net = build_cooccurrence_network(event_df; min_count=5, alpha=0.05)
comm = detect_communities(net)
metrics = compute_network_metrics(net)
network_summary(net, comm)

# %% [markdown]
# ### Co-occurrence Network

# %%
plot_cooccurrence_network(net; communities=comm)

# %% [markdown]
# ### Community Heatmap

# %%
plot_community_heatmap(net, comm)

# %% [markdown]
# ### Centrality Bar Chart

# %%
plot_centrality_barchart(metrics)

# %% [markdown]
# ### Group-Stratified Network

# %%
net_strat = stratified_network_analysis(event_df; min_count=5, alpha=0.05)
if nv(net_strat.group_a_net.graph) > 0 && nv(net_strat.group_b_net.graph) > 0
    net_comp = compare_networks(net_strat.group_a_net, net_strat.group_b_net,
                                 net_strat.group_a_communities, net_strat.group_b_communities)
    plot_group_stratified_network(net_comp, net_strat.group_a_net, net_strat.group_b_net)
end

# %% [markdown]
# ### Side-by-Side Network Comparison

# %%
if nv(net_strat.group_a_net.graph) > 0 && nv(net_strat.group_b_net.graph) > 0
    plot_network_comparison(net_strat.group_a_net, net_strat.group_b_net;
                             group_a_comm=net_strat.group_a_communities,
                             group_b_comm=net_strat.group_b_communities)
end

# %% [markdown]
# ### Centrality Comparison

# %%
if nv(net_strat.group_a_net.graph) > 0 && nv(net_strat.group_b_net.graph) > 0
    group_a_metrics = compute_network_metrics(net_strat.group_a_net)
    group_b_metrics = compute_network_metrics(net_strat.group_b_net)
    plot_centrality_comparison(group_a_metrics, group_b_metrics)
end

# %% [markdown]
# ## 4. Bayesian Clustering Visualizations
#
# ### Fit Bernoulli Mixture Model

# %%
clustering = bernoulli_clustering(event_df; K_range=2:6, n_init=5)
clustering_summary(clustering)

# %% [markdown]
# ### BIC Elbow Plot

# %%
plot_bic_elbow(clustering)

# %% [markdown]
# ### Class Probability Heatmap

# %%
plot_class_probabilities(clustering)

# %% [markdown]
# ### Class Profiles (Top Items per Class)

# %%
plot_class_profiles(clustering)

# %% [markdown]
# ### Group-Stratified Clustering Comparison

# %%
clust_strat = stratified_bernoulli_clustering(event_df; K_range=2:6, n_init=5)
plot_clustering_comparison(clust_strat.group_a_result, clust_strat.group_b_result)

# %%
