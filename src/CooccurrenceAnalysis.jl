"""
Association Rule Mining for Item Co-occurrence Analysis

Discovers item combinations that co-occur more often than expected by
chance using FP-Growth frequent itemset mining with statistical validation.
Supports group-stratified analysis for identifying group-specific vs universal
item associations.
"""
module CooccurrenceAnalysis

using DataFrames, StatsBase
using OrderedCollections: OrderedDict
import Arrow
using RuleMiner
using HypothesisTests, MultipleTesting
import DataFrames: combine, groupby
using Printf
using Graphs, SimpleWeightedGraphs
using CairoMakie, GraphMakie, NetworkLayout
using Distributions: Beta, quantile

# Association rule mining exports
export load_event_data, get_record_summary, build_transactions, build_transactions_with_ids,
       filter_event_by_timing,
       mine_frequent_itemsets, mine_association_rules,
       build_contingency_table, odds_ratio, test_association, validate_rules,
       adjust_pvalues,
       stratify_by, stratified_analysis, compare_strata,
       run_full_pipeline, FullPipelineResult,
       format_itemset, significance_stars, results_summary,
       plot_arm_scatter, plot_arm_comparison, plot_arm_matrix

# Network analysis exports
export CooccurrenceNetwork, CommunityResult, NetworkComparisonResult,
       phi_coefficient, compute_pairwise_associations,
       build_cooccurrence_network,
       detect_communities, compute_network_metrics, network_summary,
       stratified_network_analysis, compare_networks,
       plot_cooccurrence_network, plot_network_comparison,
       plot_community_heatmap,
       plot_group_stratified_network, plot_centrality_barchart,
       plot_centrality_comparison,
       run_network_pipeline, NetworkPipelineResult

# Presentation vocabulary exports
export Vocabulary, VOCAB, set_vocabulary!, reset_vocabulary!

# HDP clustering exports
export HDPBernoulliResult, HDPClusteringResult,
       fit_hdp_bernoulli_mixture,
       hdp_clustering,
       hdp_cluster_categorization,
       plot_hdp_stick_weights,
       plot_hdp_class_profiles,
       plot_hdp_sharing_heatmap,
       plot_hdp_cluster_butterfly

# Bayesian clustering exports
export BernoulliMixtureResult, BernoulliMixtureModelSelection,
       CooccurrenceAnalysisResult, ClusteringComparisonResult,
       BernoulliMixturePosterior,
       fit_bernoulli_mixture, select_K,
       empirical_bayes_priors,
       fit_bernoulli_mixture_turing, posterior_summary,
       bernoulli_clustering_turing,
       fit_bernoulli_mixture_advi, bernoulli_clustering_advi,
       transactions_to_matrix,
       bernoulli_clustering,
       stratified_bernoulli_clustering, compare_clusterings,
       clustering_summary,
       plot_class_probabilities, plot_bic_elbow,
       plot_class_profiles, plot_clustering_comparison,
       run_clustering_pipeline, ClusteringPipelineResult,
       plot_upset, plot_record_heatmap,
       plot_item_prevalence, plot_cooccurrence_heatmap

include("vocabulary.jl")
include("utils.jl")
include("plot_utils.jl")
include("data_preparation.jl")
include("mining.jl")
include("statistical_validation.jl")
include("group_stratification.jl")
include("network_construction.jl")
include("network_metrics.jl")
include("network_group_stratification.jl")
include("network_visualization.jl")
include("bayesian_clustering.jl")
include("hdp_clustering.jl")
include("hdp_wrappers.jl")
include("bayesian_wrappers.jl")
include("bayesian_turing.jl")
include("bayesian_visualization.jl")
include("arm_visualization.jl")
include("overview_visualization.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline result types
#
# Concrete structs (rather than a Dict→NamedTuple whose shape varied by keyword)
# so the return type is stable and discoverable. `stratified`/`comparison` are
# `nothing` when `stratify_by_group=false`. `comparison` maps "X_vs_Y" → the
# pairwise comparison for that group pair.
# ──────────────────────────────────────────────────────────────────────────────

struct FullPipelineResult
    rules::DataFrame
    itemsets::DataFrame
    n_records::Int
    n_total_records::Int
    stratified::Union{Nothing, OrderedDict{String, Any}}
    comparison::Union{Nothing, OrderedDict{String, DataFrame}}
end

struct NetworkPipelineResult
    network::CooccurrenceNetwork
    communities::CommunityResult
    metrics::DataFrame
    stratified::Union{Nothing, OrderedDict{String, Any}}
    comparison::Union{Nothing, OrderedDict{String, NetworkComparisonResult}}
end

struct ClusteringPipelineResult
    clustering::CooccurrenceAnalysisResult
    stratified::Union{Nothing, OrderedDict{String, Any}}
    comparison::Union{Nothing, OrderedDict{String, ClusteringComparisonResult}}
end

# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_full_pipeline(data_path::String;
                      min_support::Float64=0.005,
                      min_confidence::Float64=0.1,
                      min_count::Union{Int, Nothing}=30,
                      max_length::Int=4,
                      test::Symbol=:fisher,
                      correction::Symbol=:bh,
                      stratify_by_group::Bool=true) -> FullPipelineResult

Run the complete analysis pipeline:
1. Load data from Arrow file
2. Build transactions (multi-item records)
3. Mine frequent itemsets and association rules
4. Validate with statistical tests + multiple testing correction
5. Optionally stratify by group and compare

# Returns
`FullPipelineResult` with fields:
- `rules`: Validated rules for the full population
- `itemsets`: Frequent itemsets for the full population
- `n_records`: Number of multi-item records
- `n_total_records`: Total records in dataset
- `stratified` (if stratify_by_group): Output from `stratified_analysis`
- `comparison` (if stratify_by_group): Output from `compare_strata`
"""
function run_full_pipeline(data_path::String; verbose::Bool=true, kwargs...)
    verbose && println("Loading data from $data_path...")
    return run_full_pipeline(load_event_data(data_path); verbose, kwargs...)
end

function run_full_pipeline(event_df::DataFrame;
                           min_support::Float64=0.005,
                           min_confidence::Float64=0.1,
                           min_count::Union{Int, Nothing}=30,
                           max_length::Int=4,
                           test::Symbol=:fisher,
                           correction::Symbol=:bh,
                           stratify_by_group::Bool=true,
                           timing_filter::Symbol=:all,
                           concurrent_window::Int=0,
                           exclusive_items::Union{Nothing, AbstractDict}=nothing,
                           verbose::Bool=true)
    vprintln(args...) = verbose && println(args...)
    n_total = length(unique(event_df.id))
    vprintln("  $n_total records, $(nrow(event_df)) event rows")

    # Full population analysis
    vprintln("\n═══ Full Population Analysis ═══")
    txns_df = build_transactions(event_df; min_items=2,
                                  timing_filter, concurrent_window)
    n_multi = nrow(txns_df)
    vprintln("  Multi-item records: $n_multi ($(round(n_multi/n_total*100, digits=1))%)")

    itemsets = mine_frequent_itemsets(txns_df;
        min_support, min_count, max_length)
    rules = mine_association_rules(txns_df;
        min_support, min_confidence, min_count, max_length)
    vprintln("  Frequent itemsets (≥2 items): $(nrow(itemsets))")
    vprintln("  Association rules: $(nrow(rules))")

    if nrow(rules) > 0
        rules = validate_rules(rules, event_df; test, correction)
        sig = count(rules.significant)
        vprintln("  Significant rules: $sig / $(nrow(rules))")
    end

    stratified = nothing
    comparison = nothing
    if stratify_by_group
        vprintln("\n═══ Group-Stratified Analysis ═══")
        stratified = stratified_analysis(event_df;
            min_support, min_confidence, min_count, max_length, test, correction,
            timing_filter, concurrent_window, exclusive_items, verbose)

        # Pairwise cross-strata comparison over every group pair that has rules.
        labels = collect(keys(stratified))
        comparisons = OrderedDict{String, DataFrame}()
        for i in 1:length(labels), j in (i+1):length(labels)
            lx, ly = labels[i], labels[j]
            (nrow(stratified[lx].rules) > 0 && nrow(stratified[ly].rules) > 0) || continue
            comparisons["$(lx)_vs_$(ly)"] = compare_strata(
                stratified[lx].rules, stratified[ly].rules; labels=(lx, ly), exclusive_items)
        end
        if !isempty(comparisons)
            vprintln("\n═══ Cross-Strata Comparison ($(length(comparisons)) pair(s)) ═══")
            comparison = comparisons
        end
    end

    return FullPipelineResult(rules, itemsets, n_multi, n_total, stratified, comparison)
end

# ──────────────────────────────────────────────────────────────────────────────
# Network analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_network_pipeline(data_path::String;
                          weight_metric::Symbol=:lift,
                          min_count::Int=30,
                          alpha::Float64=0.05,
                          test::Symbol=:fisher,
                          correction::Symbol=:bh,
                          community_method::Symbol=:louvain,
                          stratify_by_group::Bool=true) -> NetworkPipelineResult

Run the complete network analysis pipeline:
1. Load data and build co-occurrence network
2. Detect communities and compute metrics
3. Optionally build group-stratified networks and compare

# Returns
`NetworkPipelineResult` with fields:
- `network`: CooccurrenceNetwork for the full population
- `communities`: CommunityResult
- `metrics`: Per-node metrics DataFrame
- `stratified` (if stratify_by_group): Output from stratified_network_analysis
- `comparison` (if stratify_by_group): NetworkComparisonResult
"""
function run_network_pipeline(data_path::String; verbose::Bool=true, kwargs...)
    verbose && println("Loading data from $data_path...")
    return run_network_pipeline(load_event_data(data_path); verbose, kwargs...)
end

function run_network_pipeline(event_df::DataFrame;
                               weight_metric::Symbol=:lift,
                               min_count::Int=30,
                               alpha::Float64=0.05,
                               test::Symbol=:fisher,
                               correction::Symbol=:bh,
                               community_method::Symbol=:louvain,
                               stratify_by_group::Bool=true,
                               timing_filter::Symbol=:all,
                               concurrent_window::Int=0,
                               exclusive_items::Union{Nothing, AbstractDict}=nothing,
                               verbose::Bool=true)
    vprintln(args...) = verbose && println(args...)
    n_total = length(unique(event_df.id))
    vprintln("  $n_total records, $(nrow(event_df)) event rows")

    # Full population network
    vprintln("\n═══ Full Population Network ═══")
    net = build_cooccurrence_network(event_df;
        weight_metric, min_count, alpha, test, correction,
        timing_filter, concurrent_window)
    comm = detect_communities(net; method=community_method)
    metrics = compute_network_metrics(net)
    verbose && network_summary(net, comm)

    stratified = nothing
    comparison = nothing
    if stratify_by_group
        vprintln("\n═══ Group-Stratified Networks ═══")
        stratified = stratified_network_analysis(event_df;
            weight_metric, min_count, alpha, test, correction,
            community_method, timing_filter, concurrent_window, exclusive_items, verbose)

        # Pairwise network comparison over every group pair with non-empty graphs.
        labels = collect(keys(stratified))
        comparisons = OrderedDict{String, NetworkComparisonResult}()
        for i in 1:length(labels), j in (i+1):length(labels)
            lx, ly = labels[i], labels[j]
            (nv(stratified[lx].net.graph) > 0 && nv(stratified[ly].net.graph) > 0) || continue
            comparisons["$(lx)_vs_$(ly)"] = compare_networks(
                stratified[lx].net, stratified[ly].net,
                stratified[lx].communities, stratified[ly].communities)
        end
        if !isempty(comparisons)
            vprintln("\n═══ Network Comparison ($(length(comparisons)) pair(s)) ═══")
            comparison = comparisons
        end
    end

    return NetworkPipelineResult(net, comm, metrics, stratified, comparison)
end

# ──────────────────────────────────────────────────────────────────────────────
# Bayesian clustering pipeline
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_clustering_pipeline(data_path::String;
                             K_range=2:8, min_items=2,
                             prior=:flat,
                             alpha_prior=1.0, beta_prior=1.0,
                             concentration=10.0, floor=1.0,
                             dirichlet_prior=1.0, n_init=5,
                             stratify_by_group=true) -> ClusteringPipelineResult

Run the complete Bayesian clustering pipeline:
1. Load data and fit Bernoulli mixture models across K values
2. Select best K by BIC
3. Optionally cluster group_a and group_b cohorts separately and compare

Set `prior=:empirical_bayes` for sparse data (Ye 2018-style per-feature priors).

# Returns
`ClusteringPipelineResult` with fields:
- `clustering`: CooccurrenceAnalysisResult for full population
- `stratified` (if stratify_by_group): NamedTuple with group_a_result, group_b_result
- `comparison` (if stratify_by_group): ClusteringComparisonResult
"""
function run_clustering_pipeline(data_path::String; verbose::Bool=true, kwargs...)
    verbose && println("Loading data from $data_path...")
    return run_clustering_pipeline(load_event_data(data_path); verbose, kwargs...)
end

function run_clustering_pipeline(event_df::DataFrame;
                                  K_range::UnitRange{Int}=2:8,
                                  min_items::Int=2,
                                  prior::Symbol=:flat,
                                  alpha_prior::Float64=1.0,
                                  beta_prior::Float64=1.0,
                                  concentration::Float64=10.0,
                                  floor::Float64=1.0,
                                  dirichlet_prior::Float64=1.0,
                                  n_init::Int=5,
                                  stratify_by_group::Bool=true,
                                  exclusive_items::Union{Nothing, AbstractDict}=nothing,
                                  verbose::Bool=true)
    vprintln(args...) = verbose && println(args...)
    n_total = length(unique(event_df.id))
    vprintln("  $n_total records, $(nrow(event_df)) event rows")

    vprintln("\n═══ Full Population Bayesian Clustering ═══")
    clustering = bernoulli_clustering(event_df;
        K_range, min_items, prior, alpha_prior, beta_prior,
        concentration, floor, dirichlet_prior, n_init)
    verbose && clustering_summary(clustering)

    stratified = nothing
    comparison = nothing
    if stratify_by_group
        vprintln("\n═══ Group-Stratified Bayesian Clustering ═══")
        stratified = stratified_bernoulli_clustering(event_df;
            K_range, min_items, prior, alpha_prior, beta_prior,
            concentration, floor, dirichlet_prior, n_init, exclusive_items, verbose)

        # Pairwise clustering comparison over every group pair.
        labels = collect(keys(stratified))
        comparisons = OrderedDict{String, ClusteringComparisonResult}()
        for i in 1:length(labels), j in (i+1):length(labels)
            lx, ly = labels[i], labels[j]
            comparisons["$(lx)_vs_$(ly)"] = compare_clusterings(stratified[lx], stratified[ly])
        end
        if !isempty(comparisons)
            vprintln("\n═══ Clustering Comparison ($(length(comparisons)) pair(s)) ═══")
            comparison = comparisons
        end
    end

    return ClusteringPipelineResult(clustering, stratified, comparison)
end

end # module
