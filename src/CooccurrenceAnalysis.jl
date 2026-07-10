"""
Association Rule Mining for Item Co-occurrence Analysis

Discovers item combinations that co-occur more often than expected by
chance using FP-Growth frequent itemset mining with statistical validation.
Supports group-stratified analysis for identifying group-specific vs universal
item associations.
"""
module CooccurrenceAnalysis

using DataFrames, StatsBase
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
       stratified_analysis, compare_strata,
       run_full_pipeline,
       format_itemset, significance_stars, results_summary,
       GROUP_A_ONLY_ITEMS, GROUP_B_ONLY_ITEMS,
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
       run_network_pipeline

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
       run_clustering_pipeline,
       plot_upset, plot_record_heatmap,
       plot_item_prevalence, plot_cooccurrence_heatmap

include("utils.jl")
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
                      stratify_by_group::Bool=true) -> NamedTuple

Run the complete analysis pipeline:
1. Load data from Arrow file
2. Build transactions (multi-item records)
3. Mine frequent itemsets and association rules
4. Validate with statistical tests + multiple testing correction
5. Optionally stratify by group and compare

# Returns
NamedTuple with:
- `rules`: Validated rules for the full population
- `itemsets`: Frequent itemsets for the full population
- `n_records`: Number of multi-item records
- `n_total_records`: Total records in dataset
- `stratified` (if stratify_by_group): Output from `stratified_analysis`
- `comparison` (if stratify_by_group): Output from `compare_strata`
"""
function run_full_pipeline(data_path::String;
                           min_support::Float64=0.005,
                           min_confidence::Float64=0.1,
                           min_count::Union{Int, Nothing}=30,
                           max_length::Int=4,
                           test::Symbol=:fisher,
                           correction::Symbol=:bh,
                           stratify_by_group::Bool=true,
                           timing_filter::Symbol=:all,
                           concurrent_window::Int=0)
    println("Loading data from $data_path...")
    event_df = load_event_data(data_path)
    n_total = length(unique(event_df.id))
    println("  $n_total records, $(nrow(event_df)) event rows")

    # Full population analysis
    println("\n═══ Full Population Analysis ═══")
    txns_df = build_transactions(event_df; min_items=2,
                                  timing_filter, concurrent_window)
    n_multi = nrow(txns_df)
    println("  Multi-item records: $n_multi ($(round(n_multi/n_total*100, digits=1))%)")

    itemsets = mine_frequent_itemsets(txns_df;
        min_support, min_count, max_length)
    rules = mine_association_rules(txns_df;
        min_support, min_confidence, min_count, max_length)
    println("  Frequent itemsets (≥2 items): $(nrow(itemsets))")
    println("  Association rules: $(nrow(rules))")

    if nrow(rules) > 0
        rules = validate_rules(rules, event_df; test, correction)
        sig = count(rules.significant)
        println("  Significant rules: $sig / $(nrow(rules))")
    end

    result = Dict{Symbol, Any}(
        :rules => rules,
        :itemsets => itemsets,
        :n_records => n_multi,
        :n_total_records => n_total
    )

    if stratify_by_group
        println("\n═══ Group-Stratified Analysis ═══")
        strat = stratified_analysis(event_df;
            min_support, min_confidence, min_count, max_length, test, correction,
            timing_filter, concurrent_window)
        result[:stratified] = strat

        if nrow(strat.group_a_rules) > 0 && nrow(strat.group_b_rules) > 0
            println("\n═══ Cross-Strata Comparison ═══")
            comp = compare_strata(strat.group_a_rules, strat.group_b_rules)
            n_universal = count(comp.category .== :universal)
            n_group_a = count(comp.category .== :group_a_only)
            n_group_b = count(comp.category .== :group_b_only)
            n_group_spec = count(comp.category .== :group_specific_item)
            println("  Universal: $n_universal | Group A-only: $n_group_a | Group B-only: $n_group_b | Group-specific: $n_group_spec")
            result[:comparison] = comp
        end
    end

    return NamedTuple(result)
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
                          stratify_by_group::Bool=true) -> NamedTuple

Run the complete network analysis pipeline:
1. Load data and build co-occurrence network
2. Detect communities and compute metrics
3. Optionally build group-stratified networks and compare

# Returns
NamedTuple with:
- `network`: CooccurrenceNetwork for the full population
- `communities`: CommunityResult
- `metrics`: Per-node metrics DataFrame
- `stratified` (if stratify_by_group): Output from stratified_network_analysis
- `comparison` (if stratify_by_group): NetworkComparisonResult
"""
function run_network_pipeline(data_path::String;
                               weight_metric::Symbol=:lift,
                               min_count::Int=30,
                               alpha::Float64=0.05,
                               test::Symbol=:fisher,
                               correction::Symbol=:bh,
                               community_method::Symbol=:louvain,
                               stratify_by_group::Bool=true,
                               timing_filter::Symbol=:all,
                               concurrent_window::Int=0)
    println("Loading data from $data_path...")
    event_df = load_event_data(data_path)
    n_total = length(unique(event_df.id))
    println("  $n_total records, $(nrow(event_df)) event rows")

    # Full population network
    println("\n═══ Full Population Network ═══")
    net = build_cooccurrence_network(event_df;
        weight_metric, min_count, alpha, test, correction,
        timing_filter, concurrent_window)
    comm = detect_communities(net; method=community_method)
    metrics = compute_network_metrics(net)
    network_summary(net, comm)

    result = Dict{Symbol, Any}(
        :network => net,
        :communities => comm,
        :metrics => metrics
    )

    if stratify_by_group
        println("\n═══ Group-Stratified Networks ═══")
        strat = stratified_network_analysis(event_df;
            weight_metric, min_count, alpha, test, correction,
            community_method, timing_filter, concurrent_window)
        result[:stratified] = strat

        if nv(strat.group_a_net.graph) > 0 && nv(strat.group_b_net.graph) > 0
            println("\n═══ Network Comparison ═══")
            comp = compare_networks(strat.group_a_net, strat.group_b_net,
                                    strat.group_a_communities, strat.group_b_communities)
            println("  Shared edges: $(nrow(comp.shared_edges))")
            println("  Group A-only edges: $(nrow(comp.group_a_only_edges))")
            println("  Group B-only edges: $(nrow(comp.group_b_only_edges))")
            ari_str = ismissing(comp.community_ari) ? "n/a (<2 shared items)" :
                      string(round(comp.community_ari, digits=4))
            println("  Community ARI: $ari_str")
            result[:comparison] = comp
        end
    end

    return NamedTuple(result)
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
                             stratify_by_group=true) -> NamedTuple

Run the complete Bayesian clustering pipeline:
1. Load data and fit Bernoulli mixture models across K values
2. Select best K by BIC
3. Optionally cluster group_a and group_b cohorts separately and compare

Set `prior=:empirical_bayes` for sparse data (Ye 2018-style per-feature priors).

# Returns
NamedTuple with:
- `clustering`: CooccurrenceAnalysisResult for full population
- `stratified` (if stratify_by_group): NamedTuple with group_a_result, group_b_result
- `comparison` (if stratify_by_group): ClusteringComparisonResult
"""
function run_clustering_pipeline(data_path::String;
                                  K_range::UnitRange{Int}=2:8,
                                  min_items::Int=2,
                                  prior::Symbol=:flat,
                                  alpha_prior::Float64=1.0,
                                  beta_prior::Float64=1.0,
                                  concentration::Float64=10.0,
                                  floor::Float64=1.0,
                                  dirichlet_prior::Float64=1.0,
                                  n_init::Int=5,
                                  stratify_by_group::Bool=true)
    println("Loading data from $data_path...")
    event_df = load_event_data(data_path)
    n_total = length(unique(event_df.id))
    println("  $n_total records, $(nrow(event_df)) event rows")

    println("\n═══ Full Population Bayesian Clustering ═══")
    clustering = bernoulli_clustering(event_df;
        K_range, min_items, prior, alpha_prior, beta_prior,
        concentration, floor, dirichlet_prior, n_init)
    clustering_summary(clustering)

    result = Dict{Symbol, Any}(:clustering => clustering)

    if stratify_by_group
        println("\n═══ Group-Stratified Bayesian Clustering ═══")
        strat = stratified_bernoulli_clustering(event_df;
            K_range, min_items, prior, alpha_prior, beta_prior,
            concentration, floor, dirichlet_prior, n_init)
        result[:stratified] = strat

        println("\n═══ Clustering Comparison ═══")
        comp = compare_clusterings(strat.group_a_result, strat.group_b_result)
        println("  Shared high-prob items: $(nrow(comp.shared_high_prob_items))")
        println("  Group-specific cluster signatures: $(nrow(comp.group_specific_clusters))")
        result[:comparison] = comp
    end

    return NamedTuple(result)
end

end # module
