# Clustering Analysis

Core analysis package implementing four methods for discovering item co-occurrence patterns, with comprehensive visualization support.

## Analysis Methods

### HDP Bernoulli Mixture (`hdp_clustering.jl`, `hdp_wrappers.jl`)

**Main method.** Fits a Hierarchical Dirichlet Process Bernoulli mixture jointly across record strata (by group, etc.) using coordinate-ascent variational inference (CAVI) with truncated stick-breaking. Learns a shared set of cluster atoms and per-group mixing weights, directly answering "is this cluster universal or group-specific?" with a variational posterior. K is not pre-specified — the posterior over stick weights reveals how many clusters are actually used.

- ~30 seconds on 190K records, 41 items, K_max=20 (3 restarts)
- Converges in ~100 CAVI iterations; all updates closed-form (no autodiff)
- `hdp_cluster_categorization` outputs a per-cluster table: `universal` / `group_a_only` / `group_b_only` / `both_present_but_unequal` / `negligible`

Key functions: `hdp_clustering`, `hdp_cluster_categorization`, `fit_hdp_bernoulli_mixture`

### Flat-K Bayesian Bernoulli Mixture (`bayesian_clustering.jl`, `bayesian_wrappers.jl`)

Fits K-component Bernoulli mixture models via EM with BIC model selection. Each component is characterized by a probability vector over items. Ye 2018 empirical Bayes priors available for sparse data. Useful as a per-stratum baseline and for posterior credible intervals via Turing.jl NUTS on small subsets.

Key functions: `bernoulli_clustering`, `select_K`, `fit_bernoulli_mixture`

### Association Rule Mining (`mining.jl`, `statistical_validation.jl`)

Finds item combinations that co-occur more often than expected using FP-Growth frequent itemset mining. Rules are validated with Fisher's exact test (or chi-square) and p-values are adjusted for multiple testing (Benjamini-Hochberg or Bonferroni).

Key functions: `build_transactions`, `mine_frequent_itemsets`, `mine_association_rules`, `validate_rules`

### Network Analysis (`network_construction.jl`, `network_metrics.jl`)

Builds a weighted co-occurrence graph where nodes are items and edges encode association strength (lift or phi coefficient). Community detection (Louvain or label propagation) identifies clusters. Per-node centrality metrics (degree, strength, betweenness, clustering coefficient) highlight the most connected items.

Key functions: `build_cooccurrence_network`, `detect_communities`, `compute_network_metrics`

### Group Stratification (`group_stratification.jl`, `network_group_stratification.jl`)

All methods support group stratification over **any number of groups**. The HDP handles it natively (joint fit with per-group weights). ARM, network, and flat-K clustering run separately per group via the generic `stratify_by` driver, then compare strata **pairwise**.

`stratify_by(analysis_fn, event_df; group_col=:Group, kwargs...)` iterates `sort(unique(event_df[!, group_col]))` and returns an `OrderedDict` keyed by group value. Each stratified function returns such a dict:

- `stratified_analysis(event_df)["A"]` → `(rules, itemsets, n_records)`
- `stratified_network_analysis(event_df)["A"]` → `(net, communities, metrics)`
- `stratified_bernoulli_clustering(event_df)["A"]` → `CooccurrenceAnalysisResult`

Key functions: `stratify_by`, `stratified_analysis`, `compare_strata`, `stratified_network_analysis`, `compare_networks`, `stratified_bernoulli_clustering`, `compare_clusterings`

**Domain knowledge stays with the caller.** Items exclusive to a group are not baked into the package. Pass `exclusive_items = Dict("A" => Set(["GA1", ...]), "B" => Set([...]))` to `build_transactions` / `compute_pairwise_associations` / `build_cooccurrence_network` (and the stratified/pipeline functions that wrap them). When filtering to a group, items exclusive to *other* groups are dropped.

#### Migration from the two-group API (breaking change)

The per-group ARM/network/flat-K results are now keyed by group value in a dict instead of spliced into `NamedTuple` field names, which removes the old two-group ceiling:

| Old (≤ v0.1) | New |
|--------------|-----|
| `strat.group_a_rules` | `strat["A"].rules` |
| `strat.group_b_n_records` | `strat["B"].n_records` |
| `strat.group_a_net` / `.group_a_communities` | `strat["A"].net` / `strat["A"].communities` |
| `strat.group_a_result` | `strat["A"]` |
| `GROUP_A_ONLY_ITEMS` / `GROUP_B_ONLY_ITEMS` (exported) | caller-supplied `exclusive_items` keyword |

`compare_strata(rules_x, rules_y; labels=("A","B"))` now takes two dict entries and derives its column prefixes and category symbols from `labels`. Comparisons remain explicitly pairwise; the pipelines compare every group pair.

## Visualization

All plot functions return a `Makie.Figure` and follow consistent conventions (wong_colors for categorical, :YlOrRd for heatmaps).

| Function | Description |
|----------|-------------|
| `plot_arm_scatter` | Support vs Confidence scatter, colored by Lift |
| `plot_arm_matrix` | Item-by-item heatmap of pairwise Lift |
| `plot_arm_comparison` | Group A vs group_b grouped bar chart |
| `plot_cooccurrence_network` | Force-directed network graph |
| `plot_community_heatmap` | Association matrix ordered by community |
| `plot_network_comparison` | Side-by-side group_a/group_b networks |
| `plot_group_stratified_network` | Union graph with group-colored edges |
| `plot_centrality_barchart` | Ranked centrality bar chart |
| `plot_centrality_comparison` | Side-by-side group_a/group_b centrality |
| `plot_class_probabilities` | Latent class probability heatmap |
| `plot_bic_elbow` | BIC vs K line plot |
| `plot_class_profiles` | Top items per class bar charts |
| `plot_clustering_comparison` | Side-by-side group_a/group_b class heatmaps |
| `plot_hdp_stick_weights` | Global β and per-group π grouped bar chart |
| `plot_hdp_class_profiles` | HDP cluster × item heatmap with category labels |
| `plot_hdp_sharing_heatmap` | Cluster × group presence matrix |
| `plot_upset` | UpSet plot of item combination frequencies |
| `plot_record_heatmap` | Binary record-by-item matrix |

## End-to-End Pipelines

Each method has a one-call pipeline function:

```julia
using CooccurrenceAnalysis

arm_results   = run_full_pipeline("../ddata/synthetic_events.arrow")
net_results   = run_network_pipeline("../ddata/synthetic_events.arrow")
clust_results = run_clustering_pipeline("../ddata/synthetic_events.arrow")

# HDP (main method — joint group-stratified fit)
hdp_result = hdp_clustering(event_df; group_by=:Group, K_max=20,
                                   prior=:empirical_bayes)
hdp_cluster_categorization(hdp_result)   # universal / group_a_only / group_b_only table
```

## Notebooks

Jupytext percent-format Julia scripts (convert with `jupytext --to notebook <file>.jl`):

- `analysis_notebook.jl` — Association rule mining walkthrough
- `network_notebook.jl` — Network analysis walkthrough
- `visualization_notebook.jl` — Gallery demonstrating all plot functions

## Project Structure

```
cooccurrence-analysis/
  src/
    CooccurrenceAnalysis.jl   # Main module (includes + exports)
    utils.jl                    # Formatting helpers and item constants
    data_preparation.jl         # Data loading and transaction building
    mining.jl                   # FP-Growth itemset and rule mining
    statistical_validation.jl   # Hypothesis testing and p-value adjustment
    group_stratification.jl       # ARM group-stratified analysis
    network_construction.jl     # Co-occurrence network building
    network_metrics.jl          # Centrality and community detection
    network_group_stratification.jl
    network_visualization.jl    # Network and centrality plots
    bayesian_clustering.jl      # Flat-K Bernoulli mixture EM + EB priors
    bayesian_wrappers.jl        # Item-domain wrappers for flat-K mixture
    bayesian_turing.jl          # Turing.jl NUTS/ADVI (small-data posteriors)
    hdp_clustering.jl           # HDP CAVI core (HDPBernoulliResult)
    hdp_wrappers.jl             # Item-domain wrappers for HDP
    bayesian_visualization.jl   # Class profile, BIC, and HDP plots
    arm_visualization.jl        # ARM scatter, matrix, comparison plots
    overview_visualization.jl   # UpSet plot and record heatmap
  test/
    runtests.jl                 # 306 tests
  scripts/
    run_hdp_demo.jl             # HDP validation on full synthetic cohort
    compare_priors.jl           # Flat vs empirical-Bayes EM comparison
    turing_stratified.jl        # Turing ADVI (small-data only)
  Project.toml
```

## Running Tests

```bash
julia --project=. -e 'include("test/runtests.jl")'
```

## Dependencies

Arrow, CSV, DataFrames, StatsBase, RuleMiner, HypothesisTests, MultipleTesting, Graphs, SimpleWeightedGraphs, CairoMakie, GraphMakie, NetworkLayout, SpecialFunctions, Distributions, Turing, MCMCChains, ReverseDiff.

## License

MIT. See [LICENSE](LICENSE).
