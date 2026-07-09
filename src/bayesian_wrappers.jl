# ──────────────────────────────────────────────────────────────────────────────
# Item-specific wrappers for Bayesian Bernoulli mixture clustering
# ──────────────────────────────────────────────────────────────────────────────

"""
    CooccurrenceAnalysisResult

Item-domain wrapper around BernoulliMixtureModelSelection.

# Fields
- `model_selection`: Full model selection output across K values
- `item_names`: Item names matching columns of the data matrix
- `class_profiles`: DataFrame — K rows, one column per item, values are θ_{k,d}
- `record_assignments`: DataFrame with id, assignment, max_responsibility
- `n_records`: Number of records clustered
"""
struct CooccurrenceAnalysisResult
    model_selection::BernoulliMixtureModelSelection
    item_names::Vector{String}
    class_profiles::DataFrame
    record_assignments::DataFrame
    n_records::Int
end

"""
    ClusteringComparisonResult

Comparison of group_a vs group_b Bayesian clustering results.

# Fields
- `group_a_result`, `group_b_result`: Per-group clustering results
- `shared_high_prob_items`: Items with high probability in at least one class in both groups
- `group_specific_clusters`: Summary of clusters unique to one group
"""
struct ClusteringComparisonResult
    group_a_result::CooccurrenceAnalysisResult
    group_b_result::CooccurrenceAnalysisResult
    shared_high_prob_items::DataFrame
    group_specific_clusters::DataFrame
end

"""
    transactions_to_matrix(txns_df::DataFrame) -> (Matrix{Bool}, Vector{String})

Convert a one-hot boolean DataFrame (from `build_transactions`) to a Bool matrix
and ordered vector of item names.
"""
function transactions_to_matrix(txns_df::DataFrame)
    nrow(txns_df) == 0 && return (Matrix{Bool}(undef, 0, 0), String[])
    items = names(txns_df)
    X = Matrix{Bool}(undef, nrow(txns_df), length(items))
    for (j, col) in enumerate(items)
        X[:, j] .= txns_df[!, col]
    end
    return X, items
end

"""
    bernoulli_clustering(event_df::DataFrame;
                                 group_filter=nothing,
                                 min_items=2,
                                 K_range=2:8,
                                 prior=:flat,
                                 alpha_prior=1.0, beta_prior=1.0,
                                 concentration=10.0, floor=1.0,
                                 dirichlet_prior=1.0,
                                 max_iter=200, tol=1e-6, n_init=5,
                                 rng=Random.GLOBAL_RNG) -> CooccurrenceAnalysisResult

Run Bayesian Bernoulli mixture clustering on item co-occurrence data.

Prepares data via `build_transactions`, runs model selection over `K_range`,
and returns item-domain results including class profiles and record assignments.

Set `prior=:empirical_bayes` (Ye 2018) for sparse data — per-feature Beta priors
calibrated from marginal item-item frequencies, controlled by `concentration`
and `floor`.
"""
function bernoulli_clustering(event_df::DataFrame;
                                      group_filter::Union{AbstractString, Nothing}=nothing,
                                      min_items::Int=2,
                                      timing_filter::Symbol=:all,
                                      concurrent_window::Int=0,
                                      K_range::UnitRange{Int}=2:8,
                                      prior::Symbol=:flat,
                                      alpha_prior::Float64=1.0,
                                      beta_prior::Float64=1.0,
                                      concentration::Float64=10.0,
                                      floor::Float64=1.0,
                                      dirichlet_prior::Float64=1.0,
                                      max_iter::Int=200,
                                      tol::Float64=1e-6,
                                      n_init::Int=5,
                                      rng::AbstractRNG=Random.GLOBAL_RNG)
    # Build transactions and convert to matrix; ids stay row-aligned with X
    txns_df, multi_ids = build_transactions_with_ids(event_df;
        group_filter, min_items, timing_filter, concurrent_window)
    X, item_names = transactions_to_matrix(txns_df)
    N = size(X, 1)
    N == 0 && error("No records with ≥$min_items items after filtering")

    # Clamp K_range to valid range
    max_K = min(last(K_range), N - 1)
    actual_K_range = first(K_range):max_K
    length(actual_K_range) == 0 && error("Not enough records ($N) for K_range $K_range")

    # Run model selection
    ms = select_K(X, actual_K_range;
        prior, alpha_prior, beta_prior, concentration, floor,
        dirichlet_prior, max_iter, tol, n_init, rng)

    # Build class profiles DataFrame
    best = ms.best
    profiles = DataFrame()
    profiles.class = 1:best.K
    profiles.proportion = round.(best.pi .* 100, digits=1)
    for (j, item) in enumerate(item_names)
        profiles[!, item] = round.(best.theta[:, j], digits=4)
    end

    # Build record assignments DataFrame
    assignments_df = DataFrame(
        id = multi_ids,
        assignment = best.assignments,
        max_responsibility = [maximum(best.responsibilities[n, :]) for n in 1:N]
    )

    return CooccurrenceAnalysisResult(ms, item_names, profiles, assignments_df, N)
end

"""
    stratified_bernoulli_clustering(event_df::DataFrame;
                                     K_range=2:8, kwargs...) -> NamedTuple

Run Bayesian clustering separately for group_a and group_b cohorts.

Returns NamedTuple with `group_a_result` and `group_b_result`.
"""
function stratified_bernoulli_clustering(event_df::DataFrame;
                                          K_range::UnitRange{Int}=2:8,
                                          kwargs...)
    results = Dict{Symbol, CooccurrenceAnalysisResult}()
    for group in ["A", "B"]
        prefix = Symbol("group_", lowercase(group), :_result)
        println("── Bayesian clustering: $group cohort ──")
        result = bernoulli_clustering(event_df;
            group_filter=group, K_range, kwargs...)
        println("  Records: $(result.n_records), Best K: $(result.model_selection.best_K)")
        results[prefix] = result
    end
    return (group_a_result=results[:group_a_result],
            group_b_result=results[:group_b_result])
end

"""
    compare_clusterings(group_a::CooccurrenceAnalysisResult,
                        group_b::CooccurrenceAnalysisResult;
                        prob_threshold=0.3) -> ClusteringComparisonResult

Compare group_a and group_b clustering results. Identifies items with
high class probability in both groups vs items that define group-specific clusters.
"""
function compare_clusterings(group_a::CooccurrenceAnalysisResult,
                              group_b::CooccurrenceAnalysisResult;
                              prob_threshold::Float64=0.3)
    best_m = group_a.model_selection.best
    best_f = group_b.model_selection.best

    # Find high-probability items per class for each group
    group_a_high = _high_prob_items(best_m.theta, group_a.item_names, prob_threshold)
    group_b_high = _high_prob_items(best_f.theta, group_b.item_names, prob_threshold)

    # Shared: items that appear as high-prob in at least one class in both groups
    group_a_item_set = Set(vcat(group_a_high...))
    group_b_item_set = Set(vcat(group_b_high...))
    shared_items = intersect(group_a_item_set, group_b_item_set)

    shared_rows = NamedTuple[]
    for item in sort(collect(shared_items))
        m_max = _max_theta_for_item(best_m.theta, group_a.item_names, item)
        f_max = _max_theta_for_item(best_f.theta, group_b.item_names, item)
        push!(shared_rows, (item=item, group_a_max_prob=round(m_max, digits=4),
                            group_b_max_prob=round(f_max, digits=4)))
    end
    shared_df = isempty(shared_rows) ?
        DataFrame(item=String[], group_a_max_prob=Float64[], group_b_max_prob=Float64[]) :
        DataFrame(shared_rows)

    # Group-specific clusters: summarize classes whose top items don't appear in the other group
    spec_rows = NamedTuple[]
    for (group, high, other_set) in [("A", group_a_high, group_b_item_set),
                                    ("B", group_b_high, group_a_item_set)]
        for (k, items) in enumerate(high)
            unique_items = setdiff(Set(items), other_set)
            if !isempty(unique_items)
                push!(spec_rows, (group=group, class=k,
                                  unique_items=join(sort(collect(unique_items)), ", "),
                                  n_unique=length(unique_items)))
            end
        end
    end
    spec_df = isempty(spec_rows) ?
        DataFrame(group=String[], class=Int[], unique_items=String[], n_unique=Int[]) :
        DataFrame(spec_rows)

    return ClusteringComparisonResult(group_a, group_b, shared_df, spec_df)
end

"""
    _high_prob_items(theta, item_names, threshold) -> Vector{Vector{String}}

Return, for each class k, the item names where θ_{k,d} ≥ threshold.
"""
function _high_prob_items(theta::Matrix{Float64}, item_names::Vector{String},
                          threshold::Float64)
    K = size(theta, 1)
    return [item_names[theta[k, :] .>= threshold] for k in 1:K]
end

"""
    _max_theta_for_item(theta, item_names, item) -> Float64

Return the maximum θ_{k,d} across all classes for the given item name.
Returns 0.0 if the item is not found.
"""
function _max_theta_for_item(theta::Matrix{Float64}, item_names::Vector{String},
                             item::String)
    idx = findfirst(==(item), item_names)
    idx === nothing && return 0.0
    return maximum(theta[:, idx])
end

"""
    clustering_summary(result::CooccurrenceAnalysisResult; prob_threshold=0.3)

Print a formatted summary of the Bayesian clustering result.
"""
function clustering_summary(result::CooccurrenceAnalysisResult;
                             prob_threshold::Float64=0.3)
    ms = result.model_selection
    best = ms.best
    println("═══ Bayesian Bernoulli Mixture Clustering ═══")
    println("  Records: $(result.n_records)")
    println("  Items: $(length(result.item_names))")
    println("  Best K: $(ms.best_K) (BIC = $(round(best.bic, digits=1)))")
    println("  Converged: $(best.converged) ($(best.n_iter) iterations)")
    println("  Log-likelihood: $(round(best.log_likelihood, digits=2))")

    println("\n  BIC across K values:")
    for (k, bic) in ms.bic_values
        marker = k == ms.best_K ? " ← best" : ""
        println("    K=$k: $(round(bic, digits=1))$marker")
    end

    println("\n  Latent class profiles (items with P ≥ $prob_threshold):")
    for k in 1:best.K
        pct = round(best.pi[k] * 100, digits=1)
        println("\n  Class $k ($pct% of records):")
        high_idx = findall(best.theta[k, :] .>= prob_threshold)
        if isempty(high_idx)
            println("    (no items above threshold)")
        else
            sorted = sort(high_idx, by=j -> -best.theta[k, j])
            for j in sorted
                println("    • $(result.item_names[j]): $(round(best.theta[k, j], digits=3))")
            end
        end
    end
end
