# ──────────────────────────────────────────────────────────────────────────────
# Item-domain wrappers for HDP Bernoulli mixture clustering
# ──────────────────────────────────────────────────────────────────────────────

"""
    HDPClusteringResult

Item-domain wrapper around HDPBernoulliResult.

# Fields
- `hdp_result`: Core variational posterior
- `item_names`: Item names matching columns of the data matrix
- `class_profiles`: DataFrame — K_max rows × (2 + D): cluster, beta_weight, per-item θ
- `group_profiles`: DataFrame — K_max rows × (1 + n_groups): cluster, per-group π
- `record_assignments`: DataFrame with id, group_label, assignment, max_responsibility
- `n_records_per_group`: Number of records in each group
"""
struct HDPClusteringResult
    hdp_result::HDPBernoulliResult
    item_names::Vector{String}
    class_profiles::DataFrame
    group_profiles::DataFrame
    record_assignments::DataFrame
    n_records_per_group::Vector{Int}
end

"""
    hdp_clustering(event_df::DataFrame;
                          group_by=:Group,
                          min_items=2,
                          timing_filter=:all,
                          concurrent_window=0,
                          K_max=20,
                          alpha=1.0,
                          gamma=1.0,
                          prior=:flat,
                          alpha_prior=1.0, beta_prior=1.0,
                          concentration=10.0, floor=1.0,
                          max_iter=300, tol=1e-5, n_init=3,
                          rng=Random.GLOBAL_RNG) -> HDPClusteringResult

Fit an HDP Bernoulli mixture jointly across strata defined by `group_by`.

The `group_by` column must be present in `event_df`. When `group_by == :Group`,
group-inappropriate items are automatically removed from each record's
record before building the data matrix (uses `GROUP_A_ONLY_ITEMS` /
`GROUP_B_ONLY_ITEMS`). For other stratification columns the full item set is used.

Group-specific items (GA1, GB1, etc.) are included in the shared atom
pool — each atom θ_k has a probability for every item, and group_a-only items
will naturally have θ_{k,d} ≈ 0 in group_b-dominant clusters.

Returns a `HDPClusteringResult` with shared cluster atoms, per-group mixing weights,
and a categorization of clusters as universal / group_a-only / group_b-only (when
`group_by == :Group`).
"""
function hdp_clustering(event_df::DataFrame;
                                group_by::Symbol=:Group,
                                min_items::Int=2,
                                timing_filter::Symbol=:all,
                                concurrent_window::Int=0,
                                K_max::Int=20,
                                alpha::Float64=1.0,
                                gamma::Float64=1.0,
                                prior::Symbol=:flat,
                                alpha_prior::Float64=1.0,
                                beta_prior::Float64=1.0,
                                concentration::Float64=10.0,
                                floor::Float64=1.0,
                                max_iter::Int=300,
                                tol::Float64=1e-5,
                                n_init::Int=3,
                                rng::AbstractRNG=Random.GLOBAL_RNG)

    hasproperty(event_df, group_by) ||
        error("Column $group_by not found in event_df")

    # Sorted unique group values → integer mapping
    group_vals = sort(unique(event_df[!, group_by]))
    n_groups = length(group_vals)
    n_groups >= 1 || error("No groups found in column $group_by")

    group_label_map = Dict(v => i for (i, v) in enumerate(group_vals))
    group_labels = string.(group_vals)

    # Build transactions per group, then join
    all_txns = DataFrame[]
    all_ids = Int[]
    all_grp  = Int[]

    for (j, gval) in enumerate(group_vals)
        sf = (group_by == :Group) ? string(gval) : nothing
        sub_df = filter(row -> row[group_by] == gval, event_df)

        txns_j, ids_j = build_transactions_with_ids(sub_df;
            group_filter=sf, min_items, timing_filter, concurrent_window)

        if nrow(txns_j) > 0
            push!(all_txns, txns_j)
            append!(all_ids, ids_j)
            append!(all_grp,  fill(j, nrow(txns_j)))
        end
    end

    isempty(all_txns) && error("No multi-item records found after filtering")

    # Merge transactions, aligning columns (union of all item columns)
    all_items = sort(unique(reduce(vcat, names.(all_txns))))
    N_total = sum(nrow, all_txns)
    X_bool = falses(N_total, length(all_items))
    item_idx = Dict(s => i for (i, s) in enumerate(all_items))

    row_offset = 0
    for df in all_txns
        for (col, item) in enumerate(names(df))
            cidx = item_idx[item]
            X_bool[row_offset+1:row_offset+nrow(df), cidx] .= df[!, col]
        end
        row_offset += nrow(df)
    end

    # Check every group has records
    for j in 1:n_groups
        count(==(j), all_grp) > 0 ||
            error("Group $(group_labels[j]) has no multi-item records")
    end

    # Fit HDP
    println("── HDP clustering: $(N_total) records, $(length(all_items)) items, $(n_groups) groups ──")
    hdp = fit_hdp_bernoulli_mixture(X_bool, all_grp, n_groups;
        K_max, alpha, gamma, prior, alpha_prior, beta_prior,
        concentration, floor, max_iter, tol, n_init, rng)

    # Attach group labels
    result_with_labels = HDPBernoulliResult(
        hdp.K_max, hdp.n_groups, group_labels,
        hdp.theta, hdp.theta_alpha, hdp.theta_beta,
        hdp.beta_alpha, hdp.beta_beta, hdp.beta_mean,
        hdp.pi_mean, hdp.pi_alpha, hdp.pi_beta,
        hdp.responsibilities, hdp.assignments, hdp.group,
        hdp.elbo, hdp.n_iter, hdp.converged, hdp.effective_K
    )

    # Build class_profiles DataFrame
    profiles = DataFrame()
    profiles.cluster = 1:K_max
    profiles.beta_weight = round.(hdp.beta_mean, digits=4)
    for (d, item) in enumerate(all_items)
        profiles[!, item] = round.(hdp.theta[:, d], digits=4)
    end

    # Build group_profiles DataFrame
    gprofiles = DataFrame()
    gprofiles.cluster = 1:K_max
    for (j, gl) in enumerate(group_labels)
        gprofiles[!, Symbol("pi_" * gl)] = round.(hdp.pi_mean[j, :], digits=4)
    end

    # Build record_assignments DataFrame
    glabels_per_record = [group_labels[g] for g in all_grp]
    assignments_df = DataFrame(
        id             = all_ids,
        group           = glabels_per_record,
        assignment      = hdp.assignments,
        max_responsibility = [maximum(hdp.responsibilities[i, :]) for i in 1:N_total]
    )

    n_per_group = [count(==(j), all_grp) for j in 1:n_groups]

    return HDPClusteringResult(result_with_labels, all_items, profiles,
                           gprofiles, assignments_df, n_per_group)
end

"""
    hdp_cluster_categorization(result::HDPClusteringResult;
                               presence_threshold=0.05,
                               beta_threshold=0.01) -> DataFrame

Categorize each HDP cluster by its presence across groups.

Returns a DataFrame with columns:
- `cluster`: component index (sorted by global β weight)
- `beta_weight`: global stick weight E_q[β_k]
- one column per group `pi_<label>`: per-group mixing weight E_q[π_{j,k}]
- `category`: `:universal`, `:group_<j>_only`, `:both_present_but_unequal`,
  or `:negligible`

A cluster is "present in group j" if `pi_mean[j, k] > presence_threshold`.
`:negligible` means beta_weight ≤ beta_threshold (effectively unused component).

A cluster present in some but not all groups is named for the present group(s),
lowercased: `:group_a_only`, or `:group_a_and_b_only` when several (but not all)
groups share it. Fully shared clusters are `:universal`.
"""
function hdp_cluster_categorization(result::HDPClusteringResult;
                                    presence_threshold::Float64=0.05,
                                    beta_threshold::Float64=0.01)
    hdp = result.hdp_result
    K = hdp.K_max
    J = hdp.n_groups
    labels = hdp.group_labels

    rows = NamedTuple[]
    for k in 1:K
        nt_fields = Any[:cluster => k,
                        :beta_weight => round(hdp.beta_mean[k], digits=4)]
        present = Bool[]
        for j in 1:J
            pi_jk = hdp.pi_mean[j, k]
            push!(nt_fields, Symbol("pi_" * labels[j]) => round(pi_jk, digits=4))
            push!(present, pi_jk > presence_threshold)
        end

        if hdp.beta_mean[k] <= beta_threshold
            cat = :negligible
        elseif all(present)
            # All groups present — check if weights are similar
            pis = [hdp.pi_mean[j, k] for j in 1:J]
            max_ratio = maximum(pis) / max(minimum(pis), 1e-8)
            cat = max_ratio < 2.0 ? :universal : :both_present_but_unequal
        elseif !any(present)
            cat = :negligible
        else
            # Present in some but not all groups: name for the present group(s).
            # One consistent, lowercased form for any number of groups:
            #   1 present  -> :group_<label>_only          (e.g. :group_a_only)
            #   ≥2 present -> :group_<l1>_and_<l2>_only     (e.g. :group_a_and_b_only)
            present_labels = labels[present]
            cat = Symbol("group_" * join(lowercase.(present_labels), "_and_") * "_only")
        end

        push!(nt_fields, :category => cat)
        push!(rows, NamedTuple(nt_fields))
    end

    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

"""
    clustering_summary(result::HDPClusteringResult; prob_threshold=0.15)

Print a formatted summary of the HDP clustering result.
"""
function clustering_summary(result::HDPClusteringResult; prob_threshold::Float64=0.15)
    hdp = result.hdp_result
    K = hdp.K_max
    J = hdp.n_groups

    println("═══ HDP Bernoulli Mixture Clustering ═══")
    println("  Total records: $(sum(result.n_records_per_group))")
    for (j, label) in enumerate(hdp.group_labels)
        println("    $(label): $(result.n_records_per_group[j])")
    end
    println("  Items: $(length(result.item_names))")
    println("  K_max: $K, Effective K: $(hdp.effective_K)")
    println("  ELBO: $(round(hdp.elbo, digits=2))")
    println("  Converged: $(hdp.converged) ($(hdp.n_iter) iterations)")

    # Cross-strata categorization
    cat_df = hdp_cluster_categorization(result)
    non_neg = filter(row -> row.category != :negligible, cat_df)
    if nrow(non_neg) > 0
        println("\n  Active clusters: $(nrow(non_neg)) (β_weight > 0.01)")
        cats = countmap(non_neg.category)
        for (c, n) in sort(collect(cats), by=x->string(x[1]))
            println("    $c: $n")
        end
    end

    # Per-cluster detail for active clusters
    println("\n  Cluster profiles (items with θ ≥ $prob_threshold):")
    for k in 1:K
        hdp.beta_mean[k] < 0.01 && continue

        pi_str = join(["$(hdp.group_labels[j])=$(round(hdp.pi_mean[j,k], digits=3))"
                       for j in 1:J], ", ")
        println("\n  Cluster $k (β=$(round(hdp.beta_mean[k], digits=3)) | $pi_str):")

        high_idx = findall(hdp.theta[k, :] .>= prob_threshold)
        if isempty(high_idx)
            println("    (no items above threshold)")
        else
            sorted = sort(high_idx, by=d -> -hdp.theta[k, d])
            for d in sorted[1:min(8, length(sorted))]
                a = hdp.theta_alpha[k, d]
                b = hdp.theta_beta[k, d]
                ci_lo = quantile(Beta(a, b), 0.025)
                ci_hi = quantile(Beta(a, b), 0.975)
                println("    • $(result.item_names[d]): " *
                        "$(round(hdp.theta[k,d], digits=3)) " *
                        "[$(round(ci_lo, digits=3)), $(round(ci_hi, digits=3))]")
            end
        end
    end
end
