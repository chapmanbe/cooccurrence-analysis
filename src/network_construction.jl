# ──────────────────────────────────────────────────────────────────────────────
# Network construction from item co-occurrence data
# ──────────────────────────────────────────────────────────────────────────────

"""
    CooccurrenceNetwork

Weighted graph representing item co-occurrence relationships.

# Fields
- `graph`: SimpleWeightedGraph with edge weights encoding association strength
- `items`: Ordered vector of item names (vertex i ↔ items[i])
- `item_index`: Reverse lookup from item name to vertex index
- `edge_data`: DataFrame with all pairwise association statistics
- `prevalence`: Dict mapping item name to record count
- `n_records`: Total number of records in the population
"""
struct CooccurrenceNetwork
    graph::SimpleWeightedGraph{Int, Float64}
    items::Vector{String}
    item_index::Dict{String, Int}
    edge_data::DataFrame
    prevalence::Dict{String, Int}
    n_records::Int
end

"""
    phi_coefficient(ct::Matrix{Int}) -> Float64

Compute the phi coefficient from a 2×2 contingency table.

Layout:
            Has B    No B
    Has A   [n11     n10]
    No A    [n01     n00]

Phi ranges from -1 (perfect negative) to +1 (perfect positive).
Returns 0.0 if denominator is zero.
"""
function phi_coefficient(ct::Matrix{Int})
    n11, n10, n01, n00 = ct[1,1], ct[1,2], ct[2,1], ct[2,2]
    denom = sqrt(Float64(n11 + n10) * Float64(n01 + n00) *
                 Float64(n11 + n01) * Float64(n10 + n00))
    denom == 0.0 && return 0.0
    return (Float64(n11) * Float64(n00) - Float64(n10) * Float64(n01)) / denom
end

"""
    compute_pairwise_associations(event_df::DataFrame;
                                   group_filter=nothing,
                                   test::Symbol=:fisher,
                                   correction::Symbol=:bh,
                                   timing_filter::Symbol=:all,
                                   concurrent_window::Int=0) -> DataFrame

Compute association statistics for all pairs of items.

With `timing_filter=:concurrent` or `:sequential`, restricts to multi-item
records whose events match the timing criterion (single-item records are
excluded under those modes since they have no within-record timing). This
changes the implicit denominator for lift; report results in that context.

Returns a DataFrame with columns: `item_a`, `item_b`, `observed`, `expected`,
`lift`, `phi`, `odds_ratio`, `p_value`, `p_adjusted`, `n_a`, `n_b`.
"""
function compute_pairwise_associations(event_df::DataFrame;
                                        group_filter=nothing,
                                        test::Symbol=:fisher,
                                        correction::Symbol=:bh,
                                        timing_filter::Symbol=:all,
                                        concurrent_window::Int=0,
                                        exclusive_items::Union{Nothing, AbstractDict}=nothing)
    # Filter to the requested group and drop other groups' exclusive items
    df = _filter_group(event_df, group_filter, exclusive_items)

    # Apply timing filter (no-op when timing_filter == :all)
    df = filter_event_by_timing(df; timing_filter, concurrent_window)

    # Build a record × item boolean matrix once. All co-occurrence counts then
    # come from C = Xᵀ·X (BLAS) and per-item prevalence from column sums, so no
    # pair rescans the records (was O(M²·N) with two scans per pair).
    gp = groupby(df, :id)
    record_sets = [Set(g.item) for g in gp]
    n_records = length(record_sets)
    all_items = sort(unique(reduce(vcat, (collect(s) for s in record_sets); init=String[])))
    M = length(all_items)
    col_of = Dict(s => i for (i, s) in enumerate(all_items))

    X = falses(n_records, M)
    for (r, s) in enumerate(record_sets)
        for it in s
            X[r, col_of[it]] = true
        end
    end

    Xf = Float64.(X)
    # C[i,j] = # records containing both item i and item j (integer, exact:
    # 0/1 sums stay well within Float64's exact-integer range).
    C = Xf' * Xf
    prev = vec(sum(X, dims=1))  # prevalence per item (records containing it)

    # Compute all pairwise stats from the matrix
    rows = NamedTuple[]
    for i in 1:M
        for j in (i+1):M
            a, b = all_items[i], all_items[j]
            n11 = round(Int, C[i, j])

            # Skip pairs with zero co-occurrence
            n11 == 0 && continue

            n10 = prev[i] - n11
            n01 = prev[j] - n11
            n00 = n_records - n11 - n10 - n01
            ct = [n11 n10; n01 n00]

            observed = n11
            n = n_records
            row_a = prev[i]
            col_b = prev[j]
            expected = (row_a * col_b) / n
            lift = expected > 0 ? observed / expected : 0.0
            phi = phi_coefficient(ct)
            or_val = odds_ratio(ct)

            # Statistical test (from the precomputed table — no rescan)
            result = test_association(ct; test)

            push!(rows, (item_a=a, item_b=b,
                         observed=observed, expected=round(expected, digits=2),
                         lift=round(lift, digits=4), phi=round(phi, digits=4),
                         odds_ratio=round(or_val, digits=4),
                         p_value=result.p_value,
                         n_a=prev[i], n_b=prev[j]))
        end
    end

    isempty(rows) && return DataFrame(
        item_a=String[], item_b=String[], observed=Int[], expected=Float64[],
        lift=Float64[], phi=Float64[], odds_ratio=Float64[], p_value=Float64[],
        p_adjusted=Float64[], n_a=Int[], n_b=Int[])

    result_df = DataFrame(rows)

    # Multiple testing correction
    adj = adjust_pvalues(result_df.p_value; method=correction)
    result_df.p_adjusted = adj

    return result_df
end

"""
    build_cooccurrence_network(event_df::DataFrame;
                                weight_metric::Symbol=:lift,
                                min_count::Int=30,
                                alpha::Float64=0.05,
                                group_filter=nothing,
                                test::Symbol=:fisher,
                                correction::Symbol=:bh,
                                timing_filter::Symbol=:all,
                                concurrent_window::Int=0) -> CooccurrenceNetwork

Build a weighted co-occurrence network from event data.

# Arguments
- `weight_metric`: Edge weight metric — `:lift` (default) or `:phi`
- `min_count`: Minimum co-occurrence count to include an edge
- `alpha`: Significance threshold for adjusted p-values
- `group_filter`: a group value to build a group-specific network (or `nothing`)
- `exclusive_items`: optional `Dict(group_value => Set(items))`; when
  `group_filter` is set, items exclusive to OTHER groups are dropped
- `timing_filter`: `:all` / `:concurrent` / `:sequential`. With non-`:all`
  values, restricts to multi-item records matching the timing criterion;
  see `filter_event_by_timing`.
- `concurrent_window`: Year-gap threshold for concurrent classification.

# Returns
`CooccurrenceNetwork` with edges for pairs that pass all filters:
co-occurrence ≥ `min_count`, `p_adjusted` < `alpha`, and `lift` > 1.
"""
function build_cooccurrence_network(event_df::DataFrame;
                                     weight_metric::Symbol=:lift,
                                     min_count::Int=30,
                                     alpha::Float64=0.05,
                                     group_filter=nothing,
                                     test::Symbol=:fisher,
                                     correction::Symbol=:bh,
                                     timing_filter::Symbol=:all,
                                     concurrent_window::Int=0,
                                     exclusive_items::Union{Nothing, AbstractDict}=nothing)
    # Compute all pairwise associations
    edge_data = compute_pairwise_associations(event_df;
        group_filter, test, correction, timing_filter, concurrent_window, exclusive_items)

    # Filter edges
    significant = filter(row ->
        row.observed >= min_count &&
        row.p_adjusted < alpha &&
        row.lift > 1.0,
        edge_data)

    # Collect all items that appear in at least one significant edge
    items_in_edges = unique(vcat(significant.item_a, significant.item_b))

    # Also include all items from the population for complete vertex set
    df = _filter_group(event_df, group_filter, exclusive_items)
    df = filter_event_by_timing(df; timing_filter, concurrent_window)

    # Use only items that appear in edges (isolated nodes provide no info)
    items = sort(items_in_edges)
    item_index = Dict(s => i for (i, s) in enumerate(items))
    n_vertices = length(items)

    # Build graph
    g = SimpleWeightedGraph(n_vertices)
    for row in eachrow(significant)
        haskey(item_index, row.item_a) || continue
        haskey(item_index, row.item_b) || continue
        i = item_index[row.item_a]
        j = item_index[row.item_b]
        w = weight_metric == :phi ? max(row.phi, 0.001) : row.lift
        add_edge!(g, i, j, w)
    end

    # Compute prevalence
    gp = groupby(df, :id)
    record_items = [sort(unique(g.item)) for g in gp]
    n_records = length(record_items)
    prevalence = Dict{String, Int}()
    for item in items
        prevalence[item] = count(ps -> item in ps, record_items)
    end

    return CooccurrenceNetwork(g, items, item_index, edge_data,
                               prevalence, n_records)
end
