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
                                        group_filter::Union{AbstractString, Nothing}=nothing,
                                        test::Symbol=:fisher,
                                        correction::Symbol=:bh,
                                        timing_filter::Symbol=:all,
                                        concurrent_window::Int=0)
    df = event_df

    # Apply group filter
    if group_filter !== nothing
        df = filter(row -> row.Group == group_filter, df)
    end
    if group_filter == "A"
        df = filter(row -> !(row.item in GROUP_B_ONLY_ITEMS), df)
    elseif group_filter == "B"
        df = filter(row -> !(row.item in GROUP_A_ONLY_ITEMS), df)
    end

    # Apply timing filter (no-op when timing_filter == :all)
    df = filter_event_by_timing(df; timing_filter, concurrent_window)

    # Build record item lists
    gp = groupby(df, :id)
    record_items = [sort(unique(g.item)) for g in gp]
    n_records = length(record_items)

    # Count prevalence per item
    all_items = sort(unique(vcat(record_items...)))
    prevalence = Dict{String, Int}()
    for item in all_items
        prevalence[item] = count(ps -> item in ps, record_items)
    end

    # Compute all pairwise stats
    rows = NamedTuple[]
    for i in 1:length(all_items)
        for j in (i+1):length(all_items)
            a, b = all_items[i], all_items[j]
            ct = build_contingency_table(a, b, record_items)
            observed = ct[1, 1]

            # Skip pairs with zero co-occurrence
            observed == 0 && continue

            n = sum(ct)
            row_a = ct[1, 1] + ct[1, 2]
            col_b = ct[1, 1] + ct[2, 1]
            expected = (row_a * col_b) / n
            lift = expected > 0 ? observed / expected : 0.0
            phi = phi_coefficient(ct)
            or_val = odds_ratio(ct)

            # Statistical test
            result = test_association(a, b, record_items; test)

            push!(rows, (item_a=a, item_b=b,
                         observed=observed, expected=round(expected, digits=2),
                         lift=round(lift, digits=4), phi=round(phi, digits=4),
                         odds_ratio=round(or_val, digits=4),
                         p_value=result.p_value,
                         n_a=prevalence[a], n_b=prevalence[b]))
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
- `group_filter`: `"A"` or `"B"` for group-specific networks
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
                                     group_filter::Union{AbstractString, Nothing}=nothing,
                                     test::Symbol=:fisher,
                                     correction::Symbol=:bh,
                                     timing_filter::Symbol=:all,
                                     concurrent_window::Int=0)
    # Compute all pairwise associations
    edge_data = compute_pairwise_associations(event_df;
        group_filter, test, correction, timing_filter, concurrent_window)

    # Filter edges
    significant = filter(row ->
        row.observed >= min_count &&
        row.p_adjusted < alpha &&
        row.lift > 1.0,
        edge_data)

    # Collect all items that appear in at least one significant edge
    items_in_edges = unique(vcat(significant.item_a, significant.item_b))

    # Also include all items from the population for complete vertex set
    df = event_df
    if group_filter !== nothing
        df = filter(row -> row.Group == group_filter, df)
    end
    if group_filter == "A"
        df = filter(row -> !(row.item in GROUP_B_ONLY_ITEMS), df)
    elseif group_filter == "B"
        df = filter(row -> !(row.item in GROUP_A_ONLY_ITEMS), df)
    end
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
