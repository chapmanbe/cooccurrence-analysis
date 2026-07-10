# ──────────────────────────────────────────────────────────────────────────────
# Group-stratified network analysis and comparison
# ──────────────────────────────────────────────────────────────────────────────

"""
    NetworkComparisonResult

Result of comparing group_a and group_b co-occurrence networks.

# Fields
- `shared_edges`: DataFrame of edges present in both networks
- `group_a_only_edges`: DataFrame of edges unique to the group_a network
- `group_b_only_edges`: DataFrame of edges unique to the group_b network
- `community_ari`: Adjusted Rand Index comparing community structures, or
  `missing` when the two networks share fewer than 2 items (ARI undefined)
"""
struct NetworkComparisonResult
    shared_edges::DataFrame
    group_a_only_edges::DataFrame
    group_b_only_edges::DataFrame
    community_ari::Union{Missing, Float64}
end

"""
    stratified_network_analysis(event_df::DataFrame;
                                 weight_metric::Symbol=:lift,
                                 min_count::Int=30,
                                 alpha::Float64=0.05,
                                 test::Symbol=:fisher,
                                 correction::Symbol=:bh,
                                 community_method::Symbol=:louvain) -> NamedTuple

Build and analyze co-occurrence networks separately for group_a and group_b cohorts.

# Returns
NamedTuple with:
- `group_a_net`, `group_b_net`: CooccurrenceNetwork for each group
- `group_a_communities`, `group_b_communities`: CommunityResult for each group
- `group_a_metrics`, `group_b_metrics`: Per-node metrics DataFrames
"""
# One network stratum: build + analyze the co-occurrence network for a group.
function _network_stratum(event_df::DataFrame; group_filter,
                          weight_metric::Symbol, min_count::Int, alpha::Float64,
                          test::Symbol, correction::Symbol, community_method::Symbol,
                          timing_filter::Symbol, concurrent_window::Int,
                          exclusive_items::Union{Nothing, AbstractDict})
    net = build_cooccurrence_network(event_df;
        weight_metric, min_count, alpha, group_filter, test, correction,
        timing_filter, concurrent_window, exclusive_items)
    comm = detect_communities(net; method=community_method)
    metrics = compute_network_metrics(net)
    return (net=net, communities=comm, metrics=metrics)
end

function stratified_network_analysis(event_df::DataFrame;
                                      weight_metric::Symbol=:lift,
                                      min_count::Int=30,
                                      alpha::Float64=0.05,
                                      test::Symbol=:fisher,
                                      correction::Symbol=:bh,
                                      community_method::Symbol=:louvain,
                                      timing_filter::Symbol=:all,
                                      concurrent_window::Int=0,
                                      exclusive_items::Union{Nothing, AbstractDict}=nothing,
                                      verbose::Bool=true)
    return stratify_by(_network_stratum, event_df; verbose,
        weight_metric, min_count, alpha, test, correction, community_method,
        timing_filter, concurrent_window, exclusive_items)
end

"""
    _edge_set(net::CooccurrenceNetwork) -> Set{Tuple{String, String}}

Extract the set of edges as canonical (sorted) item name pairs.
"""
function _edge_set(net::CooccurrenceNetwork)
    edge_pairs = Set{Tuple{String, String}}()
    for e in edges(net.graph)
        a = net.items[src(e)]
        b = net.items[dst(e)]
        push!(edge_pairs, a < b ? (a, b) : (b, a))
    end
    return edge_pairs
end

"""
    _edge_weight_lookup(net::CooccurrenceNetwork) -> Dict{Tuple{String, String}, Float64}

Build a lookup from canonical edge pair to weight.
"""
function _edge_weight_lookup(net::CooccurrenceNetwork)
    lookup = Dict{Tuple{String, String}, Float64}()
    for e in edges(net.graph)
        a = net.items[src(e)]
        b = net.items[dst(e)]
        key = a < b ? (a, b) : (b, a)
        lookup[key] = weight(e)
    end
    return lookup
end

"""
    _adjusted_rand_index(labels_a::Vector{Int}, labels_b::Vector{Int}) -> Float64

Compute the Adjusted Rand Index (ARI) between two clusterings of the same elements.

ARI = (RI - Expected_RI) / (max_RI - Expected_RI)

Only compares items that appear in both networks. Returns 0.0 if fewer than
2 shared items exist.
"""
function _adjusted_rand_index(labels_a::Vector{Int}, labels_b::Vector{Int})
    n = length(labels_a)
    n == length(labels_b) || error("Label vectors must have same length")
    n < 2 && return 0.0

    # Build contingency table
    ca = sort(unique(labels_a))
    cb = sort(unique(labels_b))
    ca_map = Dict(c => i for (i, c) in enumerate(ca))
    cb_map = Dict(c => i for (i, c) in enumerate(cb))

    nij = zeros(Int, length(ca), length(cb))
    for k in 1:n
        nij[ca_map[labels_a[k]], cb_map[labels_b[k]]] += 1
    end

    # Row and column sums
    ai = vec(sum(nij, dims=2))
    bj = vec(sum(nij, dims=1))

    # Compute index using combinatorial formulation
    _c2(x) = x * (x - 1) ÷ 2

    sum_nij_c2 = sum(_c2(nij[i, j]) for i in 1:length(ca) for j in 1:length(cb))
    sum_ai_c2 = sum(_c2(a) for a in ai)
    sum_bj_c2 = sum(_c2(b) for b in bj)
    n_c2 = _c2(n)

    n_c2 == 0 && return 0.0

    expected = sum_ai_c2 * sum_bj_c2 / n_c2
    max_index = 0.5 * (sum_ai_c2 + sum_bj_c2)
    denom = max_index - expected

    denom == 0.0 && return 1.0  # Perfect agreement when both are trivial
    return (sum_nij_c2 - expected) / denom
end

"""
    compare_networks(group_a_net::CooccurrenceNetwork,
                     group_b_net::CooccurrenceNetwork,
                     group_a_comm::CommunityResult,
                     group_b_comm::CommunityResult) -> NetworkComparisonResult

Compare group_a and group_b co-occurrence networks.

Identifies shared and unique edges, and computes the Adjusted Rand Index (ARI)
for community agreement on shared items.
"""
function compare_networks(group_a_net::CooccurrenceNetwork,
                          group_b_net::CooccurrenceNetwork,
                          group_a_comm::CommunityResult,
                          group_b_comm::CommunityResult)
    group_a_edges = _edge_set(group_a_net)
    group_b_edges = _edge_set(group_b_net)
    group_a_weights = _edge_weight_lookup(group_a_net)
    group_b_weights = _edge_weight_lookup(group_b_net)

    shared = intersect(group_a_edges, group_b_edges)
    a_only = setdiff(group_a_edges, group_b_edges)
    b_only = setdiff(group_b_edges, group_a_edges)

    # Build DataFrames
    shared_df = _edges_to_df(shared, group_a_weights, group_b_weights)
    group_a_only_df = _edges_to_df(a_only, group_a_weights, nothing)
    group_b_only_df = _edges_to_df(b_only, nothing, group_b_weights)

    # ARI on shared items
    shared_items = intersect(Set(group_a_net.items), Set(group_b_net.items))
    if length(shared_items) >= 2
        shared_items_sorted = sort(collect(shared_items))
        group_a_labels = [group_a_comm.assignments[group_a_net.item_index[s]]
                       for s in shared_items_sorted]
        group_b_labels = [group_b_comm.assignments[group_b_net.item_index[s]]
                         for s in shared_items_sorted]
        ari = _adjusted_rand_index(group_a_labels, group_b_labels)
    else
        # Fewer than 2 shared items: ARI is undefined. Return missing rather
        # than 0.0, which would falsely read as "chance-level agreement".
        ari = missing
    end

    return NetworkComparisonResult(shared_df, group_a_only_df, group_b_only_df, ari)
end

function _edges_to_df(edge_set, group_a_weights, group_b_weights)
    sorted_edges = sort(collect(edge_set))
    item_a = [e[1] for e in sorted_edges]
    item_b = [e[2] for e in sorted_edges]

    df = DataFrame(item_a=item_a, item_b=item_b)
    if group_a_weights !== nothing
        df.group_a_weight = [get(group_a_weights, e, missing) for e in sorted_edges]
    end
    if group_b_weights !== nothing
        df.group_b_weight = [get(group_b_weights, e, missing) for e in sorted_edges]
    end
    return df
end
