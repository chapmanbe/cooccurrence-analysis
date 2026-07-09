# ──────────────────────────────────────────────────────────────────────────────
# Frequent itemset mining and association rule generation
# ──────────────────────────────────────────────────────────────────────────────

"""
    mine_frequent_itemsets(txns_df::DataFrame;
                           min_support::Float64=0.005,
                           min_count::Union{Int, Nothing}=nothing,
                           max_length::Int=4) -> DataFrame

Find frequent item combinations using FP-Growth.

# Arguments
- `txns_df`: One-hot boolean DataFrame from `build_transactions`
- `min_support`: Minimum relative support (fraction of transactions)
- `min_count`: If provided, overrides `min_support` with an absolute count threshold
  (converted to relative support internally)
- `max_length`: Maximum itemset size (default 4)

# Returns
DataFrame with columns: `Itemset`, `Support`, `N`, `Length`
"""
function mine_frequent_itemsets(txns_df::DataFrame;
                                min_support::Float64=0.005,
                                min_count::Union{Int, Nothing}=nothing,
                                max_length::Int=4)
    nrow(txns_df) == 0 && return DataFrame(
        Itemset=Vector{String}[], Support=Float64[], N=Int[], Length=Int[])

    # Convert absolute count to relative support
    if min_count !== nothing
        min_support = min_count / nrow(txns_df)
    end

    txns = Txns(txns_df)
    result = fpgrowth(txns, min_support)

    # Filter to itemsets with 2+ items and within max_length
    return filter(row -> row.Length >= 2 && row.Length <= max_length, result)
end

"""
    mine_association_rules(txns_df::DataFrame;
                           min_support::Float64=0.005,
                           min_confidence::Float64=0.1,
                           min_count::Union{Int, Nothing}=nothing,
                           max_length::Int=4) -> DataFrame

Generate association rules with support, confidence, lift, and coverage metrics.

# Arguments
- `txns_df`: One-hot boolean DataFrame from `build_transactions`
- `min_support`: Minimum relative support
- `min_confidence`: Minimum confidence for rules
- `min_count`: If provided, overrides `min_support` with absolute count threshold
- `max_length`: Maximum rule length (LHS + RHS items)

# Returns
DataFrame with columns: `LHS`, `RHS`, `Support`, `Confidence`, `Coverage`,
`Lift`, `N`, `Length`
"""
function mine_association_rules(txns_df::DataFrame;
                                min_support::Float64=0.005,
                                min_confidence::Float64=0.1,
                                min_count::Union{Int, Nothing}=nothing,
                                max_length::Int=4)
    nrow(txns_df) == 0 && return DataFrame(
        LHS=Vector{String}[], RHS=String[], Support=Float64[],
        Confidence=Float64[], Coverage=Float64[], Lift=Float64[],
        N=Int[], Length=Int[])

    if min_count !== nothing
        min_support = min_count / nrow(txns_df)
    end

    txns = Txns(txns_df)
    result = apriori(txns, min_support, min_confidence, max_length)

    # Filter out rules with empty LHS (these are just marginal frequencies)
    return filter(row -> !isempty(row.LHS), result)
end
