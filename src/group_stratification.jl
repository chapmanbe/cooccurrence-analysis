# ──────────────────────────────────────────────────────────────────────────────
# Group-stratified analysis and cross-strata comparison
# ──────────────────────────────────────────────────────────────────────────────

"""
    stratify_by(analysis_fn, event_df::DataFrame;
                group_col::Symbol=:Group, verbose::Bool=true, kwargs...)
        -> OrderedDict{String, Any}

Generic per-group stratification driver. Runs `analysis_fn` once for every group
value in `sort(unique(event_df[!, group_col]))`, calling
`analysis_fn(event_df; group_filter=g, kwargs...)`, and returns the results keyed
by the string form of each group value.

This is the single driver behind the per-group ARM, network, and flat-K
clustering stratifications. Keying results by group value in a Dict — rather than
splicing values into Symbol field names — removes the old two-group ceiling and
the lowercase-collision problem, and makes N groups work with no code change.
"""
function stratify_by(analysis_fn, event_df::DataFrame;
                     group_col::Symbol=:Group, verbose::Bool=true, kwargs...)
    groups = sort(unique(event_df[!, group_col]))
    out = OrderedDict{String, Any}()
    for g in groups
        verbose && println("── stratum $(g) ──")
        out[string(g)] = analysis_fn(event_df; group_filter=g, kwargs...)
    end
    return out
end

# One ARM stratum: mine + validate rules for a single group.
function _arm_stratum(event_df::DataFrame; group_filter,
                      min_support::Float64, min_confidence::Float64,
                      min_count::Union{Int, Nothing}, max_length::Int,
                      test::Symbol, correction::Symbol,
                      timing_filter::Symbol, concurrent_window::Int,
                      exclusive_items::Union{Nothing, AbstractDict})
    txns_df = build_transactions(event_df; group_filter, min_items=2,
                                 timing_filter, concurrent_window, exclusive_items)
    n_records = nrow(txns_df)
    n_records == 0 && return (rules=DataFrame(), itemsets=DataFrame(), n_records=0)

    itemsets = mine_frequent_itemsets(txns_df; min_support, min_count, max_length)
    rules = mine_association_rules(txns_df; min_support, min_confidence, min_count, max_length)

    # Validate against the same group-filtered event data
    group_event_df = _filter_group(event_df, group_filter, exclusive_items)
    if nrow(rules) > 0
        rules = validate_rules(rules, group_event_df; test, correction)
    end
    return (rules=rules, itemsets=itemsets, n_records=n_records)
end

"""
    stratified_analysis(event_df::DataFrame; kwargs...) -> OrderedDict{String, Any}

Run the association-rule-mining pipeline separately for every group value, via
[`stratify_by`](@ref). Returns an `OrderedDict` keyed by group value; each entry
is a NamedTuple `(rules, itemsets, n_records)`. Access as `strat["A"].rules`.

Pass `exclusive_items = Dict(group => Set(items))` to drop each group's exclusive
items from the other groups' cohorts (domain knowledge supplied by the caller).
"""
function stratified_analysis(event_df::DataFrame;
                             min_support::Float64=0.005,
                             min_confidence::Float64=0.1,
                             min_count::Union{Int, Nothing}=30,
                             max_length::Int=4,
                             test::Symbol=:fisher,
                             correction::Symbol=:bh,
                             timing_filter::Symbol=:all,
                             concurrent_window::Int=0,
                             exclusive_items::Union{Nothing, AbstractDict}=nothing,
                             verbose::Bool=true)
    return stratify_by(_arm_stratum, event_df; verbose,
        min_support, min_confidence, min_count, max_length,
        test, correction, timing_filter, concurrent_window, exclusive_items)
end

"""
    _normalize_rule_key(lhs, rhs) -> String

Canonical string key for a rule, independent of the *internal order* of the
antecedent but preserving the antecedent/consequent split. The LHS items are
sorted (so `{X,Y}→Z` and `{Y,X}→Z` match across strata), then joined with the
RHS via `=>`. Keying the full item multiset instead — the previous behavior —
collapsed distinct rules like `{X,Y}→Z` and `{X,Z}→Y` into one, silently
dropping all but the max-lift copy.
"""
function _normalize_rule_key(lhs, rhs)
    lhs_sorted = sort(collect(lhs))
    return join(lhs_sorted, " + ") * " => " * string(rhs)
end

# Lookup from canonical rule key to the max-lift rule row.
function _rule_lookup(rules::DataFrame)
    d = Dict{String, DataFrameRow}()
    for row in eachrow(rules)
        key = _normalize_rule_key(row.LHS, row.RHS)
        if !haskey(d, key) || row.Lift > d[key].Lift
            d[key] = row
        end
    end
    return d
end

# Items appearing in a canonical rule key ("X + Y => Z" → ["X","Y","Z"]).
function _key_items(key::AbstractString)
    parts = split(key, " => ")
    lhs = split(parts[1], " + ")
    return length(parts) > 1 ? vcat(lhs, [parts[2]]) : collect(lhs)
end

"""
    compare_strata(rules_x::DataFrame, rules_y::DataFrame;
                   labels=("A", "B"), exclusive_items=nothing) -> DataFrame

Compare rules from two group cohorts (explicitly pairwise). `labels` names the
two groups and drives the dynamic column prefixes `group_<label>_*` and the
category symbols `:group_<label>_only`. Each association is categorized as
`:universal`, `:group_<label_x>_only`, `:group_<label_y>_only`, or — when
`exclusive_items` is supplied — `:group_specific_item` if any of its items is
exclusive to some group.

Returns a DataFrame with `association`, `category`, and per-group
`support`/`lift`/`N`/`p_adjusted`/`significant` columns.
"""
function compare_strata(rules_x::DataFrame, rules_y::DataFrame;
                        labels=("A", "B"),
                        exclusive_items::Union{Nothing, AbstractDict}=nothing)
    px = "group_" * lowercase(string(labels[1]))
    py = "group_" * lowercase(string(labels[2]))

    keys_x = _rule_lookup(rules_x)
    keys_y = _rule_lookup(rules_y)
    all_keys = union(keys(keys_x), keys(keys_y))

    excl = exclusive_items === nothing ? Set{String}() :
           reduce(union, (Set(String(i) for i in v) for v in values(exclusive_items));
                  init=Set{String}())

    assoc     = String[]
    category  = Symbol[]
    x_support = Union{Missing, Float64}[]; y_support = Union{Missing, Float64}[]
    x_lift    = Union{Missing, Float64}[]; y_lift    = Union{Missing, Float64}[]
    x_N       = Union{Missing, Int}[];     y_N       = Union{Missing, Int}[]
    x_padj    = Union{Missing, Float64}[]; y_padj    = Union{Missing, Float64}[]
    x_sig     = Union{Missing, Bool}[];    y_sig     = Union{Missing, Bool}[]

    # Fill one group's five metric columns for a given key.
    function fill_metrics!(sup, lft, nn, padj, sig, lookup, rules_df, key)
        if haskey(lookup, key)
            r = lookup[key]
            push!(sup, r.Support); push!(lft, r.Lift); push!(nn, r.N)
            if hasproperty(rules_df, :p_adjusted)
                push!(padj, r.p_adjusted); push!(sig, r.significant)
            else
                push!(padj, missing); push!(sig, missing)
            end
        else
            push!(sup, missing); push!(lft, missing); push!(nn, missing)
            push!(padj, missing); push!(sig, missing)
        end
    end

    for key in sort(collect(all_keys))
        push!(assoc, key)
        has_x = haskey(keys_x, key)
        has_y = haskey(keys_y, key)
        fill_metrics!(x_support, x_lift, x_N, x_padj, x_sig, keys_x, rules_x, key)
        fill_metrics!(y_support, y_lift, y_N, y_padj, y_sig, keys_y, rules_y, key)

        is_group_specific = !isempty(excl) && any(s -> String(s) in excl, _key_items(key))
        cat = if is_group_specific
            :group_specific_item
        elseif has_x && has_y
            :universal
        elseif has_x
            Symbol(px * "_only")
        else
            Symbol(py * "_only")
        end
        push!(category, cat)
    end

    return DataFrame(
        "association" => assoc, "category" => category,
        "$(px)_support" => x_support, "$(px)_lift" => x_lift, "$(px)_N" => x_N,
        "$(px)_p_adjusted" => x_padj, "$(px)_significant" => x_sig,
        "$(py)_support" => y_support, "$(py)_lift" => y_lift, "$(py)_N" => y_N,
        "$(py)_p_adjusted" => y_padj, "$(py)_significant" => y_sig,
    )
end
