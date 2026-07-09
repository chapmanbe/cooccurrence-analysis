# ──────────────────────────────────────────────────────────────────────────────
# Group-stratified analysis and cross-strata comparison
# ──────────────────────────────────────────────────────────────────────────────

"""
    stratified_analysis(event_df::DataFrame;
                        min_support::Float64=0.005,
                        min_confidence::Float64=0.1,
                        min_count::Union{Int, Nothing}=30,
                        max_length::Int=4,
                        test::Symbol=:fisher,
                        correction::Symbol=:bh) -> NamedTuple

Run the association rule mining pipeline separately for group_a and group_b cohorts.

# Returns
NamedTuple with fields:
- `group_a_rules`, `group_b_rules`: Validated rule DataFrames
- `group_a_itemsets`, `group_b_itemsets`: Frequent itemset DataFrames
- `group_a_n_records`, `group_b_n_records`: Multi-item record counts per group
"""
function stratified_analysis(event_df::DataFrame;
                             min_support::Float64=0.005,
                             min_confidence::Float64=0.1,
                             min_count::Union{Int, Nothing}=30,
                             max_length::Int=4,
                             test::Symbol=:fisher,
                             correction::Symbol=:bh,
                             timing_filter::Symbol=:all,
                             concurrent_window::Int=0)
    results = Dict{String, Any}()

    for group in ["A", "B"]
        println("── Analyzing $group cohort ──")

        # Build group-filtered transactions
        txns_df = build_transactions(event_df; group_filter=group, min_items=2,
                                      timing_filter, concurrent_window)
        n_records = nrow(txns_df)
        println("  Multi-item records: $n_records")

        if n_records == 0
            results["group_$(lowercase(group))_rules"] = DataFrame()
            results["group_$(lowercase(group))_itemsets"] = DataFrame()
            results["group_$(lowercase(group))_n_records"] = 0
            continue
        end

        # Mine itemsets and rules
        itemsets = mine_frequent_itemsets(txns_df;
            min_support, min_count, max_length)
        rules = mine_association_rules(txns_df;
            min_support, min_confidence, min_count, max_length)
        println("  Frequent itemsets: $(nrow(itemsets)), Rules: $(nrow(rules))")

        # Filter to group-appropriate event data for validation
        group_event_df = filter(row -> row.Group == group, event_df)
        if group == "A"
            group_event_df = filter(row -> !(row.item in GROUP_B_ONLY_ITEMS), group_event_df)
        else
            group_event_df = filter(row -> !(row.item in GROUP_A_ONLY_ITEMS), group_event_df)
        end

        # Validate with statistical tests
        if nrow(rules) > 0
            rules = validate_rules(rules, group_event_df; test, correction)
            sig_count = count(rules.significant)
            println("  Significant rules (after $correction correction): $sig_count")
        end

        prefix = "group_" * lowercase(group)
        results["$(prefix)_rules"] = rules
        results["$(prefix)_itemsets"] = itemsets
        results["$(prefix)_n_records"] = n_records
    end

    return (group_a_rules=results["group_a_rules"],
            group_b_rules=results["group_b_rules"],
            group_a_itemsets=results["group_a_itemsets"],
            group_b_itemsets=results["group_b_itemsets"],
            group_a_n_records=results["group_a_n_records"],
            group_b_n_records=results["group_b_n_records"])
end

"""
    _normalize_rule_key(lhs, rhs) -> String

Create a canonical string key for a rule, independent of direction.
Sorts all items alphabetically to enable cross-strata matching.
"""
function _normalize_rule_key(lhs, rhs)
    all_items = sort(vcat(collect(lhs), [rhs]))
    return join(all_items, " + ")
end

"""
    compare_strata(group_a_rules::DataFrame, group_b_rules::DataFrame) -> DataFrame

Compare rules found in group_a vs group_b cohorts. Categorizes each discovered
association as `:universal`, `:group_a_only`, `:group_b_only`, or
`:group_specific_item`.

# Returns
DataFrame with columns for both group_a and group_b metrics plus a `category` column.
"""
function compare_strata(group_a_rules::DataFrame, group_b_rules::DataFrame)
    # Build lookup by canonical key
    group_a_keys = Dict{String, DataFrameRow}()
    for row in eachrow(group_a_rules)
        key = _normalize_rule_key(row.LHS, row.RHS)
        # Keep the rule with higher lift if duplicate keys
        if !haskey(group_a_keys, key) || row.Lift > group_a_keys[key].Lift
            group_a_keys[key] = row
        end
    end

    group_b_keys = Dict{String, DataFrameRow}()
    for row in eachrow(group_b_rules)
        key = _normalize_rule_key(row.LHS, row.RHS)
        if !haskey(group_b_keys, key) || row.Lift > group_b_keys[key].Lift
            group_b_keys[key] = row
        end
    end

    all_keys = union(keys(group_a_keys), keys(group_b_keys))

    rows = Dict{String, Vector{Any}}(
        "association" => String[],
        "group_a_support" => Union{Missing, Float64}[],
        "group_a_lift" => Union{Missing, Float64}[],
        "group_a_N" => Union{Missing, Int}[],
        "group_a_p_adjusted" => Union{Missing, Float64}[],
        "group_a_significant" => Union{Missing, Bool}[],
        "group_b_support" => Union{Missing, Float64}[],
        "group_b_lift" => Union{Missing, Float64}[],
        "group_b_N" => Union{Missing, Int}[],
        "group_b_p_adjusted" => Union{Missing, Float64}[],
        "group_b_significant" => Union{Missing, Bool}[],
        "category" => Symbol[]
    )

    for key in sort(collect(all_keys))
        push!(rows["association"], key)

        has_group_a = haskey(group_a_keys, key)
        has_group_b = haskey(group_b_keys, key)

        # Group A metrics
        if has_group_a
            mr = group_a_keys[key]
            push!(rows["group_a_support"], mr.Support)
            push!(rows["group_a_lift"], mr.Lift)
            push!(rows["group_a_N"], mr.N)
            if hasproperty(group_a_rules, :p_adjusted)
                push!(rows["group_a_p_adjusted"], mr.p_adjusted)
                push!(rows["group_a_significant"], mr.significant)
            else
                push!(rows["group_a_p_adjusted"], missing)
                push!(rows["group_a_significant"], missing)
            end
        else
            push!(rows["group_a_support"], missing)
            push!(rows["group_a_lift"], missing)
            push!(rows["group_a_N"], missing)
            push!(rows["group_a_p_adjusted"], missing)
            push!(rows["group_a_significant"], missing)
        end

        # Group B metrics
        if has_group_b
            fr = group_b_keys[key]
            push!(rows["group_b_support"], fr.Support)
            push!(rows["group_b_lift"], fr.Lift)
            push!(rows["group_b_N"], fr.N)
            if hasproperty(group_b_rules, :p_adjusted)
                push!(rows["group_b_p_adjusted"], fr.p_adjusted)
                push!(rows["group_b_significant"], fr.significant)
            else
                push!(rows["group_b_p_adjusted"], missing)
                push!(rows["group_b_significant"], missing)
            end
        else
            push!(rows["group_b_support"], missing)
            push!(rows["group_b_lift"], missing)
            push!(rows["group_b_N"], missing)
            push!(rows["group_b_p_adjusted"], missing)
            push!(rows["group_b_significant"], missing)
        end

        # Categorize
        items = split(key, " + ")
        is_group_specific = any(s -> s in GROUP_A_ONLY_ITEMS || s in GROUP_B_ONLY_ITEMS, items)
        if is_group_specific
            push!(rows["category"], :group_specific_item)
        elseif has_group_a && has_group_b
            push!(rows["category"], :universal)
        elseif has_group_a
            push!(rows["category"], :group_a_only)
        else
            push!(rows["category"], :group_b_only)
        end
    end

    col_order = ["association", "category",
                 "group_a_support", "group_a_lift", "group_a_N",
                 "group_a_p_adjusted", "group_a_significant",
                 "group_b_support", "group_b_lift", "group_b_N",
                 "group_b_p_adjusted", "group_b_significant"]
    return DataFrame([col => rows[col] for col in col_order])
end
