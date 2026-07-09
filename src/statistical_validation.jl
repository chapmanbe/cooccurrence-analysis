# ──────────────────────────────────────────────────────────────────────────────
# Statistical validation of item co-occurrence associations
# ──────────────────────────────────────────────────────────────────────────────

"""
    build_contingency_table(item_a, item_b, record_items) -> Matrix{Int}

Build a 2×2 contingency table for two items across all records.

                Has item_b    No item_b
    Has item_a  [n11           n10]
    No item_a   [n01           n00]

Uses the full record population to correctly assess whether co-occurrence
exceeds chance expectation.
"""
function build_contingency_table(item_a::AbstractString, item_b::AbstractString,
                                 record_items::AbstractVector)
    n11 = n10 = n01 = n00 = 0
    for items in record_items
        has_a = item_a in items
        has_b = item_b in items
        if has_a && has_b
            n11 += 1
        elseif has_a && !has_b
            n10 += 1
        elseif !has_a && has_b
            n01 += 1
        else
            n00 += 1
        end
    end
    return [n11 n10; n01 n00]
end

"""
    test_association(item_a, item_b, record_items;
                     test::Symbol=:fisher)

Test whether two items co-occur more than expected by chance.

# Arguments
- `record_items`: Vector of item lists, one per record (full population)
- `test`: `:fisher` for Fisher's exact test, `:chisq` for chi-square test

# Returns
NamedTuple with: `observed`, `expected`, `odds_ratio`, `p_value`, `test_type`
"""
function test_association(item_a::AbstractString, item_b::AbstractString,
                          record_items::AbstractVector;
                          test::Symbol=:fisher)
    ct = build_contingency_table(item_a, item_b, record_items)
    n = sum(ct)
    observed = ct[1, 1]

    # Expected count under independence
    row_a = ct[1, 1] + ct[1, 2]
    col_b = ct[1, 1] + ct[2, 1]
    expected = (row_a * col_b) / n

    # Odds ratio
    or_val = (ct[1,1] * ct[2,2]) / max(ct[1,2] * ct[2,1], 1)

    # Statistical test
    if test == :fisher
        ft = FisherExactTest(ct[1,1], ct[1,2], ct[2,1], ct[2,2])
        p = pvalue(ft; tail=:right)  # One-sided: testing for positive association
    elseif test == :chisq
        cst = ChisqTest(ct)
        p = pvalue(cst)
    else
        error("Unknown test type: $test. Use :fisher or :chisq")
    end

    return (observed=observed, expected=expected, odds_ratio=or_val,
            p_value=p, test_type=test)
end

"""
    adjust_pvalues(p_values::Vector{Float64}; method::Symbol=:bh) -> Vector{Float64}

Apply multiple testing correction.

# Methods
- `:bh` — Benjamini-Hochberg (FDR control)
- `:bonferroni` — Bonferroni correction
- `:holm` — Holm step-down
"""
function adjust_pvalues(p_values::Vector{Float64}; method::Symbol=:bh)
    isempty(p_values) && return Float64[]
    correction = if method == :bh
        BenjaminiHochberg()
    elseif method == :bonferroni
        Bonferroni()
    elseif method == :holm
        Holm()
    else
        error("Unknown correction method: $method. Use :bh, :bonferroni, or :holm")
    end
    return MultipleTesting.adjust(p_values, correction)
end

"""
    _extract_item_pairs(rules_df::DataFrame) -> Vector{Tuple{String, String}}

Extract unique item pairs from rules or itemsets DataFrame.
For rules with LHS/RHS: pairs each LHS item with RHS.
For itemsets: generates all pairwise combinations.
"""
function _extract_item_pairs(rules_df::DataFrame)
    pairs = Set{Tuple{String, String}}()
    if hasproperty(rules_df, :LHS) && hasproperty(rules_df, :RHS)
        for row in eachrow(rules_df)
            rhs = row.RHS
            for lhs_item in row.LHS
                pair = lhs_item < rhs ? (lhs_item, rhs) : (rhs, lhs_item)
                push!(pairs, pair)
            end
        end
    elseif hasproperty(rules_df, :Itemset)
        for row in eachrow(rules_df)
            items = sort(row.Itemset)
            for i in 1:length(items), j in (i+1):length(items)
                push!(pairs, (items[i], items[j]))
            end
        end
    end
    return collect(pairs)
end

"""
    validate_rules(rules_df::DataFrame, event_df::DataFrame;
                   test::Symbol=:fisher,
                   correction::Symbol=:bh,
                   alpha::Float64=0.05) -> DataFrame

Augment rules/itemsets with statistical significance tests.

Builds contingency tables against the **full record population** and applies
multiple testing correction. Adds columns: `item_pair`, `observed`, `expected`,
`odds_ratio`, `p_value`, `p_adjusted`, `significant`.

!!! note "Correction family is selection-conditioned"
    The BH (or Bonferroni/Holm) correction here runs only over the pairs that
    survived rule mining (support/confidence thresholds), **not** over the full
    set of candidate item pairs. The adjusted p-values are therefore optimistic
    and should be read as an *exploratory ranking aid*, not as strict
    family-wise inference. Confirmatory significance for the manuscript rests on
    the pre-specified, fixed-family pair-recovery path (`known_pair_recovery.jl`,
    26 literature pairs), which is not selection-conditioned. To obtain a
    non-optimistic family here, pass a `rules_df` covering the full candidate
    pair set rather than only mined survivors.

# Arguments
- `rules_df`: Output from `mine_association_rules` or `mine_frequent_itemsets`
- `event_df`: Full event-level DataFrame (all records, not just multi-item)
- `test`: `:fisher` or `:chisq`
- `correction`: `:bh`, `:bonferroni`, or `:holm`
- `alpha`: Significance threshold after correction (default 0.05)
"""
function validate_rules(rules_df::DataFrame, event_df::DataFrame;
                        test::Symbol=:fisher,
                        correction::Symbol=:bh,
                        alpha::Float64=0.05)
    nrow(rules_df) == 0 && return rules_df

    # Build full record item lists once
    gp = groupby(event_df, :id)
    record_items = [sort(unique(g.item)) for g in gp]

    # Extract unique pairs to test
    pairs = _extract_item_pairs(rules_df)

    # Run tests for all unique pairs
    pair_results = Dict{Tuple{String,String}, NamedTuple}()
    for (a, b) in pairs
        pair_results[(a, b)] = test_association(a, b, record_items; test)
    end

    # Apply multiple testing correction across all pairs
    all_pvals = [pair_results[p].p_value for p in pairs]
    adj_pvals = adjust_pvalues(all_pvals; method=correction)
    for (i, p) in enumerate(pairs)
        old = pair_results[p]
        pair_results[p] = (observed=old.observed, expected=old.expected,
                           odds_ratio=old.odds_ratio, p_value=old.p_value,
                           p_adjusted=adj_pvals[i], test_type=old.test_type)
    end

    # Augment each rule with its test result
    # For multi-item rules, use the pair with the highest (worst) p-value
    item_pairs = String[]
    observed = Int[]
    expected = Float64[]
    odds_ratios = Float64[]
    p_values = Float64[]
    p_adjusted = Float64[]

    for row in eachrow(rules_df)
        row_pairs = if hasproperty(rules_df, :LHS) && hasproperty(rules_df, :RHS)
            rhs = row.RHS
            [(lhs < rhs ? (lhs, rhs) : (rhs, lhs)) for lhs in row.LHS]
        else
            items = sort(row.Itemset)
            [(items[i], items[j])
             for i in 1:length(items) for j in (i+1):length(items)]
        end

        # Most conservative (highest) p-value among constituent pairs
        worst_idx = argmax([pair_results[p].p_adjusted for p in row_pairs])
        result = pair_results[row_pairs[worst_idx]]

        push!(item_pairs, join([format_itemset([p...]) for p in row_pairs], "; "))
        push!(observed, result.observed)
        push!(expected, round(result.expected, digits=1))
        push!(odds_ratios, round(result.odds_ratio, digits=3))
        push!(p_values, result.p_value)
        push!(p_adjusted, result.p_adjusted)
    end

    result = copy(rules_df)
    result.item_pair = item_pairs
    result.observed = observed
    result.expected = expected
    result.odds_ratio = odds_ratios
    result.p_value = p_values
    result.p_adjusted = p_adjusted
    result.significant = p_adjusted .< alpha

    return result
end
