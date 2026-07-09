#!/usr/bin/env julia
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#   kernelspec:
#     display_name: Julia
#     language: julia
#     name: julia
# ---

# %% [markdown]
# # Item Co-occurrence Analysis: Association Rule Mining
#
# Discovers item combinations that co-occur more than expected by chance
# in the tabular synthetic dataset using FP-Growth frequent itemset mining with
# Fisher's exact test validation and Benjamini-Hochberg FDR correction.

# %%
using Pkg
Pkg.activate(@__DIR__)

# %%
using Revise

# %%
using CooccurrenceAnalysis
using DataFrames

# %% [markdown]
# ## Load data and population overview

# %%
data_path = joinpath(@__DIR__, "..", "ddata", "synthetic_events.arrow")
event_df = load_event_data(data_path)
println("Total event rows: $(nrow(event_df))")
println("Total records: $(length(unique(event_df.id)))")

# %%
# Record-level summary
ps = get_record_summary(event_df)
println("Item count distribution:")
for n in sort(unique(ps.n_items))
    pct = round(count(==(n), ps.n_items) / nrow(ps) * 100, digits=2)
    println("  $n item(s): $pct%")
end

# %% [markdown]
# ## Full population association rule mining
#
# We mine frequent itemsets from multi-item records (≥2 distinct items) using
# FP-Growth, then generate association rules. Rules are validated against the
# **full population** using Fisher's exact test with BH correction.

# %%
# Build transactions (multi-item records only)
txns_df = build_transactions(event_df; min_items=2)
println("Multi-item records for mining: $(nrow(txns_df))")

# %%
# Mine frequent itemsets (min 30 co-occurrences)
itemsets = mine_frequent_itemsets(txns_df; min_count=30)
println("Frequent itemsets (≥2 items, ≥30 occurrences): $(nrow(itemsets))")
sort!(itemsets, :N, rev=true)
println("\nTop 20 frequent itemsets:")
for row in first(eachrow(itemsets), 20)
    println("  $(format_itemset(row.Itemset)) | N=$(row.N) sup=$(round(row.Support, digits=4))")
end

# %%
# Mine association rules
rules = mine_association_rules(txns_df; min_count=30, min_confidence=0.1)
println("Association rules: $(nrow(rules))")

# %%
# Validate with Fisher's exact test + BH correction
validated = validate_rules(rules, event_df; test=:fisher, correction=:bh)
sig_rules = filter(row -> row.significant, validated)
println("Significant rules (p_adj < 0.05): $(nrow(sig_rules)) / $(nrow(validated))")

# %%
# Top associations by lift
println("\n=== Top Significant Associations by Lift ===\n")
sig_sorted = sort(sig_rules, :Lift, rev=true)
results_summary(sig_sorted; top_n=30)

# %% [markdown]
# ## Group-stratified analysis
#
# Run ARM separately for group_a and group_b cohorts, then compare to identify
# universal vs group-specific associations.

# %%
strat = stratified_analysis(event_df; min_count=20, min_confidence=0.1)

# %%
println("\n=== Group A-specific top rules (by lift) ===")
if nrow(strat.group_a_rules) > 0
    group_a_sig = filter(row -> hasproperty(strat.group_a_rules, :significant) ?
                      row.significant : true, strat.group_a_rules)
    results_summary(sort(group_a_sig, :Lift, rev=true); top_n=15)
end

# %%
println("\n=== Group B-specific top rules (by lift) ===")
if nrow(strat.group_b_rules) > 0
    group_b_sig = filter(row -> hasproperty(strat.group_b_rules, :significant) ?
                        row.significant : true, strat.group_b_rules)
    results_summary(sort(group_b_sig, :Lift, rev=true); top_n=15)
end

# %%
# Compare strata
if nrow(strat.group_a_rules) > 0 && nrow(strat.group_b_rules) > 0
    comp = compare_strata(strat.group_a_rules, strat.group_b_rules)

    println("\n=== Cross-Strata Comparison ===")
    for cat in [:universal, :group_a_only, :group_b_only, :group_specific_item]
        sub = filter(row -> row.category == cat, comp)
        println("\n$(cat) ($(nrow(sub)) associations):")
        for row in eachrow(sub)
            ml = ismissing(row.group_a_lift) ? "-" : round(row.group_a_lift, digits=2)
            fl = ismissing(row.group_b_lift) ? "-" : round(row.group_b_lift, digits=2)
            println("  $(row.association) | group_a_lift=$ml group_b_lift=$fl")
        end
    end
end
