# ──────────────────────────────────────────────────────────────────────────────
# Constants and helper functions
# ──────────────────────────────────────────────────────────────────────────────

"Items that occur only in group A"
const GROUP_A_ONLY_ITEMS = Set(["GA1", "GA2", "GA3", "GA4"])

"Items that occur only in group B"
const GROUP_B_ONLY_ITEMS = Set([
    "GB1", "GB2", "GB3",
    "GB4", "GB5", "GB6"
])

"""
    format_itemset(items) -> String

Pretty-print an itemset as "Item A + Item B + ...".
"""
format_itemset(items) = join(sort(collect(items)), " + ")

"""
    significance_stars(p::Float64) -> String

Return significance stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.
"""
function significance_stars(p::Float64)
    p < 0.001 && return "***"
    p < 0.01  && return "**"
    p < 0.05  && return "*"
    return "ns"
end

"""
    results_summary(rules_df::DataFrame; top_n::Int=20)

Print a formatted summary of the top rules by lift.
"""
function results_summary(rules_df::DataFrame; top_n::Int=20)
    nrow(rules_df) == 0 && (println("No rules found."); return)

    sorted = sort(rules_df, :Lift, rev=true)
    n = min(top_n, nrow(sorted))
    println("Top $n association rules by lift:")
    println("─"^100)

    for i in 1:n
        r = sorted[i, :]
        lhs = format_itemset(r.LHS)
        rhs = r.RHS
        sig = hasproperty(sorted, :p_adjusted) ?
              " $(significance_stars(r.p_adjusted))" : ""
        padj = hasproperty(sorted, :p_adjusted) ?
               @sprintf(" | p_adj=%.2e", r.p_adjusted) : ""
        @printf("  %2d. %s → %s | sup=%.4f conf=%.3f lift=%.2f N=%d%s%s\n",
                i, lhs, rhs, r.Support, r.Confidence, r.Lift, r.N, padj, sig)
    end
end
