# ─────────────────────────────────────────────────────────────────────────────
# Experiment: does the "no universal cluster" result survive removing the
# structural-zero group-specific items from the shared HDP atom pool?
#
# Baseline   = manuscript fit (group-only items retained; opposite-group items dropped
#              per stratum, own-group items kept → structural zeros in the shared
#              atom matrix). Settings copied verbatim from run_real_data_analysis.jl.
# Experiment = same fit but ALL group-specific items (GROUP_A_ONLY ∪ GROUP_B_ONLY)
#              removed from every record first, so the shared atom matrix has
#              no structural-zero columns. Cohort = records with ≥2 distinct
#              *group-neutral* events.
#
# Question: baseline reports 0 universal clusters. Does the experiment also
# report 0 universal clusters? If yes → group-specificity is robust. If the
# experiment now yields universal clusters → the headline was an artifact of
# the structural zeros.
# ─────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate("/Users/brainchapman/Code/Julia/cooccurrence-analysis")
push!(LOAD_PATH, "/Users/brainchapman/Code/Julia/cooccurrence-analysis/src")

using CooccurrenceAnalysis
using DataFrames, Random, Printf

const M = CooccurrenceAnalysis   # for non-exported GROUP_A_ONLY_ITEMS / GROUP_B_ONLY_ITEMS
const DATA_PATH = "/Users/brainchapman/Code/Julia/CooccurrenceAnalysis/odata/real_data.arrow"

const HDP_KW = (
    group_by = :Group, K_max = 20, prior = :empirical_bayes,
    concentration = 10.0, floor = 1.0, alpha = 1.0, gamma = 1.0,
    n_init = 3, max_iter = 300, tol = 1e-5,
)

function summarize_fit(name, res)
    hdp = res.hdp_result
    J   = hdp.n_groups
    labels = hdp.group_labels
    cat = hdp_cluster_categorization(res; beta_threshold = 0.01)
    active = filter(r -> r.category != :negligible, cat)

    println("\n" * "="^78)
    println("## $name")
    println("="^78)
    @printf "  Records: %d  (%s)\n" sum(res.n_records_per_group) join(
        ["$(labels[j])=$(res.n_records_per_group[j])" for j in 1:J], ", ")
    @printf "  Items in matrix: %d\n" length(res.item_names)
    @printf "  Effective K: %d   Converged: %s (%d iters)   ELBO: %.1f\n" hdp.effective_K hdp.converged hdp.n_iter hdp.elbo

    # Category tally among active clusters
    counts = Dict{Symbol,Int}()
    for c in active.category
        counts[c] = get(counts, c, 0) + 1
    end
    println("  Active clusters (β>0.01): ", nrow(active))
    for (c, n) in sort(collect(counts), by = x -> string(x[1]))
        println("     $c: $n")
    end
    n_universal = get(counts, :universal, 0) + get(counts, :both_present_but_unequal, 0)
    @printf "  >>> universal-ish clusters (universal + both_present_but_unequal): %d\n" n_universal

    # Per-cluster detail, sorted by β
    println("\n  Per active cluster (top items by θ):")
    order = sortperm(hdp.beta_mean, rev = true)
    for k in order
        hdp.beta_mean[k] < 0.01 && continue
        pis   = [hdp.pi_mean[j, k] for j in 1:J]
        ratio = maximum(pis) / max(minimum(pis), 1e-8)
        topd  = sortperm(hdp.theta[k, :], rev = true)[1:min(5, length(res.item_names))]
        items = join(["$(res.item_names[d])=$(round(hdp.theta[k,d], digits=2))" for d in topd], ", ")
        pistr = join(["π_$(labels[j])=$(round(pis[j], digits=3))" for j in 1:J], " ")
        catk  = cat[cat.cluster .== k, :category][1]
        @printf "   k=%2d β=%.3f %s ratio=%.1f [%s]\n        %s\n" k hdp.beta_mean[k] pistr ratio catk items
    end
    return active
end

println("═══ Loading data ═══")
event_df = load_event_data(DATA_PATH)
@printf "  rows=%d  records=%d\n" nrow(event_df) length(unique(event_df.id))

group_items = union(M.GROUP_A_ONLY_ITEMS, M.GROUP_B_ONLY_ITEMS)
println("  Excluding group-specific items: ", join(sort(collect(group_items)), ", "))

# ── Baseline (manuscript fit) ────────────────────────────────────────────────
println("\n═══ Fitting BASELINE (group-only items retained) ═══")
base = hdp_clustering(event_df; HDP_KW..., rng = MersenneTwister(2026))
summarize_fit("BASELINE — group-only items RETAINED (structural zeros present)", base)

# ── Experiment (group-only items excluded) ─────────────────────────────────────────
println("\n═══ Fitting EXPERIMENT (group-only items excluded) ═══")
event_neutral = filter(r -> !(r.item in group_items), event_df)
@printf "  rows after dropping group-specific events: %d (was %d)\n" nrow(event_neutral) nrow(event_df)
exp = hdp_clustering(event_neutral; HDP_KW..., rng = MersenneTwister(2026))
summarize_fit("EXPERIMENT — group-only items EXCLUDED (no structural zeros)", exp)

println("\n═══ DONE ═══")
