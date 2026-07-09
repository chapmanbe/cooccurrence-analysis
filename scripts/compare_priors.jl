# Compare flat-prior vs empirical-Bayes-prior Bernoulli mixture clustering on
# the tabular synthetic data. Reports cluster sizes, BIC values, top items per
# cluster, and Adjusted Rand Index between the two assignments.
#
# Usage: julia --project=clustering clustering/scripts/compare_priors.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis
using DataFrames
using Random
using Printf

const DATA_PATH = joinpath(@__DIR__, "..", "..", "ddata", "synthetic_events.arrow")

function adjusted_rand_index(a::Vector{Int}, b::Vector{Int})
    length(a) == length(b) || error("length mismatch")
    n = length(a)
    a_levels = sort(unique(a))
    b_levels = sort(unique(b))
    ct = zeros(Int, length(a_levels), length(b_levels))
    a_idx = Dict(v => i for (i, v) in enumerate(a_levels))
    b_idx = Dict(v => i for (i, v) in enumerate(b_levels))
    for i in 1:n
        ct[a_idx[a[i]], b_idx[b[i]]] += 1
    end
    sum_comb = x -> x < 2 ? 0 : x * (x - 1) ÷ 2
    sum_ij = sum(sum_comb.(ct))
    sum_ai = sum(sum_comb.(sum(ct, dims=2)))
    sum_bj = sum(sum_comb.(sum(ct, dims=1)))
    total = sum_comb(n)
    expected = sum_ai * sum_bj / total
    max_index = (sum_ai + sum_bj) / 2
    return (sum_ij - expected) / (max_index - expected)
end

function top_items(profile_row::DataFrameRow, item_names::Vector{String}; n=5)
    vals = [(s, profile_row[s]) for s in item_names]
    sort!(vals, by=last, rev=true)
    return first(vals, min(n, length(vals)))
end

function summarize(label::String, result::CooccurrenceAnalysisResult)
    best = result.model_selection.best
    K = best.K
    println("\n──── $label ────")
    println("  Best K: $K (BIC = $(round(best.bic, digits=1)))")
    println("  Records clustered: $(result.n_records)")
    println("  BIC across K:")
    for (k, bic) in result.model_selection.bic_values
        marker = k == K ? " ←" : ""
        println("    K=$k: $(round(bic, digits=1))$marker")
    end
    println("  Class profiles (top 5 items, P ≥ 0.05):")
    items = result.item_names
    for k in 1:K
        pct = round(best.pi[k] * 100, digits=1)
        println("    Class $k ($pct% of records):")
        top = top_items(result.class_profiles[k, :], items; n=10)
        shown = 0
        for (s, p) in top
            p < 0.05 && break
            @printf "      %-40s %.3f\n" s p
            shown += 1
            shown >= 5 && break
        end
        shown == 0 && println("      (no items above 0.05)")
    end
end

function main()
    println("Loading $(DATA_PATH)...")
    event_df = load_event_data(DATA_PATH)
    n_total = length(unique(event_df.id))
    println("  Total records: $n_total")
    println("  Total event rows: $(nrow(event_df))")

    txns = build_transactions(event_df; min_items=2)
    println("  Multi-item records (≥2): $(nrow(txns))")
    println("  Items in transactions: $(ncol(txns))")
    sparsity = sum(Matrix(txns)) / (nrow(txns) * ncol(txns))
    println("  Mean items per record: $(round(sum(Matrix(txns)) / nrow(txns), digits=2))")
    println("  Density: $(round(sparsity * 100, digits=2))%")

    println("\n═══ Running EM with flat prior (K = 2:6) ═══")
    flat = @time bernoulli_clustering(event_df;
        K_range=2:6, prior=:flat, n_init=5,
        rng=MersenneTwister(2026))

    println("\n═══ Running EM with empirical-Bayes prior (K = 2:6) ═══")
    eb = @time bernoulli_clustering(event_df;
        K_range=2:6, prior=:empirical_bayes, concentration=10.0, floor=1.0,
        n_init=5, rng=MersenneTwister(2026))

    summarize("Flat prior", flat)
    summarize("Empirical Bayes prior (c=10)", eb)

    # Compare assignments — only meaningful if same K and same record set
    if flat.model_selection.best_K == eb.model_selection.best_K &&
       flat.n_records == eb.n_records
        ari = adjusted_rand_index(flat.record_assignments.assignment,
                                   eb.record_assignments.assignment)
        println("\n══ Comparison ══")
        println("  Adjusted Rand Index (flat vs EB): $(round(ari, digits=4))")
        println("  (1.0 = identical clustering, 0.0 = chance agreement)")

        # Per-class assignment overlap
        K = flat.model_selection.best_K
        ct = zeros(Int, K, K)
        for (a, b) in zip(flat.record_assignments.assignment,
                          eb.record_assignments.assignment)
            ct[a, b] += 1
        end
        println("\n  Confusion matrix (rows=flat, cols=EB):")
        for k in 1:K
            print("    flat K=$k:")
            for j in 1:K
                @printf " %5d" ct[k, j]
            end
            println()
        end
    else
        println("\n══ Comparison skipped ══")
        println("  Flat best_K = $(flat.model_selection.best_K), EB best_K = $(eb.model_selection.best_K)")
    end
end

main()
