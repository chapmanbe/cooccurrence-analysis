# Phase-2 performance benchmark (P1 pairwise associations, P2 HDP CAVI hoist).
#
# Run from the parent directory with the project active:
#   julia --project=cooccurrence-analysis cooccurrence-analysis/scripts/benchmark_phase2.jl
#
# Generates a ~10K-record synthetic event dataset, then times:
#   1. compute_pairwise_associations (matrix path) vs. a local O(M^2*N) scan
#      baseline reproducing the pre-P1 double-scan, to show the speedup.
#   2. A single HDP CAVI fit (post-P2 hoist) on a 10K subset, reporting time and
#      allocations (the hoist removes 3 N*D allocations per iteration).
using CooccurrenceAnalysis, DataFrames, Random

# ── Synthetic data: N records, M items, planted block co-occurrence ────────────
function synth_events(; N=10_000, M=60, seed=1)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    blocks = [collect(b:min(b+2, M)) for b in 1:3:M]  # overlapping item blocks
    for id in 1:N
        g = rand(rng, ("A", "B"))
        blk = rand(rng, blocks)
        for it in blk
            rand(rng) < 0.85 && push!(rows, (id=id, seq=1, Group=g,
                                             item="Item$(lpad(it,2,'0'))", year=2010))
        end
        # a little singleton noise
        rand(rng) < 0.3 && push!(rows, (id=id, seq=2, Group=g,
                                        item="Item$(lpad(rand(rng,1:M),2,'0'))", year=2010))
    end
    return DataFrame(rows)
end

# Local reproduction of the pre-P1 scan path (double record scan per pair).
function pairwise_scan(event_df)
    gp = groupby(event_df, :id)
    record_items = [sort(unique(g.item)) for g in gp]
    all_items = sort(unique(reduce(vcat, record_items)))
    prevalence = Dict(it => count(ps -> it in ps, record_items) for it in all_items)
    rows = NamedTuple[]
    for i in 1:length(all_items), j in (i+1):length(all_items)
        a, b = all_items[i], all_items[j]
        ct = build_contingency_table(a, b, record_items)  # scan 1
        ct[1,1] == 0 && continue
        result = test_association(a, b, record_items; test=:fisher)  # scan 2
        push!(rows, (item_a=a, item_b=b, observed=ct[1,1], p_value=result.p_value))
    end
    return rows
end

event_df = synth_events()
println("Synthetic dataset: $(nrow(event_df)) event rows, ",
        "$(length(unique(event_df.id))) records, ",
        "$(length(unique(event_df.item))) items")

# Warm up (compile), then time.
compute_pairwise_associations(event_df)
pairwise_scan(event_df)

t_matrix = @elapsed compute_pairwise_associations(event_df)
t_scan   = @elapsed pairwise_scan(event_df)
println("\n── P1: pairwise associations (10K records) ──")
println("  scan baseline (double record scan): $(round(t_scan, digits=3)) s")
println("  matrix path (X'X)                 : $(round(t_matrix, digits=3)) s")
println("  speedup                           : $(round(t_scan / t_matrix, digits=1))x")

# ── P2: HDP CAVI on a 10K subset ───────────────────────────────────────────────
txns = build_transactions(event_df; min_items=2)
X = BitMatrix(Matrix(txns))
grp = ones(Int, size(X, 1))
# assign groups by first-seen record group
recgroup = Dict{Int,String}()
for r in eachrow(event_df); get!(recgroup, r.id, r.Group); end
# (group vector length must match X rows; rebuild via wrapper instead)
res = hdp_clustering(event_df; group_by=:Group, K_max=10, n_init=1, rng=MersenneTwister(1))  # warm up
stats = @timed hdp_clustering(event_df; group_by=:Group, K_max=10, n_init=1, rng=MersenneTwister(1))
println("\n── P2: HDP CAVI fit (10K records, K_max=10, n_init=1) ──")
println("  time       : $(round(stats.time, digits=3)) s")
println("  allocations: $(round(stats.bytes / 1e6, digits=1)) MB")
println("  (the Xf hoist removes 3 N*D float allocations per CAVI iteration)")
