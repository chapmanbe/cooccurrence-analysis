# Task 1 (review revisions): regenerate the network-centrality figure with the
# corrected (topological) betweenness, and print the corrected top-20 rankings
# so we can decide whether draft §4.4 survives.
#
# Network path only — no HDP fit. Mirrors the network settings in
# run_real_data_analysis.jl.
#
#   julia --project=clustering clustering/scripts/regen_figure8.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis
using DataFrames, Printf, CairoMakie, Graphs

const DATA_PATH  = joinpath(@__DIR__, "..", "..", "odata", "real_data.arrow")
const OUTPUT_DIR = joinpath(@__DIR__, "output", "real")

mkpath(OUTPUT_DIR)

println("═══ Loading real tabular data ═══")
event_df = load_event_data(DATA_PATH)

println("═══ Building co-occurrence network ═══")
net = build_cooccurrence_network(event_df;
    weight_metric = :phi,
    min_count     = 30,
    alpha         = 0.05,
    test          = :fisher,
    correction    = :bh)
comm = detect_communities(net; method=:louvain)
metrics = compute_network_metrics(net)

@printf "  Nodes: %d  Edges: %d\n" nv(net.graph) ne(net.graph)

function show_ranking(df, col, k=20)
    n = min(k, nrow(df))
    ord = sortperm(Float64.(df[!, col]), rev=true)[1:n]
    println("\n─── Top $n by $col ───")
    for (rank, i) in enumerate(ord)
        @printf "  %2d. %-28s %s=%.4f  deg=%d  str=%.3f  prev=%d\n" rank df.item[i] col Float64(df[i, col]) df.degree[i] df.strength[i] df.prevalence[i]
    end
end

show_ranking(metrics, :betweenness)
show_ranking(metrics, :degree)
show_ranking(metrics, :strength)

# Regenerate Figure 8 on betweenness (what §4.4 and the caption discuss).
fig_cent = plot_centrality_barchart(metrics; centrality=:betweenness, top_n=20)
save(joinpath(OUTPUT_DIR, "centrality.png"), fig_cent; px_per_unit=2)
println("\n  wrote centrality.png (betweenness)")

println("Done.")
