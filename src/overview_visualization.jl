# ──────────────────────────────────────────────────────────────────────────────
# Overview visualizations: UpSet plot and record heatmap
# ──────────────────────────────────────────────────────────────────────────────

"""
    _compute_intersections(txns::DataFrame; min_count::Int=5)

Compute set intersection data from a binary transaction DataFrame.

Returns a NamedTuple with:
- `items`: sorted item names (columns)
- `item_counts`: per-item frequency counts
- `combos`: Vector of BitVectors, each representing an item combination
- `combo_counts`: frequency of each combination
- `combo_labels`: item names for each combination
"""
function _compute_intersections(txns::DataFrame; min_count::Int=5)
    items = sort(names(txns))
    n_items = length(items)

    # Per-item frequency
    item_counts = [count(txns[!, s]) for s in items]

    # Count unique combinations
    combo_map = Dict{Vector{Bool}, Int}()
    for row in eachrow(txns)
        key = Bool[row[s] for s in items]
        combo_map[key] = get(combo_map, key, 0) + 1
    end

    # Filter by min_count and sort by count descending
    filtered = [(k, v) for (k, v) in combo_map if v >= min_count]
    sort!(filtered; by=x -> -x[2])

    combos = [x[1] for x in filtered]
    combo_counts = [x[2] for x in filtered]
    combo_labels = [items[findall(c)] for c in combos]

    return (items=items, item_counts=item_counts,
            combos=combos, combo_counts=combo_counts,
            combo_labels=combo_labels)
end

"""
    plot_upset(txns::DataFrame;
               min_count=5, max_sets=30,
               figsize=(1000, 700)) -> Figure

Custom three-panel UpSet plot for visualizing item co-occurrence patterns.

Panels:
- Bottom-left: horizontal bars showing per-item frequency
- Top-right: vertical bars showing intersection sizes (sorted by count)
- Bottom-right: dot-and-line matrix connecting items in each intersection
"""
function plot_upset(txns::DataFrame;
                     min_count::Int=5,
                     max_sets::Int=30,
                     figsize::Tuple{Int,Int}=(1000, 700))
    nrow(txns) == 0 && error("Cannot plot empty transactions DataFrame")

    data = _compute_intersections(txns; min_count)
    n_items = length(data.items)
    n_combos = min(length(data.combos), max_sets)

    combos = data.combos[1:n_combos]
    combo_counts = data.combo_counts[1:n_combos]

    fig = Figure(size=figsize)
    gl = fig[1, 1] = GridLayout()

    # ── Top-right: intersection size bars ──
    ax_bars = Axis(gl[1, 2];
        title="UpSet Plot: $(VOCAB.item) Co-occurrence Intersections",
        titlesize=16,
        ylabel="Intersection Size",
        xticks=(1:n_combos, fill("", n_combos)))
    hidedecorations!(ax_bars, label=false, ticklabels=false, ticks=false)
    hidespines!(ax_bars, :b, :r, :t)

    barplot!(ax_bars, 1:n_combos, combo_counts;
             color=:steelblue)

    # Add count labels above bars
    for (i, c) in enumerate(combo_counts)
        text!(ax_bars, Float64(i), Float64(c);
              text=string(c), fontsize=8, align=(:center, :bottom))
    end

    # ── Bottom-right: dot-and-line matrix ──
    ax_dots = Axis(gl[2, 2];
        yticks=(1:n_items, data.items),
        yticklabelsize=9,
        xticks=(1:n_combos, fill("", n_combos)))
    hidedecorations!(ax_dots, label=false, ticklabels=false, ticks=false)
    hidespines!(ax_dots)

    # Background dots (gray) — one vectorized scatter over the whole grid
    bg_x = Float64[i for i in 1:n_combos for _ in 1:n_items]
    bg_y = Float64[j for _ in 1:n_combos for j in 1:n_items]
    scatter!(ax_dots, bg_x, bg_y; color=:gray85, markersize=8)

    # Active dots (one scatter) plus per-combo connecting lines
    act_x = Float64[]
    act_y = Float64[]
    for (i, combo) in enumerate(combos)
        active = findall(combo)
        for j in active
            push!(act_x, Float64(i))
            push!(act_y, Float64(j))
        end
        if length(active) > 1
            y_min, y_max = extrema(active)
            linesegments!(ax_dots, [Float64(i), Float64(i)],
                          [Float64(y_min), Float64(y_max)];
                          color=:black, linewidth=2)
        end
    end
    scatter!(ax_dots, act_x, act_y; color=:black, markersize=10)

    ylims!(ax_dots, 0.5, n_items + 0.5)
    xlims!(ax_dots, 0.5, n_combos + 0.5)

    # ── Bottom-left: set size bars ──
    ax_sets = Axis(gl[2, 1];
        xlabel="Set Size",
        yticks=(1:n_items, fill("", n_items)))
    hidedecorations!(ax_sets, label=false, ticklabels=false, ticks=false)
    hidespines!(ax_sets, :l, :t, :r)

    barplot!(ax_sets, 1:n_items, data.item_counts;
             direction=:x, color=:gray50)

    # Reverse x-axis so bars grow leftward
    xlims!(ax_sets, maximum(data.item_counts) * 1.1, 0)
    ylims!(ax_sets, 0.5, n_items + 0.5)

    # Link y-axes
    linkyaxes!(ax_dots, ax_sets)

    # Column/row proportions
    colsize!(gl, 1, Relative(0.2))
    colsize!(gl, 2, Relative(0.8))
    rowsize!(gl, 1, Relative(0.4))
    rowsize!(gl, 2, Relative(0.6))

    return fig
end

"""
    plot_record_heatmap(txns::DataFrame;
                          max_records=200,
                          figsize=(900, 700)) -> Figure

Binary record × item heatmap. Records (rows) are sorted by number of
items descending. If there are more than `max_records`, a random sample is
taken.

Uses a two-color discrete colormap (white for absent, dark red for present).
"""
function plot_record_heatmap(txns::DataFrame;
                               max_records::Int=200,
                               figsize::Tuple{Int,Int}=(900, 700))
    nrow(txns) == 0 && error("Cannot plot empty transactions DataFrame")

    items = sort(names(txns))
    n_items = length(items)

    # Build matrix
    mat = Matrix{Int}(hcat([Int.(txns[!, s]) for s in items]...))

    # Sort by row sum descending
    row_sums = vec(sum(mat, dims=2))
    order = sortperm(row_sums, rev=true)
    mat = mat[order, :]

    # Sample if needed
    n_records = size(mat, 1)
    if n_records > max_records
        idx = sort(randperm(n_records)[1:max_records])
        mat = mat[idx, :]
        n_records = max_records
    end

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="$(VOCAB.record) × $(VOCAB.item) Matrix ($(n_records) $(VOCAB.records))",
        titlesize=16,
        xlabel="$(VOCAB.item)",
        ylabel="$(VOCAB.record) (sorted by # $(VOCAB.items))",
        xticks=(1:n_items, items),
        xticklabelrotation=π/3,
        xticklabelsize=9)

    heatmap!(ax, 1:n_items, 1:n_records, mat';
             colormap=[:white, Makie.RGB(0.7, 0.0, 0.0)])

    return fig
end

"""
    plot_item_prevalence(event_df::DataFrame;
                         top_n=30,
                         figsize=(850, 550)) -> Figure

Grouped horizontal bar chart of item prevalence.

For the `top_n` most prevalent items (by overall record count), shows the
percentage of records observed with that item in the group_b cohort (orange),
group_a cohort (blue), and overall (dark grey). Bars are sorted by overall
prevalence descending (highest at top).

`event_df` must contain columns `:id`, `:item`, `:Group`.

`colors` overrides the two group series colors as `(group_a, group_b)` — i.e. in
sorted group order. Pass it when a caller needs a specific group-to-color mapping
(for example to keep a previously published figure's colors stable).
"""
function plot_item_prevalence(event_df::DataFrame;
                              top_n::Int=30,
                              colors=(Makie.wong_colors()[1], Makie.wong_colors()[2]),
                              figsize::Tuple{Int,Int}=(850, 550))
    col_group_a = colors[1]
    col_group_b = colors[2]
    col_overall = :gray30

    # Count distinct records per item × Group
    # One row per (id, item) so we can count unique records
    unique_ps = unique(event_df[:, [:id, :item, :Group]])

    # Two-series prevalence plot: use the actual group values (no domain regex).
    # The two lowest sorted group values drive the two coloured series; the
    # overall series always covers every record. (N-group generalization is a
    # Phase 3/4 concern; this fix just makes the bars reflect real groups.)
    groupvals = sort(unique(event_df.Group))
    group_a_vals = length(groupvals) >= 1 ? [groupvals[1]] : eltype(groupvals)[]
    group_b_vals = length(groupvals) >= 2 ? [groupvals[2]] : eltype(groupvals)[]
    label_a = isempty(group_a_vals) ? "" : string(group_a_vals[1])
    label_b = isempty(group_b_vals) ? "" : string(group_b_vals[1])

    n_total_pts  = length(unique(event_df.id))
    group_b_ids  = Set(unique_ps[in.(unique_ps.Group, Ref(Set(group_b_vals))), :id])
    group_a_ids    = Set(unique_ps[in.(unique_ps.Group, Ref(Set(group_a_vals))), :id])
    n_group_b_pts = length(group_b_ids)
    n_group_a_pts   = length(group_a_ids)

    # Per-item record counts
    item_counts = combine(groupby(unique_ps, :item),
        :id => (p -> length(unique(p))) => :n_overall,
        [:id, :Group] => ((p, s) -> length(unique(p[in.(s, Ref(Set(group_b_vals)))]))) => :n_group_b,
        [:id, :Group] => ((p, s) -> length(unique(p[in.(s, Ref(Set(group_a_vals)))]))) => :n_group_a)

    sort!(item_counts, :n_overall, rev=true)
    top_items = item_counts[1:min(top_n, nrow(item_counts)), :]

    # Convert to percentages
    pct_overall = top_items.n_overall ./ n_total_pts .* 100
    pct_group_b  = top_items.n_group_b  ./ max(n_group_b_pts, 1) .* 100
    pct_group_a    = top_items.n_group_a    ./ max(n_group_a_pts, 1)   .* 100

    n = nrow(top_items)
    # y positions: 3 bars per item, grouped with gap
    bar_h   = 0.25
    spacing = 1.0
    ys_ov  = collect(1:n) .* spacing
    ys_f   = ys_ov .+ bar_h
    ys_m   = ys_ov .- bar_h

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="$(VOCAB.item) Prevalence (% of $(VOCAB.records) with each $(item_lc()))",
        titlesize=14,
        xlabel="% $(VOCAB.records)",
        yticks=(ys_ov, reverse(top_items.item)),  # top-prevalent at top
        yticklabelsize=9,
        limits=((0, max(maximum(pct_overall), maximum(pct_group_b), maximum(pct_group_a)) * 1.12),
                (0.5, n * spacing + 0.5)))

    # yticks uses reverse(items) so position 1=bottom=least prevalent, n=top=most prevalent.
    # Data must also be reversed so bar at position i matches label at position i.
    barplot!(ax, ys_ov, reverse(pct_overall); direction=:x,
             color=(col_overall, 0.7), bar_labels=nothing, label="Overall", width=bar_h)
    barplot!(ax, ys_f, reverse(pct_group_b);  direction=:x,
             color=(col_group_b, 0.8),  label=label_b,  width=bar_h)
    barplot!(ax, ys_m, reverse(pct_group_a);    direction=:x,
             color=(col_group_a, 0.8),    label=label_a,    width=bar_h)

    Legend(fig[1, 2], ax; framevisible=false, labelsize=11)
    colsize!(fig.layout, 2, Relative(0.12))

    return fig
end

"""
    plot_cooccurrence_heatmap(event_df::DataFrame;
                              group=:both,
                              min_items=2,
                              figsize=(750, 650)) -> Figure

Symmetric D×D heatmap of pairwise phi coefficients between items.

The phi coefficient φ measures the correlation between two binary variables
(presence/absence of each item across records). Only records with
at least `min_items` distinct items are included (set to 1 to use all
records for marginal counts).

`group` filters to `:group_b`, `:group_a`, or `:both` (default).

Items are ordered by greedy nearest-neighbour seriation of the phi matrix
(a cosmetic ordering that places correlated items adjacent). Cells with
|φ| > 0.02 are annotated with the rounded value.
"""
function plot_cooccurrence_heatmap(event_df::DataFrame;
                                   group::Symbol=:both,
                                   min_items::Int=2,
                                   figsize::Tuple{Int,Int}=(750, 650))
    # Filter by group. :group_a / :group_b select the first / second sorted group
    # value (no domain regex); :both uses every record. `title_str` reports the
    # group's actual value, so the figure reads in the caller's vocabulary.
    df, title_str = if group === :group_a || group === :group_b
        groupvals = sort(unique(event_df.Group))
        idx = group === :group_a ? 1 : 2
        idx > length(groupvals) && error("No group at sorted position $idx for group=$group")
        gv = groupvals[idx]
        filter(r -> r.Group == gv, event_df), string(gv)
    else
        event_df, "All $(VOCAB.records)"
    end

    # Build per-record binary matrix (all records for marginal counts)
    txns = build_transactions(df; min_items=min_items)
    isempty(txns) && error("No records with ≥$min_items items")

    items = names(txns)
    D = length(items)

    # Convert to BitMatrix for efficient column operations
    X = BitMatrix(Matrix(txns))
    N = size(X, 1)

    # Compute phi matrix
    phi = zeros(Float64, D, D)
    col_sums = Float64[sum(X[:, d]) for d in 1:D]
    for i in 1:D, j in i:D
        if i == j
            phi[i, j] = 1.0
            continue
        end
        n11 = Float64(sum(X[:, i] .& X[:, j]))
        n10 = col_sums[i] - n11
        n01 = col_sums[j] - n11
        n00 = N - n11 - n10 - n01
        denom = sqrt(col_sums[i] * col_sums[j] * (N - col_sums[i]) * (N - col_sums[j]))
        phi[i, j] = denom > 0 ? (n11 * n00 - n10 * n01) / denom : 0.0
        phi[j, i] = phi[i, j]
    end

    # Order items by greedy nearest-neighbour seriation on distance = 1 - |phi|
    dist = 1.0 .- abs.(phi)
    order = _hclust_order(dist)
    sorted_items = items[order]
    phi_sorted   = phi[order, order]

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="$(VOCAB.item) Co-occurrence (φ coefficient) — $title_str",
        titlesize=13,
        xticks=(1:D, sorted_items),
        yticks=(1:D, sorted_items),
        xticklabelrotation=π/3,
        xticklabelsize=8,
        yticklabelsize=8)

    hm = heatmap!(ax, 1:D, 1:D, phi_sorted;
                  colormap=:RdBu, colorrange=(-0.5, 0.5))
    Colorbar(fig[1, 2], hm; label="φ", width=14, labelsize=11)

    # Annotate cells with |φ| > 0.02
    for i in 1:D, j in 1:D
        v = phi_sorted[i, j]
        abs(v) > 0.02 || continue
        text!(ax, i, j;
              text=string(round(v, digits=2)),
              align=(:center, :center),
              fontsize=6,
              color=abs(v) > 0.25 ? :white : :black)
    end

    return fig
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal: greedy nearest-neighbour seriation order
# ──────────────────────────────────────────────────────────────────────────────

"""
    _hclust_order(dist::Matrix{Float64}) -> Vector{Int}

Greedy nearest-neighbour seriation: starting from item 1, repeatedly append the
nearest not-yet-placed item under `dist`. This is a cheap chain heuristic for a
visually coherent ordering — NOT true (average-linkage) hierarchical clustering,
despite what earlier comments claimed. The ordering is cosmetic (heatmap row/col
layout only), so the approximation is acceptable.
"""
function _hclust_order(dist::Matrix{Float64})
    n = size(dist, 1)
    n <= 2 && return collect(1:n)

    # Greedy nearest-neighbour chain over the distance matrix
    remaining = collect(1:n)
    order = Int[]
    push!(order, popfirst!(remaining))
    while !isempty(remaining)
        last = order[end]
        best_i = argmin([dist[last, r] for r in remaining])
        push!(order, remaining[best_i])
        deleteat!(remaining, best_i)
    end
    return order
end
