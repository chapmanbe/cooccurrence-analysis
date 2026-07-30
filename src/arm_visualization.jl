# ──────────────────────────────────────────────────────────────────────────────
# Visualization for Association Rule Mining results
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_arm_scatter(rules::DataFrame;
                      x_metric=:Support, y_metric=:Confidence,
                      color_metric=:Lift, label_top_n=5,
                      figsize=(800, 600)) -> Figure

Scatter plot of association rules: Support (x) vs Confidence (y), colored by
Lift, sized by N (number of transactions supporting the rule).

Non-significant rules (if a `significant` column exists) are shown at reduced
alpha. The top `label_top_n` rules by Lift are labeled.
"""
function plot_arm_scatter(rules::DataFrame;
                           x_metric::Symbol=:Support,
                           y_metric::Symbol=:Confidence,
                           color_metric::Symbol=:Lift,
                           label_top_n::Int=5,
                           figsize::Tuple{Int,Int}=(800, 600))
    nrow(rules) == 0 && error("Cannot plot empty rules DataFrame")

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Association Rule Quality",
        titlesize=16,
        xlabel=string(x_metric),
        ylabel=string(y_metric))

    xs = Float64.(rules[!, x_metric])
    ys = Float64.(rules[!, y_metric])
    colors = Float64.(rules[!, color_metric])

    # Size proportional to N if available, else constant
    if "N" in names(rules)
        ns = Float64.(rules.N)
        min_n, max_n = extrema(ns)
        if min_n == max_n
            sizes = fill(12.0, nrow(rules))
        else
            sizes = [6.0 + (n - min_n) / (max_n - min_n) * 18.0 for n in ns]
        end
    else
        sizes = fill(12.0, nrow(rules))
    end

    # Split significant vs non-significant for separate alpha
    if "significant" in names(rules)
        sig_mask = rules.significant
        nonsig = .!sig_mask
        if any(nonsig)
            scatter!(ax, xs[nonsig], ys[nonsig];
                color=colors[nonsig], colormap=:YlOrRd, colorrange=extrema(colors),
                markersize=sizes[nonsig], alpha=0.25)
        end
        if any(sig_mask)
            scatter!(ax, xs[sig_mask], ys[sig_mask];
                color=colors[sig_mask], colormap=:YlOrRd, colorrange=extrema(colors),
                markersize=sizes[sig_mask])
        end
    else
        scatter!(ax, xs, ys;
            color=colors, colormap=:YlOrRd,
            markersize=sizes)
    end

    Colorbar(fig[1, 2]; label=string(color_metric),
             colormap=:YlOrRd,
             limits=extrema(colors))

    # Label top rules by Lift
    top_idx = partialsortperm(colors, 1:min(label_top_n, length(colors)), rev=true)
    for i in top_idx
        lhs_str = join(sort(rules.LHS[i]), " + ")
        label = "$lhs_str → $(rules.RHS[i])"
        text!(ax, xs[i], ys[i]; text=label, fontsize=8, align=(:left, :bottom),
              offset=(4, 4))
    end

    return fig
end

"""
    plot_arm_comparison(comparison::DataFrame;
                         metric=:lift, top_n=15,
                         figsize=(900, 600)) -> Figure

Horizontal grouped bar chart comparing group_a vs group_b association rules.

Input should be the output of `compare_strata()`. Bars are grouped by
association, with group_a and group_b values side by side.
"""
function plot_arm_comparison(comparison::DataFrame;
                              metric::Symbol=:lift,
                              labels=("A", "B"),
                              top_n::Int=15,
                              figsize::Tuple{Int,Int}=(900, 600))
    nrow(comparison) == 0 && error("Cannot plot empty comparison DataFrame")

    lx, ly = string(labels[1]), string(labels[2])
    group_a_col = Symbol("group_$(lowercase(lx))_$metric")
    group_b_col = Symbol("group_$(lowercase(ly))_$metric")

    # Replace missing with 0
    group_a_vals = [ismissing(v) ? 0.0 : Float64(v) for v in comparison[!, group_a_col]]
    group_b_vals = [ismissing(v) ? 0.0 : Float64(v) for v in comparison[!, group_b_col]]

    # Sort by max of the two groups, take top_n
    max_vals = max.(group_a_vals, group_b_vals)
    order = partialsortperm(max_vals, 1:min(top_n, length(max_vals)), rev=true)

    associations = comparison.association[order]
    x_vals = group_a_vals[order]
    y_vals = group_b_vals[order]
    n = length(order)

    # Reverse for top-at-top display
    associations = reverse(associations)
    x_vals = reverse(x_vals)
    y_vals = reverse(y_vals)

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="$(VOCAB.group)-Stratified Association Comparison ($(metric))",
        titlesize=16,
        xlabel=string(metric),
        yticks=(1:n, associations),
        yticklabelsize=9)

    palette = Makie.wong_colors()

    # Grouped bars: offset first group up, second group down
    dodge_offset = 0.2
    barplot!(ax, collect(1:n) .- dodge_offset, x_vals;
             direction=:x, color=palette[1], width=0.35, label=lx)
    barplot!(ax, collect(1:n) .+ dodge_offset, y_vals;
             direction=:x, color=palette[2], width=0.35, label=ly)

    # Mark missing values with a small marker
    for (i, idx) in enumerate(reverse(order))
        if ismissing(comparison[idx, group_a_col])
            scatter!(ax, [0.0], [Float64(i) - dodge_offset];
                     marker=:xcross, color=:gray50, markersize=8)
        end
        if ismissing(comparison[idx, group_b_col])
            scatter!(ax, [0.0], [Float64(i) + dodge_offset];
                     marker=:xcross, color=:gray50, markersize=8)
        end
    end

    axislegend(ax; position=:rb)

    return fig
end

"""
    plot_arm_matrix(rules::DataFrame;
                     value_metric=:Lift,
                     figsize=(800, 700)) -> Figure

Item-by-item matrix heatmap where each cell shows the maximum Lift (or other
metric) across all rules connecting two items.

Only single-item LHS rules are used.
"""
function plot_arm_matrix(rules::DataFrame;
                          value_metric::Symbol=:Lift,
                          figsize::Tuple{Int,Int}=(800, 700))
    nrow(rules) == 0 && error("Cannot plot empty rules DataFrame")

    # Filter to single-item LHS rules
    single = filter(r -> length(r.LHS) == 1, rules)
    nrow(single) == 0 && error("No single-item LHS rules found")

    # Collect all items
    all_items = sort(unique(vcat([r.LHS[1] for r in eachrow(single)],
                                  single.RHS)))
    n = length(all_items)
    item_idx = Dict(s => i for (i, s) in enumerate(all_items))

    # Build matrix: max metric across rules connecting pairs
    mat = zeros(n, n)
    for r in eachrow(single)
        i = item_idx[r.LHS[1]]
        j = item_idx[r.RHS]
        val = Float64(r[value_metric])
        mat[i, j] = max(mat[i, j], val)
        mat[j, i] = max(mat[j, i], val)
    end

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Pairwise Association Strength ($(value_metric))",
        titlesize=16,
        xlabel="$(VOCAB.item)", ylabel="$(VOCAB.item)",
        xticks=(1:n, all_items),
        yticks=(1:n, all_items),
        xticklabelrotation=π/3,
        xticklabelsize=9,
        yticklabelsize=9)

    heatmap!(ax, 1:n, 1:n, mat; colormap=:YlOrRd)

    Colorbar(fig[1, 2]; label=string(value_metric),
             colormap=:YlOrRd,
             limits=extrema(mat))

    return fig
end
