# ──────────────────────────────────────────────────────────────────────────────
# Visualization for Bayesian Bernoulli mixture clustering
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_class_probabilities(result::CooccurrenceAnalysisResult;
                              prob_threshold=0.05,
                              figsize=(900, 500)) -> Figure

Heatmap of per-class item probabilities θ_{k,d}.

Rows are latent classes, columns are items (filtered to items where
max probability across classes ≥ `prob_threshold`). Items are ordered by the
class in which they have highest probability.
"""
function plot_class_probabilities(result::CooccurrenceAnalysisResult;
                                   prob_threshold::Float64=0.05,
                                   figsize::Tuple{Int,Int}=(900, 500))
    best = result.model_selection.best
    K, D = size(best.theta)
    items = result.item_names

    # Filter to items above threshold
    max_per_item = vec(maximum(best.theta, dims=1))
    keep = findall(max_per_item .>= prob_threshold)
    isempty(keep) && error("No items above threshold $prob_threshold")

    # Order kept items by: event class, then probability within that class
    event_class = [argmax(best.theta[:, j]) for j in keep]
    event_prob = [best.theta[event_class[i], keep[i]] for i in eachindex(keep)]
    order = sortperm(collect(zip(event_class, .-event_prob)))
    keep_ordered = keep[order]

    theta_sub = best.theta[:, keep_ordered]
    item_labels = items[keep_ordered]

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Latent Class Probability Profiles (K=$(K))",
        titlesize=16,
        xlabel="Item", ylabel="Latent Class",
        xticks=(1:length(item_labels), item_labels),
        yticks=(1:K, ["Class $k ($(round(best.pi[k]*100, digits=1))%)" for k in 1:K]),
        xticklabelrotation=π/3,
        xticklabelsize=9)

    # θ is a probability: pin both the heatmap and its colorbar to (0, 1) so the
    # rendered colors and the bar agree (they diverged when min(θ) > 0, since the
    # heatmap autoscaled while the bar was fixed at (0, max)).
    heatmap!(ax, 1:length(item_labels), 1:K, theta_sub';
             colormap=:YlOrRd, colorrange=(0.0, 1.0))

    Colorbar(fig[1, 2]; label="P(item | class)",
             colormap=:YlOrRd,
             limits=(0.0, 1.0))

    return fig
end

"""
    plot_bic_elbow(result::CooccurrenceAnalysisResult;
                    figsize=(600, 400)) -> Figure

Line plot of BIC vs K with the best K highlighted.
"""
function plot_bic_elbow(result::CooccurrenceAnalysisResult;
                         figsize::Tuple{Int,Int}=(600, 400))
    ms = result.model_selection
    ks = [p.first for p in ms.bic_values]
    bics = [p.second for p in ms.bic_values]

    # Sort by K for plotting
    order = sortperm(ks)
    ks_sorted = ks[order]
    bics_sorted = bics[order]

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Model Selection: BIC vs Number of Classes",
        titlesize=16,
        xlabel="K (number of classes)",
        ylabel="BIC")

    lines!(ax, ks_sorted, bics_sorted; color=:steelblue, linewidth=2)
    scatter!(ax, ks_sorted, bics_sorted; color=:steelblue, markersize=10)

    # Highlight best K
    best_idx = findfirst(==(ms.best_K), ks_sorted)
    scatter!(ax, [ks_sorted[best_idx]], [bics_sorted[best_idx]];
             color=:red, markersize=16, marker=:star5)

    return fig
end

"""
    plot_class_profiles(result::CooccurrenceAnalysisResult;
                         top_n=10, figsize=(800, 600)) -> Figure

Horizontal bar chart showing the top `top_n` highest-probability items per class.
"""
function plot_class_profiles(result::CooccurrenceAnalysisResult;
                              top_n::Int=10,
                              figsize::Tuple{Int,Int}=(800, 600))
    best = result.model_selection.best
    K = best.K
    items = result.item_names
    palette = Makie.wong_colors()

    # Grid layout: stack classes vertically
    n_cols = min(K, 2)
    n_rows = ceil(Int, K / n_cols)
    fig = Figure(size=figsize)

    for k in 1:K
        row = (k - 1) ÷ n_cols + 1
        col = (k - 1) % n_cols + 1

        probs = best.theta[k, :]
        top_idx = partialsortperm(probs, 1:min(top_n, length(probs)), rev=true)
        top_items = items[top_idx]
        top_probs = probs[top_idx]

        # Reverse for horizontal bars (top item at top)
        top_items_rev = reverse(top_items)
        top_probs_rev = reverse(top_probs)

        pct = round(best.pi[k] * 100, digits=1)
        ax = Axis(fig[row, col];
            title="Class $k ($pct%)",
            titlesize=14,
            xlabel="P(item | class)",
            yticks=(1:length(top_items_rev), top_items_rev),
            yticklabelsize=9)

        barplot!(ax, 1:length(top_probs_rev), top_probs_rev;
                 direction=:x,
                 color=palette[((k - 1) % length(palette)) + 1])
        xlims!(ax, 0, 1)
    end

    return fig
end

"""
    plot_clustering_comparison(group_a::CooccurrenceAnalysisResult,
                                group_b::CooccurrenceAnalysisResult;
                                prob_threshold=0.1,
                                figsize=(1400, 500)) -> Figure

Side-by-side heatmaps of class probabilities for group_a and group_b cohorts.
"""
function plot_clustering_comparison(group_a::CooccurrenceAnalysisResult,
                                    group_b::CooccurrenceAnalysisResult;
                                    prob_threshold::Float64=0.1,
                                    figsize::Tuple{Int,Int}=(1400, 500))
    fig = Figure(size=figsize)

    for (col_idx, result, group) in [(1, group_a, "A"), (2, group_b, "B")]
        best = result.model_selection.best
        K = best.K
        items = result.item_names

        # Filter items
        max_per_item = vec(maximum(best.theta, dims=1))
        keep = findall(max_per_item .>= prob_threshold)
        isempty(keep) && continue

        event_class = [argmax(best.theta[:, j]) for j in keep]
        event_prob = [best.theta[event_class[i], keep[i]] for i in eachindex(keep)]
        order = sortperm(collect(zip(event_class, .-event_prob)))
        keep_ordered = keep[order]

        theta_sub = best.theta[:, keep_ordered]
        item_labels = items[keep_ordered]

        ax = Axis(fig[1, col_idx];
            title="$group (K=$K, $(result.n_records) records)",
            titlesize=14,
            xlabel="Item", ylabel="Class",
            xticks=(1:length(item_labels), item_labels),
            yticks=(1:K, ["$k" for k in 1:K]),
            xticklabelrotation=π/3,
            xticklabelsize=8)

        heatmap!(ax, 1:length(item_labels), 1:K, theta_sub';
                 colormap=:YlOrRd, colorrange=(0, 1))
    end

    Colorbar(fig[1, 3]; label="P(item | class)",
             colormap=:YlOrRd, limits=(0, 1))

    return fig
end

# ──────────────────────────────────────────────────────────────────────────────
# HDP visualization
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_hdp_stick_weights(result::HDPClusteringResult;
                           beta_threshold=0.01,
                           figsize=(900, 500)) -> Figure

Two-panel figure showing global and per-group HDP stick weights.

*Top panel*: grouped bar chart — for each active cluster one bar per group
(π_{j,k}) plus a separate bar for the global weight (β_k).

*Bottom panel*: cluster category annotations (universal / group_a_only /
group_b_only / negligible) and top-2 items per cluster.

This is the event cross-strata interpretation figure.
"""
function plot_hdp_stick_weights(result::HDPClusteringResult;
                                beta_threshold::Float64=0.01,
                                figsize::Tuple{Int,Int}=(900, 500))
    hdp = result.hdp_result
    K = hdp.K_max
    J = hdp.n_groups
    labels = hdp.group_labels

    # Only active clusters
    active = findall(hdp.beta_mean .> beta_threshold)
    isempty(active) && error("No active clusters above beta_threshold=$beta_threshold")
    K_act = length(active)

    # Category label per active cluster
    cat_df = hdp_cluster_categorization(result; beta_threshold)
    cat_labels = [string(cat_df.category[k]) for k in active]

    # Group palette: first J+1 colours (groups + global)
    palette = Makie.wong_colors()
    group_colors = [palette[((j-1) % length(palette)) + 1] for j in 1:J]
    global_color = (:gray40, 0.8)

    fig = Figure(size=figsize)

    # ── Top: grouped bar chart ─────────────────────────────────────────────
    ax_top = Axis(fig[1, 1];
        title="HDP Stick Weights: Global β and Per-Group π",
        titlesize=15,
        ylabel="Weight",
        xticks=(1:K_act, ["C$(active[i])" for i in 1:K_act]),
        xticklabelsize=10,
        limits=(nothing, (0, nothing)))

    n_bars = J + 1           # one per group + global
    bar_width = 0.8 / n_bars
    offsets = range(-(n_bars - 1) / 2, (n_bars - 1) / 2; length=n_bars) .* bar_width

    # Global β bars
    xs_global = (1:K_act) .+ offsets[end]
    barplot!(ax_top, xs_global, hdp.beta_mean[active];
             width=bar_width, color=global_color, label="Global β")

    # Per-group π bars
    for j in 1:J
        xs_j = (1:K_act) .+ offsets[j]
        barplot!(ax_top, xs_j, hdp.pi_mean[j, active];
                 width=bar_width, color=group_colors[j], label=labels[j])
    end

    axislegend(ax_top; position=:rt, framevisible=false, labelsize=10)

    # ── Bottom: category + top-2 items text ───────────────────────────────
    ax_bot = Axis(fig[2, 1];
        xlabel="Cluster",
        xticks=(1:K_act, ["C$(active[i])" for i in 1:K_act]),
        xticklabelsize=10,
        yticksvisible=false,
        yticklabelsvisible=false,
        limits=((0.5, K_act + 0.5), (0, 1)),
        height=60)
    hidespines!(ax_bot, :t, :l, :r)

    for (i, k) in enumerate(active)
        # Top-2 items for this cluster
        top2_idx = partialsortperm(hdp.theta[k, :], 1:min(2, length(result.item_names)), rev=true)
        top2_str = join(result.item_names[top2_idx], "\n")
        cat_str = replace(cat_labels[i], "_" => " ")
        text!(ax_bot, i, 0.55; text=cat_str, align=(:center, :center),
              fontsize=8, font=:bold)
        text!(ax_bot, i, 0.2; text=top2_str, align=(:center, :center), fontsize=7)
    end

    rowsize!(fig.layout, 2, Relative(0.22))
    rowgap!(fig.layout, 6)

    return fig
end

"""
    plot_hdp_class_profiles(result::HDPClusteringResult;
                             beta_threshold=0.01,
                             prob_threshold=0.05,
                             figsize=(1000, 500)) -> Figure

Heatmap of posterior mean θ_{k,d} for active HDP clusters.

Rows = active clusters labelled with category and global β weight.
Columns = items with max(θ_{k,d}) ≥ `prob_threshold` across active
clusters, ordered by event cluster (item with highest probability).
Colour scale: white → red (YlOrRd).
"""
function plot_hdp_class_profiles(result::HDPClusteringResult;
                                  beta_threshold::Float64=0.01,
                                  prob_threshold::Float64=0.05,
                                  figsize::Tuple{Int,Int}=(1000, 500))
    hdp = result.hdp_result
    items = result.item_names
    K = hdp.K_max

    active = findall(hdp.beta_mean .> beta_threshold)
    isempty(active) && error("No active clusters above beta_threshold=$beta_threshold")
    K_act = length(active)

    theta_act = hdp.theta[active, :]     # K_act × D

    # Filter items: at least one active cluster has θ_{k,d} ≥ prob_threshold
    max_per_item = vec(maximum(theta_act, dims=1))
    keep = findall(max_per_item .>= prob_threshold)
    isempty(keep) && error("No items above prob_threshold=$prob_threshold")

    # Order items by event cluster, then probability
    event_class = [argmax(theta_act[:, d]) for d in keep]
    event_prob  = [theta_act[event_class[i], keep[i]] for i in eachindex(keep)]
    item_order    = sortperm(collect(zip(event_class, .-event_prob)))
    keep_ordered  = keep[item_order]

    theta_plot = theta_act[:, keep_ordered]    # K_act × n_items
    item_labels = items[keep_ordered]

    # Category lookup for row labels
    cat_df = hdp_cluster_categorization(result; beta_threshold)
    row_labels = Vector{String}(undef, K_act)
    for i in 1:K_act
        cat_str = replace(string(cat_df.category[active[i]]), "_" => " ")
        bw = round(hdp.beta_mean[active[i]], digits=3)
        row_labels[i] = "C$(active[i]) [$cat_str]  β=$bw"
    end

    n_items = length(keep_ordered)
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="HDP Cluster Profiles — Posterior Mean θ[k,d]",
        titlesize=15,
        xlabel="Item",
        ylabel="Cluster",
        xticks=(1:n_items, item_labels),
        yticks=(1:K_act, row_labels),
        xticklabelrotation=π/3,
        xticklabelsize=9,
        yticklabelsize=9)

    hm = heatmap!(ax, 1:n_items, 1:K_act, theta_plot';
                  colormap=:YlOrRd, colorrange=(0, 1))

    # Annotate cells where θ > 0.15
    for (ci, k) in enumerate(1:K_act)
        for (di, _) in enumerate(1:n_items)
            v = theta_plot[k, di]
            v >= 0.15 || continue
            text!(ax, di, ci; text=string(round(v, digits=2)),
                  align=(:center, :center), fontsize=7,
                  color=v > 0.6 ? :white : :black)
        end
    end

    Colorbar(fig[1, 2]; label="E_q[θ[k,d]]", colormap=:YlOrRd, limits=(0, 1))

    return fig
end

"""
    plot_hdp_sharing_heatmap(result::HDPClusteringResult;
                              beta_threshold=0.01,
                              figsize=(600, 500)) -> Figure

Cluster × group heatmap of per-group mixing weights π_{j,k}.

Rows are active clusters labelled with their top-3 items.
Columns are strata (Group A / Group B or other groups).
Cell colour = π_{j,k}; cell text = numeric value.
Right-margin column annotates each row's category.

This is the compact answer to "which clusters are shared vs group-specific?"
"""
function plot_hdp_sharing_heatmap(result::HDPClusteringResult;
                                   beta_threshold::Float64=0.01,
                                   figsize::Tuple{Int,Int}=(600, 500))
    hdp = result.hdp_result
    J = hdp.n_groups
    labels = hdp.group_labels

    active = findall(hdp.beta_mean .> beta_threshold)
    isempty(active) && error("No active clusters above beta_threshold=$beta_threshold")
    K_act = length(active)

    # Row labels: cluster index + top-3 items
    function top_items_str(k, n=3)
        idx = partialsortperm(hdp.theta[k, :], 1:min(n, length(result.item_names)), rev=true)
        join(result.item_names[idx], ", ")
    end
    row_labels = ["C$(active[i]): $(top_items_str(active[i]))" for i in 1:K_act]

    # pi_plot: K_act × J matrix
    pi_plot = hdp.pi_mean[1:J, active]'   # K_act × J

    # Category per active cluster
    cat_df = hdp_cluster_categorization(result; beta_threshold)
    _shorten(s) = replace(s, "both_present_but_unequal" => "unequal", "_" => " ")
    cat_labels = [_shorten(string(cat_df.category[active[i]])) for i in 1:K_act]

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="HDP Cross-Strata Cluster Sharing (π[j,k])",
        titlesize=14,
        xlabel="Group",
        ylabel="Cluster",
        xticks=(1:J, labels),
        yticks=(1:K_act, row_labels),
        xticklabelsize=11,
        yticklabelsize=9)

    heatmap!(ax, 1:J, 1:K_act, pi_plot';
             colormap=:YlOrRd, colorrange=(0, maximum(pi_plot) + 1e-6))

    # Cell value annotations
    for ci in 1:K_act
        for gi in 1:J
            v = pi_plot[ci, gi]
            text!(ax, gi, ci; text=string(round(v, digits=3)),
                  align=(:center, :center), fontsize=9,
                  color=v > 0.3 * maximum(pi_plot) ? :white : :black)
        end
    end

    Colorbar(fig[1, 2]; label="π[j,k]", colormap=:YlOrRd,
             limits=(0, maximum(pi_plot) + 1e-6))

    # Category annotation column
    ax_cat = Axis(fig[1, 3];
        yticks=(1:K_act, fill("", K_act)),
        xticks=(1:1, ["Category"]),
        xticklabelsize=10,
        yticklabelsvisible=false,
        limits=((0.5, 1.5), (0.5, K_act + 0.5)))
    hidespines!(ax_cat, :t, :b, :l, :r)
    for ci in 1:K_act
        text!(ax_cat, 1, ci; text=cat_labels[ci],
              align=(:center, :center), fontsize=8)
    end

    colsize!(fig.layout, 3, Relative(0.22))

    return fig
end

"""
    plot_hdp_cluster_butterfly(result::HDPClusteringResult;
                               beta_threshold=0.01,
                               figsize=(850, 480)) -> Figure

Butterfly (tornado) chart comparing per-group HDP mixing weights π[j,k].

Requires exactly 2 groups (typically Group B / Group A). For each active cluster
(β_k > `beta_threshold`), a pair of horizontal bars is drawn: the first group
extends to the **left** (negative x) and the second group extends to the
**right** (positive x). Bar length equals the posterior mean π[j,k].

Clusters are ordered top-to-bottom by global stick weight β_k descending.
Bars are coloured by cluster category from `hdp_cluster_categorization`
(universal=blue, first-group-only=orange, second-group-only=green,
both-present-but-unequal=purple, negligible=grey).
"""
function plot_hdp_cluster_butterfly(result::HDPClusteringResult;
                                    beta_threshold::Float64=0.01,
                                    figsize::Tuple{Int,Int}=(850, 480))
    hdp = result.hdp_result
    J, K = size(hdp.pi_mean)
    J == 2 || error("Butterfly chart requires exactly 2 groups; got $J")

    group_labels = hdp.group_labels
    palette = Makie.wong_colors()

    # Active clusters ordered by beta_mean descending
    active = findall(hdp.beta_mean .>= beta_threshold)
    isempty(active) && error("No active clusters above threshold $beta_threshold")
    active = active[sortperm(hdp.beta_mean[active], rev=true)]
    K_act = length(active)

    # Category colours, driven by hdp_cluster_categorization (like the sharing
    # heatmap) rather than a nonexistent :category column on group_profiles.
    # Keys must match the actual category symbols, i.e. lowercased :group_<x>_only.
    cat_df = hdp_cluster_categorization(result; beta_threshold)
    cat_by_cluster = Dict(row.cluster => row.category for row in eachrow(cat_df))
    g1 = lowercase(string(group_labels[1]))
    g2 = lowercase(string(group_labels[2]))
    cat_map = Dict{Symbol, Any}(
        :universal                => palette[1],   # blue
        :negligible               => :lightgray,
        :both_present_but_unequal => palette[4],   # reddish-purple
        Symbol("group_$(g1)_only") => palette[2],  # orange
        Symbol("group_$(g2)_only") => palette[3],  # green
    )

    function cluster_color(ki)
        cat = get(cat_by_cluster, ki, :universal)
        return get(cat_map, cat, palette[1])
    end

    # Build plot arrays (y-position = row index, 1 = top cluster)
    ys      = collect(1:K_act)
    pi_g1   = [hdp.pi_mean[1, active[i]] for i in 1:K_act]  # group 1 → left
    pi_g2   = [hdp.pi_mean[2, active[i]] for i in 1:K_act]  # group 2 → right
    colors  = [cluster_color(active[i]) for i in 1:K_act]
    labels  = ["C$(active[i])  β=$(round(hdp.beta_mean[active[i]], digits=3))"
               for i in 1:K_act]

    xmax = max(maximum(pi_g1), maximum(pi_g2)) * 1.15

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="HDP Cluster Mixing Weights: $(group_labels[1]) ← | → $(group_labels[2])",
        titlesize=14,
        xlabel="Mixing weight π",
        yticks=(ys, labels),
        yticklabelsize=9,
        limits=((-xmax, xmax), (0.5, K_act + 0.5)))

    # Vertical centre line
    vlines!(ax, 0; color=:black, linewidth=1)

    # Group label annotations
    text!(ax, -xmax * 0.95, K_act + 0.3;
          text=group_labels[1], align=(:left, :bottom), fontsize=11, font=:bold)
    text!(ax, xmax * 0.95, K_act + 0.3;
          text=group_labels[2], align=(:right, :bottom), fontsize=11, font=:bold)

    # Bars
    for i in 1:K_act
        barplot!(ax, [ys[i]], [-pi_g1[i]];
                 direction=:x, color=(colors[i], 0.85), strokewidth=0.5)
        barplot!(ax, [ys[i]], [pi_g2[i]];
                 direction=:x, color=(colors[i], 0.85), strokewidth=0.5)
    end

    # Tick marks on x-axis: show absolute values
    xt_vals = range(0, xmax * 0.9, length=5)
    xt_neg  = -reverse(xt_vals[2:end])
    xt_pos  = xt_vals[2:end]
    all_xt  = [xt_neg; 0.0; xt_pos]
    ax.xticks = (all_xt, [string(round(abs(v), digits=3)) for v in all_xt])
    ax.xticklabelsize = 8

    return fig
end
