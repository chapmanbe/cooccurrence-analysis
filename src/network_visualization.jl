# ──────────────────────────────────────────────────────────────────────────────
# Network visualization with CairoMakie + GraphMakie
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_cooccurrence_network(net::CooccurrenceNetwork;
                               communities::Union{CommunityResult, Nothing}=nothing,
                               layout::Symbol=:stress,
                               title::String="$(VOCAB.item) Co-occurrence Network",
                               node_size_range=(15, 50),
                               edge_width_range=(0.5, 5.0),
                               figsize=(900, 700)) -> Figure

Plot the co-occurrence network with CairoMakie + GraphMakie.

Node size ∝ prevalence, node color = community membership, edge width ∝ weight.
"""
function plot_cooccurrence_network(net::CooccurrenceNetwork;
                                    communities::Union{CommunityResult, Nothing}=nothing,
                                    layout::Symbol=:stress,
                                    title::String="$(VOCAB.item) Co-occurrence Network",
                                    node_size_range=(15, 50),
                                    edge_width_range=(0.5, 5.0),
                                    figsize=(900, 700))
    g = net.graph
    n = nv(g)
    n == 0 && error("Cannot plot an empty network")

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; title, titlesize=18)
    hidedecorations!(ax)
    hidespines!(ax)

    # Node sizes proportional to prevalence
    prevs = Float64[get(net.prevalence, s, 1) for s in net.items]
    node_sizes = _minmax_scale(prevs, node_size_range[1], node_size_range[2])

    # Node colors from community assignments
    if communities !== nothing && !isempty(communities.assignments)
        comm_colors = communities.assignments
    else
        comm_colors = ones(Int, n)
    end

    # Color palette for communities
    palette = Makie.wong_colors()
    n_colors = length(palette)
    node_color = [palette[((c - 1) % n_colors) + 1] for c in comm_colors]

    # Edge widths proportional to weight
    edge_weights = ne(g) > 0 ? Float64[weight(e) for e in edges(g)] : Float64[]
    edge_widths = _minmax_scale(edge_weights, edge_width_range[1], edge_width_range[2])

    # Layout
    layout_fn = _layout_fn(layout)

    graphplot!(ax, g;
        layout=layout_fn,
        node_size=node_sizes,
        node_color=node_color,
        edge_width=edge_widths,
        edge_color=(:gray70, 0.6),
        nlabels=net.items,
        nlabels_fontsize=9,
        nlabels_distance=5.0)

    # Legend for communities
    if communities !== nothing && communities.n_communities > 1
        legend_entries = [MarkerElement(color=palette[((c - 1) % n_colors) + 1],
                                        marker=:circle, markersize=12)
                          for c in 1:communities.n_communities]
        legend_labels = ["Community $c ($(length(get(communities.communities, c, String[]))))"
                         for c in 1:communities.n_communities]
        Legend(fig[1, 2], legend_entries, legend_labels; framevisible=false)
    end

    return fig
end

"""
    plot_network_comparison(group_a_net::CooccurrenceNetwork,
                            group_b_net::CooccurrenceNetwork;
                            group_a_comm=nothing, group_b_comm=nothing,
                            labels=("A", "B"),
                            layout::Symbol=:stress,
                            figsize=(1600, 700)) -> Figure

Plot group_a and group_b co-occurrence networks side by side.

`labels` names the two panels. Pass the actual group values so the figure reads
in the caller's vocabulary rather than the placeholder "A"/"B".
"""
function plot_network_comparison(group_a_net::CooccurrenceNetwork,
                                 group_b_net::CooccurrenceNetwork;
                                 group_a_comm::Union{CommunityResult, Nothing}=nothing,
                                 group_b_comm::Union{CommunityResult, Nothing}=nothing,
                                 labels::Tuple{AbstractString, AbstractString}=("A", "B"),
                                 layout::Symbol=:stress,
                                 figsize=(1600, 700))
    fig = Figure(size=figsize)

    for (col, net, comm, group) in [(1, group_a_net, group_a_comm, labels[1]),
                                   (2, group_b_net, group_b_comm, labels[2])]
        g = net.graph
        n = nv(g)
        n == 0 && continue

        ax = Axis(fig[1, col]; title="$group Network ($(nv(g)) $(VOCAB.items), $(ne(g)) edges)",
                  titlesize=16)
        hidedecorations!(ax)
        hidespines!(ax)

        # Node sizes
        prevs = Float64[get(net.prevalence, s, 1) for s in net.items]
        node_sizes = _minmax_scale(prevs, 15.0, 50.0)

        # Community colors
        palette = Makie.wong_colors()
        if comm !== nothing && !isempty(comm.assignments)
            node_color = [palette[((c - 1) % length(palette)) + 1]
                          for c in comm.assignments]
        else
            node_color = fill(palette[1], n)
        end

        # Edge widths
        ew = ne(g) > 0 ? Float64[weight(e) for e in edges(g)] : Float64[]
        edge_widths = _minmax_scale(ew, 0.5, 5.0)

        layout_fn = _layout_fn(layout)

        graphplot!(ax, g;
            layout=layout_fn,
            node_size=node_sizes,
            node_color=node_color,
            edge_width=edge_widths,
            edge_color=(:gray70, 0.6),
            nlabels=net.items,
            nlabels_fontsize=8,
            nlabels_distance=4.0)
    end

    return fig
end

"""
    plot_community_heatmap(net::CooccurrenceNetwork,
                           comm::CommunityResult;
                           weight_metric::Symbol=:lift,
                           figsize=(800, 700)) -> Figure

Plot a heatmap of pairwise association strengths, ordered by community membership.

Reveals block-diagonal structure when communities capture real clusters.
"""
function plot_community_heatmap(net::CooccurrenceNetwork,
                                comm::CommunityResult;
                                weight_metric::Symbol=:lift,
                                figsize=(800, 700))
    n = nv(net.graph)
    n == 0 && error("Cannot plot heatmap for empty network")

    # Order items by community, then alphabetically within community
    order = sortperm(collect(zip(comm.assignments, net.items));
                     by=x -> (x[1], x[2]))
    ordered_items = net.items[order]

    # Build weight matrix in ordered form
    mat = zeros(n, n)
    g = net.graph
    for e in edges(g)
        i, j = src(e), dst(e)
        w = weight(e)
        mat[i, j] = w
        mat[j, i] = w
    end
    mat_ordered = mat[order, order]

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Co-occurrence Strength by Community",
        titlesize=16,
        xlabel="$(VOCAB.item)", ylabel="$(VOCAB.item)",
        xticks=(1:n, ordered_items),
        yticks=(1:n, ordered_items),
        xticklabelrotation=π/3,
        xticklabelsize=9,
        yticklabelsize=9)

    heatmap!(ax, 1:n, 1:n, mat_ordered;
             colormap=:YlOrRd)

    # Draw community boundaries
    boundaries = Float64[]
    prev_comm = comm.assignments[order[1]]
    for (idx, orig_idx) in enumerate(order)
        c = comm.assignments[orig_idx]
        if c != prev_comm
            push!(boundaries, idx - 0.5)
            prev_comm = c
        end
    end
    for b in boundaries
        hlines!(ax, [b]; color=:black, linewidth=1.5, linestyle=:dash)
        vlines!(ax, [b]; color=:black, linewidth=1.5, linestyle=:dash)
    end

    Colorbar(fig[1, 2]; label=string(weight_metric),
             colormap=:YlOrRd,
             limits=extrema(mat_ordered))

    return fig
end

# ──────────────────────────────────────────────────────────────────────────────
# Extended network visualizations
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_centrality_barchart(metrics::DataFrame;
                              centrality=:strength, top_n=20,
                              figsize=(700, 500)) -> Figure

Horizontal bar chart of network centrality metrics, sorted descending.

Input should be the output of `compute_network_metrics()`.
"""
function plot_centrality_barchart(metrics::DataFrame;
                                   centrality::Symbol=:strength,
                                   top_n::Int=20,
                                   figsize::Tuple{Int,Int}=(700, 500))
    nrow(metrics) == 0 && error("Cannot plot empty metrics DataFrame")

    n = min(top_n, nrow(metrics))
    order = partialsortperm(Float64.(metrics[!, centrality]),
                            1:n, rev=true)

    items = metrics.item[order]
    vals = Float64.(metrics[order, centrality])

    # Reverse for top-at-top display
    items = reverse(items)
    vals = reverse(vals)

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="Network Centrality: $(centrality)",
        titlesize=16,
        xlabel=string(centrality),
        yticks=(1:n, items),
        yticklabelsize=9)

    barplot!(ax, 1:n, vals; direction=:x, color=:steelblue)

    return fig
end

"""
    plot_group_stratified_network(comparison::NetworkComparisonResult,
                                 group_a_net::CooccurrenceNetwork,
                                 group_b_net::CooccurrenceNetwork;
                                 labels=("Group A", "Group B"),
                                 layout=:stress,
                                 figsize=(1000, 800)) -> Figure

Union graph of group_a and group_b networks with edges colored by group specificity:
gray for shared, `wong_colors()[1]` for group_a-only, `wong_colors()[2]` for
group_b-only. Node size is proportional to max prevalence across networks.

`labels` names the two groups in the legend.
"""
function plot_group_stratified_network(comparison::NetworkComparisonResult,
                                      group_a_net::CooccurrenceNetwork,
                                      group_b_net::CooccurrenceNetwork;
                                      labels::Tuple{AbstractString, AbstractString}=("Group A", "Group B"),
                                      layout::Symbol=:stress,
                                      figsize::Tuple{Int,Int}=(1000, 800))
    # Build union item list
    all_items = sort(unique(vcat(group_a_net.items, group_b_net.items)))
    n = length(all_items)
    item_idx = Dict(s => i for (i, s) in enumerate(all_items))

    # Build union graph
    g = SimpleWeightedGraph(n)

    # Classify edges
    shared_pairs = Set{Tuple{String,String}}()
    for r in eachrow(comparison.shared_edges)
        push!(shared_pairs, (r.item_a, r.item_b))
    end

    group_a_pairs = Set{Tuple{String,String}}()
    for r in eachrow(comparison.group_a_only_edges)
        push!(group_a_pairs, (r.item_a, r.item_b))
    end

    group_b_pairs = Set{Tuple{String,String}}()
    for r in eachrow(comparison.group_b_only_edges)
        push!(group_b_pairs, (r.item_a, r.item_b))
    end

    # Add all edges to union graph
    edge_colors = Tuple{Int,Int,Symbol}[]  # (src, dst, category)
    for (pairs, cat) in [(shared_pairs, :shared),
                          (group_a_pairs, :group_a),
                          (group_b_pairs, :group_b)]
        for (a, b) in pairs
            i, j = item_idx[a], item_idx[b]
            add_edge!(g, i, j, 1.0)
            push!(edge_colors, (i, j, cat))
        end
    end

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        title="$(VOCAB.group)-Stratified Co-occurrence Network",
        titlesize=18)
    hidedecorations!(ax)
    hidespines!(ax)

    # Node sizes proportional to max prevalence
    node_sizes = Float64[]
    for s in all_items
        m_prev = get(group_a_net.prevalence, s, 0)
        f_prev = get(group_b_net.prevalence, s, 0)
        push!(node_sizes, Float64(max(m_prev, f_prev)))
    end
    node_sizes = _minmax_scale(node_sizes, 15.0, 50.0)

    # Edge colors
    palette = Makie.wong_colors()
    cat_colors = Dict(:shared => :gray60, :group_a => palette[1], :group_b => palette[2])

    # Build edge color vector matching graph edge order
    edge_idx_map = Dict{Tuple{Int,Int}, Symbol}()
    for (i, j, cat) in edge_colors
        key = i < j ? (i, j) : (j, i)
        edge_idx_map[key] = cat
    end

    e_colors = [cat_colors[edge_idx_map[(min(src(e), dst(e)), max(src(e), dst(e)))]]
                for e in edges(g)]

    layout_fn = _layout_fn(layout)

    graphplot!(ax, g;
        layout=layout_fn,
        node_size=node_sizes,
        node_color=:gray30,
        edge_width=2.0,
        edge_color=e_colors,
        nlabels=all_items,
        nlabels_fontsize=9,
        nlabels_distance=5.0)

    # Legend
    legend_entries = [LineElement(color=:gray60, linewidth=3),
                      LineElement(color=palette[1], linewidth=3),
                      LineElement(color=palette[2], linewidth=3)]
    Legend(fig[1, 2], legend_entries,
           ["Shared", "$(labels[1])-only", "$(labels[2])-only"];
           framevisible=false)

    return fig
end

"""
    plot_centrality_comparison(group_a_metrics::DataFrame,
                                group_b_metrics::DataFrame;
                                centrality=:strength, top_n=15,
                                labels=("A", "B"),
                                figsize=(1200, 500)) -> Figure

Side-by-side horizontal bar charts comparing centrality metrics for group_a and
group_b networks. `labels` names the two panels.
"""
function plot_centrality_comparison(group_a_metrics::DataFrame,
                                    group_b_metrics::DataFrame;
                                    centrality::Symbol=:strength,
                                    top_n::Int=15,
                                    labels::Tuple{AbstractString, AbstractString}=("A", "B"),
                                    figsize::Tuple{Int,Int}=(1200, 500))
    palette = Makie.wong_colors()
    fig = Figure(size=figsize)

    for (col, metrics, group, color) in [(1, group_a_metrics, labels[1], palette[1]),
                                        (2, group_b_metrics, labels[2], palette[2])]
        nrow(metrics) == 0 && continue
        n = min(top_n, nrow(metrics))
        order = partialsortperm(Float64.(metrics[!, centrality]),
                                1:n, rev=true)

        items = reverse(metrics.item[order])
        vals = reverse(Float64.(metrics[order, centrality]))

        ax = Axis(fig[1, col];
            title="$group — $(centrality)",
            titlesize=14,
            xlabel=string(centrality),
            yticks=(1:n, items),
            yticklabelsize=9)

        barplot!(ax, 1:n, vals; direction=:x, color=color)
    end

    return fig
end
