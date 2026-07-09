# ──────────────────────────────────────────────────────────────────────────────
# Network metrics and community detection
# ──────────────────────────────────────────────────────────────────────────────

"""
    CommunityResult

Result of community detection on a co-occurrence network.

# Fields
- `assignments`: Vector mapping vertex index to community ID
- `communities`: Dict mapping community ID to vector of item names
- `modularity`: Modularity score of the partition
- `n_communities`: Number of communities detected
"""
struct CommunityResult
    assignments::Vector{Int}
    communities::Dict{Int, Vector{String}}
    modularity::Float64
    n_communities::Int
end

"""
    _modularity(g::SimpleWeightedGraph, assignments::Vector{Int}) -> Float64

Compute the modularity Q of a partition on a weighted graph.

Q = (1/2m) Σᵢⱼ [Aᵢⱼ - (sᵢ sⱼ)/(2m)] δ(cᵢ, cⱼ)

where m = total edge weight, sᵢ = strength of node i, Aᵢⱼ = edge weight.

Computed in the equivalent per-community form Q = Σ_c [L_c/m - (D_c/2m)²],
where L_c is the internal edge weight of community c and D_c is the total node
strength in c. The `-(D_c/2m)²` term includes the Newman diagonal contribution
(`-(sᵢ/2m)²` self-terms) and the null-model penalty for *all* same-community
node pairs, not only those joined by an edge — an earlier version summed the
null term over edges alone, which inflated Q (a trivial one-community graph
returned 0.5 instead of 0). Community *assignments* are unaffected (the Louvain
gain does not depend on the omitted terms); only the reported Q changes.
"""
function _modularity(g::SimpleWeightedGraph, assignments::Vector{Int})
    n = nv(g)
    n == 0 && return 0.0

    # Total weight (2m = sum over ordered edge endpoints = 2 × total edge weight)
    m2 = 0.0
    for e in edges(g)
        m2 += 2.0 * weight(e)
    end
    m2 == 0.0 && return 0.0

    # Node strengths (weighted degree)
    strengths = zeros(n)
    for e in edges(g)
        w = weight(e)
        strengths[src(e)] += w
        strengths[dst(e)] += w
    end

    # Per-community total strength D_c and internal edge weight L_c (each internal
    # edge counted once). Community labels may be arbitrary integers → use Dicts.
    d_tot = Dict{Int, Float64}()
    for v in 1:n
        c = assignments[v]
        d_tot[c] = get(d_tot, c, 0.0) + strengths[v]
    end

    l_in = Dict{Int, Float64}()
    for e in edges(g)
        if assignments[src(e)] == assignments[dst(e)]
            c = assignments[src(e)]
            l_in[c] = get(l_in, c, 0.0) + weight(e)
        end
    end

    Q = 0.0
    for (c, dc) in d_tot
        lc = get(l_in, c, 0.0)
        Q += 2.0 * lc / m2 - (dc / m2)^2   # L_c/m = 2·L_c/2m ; (D_c/2m)²
    end

    return Q
end

"""
    _modularity_gain(g::SimpleWeightedGraph, node::Int, community::Int,
                     assignments::Vector{Int}, strengths::Vector{Float64},
                     m2::Float64) -> Float64

Compute the modularity gain from moving `node` into `community`.
"""
function _modularity_gain(g::SimpleWeightedGraph, node::Int, community::Int,
                          assignments::Vector{Int}, strengths::Vector{Float64},
                          m2::Float64)
    # Sum of weights from node to members of target community
    k_in = 0.0
    for nb in neighbors(g, node)
        if assignments[nb] == community
            k_in += g.weights[node, nb]
        end
    end

    # Sum of strengths in target community
    s_tot = 0.0
    for v in 1:nv(g)
        if assignments[v] == community
            s_tot += strengths[v]
        end
    end

    k_i = strengths[node]
    return k_in / m2 - (s_tot * k_i) / (m2 * m2) * 2.0
end

"""
    detect_communities(net::CooccurrenceNetwork;
                       method::Symbol=:louvain,
                       max_iter::Int=100) -> CommunityResult

Detect communities in a co-occurrence network.

# Methods
- `:louvain` — Louvain algorithm (default), good for weighted networks
- `:label_propagation` — Label propagation from Graphs.jl, fast but nondeterministic

# Returns
`CommunityResult` with community assignments, membership dict, modularity, and count.
"""
function detect_communities(net::CooccurrenceNetwork;
                            method::Symbol=:louvain,
                            max_iter::Int=100)
    g = net.graph
    n = nv(g)

    if n == 0
        return CommunityResult(Int[], Dict{Int, Vector{String}}(), 0.0, 0)
    end

    if method == :louvain
        assignments = _louvain(g, max_iter)
    elseif method == :label_propagation
        assignments = Graphs.label_propagation(g)[1]
    else
        error("Unknown community detection method: $method. Use :louvain or :label_propagation")
    end

    # Renumber communities to be contiguous starting at 1
    unique_comm = sort(unique(assignments))
    remap = Dict(c => i for (i, c) in enumerate(unique_comm))
    assignments = [remap[c] for c in assignments]

    # Build communities dict
    communities = Dict{Int, Vector{String}}()
    for (i, c) in enumerate(assignments)
        if !haskey(communities, c)
            communities[c] = String[]
        end
        push!(communities[c], net.items[i])
    end
    for k in keys(communities)
        sort!(communities[k])
    end

    mod = _modularity(g, assignments)
    return CommunityResult(assignments, communities, mod, length(unique_comm))
end

"""
    _louvain(g::SimpleWeightedGraph, max_iter::Int) -> Vector{Int}

Simple Louvain community detection (phase 1 only — sufficient for small graphs).

Iteratively moves each node to the neighboring community that maximizes
modularity gain, until no improvement is possible.
"""
function _louvain(g::SimpleWeightedGraph, max_iter::Int)
    n = nv(g)
    assignments = collect(1:n)  # Each node starts in its own community

    # Precompute
    m2 = 0.0
    for e in edges(g)
        m2 += 2.0 * weight(e)
    end
    m2 == 0.0 && return assignments

    strengths = zeros(n)
    for e in edges(g)
        w = weight(e)
        strengths[src(e)] += w
        strengths[dst(e)] += w
    end

    for _ in 1:max_iter
        improved = false
        for node in 1:n
            current_comm = assignments[node]

            # Find neighboring communities
            neighbor_comms = Set{Int}()
            for nb in neighbors(g, node)
                push!(neighbor_comms, assignments[nb])
            end

            # Temporarily remove node from its community for gain calculation
            best_comm = current_comm
            best_gain = 0.0

            for comm in neighbor_comms
                comm == current_comm && continue
                gain = _modularity_gain(g, node, comm, assignments, strengths, m2)
                # Also compute the loss from leaving current community
                loss = _modularity_gain(g, node, current_comm, assignments, strengths, m2)
                net_gain = gain - loss
                if net_gain > best_gain
                    best_gain = net_gain
                    best_comm = comm
                end
            end

            if best_comm != current_comm
                assignments[node] = best_comm
                improved = true
            end
        end

        improved || break
    end

    return assignments
end

"""
    compute_network_metrics(net::CooccurrenceNetwork) -> DataFrame

Compute per-node network metrics.

# Returns
DataFrame with columns: `item`, `degree`, `strength`, `betweenness`,
`clustering_coeff`, `prevalence`.
"""
function compute_network_metrics(net::CooccurrenceNetwork)
    g = net.graph
    n = nv(g)

    n == 0 && return DataFrame(
        item=String[], degree=Int[], strength=Float64[],
        betweenness=Float64[], clustering_coeff=Float64[], prevalence=Int[])

    # Degree
    deg = [Graphs.degree(g, v) for v in 1:n]

    # Strength (weighted degree)
    str = zeros(n)
    for e in edges(g)
        w = weight(e)
        str[src(e)] += w
        str[dst(e)] += w
    end

    # Betweenness centrality on the unweighted topology. Edge weights here are
    # association strengths (phi / lift), NOT path lengths — passing the weighted
    # graph makes Graphs.betweenness_centrality run Dijkstra with weights as
    # distances, which inverts the intent (a strong association becomes a long
    # path). Convert to an unweighted SimpleGraph so betweenness is purely
    # topological, consistent with the degree and clustering-coefficient metrics.
    ug = Graphs.SimpleGraph(nv(g))
    for e in edges(g)
        Graphs.add_edge!(ug, src(e), dst(e))
    end
    bc = Graphs.betweenness_centrality(ug)

    # Local clustering coefficient
    cc = Graphs.local_clustering_coefficient(g)

    return DataFrame(
        item = net.items,
        degree = deg,
        strength = round.(str, digits=3),
        betweenness = round.(bc, digits=4),
        clustering_coeff = round.(cc, digits=4),
        prevalence = [get(net.prevalence, s, 0) for s in net.items]
    )
end

"""
    network_summary(net::CooccurrenceNetwork, comm::CommunityResult)

Print a formatted summary of the network and its community structure.
"""
function network_summary(net::CooccurrenceNetwork, comm::CommunityResult)
    g = net.graph
    println("═══ Co-occurrence Network Summary ═══")
    println("  Nodes (items): $(nv(g))")
    println("  Edges (significant associations): $(ne(g))")
    println("  Records in population: $(net.n_records)")

    if ne(g) > 0
        weights = [weight(e) for e in edges(g)]
        println("  Edge weight range: $(round(minimum(weights), digits=2)) – $(round(maximum(weights), digits=2))")
        println("  Mean edge weight: $(round(mean(weights), digits=2))")
    end

    println("\n  Communities detected: $(comm.n_communities)")
    println("  Modularity: $(round(comm.modularity, digits=4))")

    for c in sort(collect(keys(comm.communities)))
        members = comm.communities[c]
        println("\n  Community $c ($(length(members)) items):")
        for item in members
            prev = get(net.prevalence, item, 0)
            println("    • $item (n=$prev)")
        end
    end
end
