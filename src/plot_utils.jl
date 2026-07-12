# ──────────────────────────────────────────────────────────────────────────────
# Shared visualization helpers
#
# Small, reusable pieces factored out of the four visualization files: the
# colormap conventions, the repeated "filter items by probability then order
# them" block, min–max scaling for node/edge sizes, and a uniform empty-input
# policy. Included before the visualization files (see the include manifest).
# ──────────────────────────────────────────────────────────────────────────────

# Colormap conventions (README/CLAUDE.md): categorical → wong_colors, heatmaps → :YlOrRd.
const HEATMAP_COLORMAP = :YlOrRd
categorical_colors() = Makie.wong_colors()

"""
    _require_nonempty(x, what::AbstractString)

Uniform empty-input policy for plot functions: error with a clear message rather
than silently returning a blank figure.
"""
function _require_nonempty(x, what::AbstractString)
    (x === nothing || isempty(x)) && error("Cannot plot: $what is empty")
    return x
end

"""
    _filter_and_order_items(theta::AbstractMatrix, items::AbstractVector, threshold::Real)
        -> (keep_ordered::Vector{Int}, theta_sub::AbstractMatrix, item_labels::AbstractVector)

Given a K×D probability matrix `theta` and item names, keep the item columns
whose maximum probability across the K classes is ≥ `threshold`, then order them
by (the class in which each item is highest, descending probability within that
class). Returns the kept column indices, the corresponding `theta` sub-matrix,
and the ordered item labels. When no item passes, `keep_ordered` is empty (the
caller decides whether that is an error or a skip).
"""
function _filter_and_order_items(theta::AbstractMatrix, items::AbstractVector, threshold::Real)
    max_per_item = vec(maximum(theta, dims=1))
    keep = findall(max_per_item .>= threshold)
    if isempty(keep)
        return (Int[], theta[:, Int[]], items[Int[]])
    end
    event_class = [argmax(theta[:, j]) for j in keep]
    event_prob = [theta[event_class[i], keep[i]] for i in eachindex(keep)]
    order = sortperm(collect(zip(event_class, .-event_prob)))
    keep_ordered = keep[order]
    return (keep_ordered, theta[:, keep_ordered], items[keep_ordered])
end

"""
    _minmax_scale(values, lo::Real, hi::Real) -> Vector{Float64}

Linearly rescale `values` into `[lo, hi]`. A constant input maps to the midpoint,
so node/edge sizes stay well-defined when every weight is equal.
"""
function _minmax_scale(values, lo::Real, hi::Real)
    isempty(values) && return Float64[]
    vmin, vmax = extrema(values)
    if vmax == vmin
        return fill((lo + hi) / 2, length(values))
    end
    return [lo + (hi - lo) * (v - vmin) / (vmax - vmin) for v in values]
end
