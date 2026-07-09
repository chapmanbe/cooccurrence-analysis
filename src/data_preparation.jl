# ──────────────────────────────────────────────────────────────────────────────
# Data loading and transaction construction
# ──────────────────────────────────────────────────────────────────────────────

"""
    load_event_data(path::String) -> DataFrame

Load event-level data from an Arrow file.
"""
function load_event_data(path::String)
    return DataFrame(Arrow.Table(path))  # Arrow imported at module level
end

"""
    get_record_summary(event_df::DataFrame) -> DataFrame

Return per-record summary: id, group, number of items, and list of items.
"""
function get_record_summary(event_df::DataFrame)
    gp = groupby(event_df, :id)
    return combine(gp,
        :Group => first => :Group,
        nrow => :n_items,
        :item => (s -> [sort(unique(s))]) => :items
    )
end

"""
    build_transactions(event_df::DataFrame;
                       group_filter::Union{AbstractString, Nothing}=nothing,
                       min_items::Int=2,
                       timing_filter::Symbol=:all,
                       concurrent_window::Int=0,
                       year_col::Symbol=:year) -> DataFrame

Transform event-level data into a one-hot record × item boolean DataFrame
suitable for RuleMiner.jl's `Txns()` constructor.

# Arguments
- `event_df`: Raw event DataFrame with at least `:id`, `:item`, `:Group` columns
- `group_filter`: If `"A"` or `"B"`, keep only that group
- `min_items`: Minimum number of distinct items per record (default 2).
  Records with fewer distinct items are excluded since they cannot contribute
  to co-occurrence patterns.
- `timing_filter`: Stratify multi-item records by timing of their events.
    * `:all` (default) — include all multi-item records
    * `:concurrent` — include only records whose events all fall within
      `concurrent_window` years of each other (max pairwise gap ≤ window)
    * `:sequential` — include only records with at least one pair of
      events more than `concurrent_window` years apart
- `concurrent_window`: Year-gap threshold for concurrent classification
  (default 0 → same event year). With month-level dates, set fractionally
  via the `year_col` value. prior domain work used 6 months at the pair level; we
  approximate at the record level with year resolution.
- `year_col`: Column holding the event year (default `:year`).

# Returns
A boolean DataFrame where each row is a record and each column is an item.

# Note on concurrent-vs-sequential semantics
A record with 3 events in years (2010, 2010, 2018) has max gap = 8 →
sequential. A record with events in (2010, 2010, 2010) → concurrent.
Records with only one event (after filtering) have no gap; they are
excluded by `min_items` and don't reach the timing level. Record-level
classification is appropriate for our record-by-item Bernoulli mixture;
true pair-level analysis would require restructuring to a pair-by-item
representation.
"""
function build_transactions(event_df::DataFrame;
                            group_filter::Union{AbstractString, Nothing}=nothing,
                            min_items::Int=2,
                            timing_filter::Symbol=:all,
                            concurrent_window::Int=0,
                            year_col::Symbol=:year)
    onehot, _ = _build_transactions_internal(event_df;
        group_filter, min_items, timing_filter, concurrent_window, year_col)
    return onehot
end

"""
    build_transactions_with_ids(event_df::DataFrame; kwargs...)
        -> (onehot::DataFrame, ids::Vector)

Same as `build_transactions` but also returns the row-aligned vector of
record IDs. Useful for callers that need to track which records ended
up in the binary matrix (e.g. attaching cluster assignments back to ids).
"""
function build_transactions_with_ids(event_df::DataFrame;
                                       group_filter::Union{AbstractString, Nothing}=nothing,
                                       min_items::Int=2,
                                       timing_filter::Symbol=:all,
                                       concurrent_window::Int=0,
                                       year_col::Symbol=:year)
    return _build_transactions_internal(event_df;
        group_filter, min_items, timing_filter, concurrent_window, year_col)
end

"""
    filter_event_by_timing(event_df::DataFrame;
                            timing_filter::Symbol=:all,
                            concurrent_window::Int=0,
                            year_col::Symbol=:year) -> DataFrame

Filter `event_df` to the rows belonging to records whose multi-item timing
profile matches `timing_filter`. With `:concurrent` or `:sequential`,
single-item records are excluded (they have no within-record timing to
classify). With `:all` (default), the input is returned unchanged.

# Semantics
A record is **concurrent** iff the max pairwise year-gap among their
events is ≤ `concurrent_window`; **sequential** otherwise.
"""
function filter_event_by_timing(event_df::DataFrame;
                                 timing_filter::Symbol=:all,
                                 concurrent_window::Int=0,
                                 year_col::Symbol=:year)
    timing_filter in (:all, :concurrent, :sequential) ||
        error("timing_filter must be :all, :concurrent, or :sequential; got :$timing_filter")
    timing_filter === :all && return event_df

    year_col in propertynames(event_df) ||
        error("Column $year_col not found in event_df; cannot apply timing_filter")

    # Per-record summary: distinct items + year list
    record_summary = combine(groupby(event_df, :id),
        :item => (s -> length(unique(s))) => :n_items,
        year_col => (y -> [collect(y)]) => :years
    )

    # Multi-item only — single-item records have no timing to classify
    multi = filter(row -> row.n_items >= 2, record_summary)
    _max_year_gap(years) = isempty(years) ? 0 : maximum(years) - minimum(years)

    keep_ids = if timing_filter === :concurrent
        Set(filter(row -> _max_year_gap(row.years) <= concurrent_window, multi).id)
    else  # :sequential
        Set(filter(row -> _max_year_gap(row.years) > concurrent_window, multi).id)
    end

    return filter(row -> row.id in keep_ids, event_df)
end

function _build_transactions_internal(event_df::DataFrame;
                                       group_filter::Union{AbstractString, Nothing}=nothing,
                                       min_items::Int=2,
                                       timing_filter::Symbol=:all,
                                       concurrent_window::Int=0,
                                       year_col::Symbol=:year)
    timing_filter in (:all, :concurrent, :sequential) ||
        error("timing_filter must be :all, :concurrent, or :sequential; got :$timing_filter")

    df = event_df

    # Filter by group if requested
    if group_filter !== nothing
        df = filter(row -> row.Group == group_filter, df)
    end

    # Remove group-inappropriate items when filtering by group
    if group_filter == "A"
        df = filter(row -> !(row.item in GROUP_B_ONLY_ITEMS), df)
    elseif group_filter == "B"
        df = filter(row -> !(row.item in GROUP_A_ONLY_ITEMS), df)
    end

    # Aggregate per record: unique items and (if needed) per-event years
    if timing_filter === :all
        record_items = combine(groupby(df, :id),
            :item => (s -> [sort(unique(s))]) => :items
        )
    else
        year_col in propertynames(df) ||
            error("Column $year_col not found in event_df; cannot apply timing_filter")
        record_items = combine(groupby(df, :id),
            :item => (s -> [sort(unique(s))]) => :items,
            year_col => (y -> [collect(y)]) => :years
        )
    end

    # Filter by minimum item count
    record_items = filter(row -> length(row.items) >= min_items, record_items)

    # Apply timing filter
    if timing_filter !== :all
        _max_year_gap(years) = isempty(years) ? 0 : maximum(years) - minimum(years)
        if timing_filter === :concurrent
            record_items = filter(row -> _max_year_gap(row.years) <= concurrent_window,
                                   record_items)
        else  # :sequential
            record_items = filter(row -> _max_year_gap(row.years) > concurrent_window,
                                   record_items)
        end
    end

    nrow(record_items) == 0 && return (DataFrame(), Int[])

    # Determine all items present in the filtered data
    all_items = sort(unique(vcat(record_items.items...)))

    # Build one-hot boolean matrix
    onehot = DataFrame()
    for item in all_items
        onehot[!, item] = [item in ps for ps in record_items.items]
    end

    return onehot, record_items.id
end
