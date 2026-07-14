# Implementation Plan: CooccurrenceAnalysis.jl deferred items
*Written 2026-07-14 at the close of the five-phase refactor session. Read this once; implement against it.*

## Objective

Finish the two items intentionally left out of the completed five-phase refactor
(`plans/refactor-cooccurrence.md`, now fully landed on git `main`):

1. Deduplicate the node-size / edge-width scaling and layout-selection code in
   `src/network_visualization.jl` onto the existing `src/plot_utils.jl` helpers.
2. Resolve the `scripts/turing_stratified.jl` `SUBSAMPLE=10_000` runtime issue
   (needs a human decision — see Open Questions).

Definition of done: item 1 landed with the full suite still green (currently
**418 tests**, run with `julia --project=. -e 'include("test/runtests.jl")'`),
node/edge sizes visually unchanged for the non-degenerate case; item 2 either
resolved or explicitly left with a warning per the maintainer's choice.

## Context you need (the rest of the refactor is DONE)

The package was fully refactored across five phases; do **not** redo any of it.
Relevant to this work:
- `src/plot_utils.jl` already exists and is included **before** the visualization
  files. It exports (internally) `_minmax_scale(values, lo, hi)`,
  `_filter_and_order_items`, `_require_nonempty`, `HEATMAP_COLORMAP`,
  `categorical_colors()`. `_minmax_scale` linearly maps into `[lo, hi]` and
  returns the **midpoint `(lo+hi)/2`** for constant input (empty → `Float64[]`).
- Plot convention (README / repo CLAUDE.md): `wong_colors` categorical,
  `:YlOrRd` heatmaps. Plot functions return a `Makie.Figure`.
- Tests for these plot functions only assert `fig isa Figure` — they will **not**
  catch a numeric regression in sizes. That is the main risk here; see Constraints.

## Decisions (do not relitigate)

- **Reuse `_minmax_scale`; do not write a new scaler.** The helper was added in
  Phase 4 precisely for this. It already matches copy #1's behavior.
- **Keep the public kwargs.** `plot_cooccurrence_network` exposes
  `node_size_range=(15,50)` and `edge_width_range=(0.5,5.0)`; keep them and thread
  the same ranges into the other functions rather than re-hardcoding constants.
- **Layout selection is duplicated too** (`:stress`/`:spring`/`:shell` →
  `NetworkLayout.*`); factor it into one small helper in `plot_utils.jl`
  (e.g. `_layout_fn(sym)`), defaulting to `Stress()`.
- **Scope: `network_visualization.jl` only.** The three
  `bayesian_visualization.jl` item-filter blocks were already deduped in Phase 4.

## Architecture

Files to change:
- `src/plot_utils.jl` — add `_layout_fn(layout::Symbol)` returning the
  `NetworkLayout` object (Stress/Spring/Shell, default Stress). `_minmax_scale`
  already present; no change needed.
- `src/network_visualization.jl` — three functions carry copy-pasted scaling and
  layout logic that has already drifted:
  - `plot_cooccurrence_network` (fn @ ~L18): scaling ~L36–70, layout ~L74–81.
    This copy is **parameterized** (`node_size_range`, `edge_width_range`) and its
    constant-input fallback is already the midpoint — it matches `_minmax_scale`
    exactly. Refactor it first; it is the reference behavior.
  - `plot_network_comparison` (fn @ ~L116): scaling ~L137–160 is **hardcoded**
    (`15.0 + …*35.0`, edges `0.5 + …*4.5`) with constant-input fallbacks `25.0`
    (nodes) and `2.0` (edges); layout ~L166–168.
  - `plot_group_stratified_network` (fn @ ~L307): audit for the same scaling
    pattern (CODE_REVIEW.md flagged a copy around the original L384) and refactor
    if present.
  Replace each hand-rolled `[lo + (v-min)/(max-min)*(hi-lo) …]` block with
  `_minmax_scale(values, lo, hi)` and each layout `if/elseif` with `_layout_fn`.

## Interfaces / contracts

```julia
# New helper in src/plot_utils.jl
_layout_fn(layout::Symbol) -> NetworkLayout.AbstractLayout
# :stress → Stress()  (also the default for unknown), :spring → Spring(), :shell → Shell()

# Existing helper, reuse as-is:
_minmax_scale(values, lo::Real, hi::Real) -> Vector{Float64}
#   linear map into [lo,hi]; constant input → fill((lo+hi)/2); empty → Float64[]

# Call-site shape after refactor (illustrative):
node_sizes = _minmax_scale(prevs, node_size_range[1], node_size_range[2])
edge_widths = isempty(edge_weights) ? Float64[] :
              _minmax_scale(edge_weights, edge_width_range[1], edge_width_range[2])
layout_fn = _layout_fn(layout)
```

## Constraints

- **Behavior-preservation is on you, not the tests.** The `isa Figure` tests
  won't flag a numeric change. Before committing, verify sizes another way:
  compute `node_sizes`/`edge_widths` for the fixture net old-vs-new in the REPL
  and compare, OR add a small test that asserts node sizes fall within
  `node_size_range` and are non-degenerate. Reason explicitly about the
  degenerate (all-equal weights) branch.
- **Known fallback drift — resolve it deliberately.** For constant input,
  `_minmax_scale` returns the midpoint: nodes `(15+50)/2 = 32.5`, edges
  `(0.5+5.0)/2 = 2.75`. `plot_network_comparison` currently hardcodes `25.0` and
  `2.0` for that case. Unifying on `_minmax_scale` therefore **changes** the
  constant-weight rendering of that one function (32.5 vs 25.0, 2.75 vs 2.0).
  This only affects the visually-uninteresting "every weight identical" case.
  Recommended: unify on the midpoint (one code path) and note the change in the
  commit message. Do not add a special case to preserve `25.0`/`2.0`.
- Respect the core/wrapper split and plot conventions in repo CLAUDE.md (don't
  restate them here). Keep `Makie.Figure` returns.
- Commit as one focused change; run the full suite before and after.

## Non-goals / rejected alternatives

- **Do not touch the completed refactor** (Phases 1–5). The review docs
  (`CODE_REVIEW.md`, `plans/refactor-cooccurrence.md`) describe the *old* state.
- **Do not "fix" Turing/ADVI scaling** — documented boundary in
  `scripts/README.md`; item 2 below is only about the script's default, not the
  underlying algorithm.
- **Do not generalize the network plots to N groups.** They stay pairwise
  (first/second group); that was the Phase 3 decision.
- **Do not add a new scaling helper or change `_minmax_scale`'s semantics.**

## Open questions for the implementer

- [ ] **Item 2 needs a maintainer decision, not code judgment.**
  `scripts/turing_stratified.jl` sets `SUBSAMPLE=10_000`, which `scripts/README.md`
  reports never finishes (the marginalized ADVI likelihood doesn't scale). Ask the
  maintainer which they want, then do exactly that:
  (a) lower the default to a size that completes (~1–2k), or
  (b) leave `10_000` but add a loud warning comment / runtime `@warn`, or
  (c) leave as-is.
  Scripts are otherwise out of scope — make only the chosen one-line change.

## Task list (ordered)

1. Add `_layout_fn(layout::Symbol)` to `src/plot_utils.jl`.
2. Refactor `plot_cooccurrence_network` (the parameterized reference copy) onto
   `_minmax_scale` + `_layout_fn`. Run the suite (expect 418 green).
3. Refactor `plot_network_comparison`; accept the constant-input fallback change
   per Constraints. Audit `plot_group_stratified_network` and refactor if it
   carries the same pattern.
4. Verify node/edge sizes numerically for the fixture net (REPL diff or a new
   in-range assertion), then run the full suite.
5. Commit item 1 (one commit, message noting the `plot_network_comparison`
   constant-fallback change).
6. Ask the maintainer the item-2 question; apply their chosen one-line change to
   `scripts/turing_stratified.jl`; commit separately.

## Suggested implementer model

**Sonnet.** Julia fails silently here — Makie will happily render wrong sizes with
no error, and the tests only check `isa Figure`, so the numeric-equivalence
reasoning (especially the degenerate branch and the fallback-drift decision) is
where care is needed. Routine mechanical refactor otherwise; no need for Opus.
