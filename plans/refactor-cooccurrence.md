# Implementation Plan: CooccurrenceAnalysis.jl refactor
*Written 2026-07-09 at the close of a code-review/design session. Read this once; implement against it. Finding details and rationale live in `CODE_REVIEW.md` (repo root) — finding IDs below (C1, D2, P1, T2, …) refer to its tables.*

## Objective

Execute the five-phase refactor from `CODE_REVIEW.md`: fix three silently-wrong-output bugs plus the remaining correctness items, remove the two performance hot spots, make the N-group/domain-neutral contract real, consolidate duplicated structure, and harden the tests. Definition of done: all phases landed, the full suite passes non-vacuously (including new planted-truth tests), and a 3-group dataset runs through all four methods end to end.

Baseline: 320/320 tests pass today (53s) — a green suite that currently hides the bugs. Run tests after every task:
`julia --project=. -e 'include("test/runtests.jl")'`

## Decisions (do not relitigate)

- **Phase order is 1→5; do not pull Phase 3 before Phase 1** — the category-symbol fix (C3) changes what tests must assert, and its planted-truth test lands with it.
- **Phases 1–2 are strictly API-compatible; Phase 3 is the single breaking change**, landed as one coordinated PR-sized change with README/notebooks/tests updated together.
- **Stratification results keyed by group value in a dict, not spliced into Symbol field names** (`strat["A"].rules`, not `strat.group_a_rules`) — kills the `lowercase` collision, the illusory dynamic naming, and the 2-group ceiling in one move.
- **`compare_strata`/`compare_networks`/`compare_clusterings` stay explicitly pairwise** — they take two named entries from the dict; N-way comparison is out of scope.
- **Fixture item constants move out of `src/` entirely** — caller supplies exclusion knowledge via keyword (Buchanan separation principle, per project CLAUDE.md). Test fixture constants move into `test/runtests.jl`.
- **Odds ratio uses Haldane–Anscombe (+0.5 all cells) when any cell is 0**, via one shared helper — not `Inf`, so downstream sorting/plotting keeps working.
- **HDP stopping rule: best-ELBO plateau over a 5-iteration window** — the documented non-monotone global-stick update makes plain `|Δ| < tol` unsound.
- **Louvain gain fix uses the standard form and incremental per-community strength sums** — fixes C1, C2, and P4 in one rewrite rather than patching three times.
- **No submodule split** — flat include-manifest stays; the core/wrapper convention is the layering. Preserve it: algorithm changes in core files, naming/grouping in wrappers.
- **Do not "fix" Turing/ADVI scaling and do not implement two-level HDP (table indicators)** — documented boundaries, tests only.
- **Stochastic tests move to `StableRNGs`** — MersenneTwister-pinned thresholds break on Julia RNG-stream changes.
- **Pipelines get concrete result structs and a `verbose::Bool=true` kwarg** — Dict→NamedTuple returns are type-unstable and undiscoverable.

## Architecture

No new modules. Files touched, by phase:

- **Phase 1 (correctness):** `src/network_metrics.jl` (Louvain rewrite), `src/hdp_wrappers.jl` (category symbols), `src/overview_visualization.jl` (group detection, docstring), `src/statistical_validation.jl` + `src/network_construction.jl` (shared OR helper — helper lives in statistical_validation.jl, included first), `src/bayesian_clustering.jl` (EM guards, dead code), `src/hdp_clustering.jl` (stopping rule, NaN-restart guard), `src/group_stratification.jl` (rule key), `src/network_group_stratification.jl` (ARI missing), `src/bayesian_turing.jl` (quantiles), `src/bayesian_visualization.jl` (butterfly wiring, colorrange).
- **Phase 2 (performance):** `src/network_construction.jl` + `src/statistical_validation.jl` (matrix-based pairwise counts), `src/hdp_clustering.jl` (hoist `Xf`), `src/bayesian_clustering.jl` (precompute `1 .- X_f`), `src/overview_visualization.jl` (vectorized scatter).
- **Phase 3 (N-group):** `src/utils.jl` (delete constants), `src/data_preparation.jl`, `src/network_construction.jl` (thread `exclusive_items`), `src/group_stratification.jl` (new generic `stratify_by` driver), `src/network_group_stratification.jl`, `src/bayesian_wrappers.jl` (become thin calls), `src/CooccurrenceAnalysis.jl` (pipelines + exports), `src/arm_visualization.jl`, README, notebooks.
- **Phase 4 (hygiene):** new `src/plot_utils.jl` (added to the include manifest before the visualization files; include order matters — utils → data_preparation → mining/network/bayesian), pipeline structs in `src/CooccurrenceAnalysis.jl`, kwarg promotion in `src/hdp_wrappers.jl`/`src/hdp_clustering.jl`.
- **Phase 5 (tests):** `test/runtests.jl` only.

## Interfaces / contracts

Agreed signatures (names final unless the implementer finds a conflict):

```julia
# Phase 1 — shared helper in statistical_validation.jl, used by both call sites
odds_ratio(ct::Matrix{Int}; correction::Symbol=:haldane) -> Float64
# :haldane → +0.5 to all cells iff any cell is 0; :none → raw (may be Inf/NaN)

# Phase 3 — exclusion knowledge supplied by caller, replaces GROUP_*_ONLY_ITEMS
# threaded through build_transactions / build_transactions_with_ids /
# compute_pairwise_associations / build_cooccurrence_network:
exclusive_items::Union{Nothing, Dict{<:AbstractString, <:AbstractSet}} = nothing
# Semantics preserved from current behavior: when group_filter == g,
# drop items belonging to OTHER groups' exclusive sets.

# Phase 3 — the one generic stratification driver (group_stratification.jl)
stratify_by(analysis_fn, event_df::DataFrame;
            group_col::Symbol=:Group, verbose::Bool=true, kwargs...)
    -> OrderedDict{String, Any}
# iterates sort(unique(event_df[!, group_col])); analysis_fn(event_df; group_filter=g, kwargs...)
# OrderedCollections is already in the dependency closure via DataFrames; add to
# Project.toml [deps] explicitly.

# Phase 3 — comparisons stay pairwise, taking values from that dict
compare_strata(rules_x, rules_y; labels=("A","B"))          # dynamic column prefixes from labels
compare_networks(net_x, net_y, comm_x, comm_y; labels=...)
compare_clusterings(res_x, res_y; labels=...)

# Phase 3 — category symbols (C3 fix, lands in Phase 1 with the naming decision)
# 2 groups:  :group_<lowercase(label)>_only   e.g. :group_a_only
# >2 groups: lowercase consistently: Symbol("group_" * join(lowercase.(present_labels), "_and_") * "_only")

# Phase 4 — pipeline returns
struct FullPipelineResult      # rules, itemsets, n_records, n_total_records,
                               # stratified::Union{Nothing,OrderedDict}, comparison::Union{Nothing,DataFrame}
struct NetworkPipelineResult   # network, communities, metrics, stratified, comparison
struct ClusteringPipelineResult
```

Migration contract (document in README): `strat.group_a_rules` → `strat["A"].rules`; `GROUP_A_ONLY_ITEMS`/`GROUP_B_ONLY_ITEMS` exports deleted.

## Constraints

- Julia; `Revise` hot-reloads src edits in a live session. Always run tests with the project active (see command above). There is no lint/build step.
- **This directory is not a git repository.** Before starting, `git init` and commit the pristine state so each phase-1 fix can be a separate commit (the plan depends on small, revertable diffs). Confirm with the user only if they object to git init; it is the expected workflow.
- One finding = one commit in Phase 1, each with its regression test in the same commit.
- Phase 2 changes must be behavior-identical: same RNG ⇒ bit-identical ELBO trajectory for the HDP hoist; assert equality of pairwise-association DataFrames old-vs-new on the test fixture before deleting the old path.
- Respect the core/wrapper split (per repo CLAUDE.md): math in `*_clustering.jl`/`mining.jl`/`network_construction.jl`, naming/group handling in wrappers.
- Plot functions return `Makie.Figure`; `wong_colors` categorical, `:YlOrRd` heatmaps (fix the `:Blues` violations while touching those files).
- Keep README.md authoritative: update its method/function catalog in the same change that alters any export.

## Non-goals / rejected alternatives

- **Submodule split** — rejected: ~4,600 lines doesn't warrant it; include-order docs suffice.
- **Fixing Turing/ADVI scaling** — rejected: documented boundary in `scripts/README.md`; add tests only.
- **Two-level HDP (CRF table counts)** — rejected: the single-level approximation is documented and acceptable; only the stopping rule changes.
- **Returning `Inf` odds ratios** — rejected in favor of Haldane–Anscombe: keeps sorting/plotting total.
- **N-way compare functions** — rejected for now: pairwise only, documented.
- **Replacing greedy `_hclust_order` with real average-linkage** — rejected: fix the docstring instead (C13); the ordering is cosmetic.

## Open questions for the implementer

- [ ] `scripts/turing_stratified.jl` sets `SUBSAMPLE=10_000`, which `scripts/README.md` reports as never finishing. Confirm with the user whether to lower the default or add a warning comment — scripts are otherwise out of scope.
- [ ] Phase 3: `run_full_pipeline`'s cross-strata comparison currently only runs for exactly the "A"/"B" pair. With N groups, decide: compare all pairs, or first two sorted values with a documented note. (Recommend: all pairs into `OrderedDict{Tuple{String,String},DataFrame}`; flag if it explodes for large N.)
- [ ] Whether `hdp_wrappers.jl:88`'s id-vector fix should use `eltype`-driven construction or just collect `Any` — pick whichever keeps `assignments_df` column types concrete.

## Task list (ordered)

**Phase 0**
1. `git init`, commit pristine tree, run suite to confirm 320/320 baseline.

**Phase 1 — correctness (one commit each, test included)**
2. Rewrite Louvain `_modularity_gain` + move loop (`network_metrics.jl:104–118, ~205`): remove node from its community, incremental per-community strength sums, standard ΔQ form. Regression test: two-clique graph partition maximizes `_modularity`; accepted moves never decrease `_modularity`. Replace the tautology at `test/runtests.jl:476`. (C1, C2, P4, T6)
3. Fix category symbols in `hdp_cluster_categorization` (`hdp_wrappers.jl:224–232`) to the contract above; lowercase the >2-group branch. Add planted-truth test: fixture with a group-exclusive cluster must yield `:group_<x>_only`; universal cluster must yield `:universal`. (C3, D6, T2)
4. Replace `^f`/`^m` and `^a`/`^b` regex group detection (`overview_visualization.jl:227–228, 308–311`) with iteration over `unique(event_df.Group)`. Test: prevalence bars nonzero for fixture groups A/B. (C4)
5. Add `odds_ratio(ct; correction=:haldane)` helper; use from `statistical_validation.jl:62` and `network_construction.jl:116`. Test zero-cell tables. (C5)
6. EM guards (`bayesian_clustering.jl`): clamp `N_k`/`pi_k` ≥ `eps()`, delete dead lines 171–172, recompute final LL after last M-step so params/responsibilities/LL/BIC are consistent. Test: fit with K much larger than true clusters produces finite LL and no NaN θ. (C6, C7, C8)
7. HDP stopping rule (`hdp_clustering.jl:401–433`): best-ELBO 5-iteration plateau; error informatively if all restarts NaN. (C9)
8. `_normalize_rule_key` → key on `(Set(LHS), RHS)` (`group_stratification.jl:91–94`); test that `{X,Y}→Z` and `{X,Z}→Y` both survive. (C10)
9. ARI → `missing` for <2 shared items (`network_group_stratification.jl:195`). (C11)
10. `Statistics.quantile` for CIs (`bayesian_turing.jl:233, 251`). (C12)
11. Wire butterfly plot to `hdp_cluster_categorization` (`bayesian_visualization.jl:484–504`); fix colorrange mismatch at `:47–52`; `:Blues` → `:YlOrRd` at `:423, 484`. (C14, C15)
12. Fix `_hclust_order` docstring (`overview_visualization.jl:345`); `group_label`→`group` docstring drift (`hdp_wrappers.jl:15`). (C13)

**Phase 2 — performance**
13. Matrix-based pairwise associations: build BitMatrix once in `compute_pairwise_associations`, `C = X'X`, derive contingency cells, pass table into `test_association` (no rescan). Assert result-DataFrame equality with old path on fixture, then delete old path. (P1, P5)
14. Hoist `Xf`/`1 .- Xf` and preallocate N×K buffer in `_run_hdp_cavi`; pass into e-step/atoms/ELBO. Bit-identical ELBO check with fixed rng. (P2)
15. Precompute `1 .- X_f` in `_em_single_run`; vectorize the prevalence scatter loops. (P3, P6)
16. Benchmark 13–14 on a ~10K-record synthetic set; record numbers in `scripts/README.md`.

**Phase 3 — N-group contract (single coordinated change)**
17. Delete `GROUP_A_ONLY_ITEMS`/`GROUP_B_ONLY_ITEMS` + exports; add `exclusive_items` kwarg per contract; move constants into `test/runtests.jl`; collapse the three copy-pasted filter blocks into one helper in `data_preparation.jl`. (D1)
18. Implement `stratify_by`; rewrite the three drivers (`group_stratification.jl:22–83`, `network_group_stratification.jl:40–75`, `bayesian_wrappers.jl:139–153`) as thin calls. (D2)
19. Update `compare_*` to the pairwise-from-dict signatures with dynamic label prefixes; update `run_*_pipeline` (D3) and `plot_arm_comparison` (D4); purge `m_`/`f_` names (D5).
20. Add a 3-group fixture test through all four methods; update README migration notes + notebooks.

**Phase 4 — hygiene**
21. Create `src/plot_utils.jl` (item filter/order, min-max scaling, layout selection, uniform empty-input error policy); refactor the four visualization files onto it.
22. Pipeline result structs + `verbose` kwarg; route prints through it.
23. Promote magic numbers to kwargs: `universal_ratio=2.0`, `negligible_threshold=0.01` (single comparison operator), effective-K `threshold=0.95`; fix id-eltype and `Vector{Any}` columns; guard item-name collisions with metadata columns.

**Phase 5 — tests (interleave where noted)**
24. Convert the six `if nrow(...) > 0` guards (`test/runtests.jl:304, 311, 322, 531, 920, 943`) to `@test nrow(...) > 0`. (T1)
25. EM planted-recovery ARI test; ADVI smoke test; `:holm` branch; `significance_stars` boundaries. (T3, T4)
26. Migrate stochastic tests to `StableRNGs` (add to test deps). (T5)

## Suggested implementer model

Sonnet — Julia fails silently (NaN propagation, `Vector{Any}` erosion, wrong-but-plausible math), and Phases 1–2 are numerical-correctness work; don't drop below Sonnet. Phase 3's coordinated breaking change is the riskiest single step — consider Opus for that one session if budget allows; Phases 4–5 are routine Sonnet work.
