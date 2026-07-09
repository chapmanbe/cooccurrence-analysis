# Code Review & Refactoring Plan — CooccurrenceAnalysis.jl

*Review date: 2026-07-09. Scope: all of `src/` (~4,600 lines), `test/runtests.jl`,
key scripts. Method: full read of the mathematical core (EM, HDP CAVI, network
construction, mining, pipelines) plus dedicated review passes over the
visualization, wrapper/stratification, and test/Turing layers. All high-severity
findings were verified against the source before inclusion.*

---

## 1. Summary assessment

The mathematical core is in good shape: the HDP CAVI updates, the EM
Bernoulli-mixture machinery, the marginalized Turing likelihood, and the
multiple-testing pipeline are essentially correct, and the one deliberate
approximation (the HDP global-stick update) is documented with unusual honesty.
The test suite is genuinely assertion-rich for data preparation, timing filters,
and HDP recovery.

The package's real problems are concentrated in three places:

1. **Three confirmed correctness bugs** that silently produce wrong output
   (Louvain modularity gain, HDP category symbols, group-detection regexes).
2. **The domain-neutrality claim is only true for HDP.** ARM, network, and
   flat-K stratification hardcode two groups named `"A"`/`"B"`, fixture item
   constants live in `src/utils.jl`, and visualization code still contains
   regexes and variable names from a prior male/female domain.
3. **Copy-paste structure**: the per-group stratification driver exists three
   nearly identical times; group-filter logic, item-ordering blocks, and
   node-scaling code are each duplicated 2–3×. This is the main obstacle to
   extension (any new method or a third group requires touching many files).

Performance has two genuine hot spots (pairwise association loop, HDP per-
iteration allocations); everything else is fine at current scale.

---

## 2. Findings

Severity: **HIGH** = wrong results or broken contract, silently. **MED** =
wrong in edge cases, misleading output, or structural debt with concrete cost.
**LOW** = polish, latent traps.

### 2.1 Correctness and mathematical errors

| # | Location | Sev | Finding |
|---|----------|-----|---------|
| C1 | `src/network_metrics.jl:112` | HIGH | Louvain `_modularity_gain` under-weights the internal-edge reward by 2× relative to the degree penalty. Per the file's own `_modularity`, ΔQ = `2k_in/m2 − 2·s_tot·k_i/m2²`; the code computes `k_in/m2 − 2·s_tot·k_i/m2²`. This can change which community wins a move and biases toward over-fragmentation — community results are quantitatively suspect. |
| C2 | `src/network_metrics.jl:104–109` | MED | In the same gain computation the node is never removed from its current community (contradicting the comment near line 205), so `s_tot` includes `k_i` when evaluating the node's own community — an inconsistent offset across candidates. Fix together with C1. |
| C3 | `src/hdp_wrappers.jl:227–228` | HIGH | Two-group categorization emits `Symbol(lowercase(label)*"_only")` → `:a_only` / `:b_only`, contradicting the docstring, README, and the test's allowed list (`:group_a_only`…). The test at `test/runtests.jl:1194–1209` passes only because no single-group cluster arises for that seed (vacuous). |
| C4 | `src/overview_visualization.jl:227–228` | HIGH | `plot_item_prevalence` detects groups with regexes `^f` / `^m` (leftover male/female logic). With this package's `"A"`/`"B"` groups both match nothing → all group bars silently render zero. `plot_cooccurrence_heatmap` (`:308–311`) uses *different* regexes `^a`/`^b` in the same file. |
| C5 | `src/statistical_validation.jl:62`, `src/network_construction.jl:116` | MED | Odds ratio uses `max(denominator, 1)`: a zero denominator (OR = ∞/undefined) is silently reported as a large finite value. Use the Haldane–Anscombe correction (+0.5 all cells) or return `Inf`. Two independent copies of the same hack. |
| C6 | `src/bayesian_clustering.jl:249–253, 258–263` | MED | EM M-step has no empty-cluster guard: with the default flat prior the θ update is `s_kd/N_k`; if a component's `N_k → 0` this is 0/0 = NaN, and `clamp(NaN, …) == NaN` in Julia, so NaN propagates through `log.(theta)` and poisons the fit. Similarly `dirichlet_prior < 1` can drive `pi_k` negative → `log` of a negative number. Clamp `N_k` / `pi_k` and reinitialize or freeze empty components. |
| C7 | `src/bayesian_clustering.jl:165–178` | LOW | Class-reordering block computes `assignments_sorted` twice; the first computation (`findfirst(==(order[…]), 1:K)`) is *wrong* but dead — immediately overwritten by the correct `inv_order` version. Delete the dead lines before someone "simplifies" the wrong way. |
| C8 | `src/bayesian_clustering.jl:208–265` | LOW | Returned parameters are one M-step ahead of the reported log-likelihood/responsibilities/assignments (M-step runs after the last E-step's LL). Also, convergence is checked on the *marginal* LL while EM optimizes the MAP objective — the tracked quantity is not guaranteed monotone. Recompute LL (or return pre-update params) at exit. |
| C9 | `src/hdp_clustering.jl:241–269, 426` | MED | The documented global-stick approximation makes the ELBO non-monotone, yet `abs(elbo − prev_elbo) < tol·(1+|prev_elbo|)` is the sole stopping rule — CAVI can stop on an oscillation crossing rather than a plateau. Track best-ELBO-so-far or use a windowed criterion; also `fit_hdp_bernoulli_mixture` returns `nothing` if every restart yields NaN ELBO (no guard at `:519–528`). |
| C10 | `src/group_stratification.jl:91–94` | MED | `_normalize_rule_key` merges any rules sharing the same item *multiset* — `{X,Y}→Z` and `{X,Z}→Y` collapse to one row keeping max lift. Silent drop, undocumented. Key on `(sorted LHS, RHS)` unordered *pair* instead, or document the merge. |
| C11 | `src/network_group_stratification.jl:195` | LOW | ARI reported as `0.0` when < 2 shared items — conflates "no overlap" with "chance agreement". Return `missing`/`NaN`. |
| C12 | `src/bayesian_turing.jl:233, 251` | LOW | Credible-interval quantiles via `ceil(Int, α·n)` without interpolation (n=100, 95% → 3rd/98th order statistics, both bounds shifted). Use `Statistics.quantile`. Also `:185,194`: chain values extracted by hardcoded VarName strings (`"θ[$k, $d]"`, space-sensitive) — breaks silently on a DynamicPPL naming change. |
| C13 | `src/overview_visualization.jl:386–401` | LOW | `_hclust_order` is a greedy nearest-neighbor chain, not the "average-linkage hierarchical clustering" its docstring claims. Fix the docstring (or the algorithm). |
| C14 | `src/bayesian_visualization.jl:491–504` | MED | Butterfly-plot category coloring is dead code: `group_profiles` never carries a `:category` column, so bars are always blue; and the `cat_map` keys wouldn't match the actual (buggy, see C3) symbols anyway. Wire it to `hdp_cluster_categorization` like the other HDP plots. |
| C15 | `src/bayesian_visualization.jl:47–52` | MED | `plot_class_probabilities` heatmap autoscales color while its Colorbar is fixed at `(0, max)` — colors and bar disagree whenever `min(θ) > 0`. (Done correctly at `:183–187` with `colorrange=(0,1)`.) |

### 2.2 Domain leak: the neutral data model is not actually neutral

The CLAUDE.md/README contract — *"fully domain-neutral: swap in any application
domain without touching the math"* and *"per-group field names built
dynamically"* — holds only for the HDP path.

| # | Location | Sev | Finding |
|---|----------|-----|---------|
| D1 | `src/utils.jl:6–12` + exports at `CooccurrenceAnalysis.jl:30` | HIGH | `GROUP_A_ONLY_ITEMS` / `GROUP_B_ONLY_ITEMS` are fixture item names baked into package source *and exported*. They're consulted inside `_build_transactions_internal` (`data_preparation.jl:160–164`) and twice in `network_construction.jl` (`:78–82`, `:199–203`) — three copies of the same hardcoded filter. Violates the project's own "domain knowledge in data files, not code" convention. |
| D2 | `src/group_stratification.jl:33`, `src/network_group_stratification.jl:51`, `src/bayesian_wrappers.jl:143` | HIGH | Three separate stratification drivers each loop over the literal `["A", "B"]`. Any other group values are silently ignored; > 2 groups impossible. The "dynamic `group_<lowercase(value)>_*`" naming is illusory — only ever produces `group_a_*`/`group_b_*`. |
| D3 | `CooccurrenceAnalysis.jl:171–179, 250–258` | MED | `run_full_pipeline` / `run_network_pipeline` consume `strat.group_a_rules` etc. by literal field name — pipelines break for any group renaming even if D2 were fixed. |
| D4 | `src/arm_visualization.jl:101–102` | MED | Hardcoded `group_a_` / `group_b_` column prefixes; other group names throw `ArgumentError`. |
| D5 | `src/bayesian_wrappers.jl:166–181`, `src/network_group_stratification.jl:177–178` | LOW | Leftover `best_m`/`best_f`, `m_only`/`f_only` naming from the male/female origin — incomplete de-domaining, misleading to readers. |
| D6 | `src/hdp_wrappers.jl:231`, `:227` | MED | Group values are `lowercase`d into Symbols: groups `"A"` and `"a"` collide; the > 2-group branch doesn't lowercase (inconsistent with the 2-group branch). No sanitization for spaces/punctuation. |

### 2.3 Performance

| # | Location | Sev | Finding |
|---|----------|-----|---------|
| P1 | `src/network_construction.jl:100–127` + `statistical_validation.jl` | HIGH (at real-data scale) | The pairwise loop is O(M²) pairs, and each pair scans all records *twice* — `build_contingency_table` once for the stats and again inside `test_association`. Membership tests use `in` on sorted `Vector`s (O(k) each). One boolean record×item matrix `X` computed once gives *all* co-occurrence counts as `X'X` (BLAS); marginals give the other three cells. Expected 10–100× on realistic M, N. |
| P2 | `src/hdp_clustering.jl:164, 194, 288` | HIGH (flagship method) | `Xf = Float64.(X)` is re-materialized in `_hdp_e_step!`, `_hdp_update_atoms!`, *and* `_hdp_compute_elbo` — three fresh N×D allocations per CAVI iteration (the header advertises ~190K records). Hoist `Xf` (and `1 .- Xf`) into `_run_hdp_cavi` once; preallocate the N×K `log_lik` buffer. This is the single biggest speedup available. |
| P3 | `src/bayesian_clustering.jl:217` | MED | `(1.0 .- X_f) * log_1m_theta'` allocates an N×D temporary every EM iteration; precompute `1 .- X_f` once per run. |
| P4 | `src/network_metrics.jl:104–109, ~213` | MED | Louvain rescans all n nodes to recompute `s_tot` for every candidate community of every node (O(n²·deg) per sweep). Maintain per-community strength sums incrementally (this also fixes C2). |
| P5 | `src/data_preparation.jl:201–204`, `network_construction.jl:93–97` | LOW | One-hot construction and prevalence counts do `item in ps` linear scans per record per item — O(N·M·k). Use `Set`s per record. Fine at fixture scale; matters at 190K×hundreds. |
| P6 | `src/overview_visualization.jl:99–110` | LOW | Per-dot `scatter!` in nested loops creates up to ~1,200 plot objects; two vectorized scatters suffice. |

### 2.4 Tests

| # | Location | Sev | Finding |
|---|----------|-----|---------|
| T1 | `test/runtests.jl:304, 311, 322, 531, 920, 943` | HIGH | `if nrow(...) > 0` guards make the `compare_strata`, `compare_networks`, and comparison-plot tests pass *vacuously* if a regression yields empty results. Assert non-emptiness. |
| T2 | `test/runtests.jl:1194–1209` | HIGH | `hdp_cluster_categorization` — the package's headline capability — is only checked for "category is a valid symbol", on a fixture where the interesting branch never fires. That is exactly why bug C3 survived. Plant a group-exclusive cluster (ground truth exists at `:1029–1093`) and assert the actual labels. |
| T3 | `test/runtests.jl:617–628, 751–768` | MED | EM planted-cluster and NUTS tests check shapes/simplex constraints only — no ARI vs truth, no θ-recovery check. Only HDP and `select_K` test statistical correctness. |
| T4 | exports | MED | `fit_bernoulli_mixture_advi` / `bernoulli_clustering_advi` are exported with zero tests (and the `raw[s].data` unpacking at `bayesian_turing.jl:365–367` is version-fragile). Also untested: `results_summary`, `network_summary`, `plot_network_comparison`, `:holm` correction, > 2-group HDP. |
| T5 | throughout | LOW | Every stochastic test pins `MersenneTwister` seeds with hard thresholds — a Julia RNG-stream change flips them. Use `StableRNGs`. |
| T6 | `test/runtests.jl:476` | LOW | `modularity >= 0.0 \|\| n_communities == 1` is a near-tautology (modularity can legitimately be negative). |

### 2.5 Readability / extensibility

- **Duplication.** The per-group driver skeleton is copied three times (~100 of
  ~370 driver lines); the group-item filter block three times (D1); the item
  filter/order block three times in `bayesian_visualization.jl` (`:24–35,
  160–171, 316–327`); node-size/edge-width scaling and layout selection twice+
  in `network_visualization.jl` (`:34–82` vs `:135–168`, `:384`) — and the
  copies have already drifted (hardcoded 15/35 vs parameterized).
- **Magic numbers without kwargs**: universal-vs-unequal ratio `2.0` and floor
  `1e-8` (`hdp_wrappers.jl:218–219`), negligible-cluster `0.01` hardcoded twice
  with mismatched `<` vs `<=` (`hdp_wrappers.jl:266,276`), effective-K `0.95`
  (`hdp_clustering.jl:108`).
- **Pipelines** build a `Dict{Symbol,Any}` then convert to `NamedTuple` — the
  return type varies by keyword flags, is type-unstable, and undiscoverable.
  `println`-based progress with no `verbose` switch pollutes non-interactive use.
- **Contract drift**: `HDPClusteringResult` docstring promises a `group_label`
  column, code emits `group` (`hdp_wrappers.jl:15` vs `:163`); colormap
  convention (`:YlOrRd`) violated by `:Blues` at `bayesian_visualization.jl:423,484`;
  `weight_metric` label on the comparison colorbar is cosmetic
  (`network_visualization.jl:246`); empty-input behavior differs between sibling
  plot functions (error vs blank figure).
- **Type fragility**: `hdp_wrappers.jl:88` assumes `Int` record ids;
  `group_stratification.jl:127–140` builds columns as `Vector{Any}`, discarding
  the intended `Union{Missing,Float64}` eltypes; item names become raw DataFrame
  columns and can shadow metadata columns like `cluster` (`hdp_wrappers.jl:148–150`).
- **Positives worth keeping**: the core/wrapper split is real and consistently
  followed; the HDP header comment and the ADVI/NUTS scaling write-up in
  `scripts/README.md` are exemplary; docstrings are near-universal; the
  timing-filter semantics are carefully documented and hand-verified in tests.

---

## 3. Refactoring plan

Ordered so that every phase leaves the package releasable. Phases 1–2 are
pure fixes (no API change); Phase 3 is the one breaking change; Phases 4–5
consolidate. Do not reorder 3 before 1 — the categorization fix (C3) changes
what tests must assert, and T2's non-vacuous test should land *with* it.

### Phase 1 — Correctness fixes (small diffs, one per commit, each with a regression test)

1. **Louvain gain (C1 + C2 + P4 together).** Rewrite `_modularity_gain` as the
   standard form: remove the node from its community first, maintain
   per-community strength sums in a vector updated on each move, and use
   `2k_in/m2 − 2·s_tot·k_i/m2²` (or equivalently drop the common factor 2 from
   *both* terms). Regression test: on a small two-clique graph, assert the
   detected partition maximizes the file's own `_modularity`, and that a move
   accepted by `_modularity_gain` never decreases `_modularity`.
2. **HDP category symbols (C3).** Emit the documented `:group_<value>_only`
   form (and lowercase consistently in the >2-group branch, D6). Land with the
   T2 planted-truth test so the fix is actually exercised.
3. **Group-detection regexes (C4).** Replace `^f`/`^m` and `^a`/`^b` regex
   detection with direct iteration over `unique(event_df.Group)` — this also
   removes the last silent male/female leftovers.
4. **Odds ratio (C5).** One shared `odds_ratio(ct; correction=:haldane)` helper
   in `statistical_validation.jl`; use it from both call sites.
5. **EM guards (C6, C7, C8).** Clamp `N_k` and `pi_k` to `eps()`, guard/reinit
   empty components, delete the dead reordering lines, and recompute the final
   LL after the last M-step (or return the pre-M-step parameters) so params,
   responsibilities, LL, and BIC describe the same state.
6. **HDP convergence (C9).** Stop on best-ELBO plateau (e.g., no improvement
   over a 5-iteration window) given the documented non-monotone update; guard
   the all-NaN-restart case with an informative error.
7. **Small ones**: rule-key collision (C10 — key on `(Set(LHS), RHS)`), ARI
   `missing` for <2 shared nodes (C11), `Statistics.quantile` for CIs (C12),
   butterfly categorization wiring (C14), colorbar/colorrange mismatch (C15),
   `_hclust_order` docstring (C13).

*Everything in Phase 1 is API-compatible. Estimated size: ~10 focused commits.*

### Phase 2 — Performance (API-compatible)

1. **Vectorize pairwise associations (P1).** Inside
   `compute_pairwise_associations`, build the record×item `BitMatrix` once
   (reuse `_build_transactions_internal`), compute `C = X'X`, and derive all
   contingency cells from `C` and column sums. Pass the precomputed table into
   `test_association` instead of re-scanning records. This subsumes P5.
2. **Hoist HDP allocations (P2).** `_run_hdp_cavi` computes `Xf` and `OneMinusXf`
   once and passes them to the E-step, atom update, and ELBO; preallocate the
   N×K work buffer. Verify ELBO trajectory unchanged (same rng ⇒ bit-identical).
3. **EM temporary (P3)** and the vectorized prevalence scatter (P6).

*Benchmark before/after with a ~10K-record synthetic set; record numbers in
`scripts/README.md`.*

### Phase 3 — Make the domain-neutral, N-group contract real (the breaking change)

1. **Evict fixture knowledge (D1).** Delete `GROUP_A_ONLY_ITEMS`/`GROUP_B_ONLY_ITEMS`
   and their exports. Add one keyword threaded through
   `build_transactions`/`compute_pairwise_associations`/`build_cooccurrence_network`:
   `exclusive_items::Union{Nothing,Dict{<:AbstractString,<:AbstractSet}} = nothing`
   (group value → items to *drop* when filtering to that group's complement, i.e.
   the current semantics, supplied by the caller or a data file). The test
   fixture moves its constants into `test/runtests.jl`.
2. **One generic stratification driver (D2).** New function in
   `group_stratification.jl`:
   `stratify_by(analysis_fn, event_df; group_col=:Group, kwargs...) ->
   OrderedDict{String,Any}` iterating `sort(unique(event_df[!, group_col]))`.
   The three drivers become thin calls. **Key results by group value in a Dict
   rather than splicing values into Symbol field names** — this eliminates the
   lowercase-collision problem (D6), the illusory dynamic naming, and the >2-group
   ceiling in one move.
3. **Update consumers**: `compare_strata`/`compare_networks`/`compare_clusterings`
   take two named entries from that Dict (explicitly pairwise — document it);
   pipelines (D3) and `plot_arm_comparison` (D4) iterate actual group values;
   purge `m_`/`f_` variable names (D5).
4. **Migration notes** in README: `strat.group_a_rules` → `strat["A"].rules`.
   Since HDP already carries `group_labels`, only the three per-group methods
   change shape.

*This is the phase that makes "swap in any domain" true. It should land as a
single PR with README, notebooks, and tests updated together.*

### Phase 4 — Structure and API hygiene

1. **Shared helpers file** (`src/plot_utils.jl`): item filter/order block,
   min-max node/edge scaling, layout selection, empty-input policy (always
   `error` with a clear message), colormap constants. Collapses the
   visualization duplication and drift.
2. **Pipelines return structs** (`FullPipelineResult`, etc.) with `Union{Nothing,...}`
   stratification fields instead of Dict→NamedTuple; add `verbose::Bool=true`
   (route prints through `@info` or a guarded `println`).
3. **Promote magic numbers to kwargs**: `universal_ratio=2.0`,
   `negligible_threshold=0.01` (used consistently, one comparison operator),
   `effective_K` threshold.
4. **Type robustness**: generic id eltype in `hdp_wrappers.jl:88`
   (`empty(similar(ids, 0))` or just `eltype`-driven), typed column vectors in
   `compare_strata`, prefixed metadata columns (`:_cluster`) or a reserved-name
   check against item columns.
5. Fix docstring/field drift (`group_label` vs `group`), colormap convention.

### Phase 5 — Test hardening

1. Convert the six `if nrow > 0` guards into `@test nrow(...) > 0` (T1).
2. Planted-recovery assertions for EM (ARI vs truth) and for
   `hdp_cluster_categorization` with a group-exclusive planted cluster (T2, T3
   — lands with Phase 1.2).
3. Add: ADVI smoke+shape test on a tiny matrix, `:holm` branch, a 3-group
   stratification test (post-Phase 3), boundary tests for `significance_stars`.
4. Switch stochastic tests to `StableRNGs` (T5); replace the modularity
   tautology with the Phase-1 Louvain regression test (T6).

### Explicitly out of scope / not recommended

- **Splitting into submodules.** At ~4,600 source lines the flat
  include-manifest is fine; the core/wrapper convention does the real
  layering. Revisit only if the package doubles.
- **"Fixing" the Turing/ADVI scaling.** Already correctly documented as a
  boundary in `scripts/README.md`; the plan above only adds tests around it.
- **Full two-level HDP (table indicators).** The documented single-level
  approximation is a reasonable engineering choice; Phase 1.6 merely makes the
  stopping rule honest about it.

---

## 4. Suggested sequencing

| Phase | Risk | API break | Rough size |
|-------|------|-----------|-----------|
| 1 Correctness | Low (each commit test-backed) | No | ~10 small commits |
| 2 Performance | Low (bit-identical checks) | No | 3 commits |
| 3 N-group contract | Medium (touches many files) | **Yes** | 1 coordinated PR |
| 4 Hygiene | Low | Minor (pipeline return type) | 4–5 commits |
| 5 Tests | None | No | interleave with 1 & 3 |

Phase 1 items 1–3 (Louvain gain, category symbols, prevalence regexes) are the
three bugs currently producing silently wrong output — fix those first
regardless of whether the rest of the plan is adopted.
