# Analysis Scripts

End-to-end scripts that exercise the package on the synthetic data in
`ddata/synthetic_events.arrow`. Run from the repo root:

```bash
julia --project=clustering clustering/scripts/<script>.jl
```

## `convert_odata_to_arrow.jl`

One-time conversion: reads the three raw tabular CSV subsets from `odata/` and
writes a single `odata/real_data.arrow`. The Arrow file is gitignored
alongside the source CSVs. Once created, pass it to any pipeline function
exactly as you would `ddata/synthetic_events.arrow`.

**Status:** ready to run. Handles BOM-prefixed column names (`normalizenames=true`)
and quoted strings with embedded commas (e.g. income ranges).

## `run_hdp_demo.jl`

HDP Bernoulli mixture validation on the full synthetic cohort. Loads
`ddata/synthetic_events.arrow`, fits `hdp_clustering` with
`group_by=:Group, K_max=20, prior=:empirical_bayes`, and reports:

- Dataset statistics (records, items, density, group split)
- Wall time, convergence, and effective K
- `clustering_summary` with per-cluster credible intervals
- `hdp_cluster_categorization` table (universal / group_a_only / group_b_only)
- Three figures saved to `scripts/output/`: `hdp_sticks.png`,
  `hdp_profiles.png`, `hdp_sharing.png`
- Sanity checks (β and π sums, active cluster count)

**Status:** runs end-to-end in ~30 seconds. As of 2026-04-29, effective K=6
on the full cohort: 1 group_b-only (Item01+GB3), 1 group_a-only (GA1),
3 universal (Item03, Item07, mixed), 1 group_a-leaning (Item04+Item06+Item05).

## `compare_priors.jl`

Compares Bernoulli mixture clustering with `prior=:flat` vs `prior=:empirical_bayes`
on the full multi-item cohort (~200K records, 41 items). Reports class
profiles, BIC values, and the Adjusted Rand Index between the two assignments.

**Status:** works end-to-end. Each prior takes ~30s for K=2:6 with 5 random
restarts. As of 2026-04-28, both priors select K=6, with ARI ≈ 0.95 between
their hard assignments.

## `turing_stratified.jl`

Group-stratified pipeline: EM with empirical-Bayes prior to choose K, then ADVI
on the chosen K to get credible intervals. Saves long-form posterior CIs as
`ddata/posterior_results/posterior_{group_a,group_b}_advi.csv`.

**Status:** works on small data (verified by unit tests) but **does not scale
to the full multi-item cohort**. As of 2026-04-28, attempts at full or
10K-subsampled ADVI did not complete in 20+ minutes regardless of AD backend
(ForwardDiff, ReverseDiff). The bottleneck is the marginalized log-likelihood
(logsumexp over K clusters inside an N-record loop), which builds a long AD
tape that does not vectorize cleanly.

To make this practical at dataset scale, future work should:
- Implement mini-batch ADVI (subsample records per ELBO evaluation)
- Vectorize the per-record loop or rewrite using `arraydist` / `loglikelihood`
- Try Mooncake or Enzyme as the AD backend
- Or, accept point estimates from EM and only run posteriors on small
  subsets (e.g., a single rare cluster's records) where uncertainty matters
  for the substantive question

The script is left in place because the infrastructure is correct; only the
runtime is the open issue.

## `benchmark_phase2.jl`

Micro-benchmark for the Phase-2 performance work (P1 matrix-based pairwise
associations; P2 hoisting `Xf` out of the HDP CAVI loop). Generates a synthetic
~10K-record event dataset (~60 items) and times the current code against an
inline scan-based baseline that reproduces the pre-P1 double record scan.

**Results (2026-07-10, Julia 1.12, Apple Silicon; ~10K records, 60 items):**

| Path | Time | Note |
|------|------|------|
| Pairwise, scan baseline (double record scan) | 1.06 s | old O(M²·N), two scans/pair |
| Pairwise, matrix path (`C = XᵀX`) | 0.34 s | **~3.1× faster here** |
| HDP CAVI fit (K_max=10, n_init=1) | 0.31 s | 228 MB allocated |

The pairwise speedup grows with the item count M (the scan cost is O(M²·N) and
each pair also did O(k) `in` lookups), and with N. On this 60-item set the
shared Fisher-exact test per pair is a floor both paths pay, so 3× understates
the win on wider data. P2 changes no numerics (verified bit-identical over
fixed-rng fits); it removes three N×D float allocations per CAVI iteration,
which dominates memory on the ~190K-record flagship path.
