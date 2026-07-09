# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`CooccurrenceAnalysis` is a Julia package for discovering co-occurrence patterns in binary (presence/absence) event data via four methods: HDP Bernoulli mixture (main), flat-K Bernoulli mixture, association rule mining, and network analysis. The data model is fully domain-neutral: **entities** (records) accumulate **events**, each event carries one **item**, and records may belong to **groups** (strata). **README.md is the authoritative catalog** of methods, key functions, and plot functions — consult it before reinventing; this file covers what the README does not.

## Commands

```bash
# Tests (320 @test assertions) — always run with the project active.
# The suite builds its own in-code fixture, so no external data file is needed.
julia --project=. -e 'include("test/runtests.jl")'

# Scripts expect a data path; run from the parent dir with this project active
julia --project=cooccurrence-analysis cooccurrence-analysis/scripts/run_hdp_demo.jl

# Notebooks are Jupytext percent-format .jl files; convert to run
jupytext --to notebook analysis_notebook.jl
```

There is no separate build or lint step. `Revise` is a dependency — src edits hot-reload in a live session.

## Data contract

All pipelines take a path to an **Arrow file** of **event-level** rows (one row per event, multiple rows per entity), with at least `:id`, `:item`, `:Group` columns; timing analysis also needs `:year`. `build_transactions` collapses event rows into a one-hot record × item boolean DataFrame — the shared input representation feeding every method. Records with `< min_items` distinct items (default 2) are excluded, since they cannot express co-occurrence.

## Architecture

Single flat module: `src/CooccurrenceAnalysis.jl` is a manifest of `include`s + `export`s; there are no submodules. Include order matters (utils → data_preparation → mining/network/bayesian). Each method follows a **core / wrapper** split you must respect when extending:

- **core** file (`hdp_clustering.jl`, `bayesian_clustering.jl`, `mining.jl`, `network_construction.jl`) — domain-agnostic algorithm operating on matrices/transactions, returns a plain `*Result` struct.
- **wrapper** file (`hdp_wrappers.jl`, `bayesian_wrappers.jl`, `group_stratification.jl`) — attaches item names and group categorization, returning a domain-facing result.

Add algorithm logic to the core; add naming/grouping handling to the wrapper. The three `run_*_pipeline` functions (in the main module file) are the top-level entry points that chain load → fit → stratify → compare for ARM, network, and flat-K clustering respectively.

### Group stratification is two different mechanisms

- **HDP** handles groups *natively*: one joint fit with per-group mixing weights (`group_by=:Group`), so `hdp_cluster_categorization` can label a cluster `universal` / `group_a_only` / `group_b_only` from a single posterior. This is why HDP is the main method.
- **ARM, network, flat-K** stratify by *running the method separately per group*, then diffing the results (`compare_strata`, `compare_networks`, `compare_clusterings`). Per-group field names are built dynamically as `group_<lowercase(value)>_*`, so the two group values in the fixture are `"A"` and `"B"`.

### Known scaling boundary

Turing.jl NUTS/ADVI paths (`bayesian_turing.jl`, `turing_stratified.jl`) are correct but **do not scale past small subsets** — the marginalized log-likelihood builds an AD tape that doesn't vectorize. Use EM (`fit_bernoulli_mixture`) or HDP CAVI for full-size work; reserve Turing for credible intervals on small data. See `scripts/README.md` for the full write-up before attempting to "fix" the runtime.

## Conventions

- Plot functions return a `Makie.Figure` (CairoMakie backend); `wong_colors` for categorical, `:YlOrRd` for heatmaps.
- The mathematical vocabulary (Bernoulli mixture, HDP, stick-breaking, Dirichlet, lift, phi coefficient, FP-Growth, BIC, CAVI, ELBO) is standard; the item/record/group naming is a deliberately neutral placeholder — swap in any application domain without touching the math.
