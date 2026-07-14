# Group-stratified posterior inference on the BIC-best K from EM.
# Uses ADVI (variational inference) for tractability at full-cohort scale.
# Reports CIs that should be read as lower bounds (meanfield underestimates
# uncertainty); for tighter manuscript figures rerun a chosen subset under
# fit_bernoulli_mixture_turing (NUTS).
#
# Saves: posterior_group_a_advi.csv, posterior_group_b_advi.csv (long-form CIs)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis
using DataFrames
using Random
using Printf
import CSV

const DATA_PATH   = joinpath(@__DIR__, "..", "..", "ddata", "synthetic_events.arrow")
const OUTPUT_DIR  = joinpath(@__DIR__, "..", "..", "ddata", "posterior_results")
const N_SAMPLES   = 1000
const MAX_ITER    = 1500
const SUBSAMPLE   = 1_500    # ADVI on a small random subset per stratum
                             # (marginalized ADVI does not scale; see scripts/README.md)
const SEED        = 2026

mkpath(OUTPUT_DIR)

function fit_and_summarize(event_df::DataFrame, group::String)
    println("\n═══ $group cohort ═══")

    # Step 1: EM with EB prior to pick K
    println("Step 1/2: EM model selection (K=2:6, prior=:empirical_bayes)...")
    em_t = @elapsed em = bernoulli_clustering(event_df;
        group_filter=group, K_range=2:6,
        prior=:empirical_bayes, concentration=10.0, floor=1.0,
        n_init=5, rng=MersenneTwister(SEED))
    K = em.model_selection.best_K
    println("  EM done in $(round(em_t, digits=1))s. Best K = $K (BIC $(round(em.model_selection.best.bic, digits=1))).")
    println("  Records: $(em.n_records), Items: $(length(em.item_names))")

    # Step 2: ADVI on the chosen K, subsampled for tractability
    println("Step 2/2: ADVI at K=$K (subsample=$SUBSAMPLE, max_iter=$MAX_ITER, n_samples=$N_SAMPLES)...")
    advi_t = @elapsed (post, items) = bernoulli_clustering_advi(
        event_df, K;
        group_filter=group,
        prior=:empirical_bayes, concentration=10.0, floor=1.0,
        subsample=SUBSAMPLE,
        max_iter=MAX_ITER, n_samples=N_SAMPLES,
        rng=MersenneTwister(SEED + 1))
    println("  ADVI done in $(round(advi_t, digits=1))s.")

    # Posterior means + CIs
    pi_mean = vec(sum(post.pi_samples, dims=1)) ./ size(post.pi_samples, 1)
    theta_mean = dropdims(sum(post.theta_samples, dims=1), dims=1) ./
                 size(post.theta_samples, 1)

    println("\n  Mixing proportions (mean [95% CI]):")
    for k in 1:post.K
        @printf "    π_%d: %.3f [%.3f, %.3f]\n" k pi_mean[k] post.pi_ci[k, 1] post.pi_ci[k, 2]
    end

    rows = NamedTuple[]
    for k in 1:post.K, d in 1:length(items)
        push!(rows, (group=group, K=K, cluster=k, item=items[d],
                     pi_mean=pi_mean[k],
                     pi_ci_low=post.pi_ci[k, 1],
                     pi_ci_high=post.pi_ci[k, 2],
                     theta_mean=theta_mean[k, d],
                     theta_ci_low=post.theta_ci[k, d, 1],
                     theta_ci_high=post.theta_ci[k, d, 2]))
    end
    df = DataFrame(rows)

    println("\n  Top 5 items per cluster (θ mean [95% CI]):")
    for k in 1:post.K
        sub = filter(r -> r.cluster == k, df)
        sort!(sub, :theta_mean, rev=true)
        @printf "    Cluster %d (%.1f%% [%.1f%%, %.1f%%]):\n" k (pi_mean[k] * 100) (post.pi_ci[k, 1] * 100) (post.pi_ci[k, 2] * 100)
        shown = 0
        for r in eachrow(sub)
            r.theta_mean < 0.05 && break
            @printf "      %-40s %.3f [%.3f, %.3f]\n" r.item r.theta_mean r.theta_ci_low r.theta_ci_high
            shown += 1
            shown >= 5 && break
        end
    end

    return em, post, df
end

function main()
    println("Loading $(DATA_PATH)...")
    event_df = load_event_data(DATA_PATH)
    println("  Total records: $(length(unique(event_df.id)))")

    em_m, post_m, df_m = fit_and_summarize(event_df, "A")
    em_f, post_f, df_f = fit_and_summarize(event_df, "B")

    out_m = joinpath(OUTPUT_DIR, "posterior_group_a_advi.csv")
    out_f = joinpath(OUTPUT_DIR, "posterior_group_b_advi.csv")
    CSV.write(out_m, df_m)
    CSV.write(out_f, df_f)
    println("\n══ Wrote results ══")
    println("  $out_m")
    println("  $out_f")
end

main()
