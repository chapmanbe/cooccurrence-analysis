# ──────────────────────────────────────────────────────────────────────────────
# Turing.jl variant: full-Bayes inference on the BIC-best Bernoulli mixture.
#
# The EM workflow in `bayesian_clustering.jl` selects K via BIC and produces
# point estimates of (π, θ). This file adds an HMC/NUTS layer that samples
# the full posterior given a chosen K, yielding credible intervals for π_k,
# θ_{k,d}, and per-record cluster membership probabilities.
#
# Cluster assignments z_n are marginalized analytically — the model is fully
# continuous, so NUTS handles it without particle Gibbs. Per-record cluster
# responsibilities are recovered post-hoc.
# ──────────────────────────────────────────────────────────────────────────────

using Turing
using Turing: AutoForwardDiff, AutoReverseDiff
using Turing.Variational: vi, q_meanfield_gaussian
using MCMCChains: Chains
using LinearAlgebra: dot
import ReverseDiff  # registers AD backend with AdvancedVI/Turing

"""
    BernoulliMixturePosterior

Posterior samples and summaries from a Turing fit of a K-component Bernoulli
mixture with continuous parameters and marginalized cluster assignments.

# Fields
- `K`: number of mixture components
- `chains`: full `MCMCChains.Chains` object (raw samples)
- `pi_samples`: S × K matrix of posterior samples for the mixing proportions
- `theta_samples`: S × K × D array of posterior samples for the Bernoulli probs
- `pi_ci`: K × 2 matrix of (lower, upper) credible intervals on π
- `theta_ci`: K × D × 2 array of credible intervals on θ
- `responsibility_means`: N × K mean per-record cluster responsibility
- `assignments`: N-vector of MAP cluster labels (argmax of responsibility_means)
- `ci_level`: credible interval level (e.g. 0.95)
"""
struct BernoulliMixturePosterior
    K::Int
    chains::Chains
    pi_samples::Matrix{Float64}
    theta_samples::Array{Float64, 3}
    pi_ci::Matrix{Float64}
    theta_ci::Array{Float64, 3}
    responsibility_means::Matrix{Float64}
    assignments::Vector{Int}
    ci_level::Float64
end

"""
    bernoulli_mixture_marginal(X, K, alpha_vec, beta_vec, dirichlet_concentration)

Turing model. Cluster assignments are marginalized — the per-record log-likelihood
is the log-sum-exp over clusters of `log(π_k) + sum_d log_bernoulli(x_{n,d} | θ_{k,d})`.
"""
@model function bernoulli_mixture_marginal(X::AbstractMatrix{Bool},
                                            K::Int,
                                            alpha_vec::Vector{Float64},
                                            beta_vec::Vector{Float64},
                                            dirichlet_concentration::Float64)
    N, D = size(X)

    π ~ Dirichlet(fill(dirichlet_concentration, K))

    # Use eltype(π) so ForwardDiff Dual types propagate through θ
    T = eltype(π)
    θ = Matrix{T}(undef, K, D)
    for k in 1:K, d in 1:D
        θ[k, d] ~ Beta(alpha_vec[d], beta_vec[d])
    end

    log_π = log.(π)
    log_θ = log.(θ)
    log_1m_θ = log.(one(T) .- θ)

    logps = Vector{T}(undef, K)
    for n in 1:N
        for k in 1:K
            ll = log_π[k]
            for d in 1:D
                ll += X[n, d] ? log_θ[k, d] : log_1m_θ[k, d]
            end
            logps[k] = ll
        end
        Turing.@addlogprob! logsumexp(logps)
    end
end

"""
    logsumexp(xs::AbstractVector{<:Real}) -> Real

Numerically stable log-sum-exp.
"""
function logsumexp(xs::AbstractVector{<:Real})
    m = maximum(xs)
    return m + log(sum(exp.(xs .- m)))
end

"""
    fit_bernoulli_mixture_turing(X::AbstractMatrix{Bool}, K::Int;
                                  prior=:flat,
                                  alpha_prior=1.0, beta_prior=1.0,
                                  concentration=10.0, floor=1.0,
                                  dirichlet_concentration=1.0,
                                  n_samples=1000, n_chains=2,
                                  ci_level=0.95,
                                  rng=Random.GLOBAL_RNG) -> BernoulliMixturePosterior

Run NUTS on a marginalized K-component Bernoulli mixture and return posterior
samples plus summaries. Use this *after* selecting K via the EM-based pipeline
(`select_K` / `bernoulli_clustering`) to get credible intervals on the
chosen model.

The posterior is subject to label-switching: cluster k=1 in one chain is
typically not the same as k=1 in another chain, and may even drift within a
chain. This implementation aligns labels by sorting each posterior sample's
clusters by `π_k * sum(θ_k)` (overall activity), which is consistent with the
sorting applied in the EM `fit_bernoulli_mixture` function. For more rigorous
relabeling, use the raw `chains` field.

# Arguments
- `prior`: `:flat` or `:empirical_bayes` (Ye 2018, see `empirical_bayes_priors`)
- `n_samples`: NUTS samples per chain (post-warmup; warmup is `n_samples ÷ 2`)
- `n_chains`: number of independent chains
- `ci_level`: credible interval level for `pi_ci`, `theta_ci`
"""
function fit_bernoulli_mixture_turing(X::AbstractMatrix{Bool}, K::Int;
                                       prior::Symbol=:flat,
                                       alpha_prior::Float64=1.0,
                                       beta_prior::Float64=1.0,
                                       concentration::Float64=10.0,
                                       floor::Float64=1.0,
                                       dirichlet_concentration::Float64=1.0,
                                       n_samples::Int=1000,
                                       n_chains::Int=2,
                                       ci_level::Float64=0.95,
                                       rng::AbstractRNG=Random.GLOBAL_RNG)
    N, D = size(X)
    N == 0 && error("Cannot fit Turing mixture to empty data")
    K < 1 && error("K must be >= 1")

    alpha_vec, beta_vec = if prior === :flat
        (fill(alpha_prior, D), fill(beta_prior, D))
    elseif prior === :empirical_bayes
        empirical_bayes_priors(X; concentration, floor)
    else
        error("Unknown prior :$prior. Use :flat or :empirical_bayes.")
    end

    model = bernoulli_mixture_marginal(X, K, alpha_vec, beta_vec,
                                        dirichlet_concentration)
    sampler = NUTS()
    n_warmup = n_samples ÷ 2

    chains = if n_chains == 1
        sample(rng, model, sampler, n_warmup + n_samples;
               discard_initial=n_warmup, progress=false)
    else
        sample(rng, model, sampler, MCMCThreads(),
               n_warmup + n_samples, n_chains;
               discard_initial=n_warmup, progress=false)
    end

    pi_samples, theta_samples = _extract_pi_theta(chains, K, D)
    pi_samples, theta_samples = _align_labels!(pi_samples, theta_samples)

    pi_ci = _credible_intervals(pi_samples, ci_level)
    theta_ci = _credible_intervals_3d(theta_samples, ci_level)

    resp_means = _posterior_responsibilities(X, pi_samples, theta_samples)
    assignments = [argmax(@view resp_means[n, :]) for n in 1:N]

    return BernoulliMixturePosterior(K, chains, pi_samples, theta_samples,
                                      pi_ci, theta_ci,
                                      resp_means, assignments, ci_level)
end

"""
    _extract_pi_theta(chains, K, D) -> (pi_samples, theta_samples)

Extract π and θ samples from a Turing chain into (S, K) and (S, K, D) arrays.
"""
function _extract_pi_theta(chains::Chains, K::Int, D::Int)
    # Pull each parameter as a flat vector across all iterations × chains
    pi_cols = [vec(Array(chains[Symbol("π[$k]")])) for k in 1:K]
    S = length(pi_cols[1])
    pi_arr = Matrix{Float64}(undef, S, K)
    for k in 1:K
        pi_arr[:, k] = pi_cols[k]
    end

    theta_arr = Array{Float64}(undef, S, K, D)
    for k in 1:K, d in 1:D
        col = vec(Array(chains[Symbol("θ[$k, $d]")]))
        theta_arr[:, k, d] = col
    end

    return pi_arr, theta_arr
end

"""
    _align_labels!(pi_samples, theta_samples) -> (pi_samples, theta_samples)

For each posterior sample, sort the K clusters by π_k * sum(θ_k) (overall
activity). This matches the post-fit sorting in `fit_bernoulli_mixture` and
gives a partial defense against label-switching across samples.
"""
function _align_labels!(pi_samples::Matrix{Float64}, theta_samples::Array{Float64, 3})
    S, K = size(pi_samples)
    for s in 1:S
        activity = [pi_samples[s, k] * sum(@view theta_samples[s, k, :]) for k in 1:K]
        order = sortperm(activity, rev=true)
        if order != 1:K
            pi_samples[s, :] = pi_samples[s, order]
            theta_samples[s, :, :] = theta_samples[s, order, :]
        end
    end
    return pi_samples, theta_samples
end

"""
    _credible_intervals(samples::AbstractMatrix, level) -> Matrix

Return K × 2 matrix of (lower, upper) credible intervals for each column.
"""
function _credible_intervals(samples::AbstractMatrix, level::Float64)
    K = size(samples, 2)
    α = (1.0 - level) / 2.0
    ci = Matrix{Float64}(undef, K, 2)
    for k in 1:K
        col = @view samples[:, k]
        ci[k, 1] = quantile(col, α)
        ci[k, 2] = quantile(col, 1.0 - α)
    end
    return ci
end

"""
    _credible_intervals_3d(samples::Array{T,3}, level) -> Array

Apply per-(k, d) credible intervals across the first dimension.
"""
function _credible_intervals_3d(samples::Array{Float64, 3}, level::Float64)
    S, K, D = size(samples)
    α = (1.0 - level) / 2.0
    ci = Array{Float64}(undef, K, D, 2)
    for k in 1:K, d in 1:D
        col = @view samples[:, k, d]
        ci[k, d, 1] = quantile(col, α)
        ci[k, d, 2] = quantile(col, 1.0 - α)
    end
    return ci
end

"""
    _posterior_responsibilities(X, pi_samples, theta_samples) -> Matrix

For each posterior sample, compute the responsibilities r_{n,k} = P(z_n=k | x_n, π, θ),
then average across samples to get the posterior mean responsibility matrix.
"""
function _posterior_responsibilities(X::AbstractMatrix{Bool},
                                      pi_samples::Matrix{Float64},
                                      theta_samples::Array{Float64, 3})
    N, D = size(X)
    S, K = size(pi_samples)
    resp_sum = zeros(N, K)

    for s in 1:S
        log_π = log.(pi_samples[s, :])
        for n in 1:N
            logps = Vector{Float64}(undef, K)
            for k in 1:K
                ll = log_π[k]
                for d in 1:D
                    θ_kd = theta_samples[s, k, d]
                    ll += X[n, d] ? log(θ_kd) : log(1.0 - θ_kd)
                end
                logps[k] = ll
            end
            m = maximum(logps)
            denom = m + log(sum(exp.(logps .- m)))
            for k in 1:K
                resp_sum[n, k] += exp(logps[k] - denom)
            end
        end
    end

    return resp_sum ./ S
end

"""
    fit_bernoulli_mixture_advi(X::AbstractMatrix{Bool}, K::Int;
                                prior=:flat,
                                alpha_prior=1.0, beta_prior=1.0,
                                concentration=10.0, floor=1.0,
                                dirichlet_concentration=1.0,
                                max_iter=2000, n_samples=500,
                                ci_level=0.95,
                                rng=Random.GLOBAL_RNG) -> BernoulliMixturePosterior

Fast variational alternative to NUTS. Fits a meanfield Gaussian variational
approximation to the posterior, then draws `n_samples` from the variational
distribution and packages them into the same `BernoulliMixturePosterior` type
returned by `fit_bernoulli_mixture_turing`.

ADVI is **orders of magnitude faster** than NUTS on small-to-moderate data,
which makes it the right tool when MCMC is intractable. The cost is
approximation: the meanfield factorization assumes posterior independence
between every variable, which typically **underestimates uncertainty** —
credible intervals from ADVI should be read as lower bounds on the true
posterior spread. For final manuscript figures on chosen subsets, prefer NUTS.

# Scaling caveat (as of 2026-04-28)
This implementation has been verified on small data (unit tests, planted
clusters with N ≤ 1000). It does **not** scale to dataset-cohort sizes
(~10K-100K multi-item records) within reasonable wall time, even with
`adtype=AutoReverseDiff()`. The bottleneck is the marginalized likelihood:
the per-record log-sum-exp over K clusters builds a long AD tape that does
not vectorize cleanly. Practical paths forward (not implemented):
- Mini-batch ADVI (subsample records per ELBO step)
- Vectorize the per-record loop using `loglikelihood` or `arraydist`
- Try Mooncake or Enzyme AD backends

# Arguments
- `max_iter`: number of variational optimization iterations (default 2000)
- `n_samples`: number of draws from the fitted variational distribution
"""
function fit_bernoulli_mixture_advi(X::AbstractMatrix{Bool}, K::Int;
                                     prior::Symbol=:flat,
                                     alpha_prior::Float64=1.0,
                                     beta_prior::Float64=1.0,
                                     concentration::Float64=10.0,
                                     floor::Float64=1.0,
                                     dirichlet_concentration::Float64=1.0,
                                     max_iter::Int=2000,
                                     n_samples::Int=500,
                                     ci_level::Float64=0.95,
                                     adtype=AutoReverseDiff(),
                                     rng::AbstractRNG=Random.GLOBAL_RNG)
    N, D = size(X)
    N == 0 && error("Cannot fit ADVI mixture to empty data")
    K < 1 && error("K must be >= 1")

    alpha_vec, beta_vec = if prior === :flat
        (fill(alpha_prior, D), fill(beta_prior, D))
    elseif prior === :empirical_bayes
        empirical_bayes_priors(X; concentration, floor)
    else
        error("Unknown prior :$prior. Use :flat or :empirical_bayes.")
    end

    model = bernoulli_mixture_marginal(X, K, alpha_vec, beta_vec,
                                        dirichlet_concentration)
    q_fit = vi(model, q_meanfield_gaussian, max_iter;
               adtype=adtype, show_progress=false)

    # Sample from the fitted variational distribution. Each draw is a
    # VarNamedTuple whose fields are accessed via the inner `data` NamedTuple.
    raw = rand(rng, q_fit, n_samples)
    pi_samples = Matrix{Float64}(undef, n_samples, K)
    theta_samples = Array{Float64}(undef, n_samples, K, D)
    for s in 1:n_samples
        d = raw[s].data
        pi_samples[s, :] = d.π
        theta_samples[s, :, :] = d.θ
    end

    pi_samples, theta_samples = _align_labels!(pi_samples, theta_samples)
    pi_ci = _credible_intervals(pi_samples, ci_level)
    theta_ci = _credible_intervals_3d(theta_samples, ci_level)

    resp_means = _posterior_responsibilities(X, pi_samples, theta_samples)
    assignments = [argmax(@view resp_means[n, :]) for n in 1:N]

    # We don't have a Chains object from VI; supply an empty placeholder
    empty_chains = Chains(zeros(0, 0, 0))

    return BernoulliMixturePosterior(K, empty_chains, pi_samples, theta_samples,
                                      pi_ci, theta_ci,
                                      resp_means, assignments, ci_level)
end

"""
    bernoulli_clustering_advi(event_df, K; kwargs...)
        -> (post::BernoulliMixturePosterior, items::Vector{String})

Item-domain wrapper around `fit_bernoulli_mixture_advi`. Same signature as
`bernoulli_clustering_turing` but uses ADVI instead of NUTS.
"""
function bernoulli_clustering_advi(event_df::DataFrame, K::Int;
                                           group_filter::Union{AbstractString, Nothing}=nothing,
                                           min_items::Int=2,
                                           timing_filter::Symbol=:all,
                                           concurrent_window::Int=0,
                                           prior::Symbol=:flat,
                                           alpha_prior::Float64=1.0,
                                           beta_prior::Float64=1.0,
                                           concentration::Float64=10.0,
                                           floor::Float64=1.0,
                                           dirichlet_concentration::Float64=1.0,
                                           max_iter::Int=2000,
                                           n_samples::Int=500,
                                           ci_level::Float64=0.95,
                                           adtype=AutoReverseDiff(),
                                           subsample::Union{Int, Nothing}=nothing,
                                           rng::AbstractRNG=Random.GLOBAL_RNG)
    txns_df = build_transactions(event_df; group_filter, min_items,
                                  timing_filter, concurrent_window)
    X, item_names = transactions_to_matrix(txns_df)
    N = size(X, 1)
    N == 0 && error("No records with ≥$min_items items after filtering")

    if subsample !== nothing && subsample < N
        idx = randperm(rng, N)[1:subsample]
        X = X[idx, :]
    end

    post = fit_bernoulli_mixture_advi(X, K;
        prior, alpha_prior, beta_prior, concentration, floor,
        dirichlet_concentration, max_iter, n_samples, ci_level, adtype, rng)

    return post, item_names
end

"""
    bernoulli_clustering_turing(event_df::DataFrame, K::Int;
                                        group_filter=nothing, min_items=2,
                                        prior=:flat,
                                        alpha_prior=1.0, beta_prior=1.0,
                                        concentration=10.0, floor=1.0,
                                        dirichlet_concentration=1.0,
                                        n_samples=1000, n_chains=2,
                                        ci_level=0.95,
                                        rng=Random.GLOBAL_RNG)
        -> (posterior::BernoulliMixturePosterior, item_names::Vector{String})

Item-domain wrapper that builds the binary record × item matrix and runs
NUTS on the Turing mixture model. Recommended workflow:

    # 1. Choose K via the EM pipeline
    em_result = bernoulli_clustering(event_df; K_range=2:6,
                                             prior=:empirical_bayes)
    K = em_result.model_selection.best_K

    # 2. Refit chosen K with Turing for posterior inference
    post, items = bernoulli_clustering_turing(event_df, K;
                                                      prior=:empirical_bayes)
    posterior_summary(post)
"""
function bernoulli_clustering_turing(event_df::DataFrame, K::Int;
                                              group_filter::Union{AbstractString, Nothing}=nothing,
                                              min_items::Int=2,
                                              timing_filter::Symbol=:all,
                                              concurrent_window::Int=0,
                                              prior::Symbol=:flat,
                                              alpha_prior::Float64=1.0,
                                              beta_prior::Float64=1.0,
                                              concentration::Float64=10.0,
                                              floor::Float64=1.0,
                                              dirichlet_concentration::Float64=1.0,
                                              n_samples::Int=1000,
                                              n_chains::Int=2,
                                              ci_level::Float64=0.95,
                                              rng::AbstractRNG=Random.GLOBAL_RNG)
    txns_df = build_transactions(event_df; group_filter, min_items,
                                  timing_filter, concurrent_window)
    X, item_names = transactions_to_matrix(txns_df)
    size(X, 1) == 0 && error("No records with ≥$min_items items after filtering")

    post = fit_bernoulli_mixture_turing(X, K;
        prior, alpha_prior, beta_prior, concentration, floor,
        dirichlet_concentration, n_samples, n_chains, ci_level, rng)

    return post, item_names
end

"""
    posterior_summary(post::BernoulliMixturePosterior; n_top=5)

Print posterior summary: π posterior means with CIs, top θ_{k,d} per cluster.
"""
function posterior_summary(post::BernoulliMixturePosterior; n_top::Int=5)
    K = post.K
    D = size(post.theta_samples, 3)
    println("═══ Bayesian Bernoulli Mixture (Turing posterior) ═══")
    println("  K: $K")
    println("  Posterior samples: $(size(post.pi_samples, 1))")
    println("  CI level: $(post.ci_level)")

    println("\n  Mixing proportions (mean [$(post.ci_level*100)% CI]):")
    pi_mean = vec(sum(post.pi_samples, dims=1)) ./ size(post.pi_samples, 1)
    for k in 1:K
        lo = round(post.pi_ci[k, 1], digits=3)
        hi = round(post.pi_ci[k, 2], digits=3)
        m = round(pi_mean[k], digits=3)
        println("    π_$k: $m [$lo, $hi]")
    end

    println("\n  Top $n_top features per cluster (θ posterior mean):")
    theta_mean = dropdims(sum(post.theta_samples, dims=1), dims=1) ./
                 size(post.theta_samples, 1)
    for k in 1:K
        idx = partialsortperm(theta_mean[k, :], 1:min(n_top, D), rev=true)
        println("    Cluster $k:")
        for d in idx
            m = round(theta_mean[k, d], digits=3)
            lo = round(post.theta_ci[k, d, 1], digits=3)
            hi = round(post.theta_ci[k, d, 2], digits=3)
            println("      feature $d: $m [$lo, $hi]")
        end
    end
end
