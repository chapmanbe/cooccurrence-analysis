# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Bernoulli Mixture Model — domain-agnostic EM core
# ──────────────────────────────────────────────────────────────────────────────

using Random, LinearAlgebra

"""
    BernoulliMixtureResult

Result of fitting a K-component Bayesian Bernoulli mixture model via EM.

# Fields
- `K`: Number of mixture components
- `theta`: K × D matrix of per-class item probabilities (posterior means)
- `pi`: K-vector of mixing proportions
- `responsibilities`: N × K matrix of posterior class memberships
- `assignments`: Hard assignments (argmax of responsibilities per record)
- `log_likelihood`: Final log-likelihood
- `bic`: Bayesian Information Criterion
- `n_iter`: Number of EM iterations to convergence
- `converged`: Whether EM converged before max_iter
"""
struct BernoulliMixtureResult
    K::Int
    theta::Matrix{Float64}
    pi::Vector{Float64}
    responsibilities::Matrix{Float64}
    assignments::Vector{Int}
    log_likelihood::Float64
    bic::Float64
    n_iter::Int
    converged::Bool
end

"""
    BernoulliMixtureModelSelection

Result of model selection across multiple K values.

# Fields
- `results`: Fitted model for each K
- `best_K`: K with lowest BIC
- `best`: The best model
- `bic_values`: Sorted (K => BIC) pairs
"""
struct BernoulliMixtureModelSelection
    results::Dict{Int, BernoulliMixtureResult}
    best_K::Int
    best::BernoulliMixtureResult
    bic_values::Vector{Pair{Int, Float64}}
end

"""
    _compute_bic(log_likelihood, K, D, N) -> Float64

BIC = -2 * LL + n_params * log(N)

Parameters: K*D theta values + (K-1) mixing proportions.
"""
function _compute_bic(log_likelihood::Float64, K::Int, D::Int, N::Int)
    n_params = K * D + (K - 1)
    return -2.0 * log_likelihood + n_params * log(N)
end

"""
    empirical_bayes_priors(X::AbstractMatrix{Bool};
                            concentration=10.0, floor=1.0)
        -> (alpha::Vector{Float64}, beta::Vector{Float64})

Compute Ye 2018-style empirical Bayes Beta hyperparameters from per-feature
marginal frequencies, returning length-D vectors:

    α_d = concentration * π̂_d + floor
    β_d = concentration * (1 - π̂_d) + floor

where π̂_d = (1/N) Σ_n X[n, d].

Larger `concentration` produces stronger shrinkage of cluster-specific θ_{k,d}
toward the marginal frequency π̂_d. The `floor` keeps the prior proper when
π̂_d is exactly 0 or 1.

# Reference
Ye, Zhang & Nie (2018). Clustering sparse binary data with hierarchical
Bayesian Bernoulli mixture model. Comput Stat Data Anal 123:32-49.
"""
function empirical_bayes_priors(X::AbstractMatrix{Bool};
                                 concentration::Float64=10.0,
                                 floor::Float64=1.0)
    N, D = size(X)
    N == 0 && error("Cannot compute empirical Bayes priors on empty data")
    pi_hat = vec(sum(X, dims=1)) ./ N
    alpha = concentration .* pi_hat .+ floor
    beta = concentration .* (1.0 .- pi_hat) .+ floor
    return alpha, beta
end

"""
    fit_bernoulli_mixture(X::AbstractMatrix{Bool}, K::Int;
                          prior=:flat,
                          alpha_prior=1.0, beta_prior=1.0,
                          concentration=10.0, floor=1.0,
                          dirichlet_prior=1.0,
                          max_iter=200, tol=1e-6, n_init=5,
                          rng=Random.GLOBAL_RNG) -> BernoulliMixtureResult

Fit a K-component Bernoulli mixture model to binary data via EM with Beta priors.

Runs `n_init` random initializations and returns the result with highest
log-likelihood.

# Arguments
- `X`: N × D binary data matrix
- `K`: Number of mixture components
- `prior`: Prior strategy on θ_{k,d}.
    * `:flat` (default) — symmetric Beta(α, β) using `alpha_prior`, `beta_prior`.
      Equivalent to the original behavior.
    * `:empirical_bayes` — per-feature Beta(α_d, β_d) calibrated from marginal
      frequencies (Ye 2018). Recommended for sparse high-dimensional data.
- `alpha_prior`, `beta_prior`: Beta hyperparameters when `prior=:flat`.
- `concentration`, `floor`: hyperparameters for `:empirical_bayes` (see
  `empirical_bayes_priors`). Larger `concentration` → stronger shrinkage.
- `dirichlet_prior`: Symmetric Dirichlet prior concentration on mixing proportions
- `max_iter`: Maximum EM iterations per run
- `tol`: Convergence threshold on log-likelihood change
- `n_init`: Number of random restarts

# Reference
Ye, Zhang & Nie (2018). Comput Stat Data Anal 123:32-49.
"""
function fit_bernoulli_mixture(X::AbstractMatrix{Bool}, K::Int;
                               prior::Symbol=:flat,
                               alpha_prior::Float64=1.0,
                               beta_prior::Float64=1.0,
                               concentration::Float64=10.0,
                               floor::Float64=1.0,
                               dirichlet_prior::Float64=1.0,
                               max_iter::Int=200,
                               tol::Float64=1e-6,
                               n_init::Int=5,
                               rng::AbstractRNG=Random.GLOBAL_RNG)
    N, D = size(X)
    N == 0 && error("Cannot fit mixture model to empty data")
    K < 1 && error("K must be >= 1")
    K > N && error("K ($K) cannot exceed number of observations ($N)")

    # Resolve prior strategy to length-D vectors
    alpha_vec, beta_vec = if prior === :flat
        (fill(alpha_prior, D), fill(beta_prior, D))
    elseif prior === :empirical_bayes
        empirical_bayes_priors(X; concentration, floor)
    else
        error("Unknown prior :$prior. Use :flat or :empirical_bayes.")
    end

    best = nothing
    for _ in 1:n_init
        result = _em_single_run(X, K, N, D;
            alpha_vec, beta_vec, dirichlet_prior, max_iter, tol, rng)
        if best === nothing || result.log_likelihood > best.log_likelihood
            best = result
        end
    end

    # Sort classes by overall activity (sum of pi*theta) for consistent ordering
    activity = [best.pi[k] * sum(best.theta[k, :]) for k in 1:K]
    order = sortperm(activity, rev=true)
    if order != 1:K
        theta_sorted = best.theta[order, :]
        pi_sorted = best.pi[order]
        resp_sorted = best.responsibilities[:, order]
        # Remap: new class i was old class order[i], so an old assignment a maps
        # to its new position inv_order[a].
        inv_order = invperm(order)
        assignments_sorted = [inv_order[best.assignments[n]] for n in 1:N]
        best = BernoulliMixtureResult(K, theta_sorted, pi_sorted, resp_sorted,
                                       assignments_sorted, best.log_likelihood,
                                       best.bic, best.n_iter, best.converged)
    end

    return best
end

"""
    _em_single_run(X, K, N, D; alpha_vec, beta_vec, ...) -> BernoulliMixtureResult

Single EM run with random initialization. `alpha_vec` and `beta_vec` are
length-D Beta hyperparameter vectors (uniform values for flat prior;
per-feature values for empirical Bayes).
"""
function _em_single_run(X::AbstractMatrix{Bool}, K::Int, N::Int, D::Int;
                         alpha_vec::Vector{Float64}, beta_vec::Vector{Float64},
                         dirichlet_prior::Float64,
                         max_iter::Int, tol::Float64,
                         rng::AbstractRNG)
    X_f = Float64.(X)  # precompute for matrix ops

    # Random initialization
    theta = clamp.(rand(rng, K, D) .* 0.6 .+ 0.2, 0.01, 0.99)
    pi_k = fill(1.0 / K, K)

    log_r = zeros(N, K)
    r = zeros(N, K)
    prev_ll = -Inf
    n_iter = 0
    converged = false

    # ── E-step (vectorized): fill `r`/`log_r` for the current theta/pi and
    # return the marginal log-likelihood. Deterministic in theta/pi (no RNG), so
    # it can be safely re-run after the final M-step to resync r/LL with params.
    function estep!()
        log_theta = log.(theta)            # K × D
        log_1m_theta = log.(1.0 .- theta)  # K × D

        # log_r[n,k] = log(pi[k]) + X[n,:]·log(theta[k,:]) + (1-X[n,:])·log(1-theta[k,:])
        mul!(log_r, X_f, log_theta')
        log_r .+= (1.0 .- X_f) * log_1m_theta'
        for k in 1:K
            @views log_r[:, k] .+= log(pi_k[k])
        end

        # Log-sum-exp normalization + log-likelihood
        ll = 0.0
        for n in 1:N
            max_lr = maximum(@view log_r[n, :])
            s = 0.0
            for k in 1:K
                s += exp(log_r[n, k] - max_lr)
            end
            log_sum = max_lr + log(s)
            ll += log_sum
            for k in 1:K
                r[n, k] = exp(log_r[n, k] - log_sum)
            end
        end
        return ll
    end

    for iter in 1:max_iter
        n_iter = iter

        ll = estep!()

        # Convergence check
        if abs(ll - prev_ll) < tol
            converged = true
            prev_ll = ll
            break
        end
        prev_ll = ll

        # ── M-step (MAP with priors) ──
        N_k = vec(sum(r, dims=1))  # effective count per class
        # Guard against empty/near-empty components: an unclamped N_k → 0 makes
        # the flat-prior θ update 0/0 = NaN (and clamp(NaN)=NaN in Julia), which
        # poisons the whole fit; dirichlet_prior < 1 can likewise drive π negative.
        N_k_safe = max.(N_k, eps())

        # Mixing proportions (Dirichlet MAP), floored at eps() so π stays positive
        for k in 1:K
            pi_k[k] = max(N_k[k] + dirichlet_prior - 1.0, eps())
        end
        pi_k ./= sum(pi_k)

        # Item probabilities (Beta MAP, per-feature priors)
        # s_kd = sum_n r[n,k] * X[n,d]
        s_kd = r' * X_f  # K × D
        for k in 1:K
            for d in 1:D
                theta[k, d] = (s_kd[k, d] + alpha_vec[d] - 1.0) /
                              (N_k_safe[k] + alpha_vec[d] + beta_vec[d] - 2.0)
                theta[k, d] = clamp(theta[k, d], 1e-10, 1.0 - 1e-10)
            end
        end
    end

    # If we exited on max_iter (not convergence), the last M-step advanced
    # theta/pi past the responsibilities and LL computed from the prior E-step.
    # Resync so params, r, LL, assignments, and BIC all describe the same state.
    if !converged
        prev_ll = estep!()
    end

    assignments = [argmax(@view r[n, :]) for n in 1:N]
    bic = _compute_bic(prev_ll, K, D, N)

    return BernoulliMixtureResult(K, copy(theta), copy(pi_k), copy(r),
                                   assignments, prev_ll, bic, n_iter, converged)
end

"""
    select_K(X::AbstractMatrix{Bool}, K_range::UnitRange{Int}; kwargs...) -> BernoulliMixtureModelSelection

Fit Bernoulli mixture models for each K and select the best by BIC.
All keyword arguments are forwarded to `fit_bernoulli_mixture`.
"""
function select_K(X::AbstractMatrix{Bool}, K_range::UnitRange{Int}; kwargs...)
    results = Dict{Int, BernoulliMixtureResult}()
    for k in K_range
        results[k] = fit_bernoulli_mixture(X, k; kwargs...)
    end

    bic_values = sort([k => results[k].bic for k in K_range], by=last)
    best_K = first(bic_values).first
    return BernoulliMixtureModelSelection(results, best_K, results[best_K], bic_values)
end
