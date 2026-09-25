# ──────────────────────────────────────────────────────────────────────────────
# HDP Bernoulli Mixture — batch CAVI with truncated stick-breaking
#
# Model: Teh, Jordan, Beal & Blei 2006 (JASA) — hierarchical Dirichlet process.
# Algorithm: in the spirit of Wang, Paisley & Blei 2011 (AISTATS) §3.1 batch
#   CAVI, adapted to Bernoulli observations. We use the single-level
#   shared-atom stick-breaking representation (Sethuraman 1994 / Sudderth 2006)
#   in which record assignment z_i selects directly from corpus-level atom k,
#   rather than the two-level Sethuraman construction with auxiliary table
#   indicators c_jt of Wang/Paisley/Blei §2 (their Eqs. 6-11). Consequences:
#     - Per-group stick prior here is Beta(α·β_k, α·Σ_{l>k}β_l) (induced from
#       π_j ~ DP(α, β)), not WPB Eq. 15-16's Beta(1, α_0).
#     - No q(c_jt); the responsibility R[i,k] plays the role of WPB's ζ·ϕ.
#     - Online/stochastic updates (WPB §3.2-3.3) are not used; this is
#       full-batch CAVI on ~190K records.
# Bernoulli atoms with empirical-Bayes Beta priors follow Ye, Zhang & Nie 2018.
# ──────────────────────────────────────────────────────────────────────────────

using Random, LinearAlgebra, SpecialFunctions

const _ψ = SpecialFunctions.digamma

# ── Type definition ───────────────────────────────────────────────────────────

"""
    HDPBernoulliResult

Variational posterior of a truncated HDP Bernoulli mixture fit by CAVI.

# Fields
- `K_max`: truncation level (components in variational posterior)
- `n_groups`: number of strata (e.g., 2 for group)
- `group_labels`: human-readable group names, length n_groups
- `theta`: K_max × D posterior means E_q[θ_{k,d}]
- `theta_alpha`, `theta_beta`: K_max × D Beta variational parameters for atoms
- `beta_alpha`, `beta_beta`: K_max vectors of global stick Beta params
- `beta_mean`: K_max-vector of E_q[β_k] global stick weights
- `pi_mean`: n_groups × K_max per-group E_q[π_{j,k}]
- `pi_alpha`, `pi_beta`: n_groups × K_max Beta params for per-group sticks
- `responsibilities`: N × K_max soft assignments q(z_{j,i}=k)
- `assignments`: N hard assignments (argmax of responsibilities)
- `group`: N-vector mapping each record to group index in 1:n_groups
- `elbo`: final ELBO value
- `n_iter`, `converged`: convergence diagnostics
- `effective_K`: fewest components whose cumulative beta_mean exceeds 0.95
"""
struct HDPBernoulliResult
    K_max::Int
    n_groups::Int
    group_labels::Vector{String}
    theta::Matrix{Float64}
    theta_alpha::Matrix{Float64}
    theta_beta::Matrix{Float64}
    beta_alpha::Vector{Float64}
    beta_beta::Vector{Float64}
    beta_mean::Vector{Float64}
    pi_mean::Matrix{Float64}
    pi_alpha::Matrix{Float64}
    pi_beta::Matrix{Float64}
    responsibilities::Matrix{Float64}
    assignments::Vector{Int}
    group::Vector{Int}
    elbo::Float64
    n_iter::Int
    converged::Bool
    effective_K::Int
end

# ── Utility functions ─────────────────────────────────────────────────────────

function _neg_kl_beta(a::Float64, b::Float64, a0::Float64, b0::Float64)
    Elogx  = _ψ(a) - _ψ(a + b)
    Elog1x = _ψ(b) - _ψ(a + b)
    logabsgamma(a)[1] + logabsgamma(b)[1] - logabsgamma(a + b)[1] -
    logabsgamma(a0)[1] - logabsgamma(b0)[1] + logabsgamma(a0 + b0)[1] -
    (a - a0) * Elogx - (b - b0) * Elog1x
end

# E_q[log π_{j,k}] via cumulative stick expectations (log-domain for stability)
function _compute_E_log_pi!(E_log_pi::Matrix{Float64},
                            pi_alpha::Matrix{Float64},
                            pi_beta::Matrix{Float64})
    n_groups, K = size(pi_alpha)
    for j in 1:n_groups
        cum = 0.0
        for k in 1:K
            E_log_pi[j, k] = _ψ(pi_alpha[j, k]) - _ψ(pi_alpha[j, k] + pi_beta[j, k]) + cum
            cum += _ψ(pi_beta[j, k]) - _ψ(pi_alpha[j, k] + pi_beta[j, k])
        end
    end
end

# E_q[β_k] from global stick variational params
function _compute_beta_mean!(beta_mean::Vector{Float64},
                             beta_alpha::Vector{Float64},
                             beta_beta::Vector{Float64})
    K = length(beta_mean)
    log_remain = 0.0
    for k in 1:K-1
        vk = beta_alpha[k] / (beta_alpha[k] + beta_beta[k])
        beta_mean[k] = exp(log(max(vk, 1e-300)) + log_remain)
        log_remain += log(max(1.0 - vk, 1e-300))
    end
    beta_mean[K] = exp(log_remain)
end

# Effective K: fewest components whose cumulative beta_mean > threshold
function _effective_K(beta_mean::Vector{Float64}; threshold::Float64=0.95)
    sorted = sort(beta_mean, rev=true)
    cum = 0.0
    for (k, v) in enumerate(sorted)
        cum += v
        cum >= threshold && return k
    end
    return length(beta_mean)
end

# ── Initialisation ────────────────────────────────────────────────────────────

function _hdp_init(X::AbstractMatrix{Bool}, group::Vector{Int}, n_groups::Int,
                   K_max::Int,
                   alpha_prior::Vector{Float64}, beta_prior::Vector{Float64},
                   rng::AbstractRNG)
    N, D = size(X)
    Xf = Float64.(X)

    # Random responsibilities, row-normalised
    R = rand(rng, Float64, N, K_max)
    R ./= sum(R, dims=2)

    # Atom params using prior + weighted counts
    theta_alpha = zeros(K_max, D)
    theta_beta  = zeros(K_max, D)
    for k in 1:K_max
        Rk = view(R, :, k)
        for d in 1:D
            theta_alpha[k, d] = alpha_prior[d] + dot(Rk, Xf[:, d])
            theta_beta[k, d]  = beta_prior[d]  + dot(Rk, 1.0 .- Xf[:, d])
        end
    end

    # Global sticks: near-uniform Beta(1, γ_init=1)
    beta_a = ones(K_max)
    beta_b = ones(K_max)
    beta_b[K_max] = 1e-6   # last stick absorbs remaining mass

    # Per-group sticks: near-uniform
    pi_a = ones(n_groups, K_max)
    pi_b = ones(n_groups, K_max)
    pi_b[:, K_max] .= 1e-6

    return R, theta_alpha, theta_beta, beta_a, beta_b, pi_a, pi_b
end

# ── CAVI update functions ─────────────────────────────────────────────────────

# E-step: update responsibilities given current variational params.
# `Xf` (= Float64.(X)) is hoisted into `_run_hdp_cavi` and passed in so it is not
# re-materialised (N×D) every iteration.
function _hdp_e_step!(R::Matrix{Float64},
                      Xf::Matrix{Float64},
                      group::Vector{Int},
                      theta_alpha::Matrix{Float64},
                      theta_beta::Matrix{Float64},
                      E_log_pi::Matrix{Float64})
    K, D = size(theta_alpha)

    # K × D expected log probabilities
    E_log_th   = _ψ.(theta_alpha) .- _ψ.(theta_alpha .+ theta_beta)
    E_log1m_th = _ψ.(theta_beta)  .- _ψ.(theta_alpha .+ theta_beta)

    # N × K log-likelihoods via matrix multiply: Xf*(E_log_th - E_log1m_th)' + row_offset
    # Xf is N×D, (E_log_th - E_log1m_th) is K×D
    log_lik = Xf * (E_log_th .- E_log1m_th)' .+ vec(sum(E_log1m_th, dims=2))'
    # log_lik is N × K

    # Add per-group prior term
    log_r = log_lik .+ E_log_pi[group, :]   # N × K

    # Row-wise log-sum-exp normalise
    log_r_max = maximum(log_r, dims=2)
    R .= exp.(log_r .- log_r_max)
    R ./= sum(R, dims=2)

    return nothing
end

# Atom update: q(θ_{k,d}) = Beta(alpha_prior_d + Σ_i r_{i,k} x_{i,d}, ...)
function _hdp_update_atoms!(theta_alpha::Matrix{Float64},
                            theta_beta::Matrix{Float64},
                            Xf::Matrix{Float64},
                            OneMinusXf::Matrix{Float64},
                            R::Matrix{Float64},
                            alpha_prior::Vector{Float64},
                            beta_prior::Vector{Float64})
    K, D = size(theta_alpha)
    # R' * Xf is K × D; R' * (1-Xf) is K × D
    weighted_x    = R' * Xf           # K × D
    weighted_1mx  = R' * OneMinusXf   # K × D
    for k in 1:K
        theta_alpha[k, :] .= alpha_prior .+ weighted_x[k, :]
        theta_beta[k, :]  .= beta_prior  .+ weighted_1mx[k, :]
    end
end

# Per-group stick update (coupled to global β)
function _hdp_update_per_group_sticks!(pi_alpha::Matrix{Float64},
                                       pi_beta::Matrix{Float64},
                                       R::Matrix{Float64},
                                       group::Vector{Int},
                                       n_groups::Int,
                                       beta_mean::Vector{Float64},
                                       alpha::Float64)
    K = size(pi_alpha, 2)
    N = size(R, 1)

    # Soft group-level counts: N_jk[j, k] = Σ_{i in group j} r_{i,k}
    N_jk = zeros(n_groups, K)
    for i in 1:N
        N_jk[group[i], :] .+= R[i, :]
    end

    # Cumulative tail of global stick: beta_tail[k] = Σ_{l>k} E[β_l]
    beta_tail = zeros(K)
    for k in K-1:-1:1
        beta_tail[k] = beta_tail[k+1] + beta_mean[k+1]
    end

    for j in 1:n_groups
        # Cumulative tail of soft counts for group j
        N_tail_j = zeros(K)
        for k in K-1:-1:1
            N_tail_j[k] = N_tail_j[k+1] + N_jk[j, k+1]
        end
        for k in 1:K
            pi_alpha[j, k] = max(alpha * beta_mean[k] + N_jk[j, k], 1e-6)
            pi_beta[j, k]  = max(alpha * beta_tail[k]  + N_tail_j[k],  1e-6)
        end
    end
end

# Global stick update.
#
# APPROXIMATION (documented, intentional): this pools the *raw soft-counts* R
# across all groups (N_k = Σ_i R[i,k]) rather than the group-level HDP *table*
# counts a full Chinese-restaurant-franchise update would use. It therefore
# treats every group's assignments as exchangeable at the top level. With
# α = γ = 1 against ~190K records the cross-group sharing prior is swamped by
# the likelihood, so the per-group mixing weights (π) end up ≈ the empirical
# per-group cluster proportions: the fit behaves closer to independent per-group
# mixtures over a shared atom pool than to a strongly-pooled hierarchy. The ELBO
# remains a valid lower bound but is NOT guaranteed monotone under this update.
# The rest of the HDP math (E-step, E_log_pi, β/π means, KL/ELBO) is exact; only
# this global-stick step is approximate. See manuscript §3.5 / §4.2 caveats.
function _hdp_update_global_sticks!(beta_alpha::Vector{Float64},
                                    beta_beta::Vector{Float64},
                                    R::Matrix{Float64},
                                    gamma::Float64)
    K = length(beta_alpha)
    N_k = vec(sum(R, dims=1))  # total soft counts per component

    for k in 1:K-1
        N_tail = sum(N_k[k+1:end])
        beta_alpha[k] = 1.0 + N_k[k]
        beta_beta[k]  = max(gamma + N_tail, 1e-6)
    end
    # Last stick: v_{K} = 1 in truncation → Beta(1, ε)
    beta_alpha[K] = 1.0
    beta_beta[K]  = 1e-6
end

# ── ELBO computation ──────────────────────────────────────────────────────────

function _hdp_compute_elbo(Xf::Matrix{Float64},
                           R::Matrix{Float64},
                           group::Vector{Int},
                           theta_alpha::Matrix{Float64},
                           theta_beta::Matrix{Float64},
                           beta_alpha::Vector{Float64},
                           beta_beta::Vector{Float64},
                           beta_mean::Vector{Float64},
                           pi_alpha::Matrix{Float64},
                           pi_beta::Matrix{Float64},
                           alpha_prior::Vector{Float64},
                           beta_prior::Vector{Float64},
                           E_log_pi::Matrix{Float64},
                           alpha::Float64,
                           gamma::Float64)
    N, K = size(R)
    D = size(Xf, 2)
    n_groups = size(pi_alpha, 1)

    E_log_th   = _ψ.(theta_alpha) .- _ψ.(theta_alpha .+ theta_beta)   # K × D
    E_log1m_th = _ψ.(theta_beta)  .- _ψ.(theta_alpha .+ theta_beta)   # K × D

    # A: E_q[log p(X | Z, θ)] + B: E_q[log p(Z | π)] + C: H[q(Z)]
    log_lik = Xf * (E_log_th .- E_log1m_th)' .+ vec(sum(E_log1m_th, dims=2))'
    log_lik_with_pi = log_lik .+ E_log_pi[group, :]  # N × K
    L_data = 0.0
    for i in 1:N
        for k in 1:K
            if R[i, k] > 1e-300
                L_data += R[i, k] * (log_lik_with_pi[i, k] - log(R[i, k]))
            end
        end
    end

    # D: -KL(q(θ) || p(θ)) summed over k, d
    L_theta = 0.0
    for k in 1:K
        for d in 1:D
            L_theta += _neg_kl_beta(theta_alpha[k,d], theta_beta[k,d],
                                    alpha_prior[d], beta_prior[d])
        end
    end

    # E: -KL(q(v^j_k) || Beta(α E[β_k], α Σ_{l>k} E[β_l]))
    beta_tail = zeros(K)
    for k in K-1:-1:1
        beta_tail[k] = beta_tail[k+1] + beta_mean[k+1]
    end
    L_pi = 0.0
    for j in 1:n_groups
        for k in 1:K-1
            a0 = max(alpha * beta_mean[k], 1e-6)
            b0 = max(alpha * beta_tail[k],  1e-6)
            L_pi += _neg_kl_beta(pi_alpha[j,k], pi_beta[j,k], a0, b0)
        end
    end

    # F: -KL(q(v^g_k) || Beta(1, γ))
    L_beta = 0.0
    for k in 1:K-1
        L_beta += _neg_kl_beta(beta_alpha[k], beta_beta[k], 1.0, gamma)
    end

    return L_data + L_theta + L_pi + L_beta
end

# ── Per-group mean mixing weights from stick-breaking ────────────────────────

function _compute_pi_mean!(pi_mean::Matrix{Float64},
                           pi_alpha::Matrix{Float64},
                           pi_beta::Matrix{Float64})
    n_groups, K = size(pi_mean)
    for j in 1:n_groups
        log_remain = 0.0
        for k in 1:K-1
            vk = pi_alpha[j,k] / (pi_alpha[j,k] + pi_beta[j,k])
            pi_mean[j, k] = exp(log(max(vk, 1e-300)) + log_remain)
            log_remain += log(max(1.0 - vk, 1e-300))
        end
        pi_mean[j, K] = exp(log_remain)
    end
end

# ── Sort components by global weight (descending) ───────────────────────────

function _hdp_sort_by_weight!(theta_alpha, theta_beta, beta_alpha, beta_beta,
                               beta_mean, pi_alpha, pi_beta, pi_mean, R)
    ord = sortperm(beta_mean, rev=true)
    theta_alpha .= theta_alpha[ord, :]
    theta_beta  .= theta_beta[ord, :]
    beta_alpha  .= beta_alpha[ord]
    beta_beta   .= beta_beta[ord]
    beta_mean   .= beta_mean[ord]
    pi_alpha    .= pi_alpha[:, ord]
    pi_beta     .= pi_beta[:, ord]
    pi_mean     .= pi_mean[:, ord]
    R           .= R[:, ord]
end

# ── Single CAVI run ───────────────────────────────────────────────────────────

function _run_hdp_cavi(X::AbstractMatrix{Bool},
                       group::Vector{Int},
                       n_groups::Int,
                       K_max::Int,
                       alpha::Float64,
                       gamma::Float64,
                       alpha_prior::Vector{Float64},
                       beta_prior::Vector{Float64},
                       max_iter::Int,
                       tol::Float64,
                       rng::AbstractRNG;
                       effective_K_threshold::Float64=0.95)

    N, D = size(X)

    # Hoist the dense float views of X out of the CAVI loop: the E-step, atom
    # update, and ELBO all consume them every iteration (previously each
    # re-materialised its own N×D copy).
    Xf = Float64.(X)
    OneMinusXf = 1.0 .- Xf

    R, theta_alpha, theta_beta, beta_a, beta_b, pi_a, pi_b =
        _hdp_init(X, group, n_groups, K_max, alpha_prior, beta_prior, rng)

    beta_mean = zeros(K_max)
    _compute_beta_mean!(beta_mean, beta_a, beta_b)

    E_log_pi = zeros(n_groups, K_max)
    _compute_E_log_pi!(E_log_pi, pi_a, pi_b)

    pi_mean = zeros(n_groups, K_max)
    _compute_pi_mean!(pi_mean, pi_a, pi_b)

    # The documented global-stick approximation makes the ELBO non-monotone, so a
    # single `|Δelbo| < tol` crossing can trigger on an oscillation rather than a
    # plateau. Require PLATEAU_WINDOW *consecutive* small steps instead.
    #
    # The test is on the per-iteration delta, NOT on improvement over the best
    # ELBO seen. Those are not interchangeable here: because the objective is
    # non-monotone, iterations that keep refining the atoms without exceeding an
    # early peak all look like "no improvement", and the window fills in a handful
    # of iterations. On a 153,700-record dataset that stopped CAVI at iteration
    # 5 with every component still sitting on the marginal item prevalence —
    # effective K of 3 undifferentiated clusters instead of 6 real ones. The
    # 20-record test fixture is far too small to expose it.
    PLATEAU_WINDOW = 5
    prev_elbo = -Inf   # last computed ELBO (describes the returned iterate)
    small_steps = 0
    converged = false
    n_iter = max_iter

    for iter in 1:max_iter
        # E-step
        _hdp_e_step!(R, Xf, group, theta_alpha, theta_beta, E_log_pi)

        # M-step: atoms
        _hdp_update_atoms!(theta_alpha, theta_beta, Xf, OneMinusXf, R, alpha_prior, beta_prior)

        # M-step: global sticks
        _hdp_update_global_sticks!(beta_a, beta_b, R, gamma)
        _compute_beta_mean!(beta_mean, beta_a, beta_b)

        # M-step: per-group sticks
        _hdp_update_per_group_sticks!(pi_a, pi_b, R, group, n_groups, beta_mean, alpha)
        _compute_E_log_pi!(E_log_pi, pi_a, pi_b)
        _compute_pi_mean!(pi_mean, pi_a, pi_b)

        # ELBO
        elbo = _hdp_compute_elbo(Xf, R, group, theta_alpha, theta_beta,
                                  beta_a, beta_b, beta_mean, pi_a, pi_b,
                                  alpha_prior, beta_prior, E_log_pi, alpha, gamma)

        if isfinite(prev_elbo) &&
           abs(elbo - prev_elbo) < tol * (1.0 + abs(prev_elbo))
            small_steps += 1
        else
            small_steps = 0
        end
        prev_elbo = elbo

        if small_steps >= PLATEAU_WINDOW
            converged = true
            n_iter = iter
            break
        end
    end

    # Sort by global weight descending
    _hdp_sort_by_weight!(theta_alpha, theta_beta, beta_a, beta_b,
                          beta_mean, pi_a, pi_b, pi_mean, R)

    assignments = [ci[2] for ci in vec(argmax(R, dims=2))]
    theta = theta_alpha ./ (theta_alpha .+ theta_beta)
    eff_K = _effective_K(beta_mean; threshold=effective_K_threshold)

    return HDPBernoulliResult(
        K_max, n_groups,
        ["Group_$j" for j in 1:n_groups],
        theta, theta_alpha, theta_beta,
        beta_a, beta_b, beta_mean,
        pi_mean, pi_a, pi_b,
        R, assignments, group,
        prev_elbo, n_iter, converged, eff_K
    )
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    fit_hdp_bernoulli_mixture(X, group, n_groups; kwargs...) -> HDPBernoulliResult

Fit an HDP Bernoulli mixture to binary record × item-item matrix `X` with
group assignments in `group` (integer vector, values in `1:n_groups`).

Uses coordinate-ascent variational inference (CAVI) with truncated
stick-breaking (Wang & Blei 2011). All updates are closed-form
Beta-Bernoulli conjugate updates — no autodiff required.

# Arguments
- `X`: N × D `Bool` matrix of binary indicators
- `group`: length-N integer vector mapping each row to a group (1-indexed)
- `n_groups`: number of distinct groups

# Keyword arguments
- `K_max=20`: truncation level (soft upper bound on effective K)
- `alpha=1.0`: per-group DP concentration. Larger → groups closer to global.
- `gamma=1.0`: global DP concentration. Smaller → fewer effective clusters.
- `prior=:flat`: atom prior strategy. `:empirical_bayes` for Ye 2018 priors.
- `alpha_prior=1.0`, `beta_prior=1.0`: flat Beta(α0, β0) prior on each atom
- `concentration=10.0`, `floor=1.0`: parameters for empirical_bayes_priors
- `max_iter=300`, `tol=1e-5`: convergence settings
- `n_init=3`: number of random restarts; best ELBO is returned
- `rng`: random number generator (default: `Random.GLOBAL_RNG`)
"""
function fit_hdp_bernoulli_mixture(X::AbstractMatrix{Bool},
                                   group::Vector{Int},
                                   n_groups::Int;
                                   K_max::Int=20,
                                   alpha::Float64=1.0,
                                   gamma::Float64=1.0,
                                   prior::Symbol=:flat,
                                   alpha_prior::Float64=1.0,
                                   beta_prior::Float64=1.0,
                                   concentration::Float64=10.0,
                                   floor::Float64=1.0,
                                   max_iter::Int=300,
                                   tol::Float64=1e-5,
                                   n_init::Int=3,
                                   effective_K_threshold::Float64=0.95,
                                   rng::AbstractRNG=Random.GLOBAL_RNG)
    N, D = size(X)
    @assert length(group) == N "group length must equal number of rows in X"
    @assert all(1 .<= group .<= n_groups) "group values must be in 1:n_groups"
    @assert n_groups >= 1 "n_groups must be at least 1"
    @assert K_max >= 2 "K_max must be at least 2"

    # Empty groups check
    group_counts = [count(==(j), group) for j in 1:n_groups]
    any(==(0), group_counts) && error(
        "Empty group detected: groups $(findall(==(0), group_counts)) have zero records")

    # Warn if N < K_max
    N < K_max && @warn "N=$N < K_max=$K_max; some components will be empty"

    # Build per-feature priors
    a_prior, b_prior = if prior == :empirical_bayes
        empirical_bayes_priors(X; concentration, floor)
    else
        fill(alpha_prior, D), fill(beta_prior, D)
    end

    # Multiple random restarts, keep best ELBO. A restart with non-finite ELBO
    # never wins (NaN > best is false), so best_result stays `nothing` iff every
    # restart failed — surface that instead of returning `nothing` silently.
    best_result = nothing
    best_elbo = -Inf
    for _ in 1:n_init
        result = _run_hdp_cavi(X, group, n_groups, K_max, alpha, gamma,
                                a_prior, b_prior, max_iter, tol, rng;
                                effective_K_threshold)
        if result.elbo > best_elbo
            best_elbo = result.elbo
            best_result = result
        end
    end

    best_result === nothing && error(
        "HDP CAVI failed: all $n_init restart(s) produced a non-finite ELBO. " *
        "Try more records, a smaller K_max, or a different rng seed.")

    return best_result
end
