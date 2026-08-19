"""sigma_dG: per-player covariance with the field's payout-rank order statistic.

Haugh & Singal's `sigma_{delta,G^(r')}` -- the P x 1 vector whose p-th entry is
Cov(delta_p, G^(r')), the covariance between a player's score and the score
that defines a payout cutoff. It appears in their objective as

    max_w  w'mu + lambda ( w'Sigma w - 2 w' sigma_dG )

and the `-2 w' sigma_dG` term is the part that matters here: it penalises
players whose good games drag the cutoff up along with them. That is a
first-principles derivation of leverage, and it is FIELD-AWARE in a way none of
our currencies are -- `leverage.py` scores the pool against ITSELF with no
opponent-field simulation, and `prj_own` is ownership arithmetic.

ROLE IN THIS PIPELINE: generation and diagnostics, NOT selection. The paper
needs sigma_dG inside the objective only because it cannot evaluate the true
portfolio reward over the full feasible set, so it substitutes a mean-variance
surrogate. We evaluate dR directly over a pool (see delta_reward.py), so adding
sigma_dG as a selector term would be re-introducing the approximation we were
able to skip. Its jobs here are:

  1. Answer whether the external pool actually SPANS the low-covariance region.
     If it does, targeted candidate generation buys nothing and is skipped.
  2. Supply a LINEAR per-player objective for an ILP if it does not -- linear
     is the whole point, since it means a plain CBC solve rather than a MIQP
     (OR-Tools is installed; Gurobi is not).

The estimator is one centred matmul: `sim_matrix` (n_sims, n_players) and the
per-world thresholds (n_sims, T) share the world axis by construction.
"""
from __future__ import annotations

import numpy as np


def tier_boundary_ranks(payout_arr: np.ndarray) -> np.ndarray:
    """The r_d of formulation (2): the LAST rank of each paying tier.

    The paper defines 0 =: r_0 < r_1 < ... < r_D with "a portfolio whose rank
    lies in (r_{d-1}, r_d] wins R_d", so r_d is a tier's END, not its start --
    you must beat the r_d-th best entry to be inside the top r_d. Using starts
    instead breaks the tier decomposition on every multi-rank tier (caught by
    tests/test_mrp_marginal_reward.py's exhaustive check over the real tables).

    DK's real tables are coarse -- `dk_rally_cap_29411` pays 6,445 positions
    through just 22 distinct tiers -- so this returns ~20 ranks, not thousands.
    That is what makes the whole payout-ladder formulation cheap.
    """
    p = np.asarray(payout_arr, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.int64)
    # Ends of constant runs: the last index before the value changes, plus the
    # final index. 1-indexed on return.
    ends = np.flatnonzero(np.diff(p) != 0.0)             # 0-indexed run ends
    ends = np.concatenate((ends, [p.size - 1]))
    ends = ends[p[ends] > 0.0]                           # paying tiers only
    return np.unique(ends + 1).astype(np.int64)


def field_order_statistics(field_sorted: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """(n_sims, T) field score at each rank-FROM-THE-TOP, per world.

    `field_sorted` is (n_sims, F) ascending -- the orientation
    `ContestScorer._build_field_sorted` produces -- so rank r maps to column
    F - r. Ranks beyond the field size are clamped to the field's minimum.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    F = field_sorted.shape[1]
    cols = np.clip(F - ranks, 0, F - 1)
    return np.asarray(field_sorted[:, cols], dtype=np.float64)


def player_field_covariance(
    sim_matrix: np.ndarray,
    thresholds: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """(n_players, T) Cov(player score, field cutoff) across simulated worlds.

    Chunked over worlds so nothing (n_players x n_sims x T)-shaped is ever
    built; the accumulators are (n_players, T) and (n_players,), both tiny.
    """
    X = sim_matrix
    W = np.asarray(thresholds, dtype=np.float64)
    S = X.shape[0]
    if W.shape[0] != S:
        raise ValueError(f"thresholds has {W.shape[0]} worlds, sim_matrix has {S}")
    P, T = X.shape[1], W.shape[1]

    sum_x = np.zeros(P, dtype=np.float64)
    sum_w = np.zeros(T, dtype=np.float64)
    sum_xw = np.zeros((P, T), dtype=np.float64)
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        xb = np.asarray(X[s0:s1], dtype=np.float64)
        wb = W[s0:s1]
        sum_x += xb.sum(axis=0)
        sum_w += wb.sum(axis=0)
        sum_xw += xb.T @ wb
    return sum_xw / S - np.outer(sum_x / S, sum_w / S)


def payout_weighted_sigma(sigma_dG: np.ndarray, payout_arr: np.ndarray,
                          ranks: np.ndarray) -> np.ndarray:
    """(n_players,) collapse of sigma_dG using the objective's own weights.

    Formulation (2) weights tier d by (R_d - R_{d+1}), so that is the weighting
    that makes a single vector faithful to the objective rather than an
    unweighted average over tiers nobody is playing for.
    """
    p = np.asarray(payout_arr, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    amounts = p[np.clip(ranks - 1, 0, len(p) - 1)]
    w = amounts - np.concatenate((amounts[1:], [0.0]))   # R_d - R_{d+1}
    total = w.sum()
    if total <= 0:
        return sigma_dG.mean(axis=1)
    return sigma_dG @ (w / total)


def assumption_52_report(sigma_dG: np.ndarray, top_t: int | None = None) -> dict:
    """Is Cov(delta_p, G^(r_d)) effectively constant across tiers d?

    Paper Assumption 5.2, with Proposition 5.1 proving it in the O -> infinity
    limit. If it holds, the whole (n_players, T) block collapses to one vector
    per slate. This reports the evidence rather than asserting it: per-player
    spread across tiers relative to the level, and the correlation of each tier
    column with the payout-weighted collapse.
    """
    sig = np.asarray(sigma_dG, dtype=np.float64)
    if top_t is not None:
        sig = sig[:, :top_t]
    mean_by_player = sig.mean(axis=1)
    sd_by_player = sig.std(axis=1)
    scale = np.abs(mean_by_player).mean()
    ref = sig.mean(axis=1)
    cols = []
    for d in range(sig.shape[1]):
        a, b = sig[:, d], ref
        if a.std() < 1e-12 or b.std() < 1e-12:
            cols.append(np.nan)
        else:
            cols.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "n_tiers": int(sig.shape[1]),
        "mean_abs_level": float(scale),
        "median_rel_spread": float(np.median(sd_by_player / np.maximum(np.abs(mean_by_player), 1e-9))),
        "min_col_corr_vs_mean": float(np.nanmin(cols)),
        "mean_col_corr_vs_mean": float(np.nanmean(cols)),
    }


def lineup_sigma_scores(sigma_vec: np.ndarray, lineup_cols: np.ndarray) -> np.ndarray:
    """(M,) w'sigma_dG per lineup -- the pool-facing currency.

    `lineup_cols` is (M, roster) column indices into the sim matrix, the same
    representation `ContestScorer._build_col_lineups` produces.
    """
    sig = np.asarray(sigma_vec, dtype=np.float64)
    cols = np.asarray(lineup_cols, dtype=np.int64)
    return sig[cols].sum(axis=1)
