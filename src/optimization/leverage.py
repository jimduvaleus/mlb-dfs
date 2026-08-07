"""Game-theoretic "optimal ownership" and leverage (plan doc
need-a-plan-to-squishy-sutherland.md), derived from the external candidate
pool and copula simulations with no opponent-field simulation and no
per-simulated-world ILP solve.

Core mechanism: score the pool against ITSELF as a self-referential field --
ep.compute_p_win(scores, scores.T, exponents) -- instead of an
ownership-sampled opponent field (ContestSimulator.generate_field, what
p_win normally uses) or a fresh optimizer solve per simulated world. The
"best" lineups in a simulated world are just the top-scoring rows of the
pool's own (M, n_sims) score matrix, which the pool already gives us.
p_opt[j] = mean_over_worlds(percentile_within_pool(j) ** exponent) is then
literally "probability lineup j is optimal," and its player-level aggregate
O[p] (a p_opt-weighted share of pool composition) is a field-size-conditioned
"optimal ownership" -- diffed against the pool's own projected ownership
(players_df["ownership"], SaberSim's "Adj Own"/"My Own" column) to get
leverage = optimal - projected.

Validated (Phase A, scripts/analyze_optimal_ownership.py, 2026-08-07):
leverage beats a naive -own_pct contrarian baseline on 76/99 real contests
across the 10-slate archive (sign test p=8.5e-8) -- see memory
project-leverage-phase-a-results for the two unexplained losing slates
(07242026, 07262026) bookmarked for later investigation.
"""
from typing import Optional

import numpy as np
import pandas as pd

# Cross-module private import: _lineup_indicator_matrix has no public
# wrapper, but external_pool.py's own compute_pool_corr/compute_pool_proj_scores
# already call it the same way internally -- reusing it here (rather than
# duplicating the indicator-matmul logic) keeps one source of truth.
from src.api.external_pool import (
    _lineup_indicator_matrix,
    compute_lineup_scores,
    compute_p_win,
)

# players_df["ownership"] is percentage points (0-100, "My Own" scale --
# build_external_players_df). Floor before a ratio division mirrors
# tests/backtest_oracle.py::lineup_features' own.fillna(0.1).clip(lower=0.01).
OWN_FLOOR_PCT = 0.1


def compute_candidate_p_opt(
    lineups: list, sim_results, field_sizes: dict[str, float],
    sharpness: float = 0.05, resolution_cap: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """{key: (M,) float64} per-candidate "probability of being optimal",
    one value per requested field-size key.

    field_sizes maps an arbitrary caller key (e.g. a contest_id) to that
    contest's real or implied entry count; the exponent per key is
    `sharpness * min(field_sizes[key], resolution_cap or M)`. The cap
    matters: the self-referential field has only M discrete support points
    (the pool scored against itself), so an exponent built off a real field
    size larger than M over-trusts resolution the construction doesn't
    have -- worse than production's real p_win, which grows its opponent
    field up to 25,000 specifically so its percentile resolves at the
    largest contest sizes. Defaulting resolution_cap to M keeps that
    ceiling honest without requiring every caller to know about it.
    """
    scores = compute_lineup_scores(lineups, sim_results)  # (M, n_sims)
    M = scores.shape[0]
    cap = resolution_cap if resolution_cap is not None else M
    exponents = {k: max(1.0, sharpness * min(fs, cap)) for k, fs in field_sizes.items()}
    return compute_p_win(scores, scores.T, exponents)


def compute_optimal_ownership(
    lineups: list, sim_results, players_df: pd.DataFrame,
    field_sizes: dict[str, float], sharpness: float = 0.05,
    resolution_cap: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """{key: (P,) float64} percentage-point optimal ownership, aligned to
    players_df row order (same order compute_leverage expects).

    O[p] is the p_opt-weighted share of pool composition attributable to
    lineups containing player p, via the same player-indicator matmul
    compute_lineup_scores/compute_pool_corr already use, scaled by 100 to
    match players_df["ownership"]'s percentage-point units.
    """
    p_opt = compute_candidate_p_opt(lineups, sim_results, field_sizes, sharpness, resolution_cap)
    I = _lineup_indicator_matrix(lineups, players_df["player_id"].tolist())
    out = {}
    for key, w in p_opt.items():
        w = w.astype(np.float64)
        denom = w.sum()
        frac = (I @ w) / denom if denom > 0 else np.zeros(I.shape[0])
        out[key] = 100.0 * frac
    return out


def compute_leverage(
    optimal_ownership: np.ndarray, players_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """(diff, ratio) leverage vectors, aligned to players_df row order
    (the order compute_optimal_ownership's output uses).

    diff = optimal - projected; ratio = diff / max(projected, OWN_FLOOR_PCT)
    -- the floor keeps a near-zero-projected player from producing a huge
    or infinite ratio.
    """
    own = players_df["ownership"].astype(float).to_numpy()
    diff = optimal_ownership - own
    ratio = diff / np.maximum(own, OWN_FLOOR_PCT)
    return diff, ratio
