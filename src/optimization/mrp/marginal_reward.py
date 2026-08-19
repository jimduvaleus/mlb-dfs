"""Joint portfolio reward R(S) with self-competition, over simulated worlds.

Haugh & Singal 2019, formulation (2): a portfolio's value is the sum over our
entries of their expected payout, where each entry is ranked against
`opponents UNION our other entries` -- not against opponents alone.

    R(W) = sum_i sum_d (R_d - R_{d+1}) P{ w_i.delta > G_{-i}^(r_d) }

Production sums INDEPENDENT per-lineup EVs (`ContestScorer.robust_payout` is
built one candidate at a time against the field), so nothing in today's
objective knows that two of our entries cannot both take first place. That
missing interaction term is the whole point of this module.

REFERENCE SEMANTICS. The realized-world truth is `tests/bt_core.grade_portfolio`
-- self-displacement, self-tie-splitting and our-dupe prize splitting, with a
clipped tie band so k=1 reduces bit-for-bit to `grade_pick`. This module
reimplements those semantics over SIMULATED worlds (src/ must not import from
tests/); `tests/test_mrp_marginal_reward.py` asserts the two agree exactly.
That test is the contract -- if it fails, this file is wrong, not bt_core.

GROSS, not net. Entry fees are sunk: the number of entries per contest is
exogenous (the DK entries file is already purchased), so the fee is a constant
that shifts every candidate equally and cannot change an argmax. Callers that
want net dollars subtract `fee * k` at the end.
"""
from __future__ import annotations

import numpy as np

# Ties with the field are ~measure-zero on continuous simulated scores but do
# occur through float32 collisions. They are counted exactly, saturating at
# this cap so the count fits a uint8 in the precomputed form.
_FTIE_CAP = 255


def _payout_cumsum(payout_arr: np.ndarray) -> np.ndarray:
    """(L+1,) float64 prefix sum, so a tie band's mean is an O(1) difference."""
    payout_arr = np.asarray(payout_arr, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(payout_arr, dtype=np.float64)))


def joint_gross_world(
    own_scores: np.ndarray,
    sorted_field: np.ndarray,
    payout_cum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gross $ and rank for every one of our k entries in ONE world.

    Mirrors `bt_core.grade_portfolio` exactly:
      n_above = (field entries > v) + (our finite entries > v)
      tie band = [n_above, n_above + f_ties + g_ties) clipped to [0, L]
      gross    = mean of payout over that band

    Parameters
    ----------
    own_scores : (k,) our entries' scores in this world. NaN entries are
        excluded from every OTHER entry's n_above and tie band (they do not
        displace anyone) and come back gross=NaN, rank=-1.
    sorted_field : (F,) opponent scores, sorted ASCENDING.
    payout_cum : (L+1,) from `_payout_cumsum`.
    """
    a = np.asarray(own_scores, dtype=np.float64)
    finite = np.isfinite(a)
    safe = np.where(finite, a, 0.0)

    n_field = len(sorted_field)
    L = len(payout_cum) - 1

    right_field = np.searchsorted(sorted_field, safe, side="right")
    left_field = np.searchsorted(sorted_field, safe, side="left")
    n_above_field = n_field - right_field
    f_ties = right_field - left_field

    own_sorted = np.sort(safe[finite])
    k_finite = len(own_sorted)
    own_right = np.searchsorted(own_sorted, safe, side="right")
    own_left = np.searchsorted(own_sorted, safe, side="left")
    own_above = k_finite - own_right
    g_ties = own_right - own_left  # includes self

    n_above = n_above_field + own_above
    rank = n_above + 1

    lo = np.clip(n_above, 0, L)
    hi = np.clip(n_above + f_ties + g_ties, 0, L)
    width = hi - lo
    gross = np.where(width > 0, (payout_cum[hi] - payout_cum[lo]) / np.maximum(width, 1), 0.0)

    gross = np.where(finite, gross, np.nan)
    rank = np.where(finite, rank, -1)
    return gross, rank.astype(np.int64)


def joint_gross_worlds(
    own_scores: np.ndarray,
    field_sorted: np.ndarray,
    payout_arr: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    """(k, S) gross $ per entry per world.

    `own_scores` is (k, S); `field_sorted` is (S, F) sorted ascending along
    axis 1 -- the orientation `ContestScorer._build_field_sorted` produces.

    Chunked over WORLDS: `field_sorted` is the large array here (2.9 GB at
    S=25,000 x F=29,411 float32) and the per-world searchsorted cannot be
    vectorised across rows, so this walks worlds in blocks rather than
    materialising anything (k, S, F)-shaped. Same per-world loop shape as
    `external_pool.compute_p_win`.
    """
    own_scores = np.asarray(own_scores, dtype=np.float64)
    if own_scores.ndim != 2:
        raise ValueError(f"own_scores must be (k, S), got shape {own_scores.shape}")
    k, S = own_scores.shape
    if field_sorted.shape[0] != S:
        raise ValueError(
            f"field_sorted has {field_sorted.shape[0]} worlds, own_scores has {S}"
        )

    payout_cum = _payout_cumsum(payout_arr)
    out = np.empty((k, S), dtype=np.float64)
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        block = field_sorted[s0:s1]
        for j in range(s1 - s0):
            out[:, s0 + j], _ = joint_gross_world(own_scores[:, s0 + j], block[j], payout_cum)
    return out


def portfolio_reward(
    own_scores: np.ndarray,
    field_sorted: np.ndarray,
    payout_arr: np.ndarray,
    chunk: int = 2048,
) -> float:
    """R(S): expected TOTAL gross dollars for the whole portfolio, one contest.

    The quantity Haugh & Singal maximise. Note this is a SUM over our entries
    (each is paid separately), not a max -- but because our entries displace
    each other in the ranking, it behaves like coverage where the payout curve
    is steep. That is the correct resolution of the sum-vs-max framing in
    memory `project-external-pool-theory`: it is neither, it interpolates.
    """
    per_entry = joint_gross_worlds(own_scores, field_sorted, payout_arr, chunk=chunk)
    return float(np.nansum(per_entry, axis=0).mean())


def precompute_field_ranks(
    cand_scores: np.ndarray,
    field_sorted: np.ndarray,
    chunk: int = 512,
    rank_cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate field rank, computed ONCE per contest.

    Returns
    -------
    n_above_field : (M, S) uint16 -- opponents strictly above candidate j in
        world s, clamped to `rank_cap`.
    f_ties : (M, S) uint8 -- exact ties with the field, saturating at 255.

    Why this exists: the greedy evaluates every candidate at every pick, so
    re-running the searchsorted per pick would be ~1.9e9 ops x 150 picks. Doing
    it once lets the per-pick cost collapse to a gather, and -- the bigger win
    -- lets the caller RELEASE `field_sorted` (2.9 GB) afterwards, since
    everything downstream needs only these two compact arrays.

    uint16 is safe because a rank cannot exceed the field size and DK's largest
    archived structure is 29,411 entries. `rank_cap` defaults to the payout
    table length, above which the payout is $0 no matter how far our own
    entries displace a candidate, so clamping there is lossless AND keeps the
    dtype valid for fields larger than 65,535.
    """
    cand_scores = np.asarray(cand_scores)
    if cand_scores.ndim != 2:
        raise ValueError(f"cand_scores must be (M, S), got shape {cand_scores.shape}")
    M, S = cand_scores.shape
    if field_sorted.shape[0] != S:
        raise ValueError(
            f"field_sorted has {field_sorted.shape[0]} worlds, cand_scores has {S}"
        )
    F = field_sorted.shape[1]
    cap = int(rank_cap) if rank_cap is not None else F
    cap = max(0, min(cap, np.iinfo(np.uint16).max))

    n_above = np.empty((M, S), dtype=np.uint16)
    f_ties = np.empty((M, S), dtype=np.uint8)
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        block = field_sorted[s0:s1]
        for j in range(s1 - s0):
            col = cand_scores[:, s0 + j]
            right = np.searchsorted(block[j], col, side="right")
            left = np.searchsorted(block[j], col, side="left")
            n_above[:, s0 + j] = np.minimum(F - right, cap).astype(np.uint16)
            f_ties[:, s0 + j] = np.minimum(right - left, _FTIE_CAP).astype(np.uint8)
    return n_above, f_ties


def tier_form_payout(rank: np.ndarray, tier_ranks: np.ndarray,
                     tier_amounts: np.ndarray) -> np.ndarray:
    """Formulation (2)'s own shape: sum_d (R_d - R_{d+1}) . 1[rank <= r_d].

    Algebraically identical to `payout_arr[rank - 1]` (proved in
    tests/test_mrp_marginal_reward.py), but written as a weighted sum of
    EXCEEDANCE INDICATORS rather than a rank lookup. That is the seam the
    smoothed estimator plugs into: each indicator can be replaced by
    P(threshold <= score) under the rank-r_d order statistic's own sampling
    distribution (`external_pool.smoothing_tau`, this branch) without touching
    anything else.

    Why that matters here rather than being an optional refinement: measured
    across the 29 archived DK payout tables, the rank-1 step carries a MEDIAN
    50% of total objective weight (range 20-70%), and the top three ranks carry
    50-92%. So this objective's precision is almost entirely the precision of
    its tightest, rarest rungs -- the ones measured at split-half rho_full 0.55
    with a hard indicator and 0.69-0.86 smoothed.
    """
    rank = np.asarray(rank)
    r = np.asarray(tier_ranks, dtype=np.int64)
    a = np.asarray(tier_amounts, dtype=np.float64)
    steps = a - np.concatenate((a[1:], [0.0]))          # R_d - R_{d+1}
    return (rank[..., None] <= r).astype(np.float64) @ steps
