"""Smoothed exceedance for the marginal-reward objective.

WHY THIS IS NOT OPTIONAL HERE. Measured across the 29 archived DK payout
tables, the rank-1 step carries a MEDIAN 50% of the objective's total weight
(range 20-70%) and the top three ranks carry 50-92%. So dR's precision is
almost entirely the precision of its tightest, rarest rungs -- the ones a hard
indicator resolves at split-half rho_full ~0.55, and smoothing lifts to
0.69-0.86 (memory project-smoothed-exceedance). A payout-ladder objective was
already REJECTED once on exactly this noise floor
(project-topn-selector-reproducibility); that rejection's premise is what this
module removes.

THE ESTIMATOR. `marginal_reward.tier_form_payout` rewrites the reward as
formulation (2)'s weighted sum of exceedance indicators -- verified identical
to the rank lookup at every rank of all 44 payout tables -- so each hard
`1[rank <= r_d]` can be replaced by `P(score > G^(r_d))` under the rank-r_d
order statistic's own sampling distribution. That is a Rao-Blackwellisation:
same target, strictly lower variance.

SELF-COMPETITION UNDER SMOOTHING. Our own entries displace a candidate by an
integer number of ranks, so the effective bar is the (r_d - own_above)-th best
field score rather than the r_d-th. We do not have thresholds at every rank --
only at the ~20 tier boundaries -- so the shift is taken along the SAME local
score-vs-rank slope that `smoothing_tau` already finite-differences to build
tau. That keeps one linearisation, not two:

    z = (score - thr_d - shift * slope_d) / tau_d
    P = expit(1.702 * z)

where `shift = own_above + 0.5 * own_ties`. The half-rank for ties is not a
fudge: smoothing works in continuous score space, where an exact tie is
measure-zero and so has no representation at all -- which would let an exact
DUPLICATE of a selected lineup escape penalty entirely, since a twin never
strictly outscores its twin. A tie is a coin flip on who ranks above, so its
expected displacement is half a rank.

`smoothing_tau` and `_rung_bracket_ranks` are imported from external_pool
rather than reimplemented, so the smoothing width here is the one whose
variance reduction was actually measured.

MEASURED BEHAVIOUR ON dR (synthetic fields, real DK tables, F = 2,972 /
11,437 / 29,411 -- consistent across all three):

    tau_scale   rho vs exact dR   level ratio
        0.5          0.988           1.045
        1.0          0.966           1.235

The level ratio is NOT a bug and must not be "corrected". The hard indicator
answers "did this beat THIS field draw"; the smoothed one answers "would it
beat a freshly drawn field of this size", which is genuinely non-zero in worlds
where this particular draw beat us. dR is used as an argmax, so a level shift
is harmless -- but it is not uniform across candidates, which is why rho falls
as tau grows, and why tau_scale is a swept knob rather than a constant.

DEFAULT IS OFF (`smooth_tau_scale=0.0`, the exact path), matching this repo's
posture on `external_pool_topn_smooth_tau_scale`: the derivation justifies the
width, only a walk-forward justifies turning it on.

KNOWN LIMITATION: an EXACT duplicate is under-penalised relative to the exact
path (retains ~77% of its standalone value vs ~37%), because a tie has no
representation in continuous score space and is modelled as half a rank rather
than an outright prize split. Unreachable in production -- the pool cull
removes every 9/10 pair -- and pinned by
tests/test_mrp_smoothing.py::test_exact_duplicates_are_penalised_harder_by_the_exact_path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from src.api.external_pool import (
    _LOGISTIC_NORMAL_SCALE,
    _SMOOTH_TAU_FLOOR,
    _rung_bracket_ranks,
    smoothing_tau,
)
from src.optimization.mrp.field_covariance import tier_boundary_ranks


@dataclass
class TierSmoothing:
    """Per-tier, per-world threshold / slope / width, plus the objective weights.

    thr, slope, tau : (T, S) float32
    steps           : (T,)   float64 -- (R_d - R_{d+1}), the weight tier d
                      carries in formulation (2).
    """

    thr: np.ndarray
    slope: np.ndarray
    tau: np.ndarray
    steps: np.ndarray

    @property
    def n_tiers(self) -> int:
        return int(self.thr.shape[0])


def build_tier_smoothing(
    field_sorted: np.ndarray,
    payout_arr: np.ndarray,
    tau_scale: float = 1.0,
) -> TierSmoothing:
    """Extract the ~20 tier rungs from a contest's field, once.

    Consumes `field_sorted` (S, F) ascending but retains only (T, S) arrays, so
    the caller can drop the multi-GB field array immediately afterwards.
    """
    payout_arr = np.asarray(payout_arr, dtype=np.float64)
    ranks = tier_boundary_ranks(payout_arr)
    if ranks.size == 0:
        raise ValueError("payout table has no paying tiers")

    S, F = field_sorted.shape
    amounts = payout_arr[np.clip(ranks - 1, 0, len(payout_arr) - 1)]
    steps = amounts - np.concatenate((amounts[1:], [0.0]))

    T = len(ranks)
    thr = np.empty((T, S), dtype=np.float32)
    slope = np.empty((T, S), dtype=np.float32)
    tau = np.empty((T, S), dtype=np.float32)
    for d, r in enumerate(ranks):
        rr = int(np.clip(int(r), 1, F))
        lo, hi = _rung_bracket_ranks(rr, F)
        thr[d] = field_sorted[:, F - rr]
        # Same finite difference smoothing_tau uses internally, kept here
        # because the self-competition shift needs the slope itself.
        slope[d] = (field_sorted[:, F - lo] - field_sorted[:, F - hi]) / float(max(hi - lo, 1))
        tau[d] = smoothing_tau(field_sorted, rr, F, tau_scale)
    np.maximum(slope, 0.0, out=slope)
    np.maximum(tau, _SMOOTH_TAU_FLOOR, out=tau)
    return TierSmoothing(thr=thr, slope=slope, tau=tau, steps=steps)


@njit(cache=True, inline="always")
def _expit(x):
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


@njit(parallel=True, cache=True)
def _smooth_main_term(cand_scores, own_above, own_ties, thr, slope, tau, steps, cutoff_z):
    """(M,) smoothed expected gross dollars for each candidate.

    Tiers are ordered tightest-rank-first, so `thr` DESCENDS in d. The loop
    therefore runs from the loosest rung inward and breaks as soon as the
    candidate is far below a threshold: every remaining tier has a higher bar
    still, so its contribution is smaller than one already deemed negligible.
    That early exit is what keeps an (M x S x T) triple loop affordable -- in a
    top-heavy contest the overwhelming majority of (candidate, world) pairs are
    nowhere near any paying rung.
    """
    M, S = cand_scores.shape
    T = thr.shape[0]
    out = np.zeros(M, dtype=np.float64)
    for m in prange(M):
        acc = 0.0
        for s in range(S):
            v = np.float64(cand_scores[m, s])
            a = np.float64(own_above[m, s]) + 0.5 * np.float64(own_ties[m, s])
            for d in range(T - 1, -1, -1):
                z = (v - np.float64(thr[d, s]) - a * np.float64(slope[d, s])) / np.float64(tau[d, s])
                if z < cutoff_z:
                    break
                acc += steps[d] * _expit(_LOGISTIC_NORMAL_SCALE * z)
        out[m] = acc / S
    return out


@njit(parallel=True, cache=True)
def _smooth_incumbent_deltas(inc_scores, inc_above, inc_ties, thr, slope, tau, steps, cutoff_z):
    """(k, S) x2: what ONE more entry above / tied with each incumbent costs it.

    Returns `(delta_gt, delta_eq)` in dollars, both normally negative. The
    displacement is +1 rank for a strictly-better new entry and +0.5 for a
    tying one, along the same local slope the thresholds use.
    """
    k, S = inc_scores.shape
    T = thr.shape[0]
    d_gt = np.zeros((k, S), dtype=np.float64)
    d_eq = np.zeros((k, S), dtype=np.float64)
    for i in prange(k):
        for s in range(S):
            v = np.float64(inc_scores[i, s])
            a = np.float64(inc_above[i, s]) + 0.5 * np.float64(inc_ties[i, s])
            acc_gt = 0.0
            acc_eq = 0.0
            for d in range(T - 1, -1, -1):
                sl = np.float64(slope[d, s])
                tv = np.float64(tau[d, s])
                base = v - np.float64(thr[d, s]) - a * sl
                z0 = base / tv
                z1 = (base - sl) / tv
                if z0 < cutoff_z and z1 < cutoff_z:
                    break
                p0 = _expit(_LOGISTIC_NORMAL_SCALE * z0)
                acc_gt += steps[d] * (_expit(_LOGISTIC_NORMAL_SCALE * z1) - p0)
                acc_eq += steps[d] * (
                    _expit(_LOGISTIC_NORMAL_SCALE * ((base - 0.5 * sl) / tv)) - p0
                )
            d_gt[i, s] = acc_gt
            d_eq[i, s] = acc_eq
    return d_gt, d_eq
