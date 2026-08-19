"""Efficient exact marginal reward: dR(j | S) = R(S u {j}) - R(S), one contest.

This is the piece production does not have. `self_play.run_contest_self_play`
already scores a candidate against `opponents u our own picks` -- the
best-response half -- but never re-scores the INCUMBENTS, so it misses that
adding j also pushes every already-selected entry down one rank in the worlds
where j beats it. That demotion term is what makes the objective submodular
and what makes diversity fall out mechanically instead of via a correlation
heuristic.

    dR(j | S) =  E_s[ payout_band(j) ]                          <- main term
               + sum_i E_s[ payout(i | j present) - payout(i) ]  <- demotion

BOTH TERMS ARE EXACT. No approximation is made anywhere in this file; the
speed comes from representation, not from dropping anything:

  * MAIN TERM. `n_above_field` is precomputed once per contest
    (`precompute_field_ranks`), so a pick costs a gather rather than an
    (M x S) searchsorted. `own_above` / `own_ties` are maintained
    INCREMENTALLY -- one broadcast compare against the newly committed entry
    per pick -- which is what keeps the greedy off an O(M . k . S) path.

  * DEMOTION TERM IS SPARSE BY CONSTRUCTION. The payout array is a step
    function over ~20 tiers, so demoting an incumbent by one rank changes its
    dollars ONLY when it sits exactly on a tier boundary in that world. We
    enumerate those (incumbent, world) cells -- a few per thousand -- and pay
    O(M) per cell instead of materialising anything (M x k x S)-shaped.

Memory, per contest, at M=5,100 candidates on a world slice of S: cand_scores
f32 (shared across contests, sliced) + n_above_field u16 + f_ties u8 +
own_above u8 + own_ties u8. `field_sorted` (the 2.9 GB array) is released
after the precompute. See CLAUDE.md's memory-conscious matrix rule -- the
naive form of this loop is exactly what killed the first self-play eval run.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.optimization.mrp.marginal_reward import _payout_cumsum, precompute_field_ranks


@njit(cache=True, inline="always")
def _band_mean(n_above, width_extra, payout_cum, L):
    """Tie-band mean payout, matching bt_core.grade_portfolio's clipped form."""
    lo = n_above
    if lo > L:
        lo = L
    hi = n_above + width_extra
    if hi > L:
        hi = L
    w = hi - lo
    if w <= 0:
        return 0.0
    return (payout_cum[hi] - payout_cum[lo]) / w


@njit(parallel=True, cache=True)
def _main_term(n_above_field, f_ties, own_above, own_ties, payout_cum, L):
    """(M,) expected gross dollars for each candidate if added to S.

    Band = [naf + own_above, naf + own_above + f_ties + own_ties + 1), the +1
    being the candidate itself (grade_portfolio's g_ties includes self).
    """
    M, S = n_above_field.shape
    out = np.zeros(M, dtype=np.float64)
    for m in prange(M):
        acc = 0.0
        for s in range(S):
            n_ab = np.int64(n_above_field[m, s]) + np.int64(own_above[m, s])
            extra = np.int64(f_ties[m, s]) + np.int64(own_ties[m, s]) + 1
            acc += _band_mean(n_ab, extra, payout_cum, L)
        out[m] = acc / S
    return out


@njit(parallel=True, cache=True)
def _demotion_term(cand_scores, inc_world, inc_score, delta_gt, delta_eq, M, S):
    """(M,) summed demotion cost, from the sparse (incumbent, world) cells.

    Cell c says: in world `inc_world[c]` an incumbent scoring `inc_score[c]`
    loses `delta_gt[c]` if one more entry finishes strictly ABOVE it, or
    `delta_eq[c]` if one more entry TIES it (widening the tie band it must
    share). Both are normally negative.

    The tie branch is not a rounding detail -- an exact duplicate of an
    already-selected lineup never outscores its twin, so without it the single
    most important case for a diversity mechanism registers zero cost.
    """
    out = np.zeros(M, dtype=np.float64)
    n_cells = inc_world.shape[0]
    for m in prange(M):
        acc = 0.0
        for c in range(n_cells):
            v = cand_scores[m, inc_world[c]]
            if v > inc_score[c]:
                acc += delta_gt[c]
            elif v == inc_score[c]:
                acc += delta_eq[c]
        out[m] = acc / S
    return out


class ContestDeltaState:
    """Incremental dR state for ONE contest. Reused across every greedy pick.

    Parameters
    ----------
    cand_scores : (M, S) float32 -- candidate scores on this contest's world
        slice. May be a view into a pool-wide array shared across contests.
    field_sorted : (S, F) float32 ascending. Consumed during construction and
        NOT retained.
    payout_arr : (L,) gross dollars by rank (index r-1), from
        `payout.payout_table_to_array`.
    """

    def __init__(self, cand_scores, field_sorted, payout_arr, chunk: int = 512):
        self.cand_scores = np.ascontiguousarray(cand_scores, dtype=np.float32)
        self.M, self.S = self.cand_scores.shape
        self.payout_arr = np.asarray(payout_arr, dtype=np.float64)
        self.payout_cum = _payout_cumsum(self.payout_arr)
        self.L = len(self.payout_arr)

        # Ranks beyond the payout table pay $0 no matter how far our own
        # entries displace them, so clamping there is lossless and keeps the
        # uint16 valid for fields larger than 65,535.
        self.n_above_field, self.f_ties = precompute_field_ranks(
            self.cand_scores, field_sorted, chunk=chunk, rank_cap=self.L
        )
        self.own_above = np.zeros((self.M, self.S), dtype=np.uint8)
        self.own_ties = np.zeros((self.M, self.S), dtype=np.uint8)
        self.selected: list[int] = []

    def marginal_gains(self) -> np.ndarray:
        """(M,) exact dR(j | S) for every candidate."""
        gains = _main_term(
            self.n_above_field, self.f_ties, self.own_above, self.own_ties,
            self.payout_cum, self.L,
        )
        if self.selected:
            cells = self._demotion_cells()
            if cells is not None:
                w, sc, d_gt, d_eq = cells
                gains = gains + _demotion_term(
                    self.cand_scores, w, sc, d_gt, d_eq, self.M, self.S
                )
        return gains

    def _demotion_cells(self):
        """Sparse (world, incumbent score, delta_gt, delta_eq) cells.

        An incumbent's dollars change under one extra entry only if that shifts
        it across a payout tier boundary (or widens a tie band it is already
        sharing), so almost every cell is exactly zero and is dropped here.

        Note the tie convention: `own_ties` at a committed entry's own row
        already counts that entry, so an incumbent's band width is
        `f_ties + own_ties` with NO further +1 -- unlike a candidate being
        evaluated for addition, which is not yet in the counter and so needs
        the +1 (see `_main_term`).
        """
        idx = np.asarray(self.selected, dtype=np.int64)
        naf = self.n_above_field[idx].astype(np.int64)
        ftie = self.f_ties[idx].astype(np.int64)
        # own_above at an incumbent's own row excludes itself (a lineup is not
        # strictly above itself); own_ties INCLUDES itself.
        oab = self.own_above[idx].astype(np.int64)
        otie = self.own_ties[idx].astype(np.int64)

        n_ab = naf + oab
        extra = ftie + otie
        before = _band_mean_vec(n_ab, extra, self.payout_cum, self.L)
        delta_gt = _band_mean_vec(n_ab + 1, extra, self.payout_cum, self.L) - before
        delta_eq = _band_mean_vec(n_ab, extra + 1, self.payout_cum, self.L) - before

        nz = np.nonzero((delta_gt != 0.0) | (delta_eq != 0.0))
        if nz[0].size == 0:
            return None
        inc_rows, inc_worlds = nz
        return (
            inc_worlds.astype(np.int64),
            self.cand_scores[idx[inc_rows], inc_worlds].astype(np.float32),
            delta_gt[inc_rows, inc_worlds].astype(np.float64),
            delta_eq[inc_rows, inc_worlds].astype(np.float64),
        )

    def commit(self, j: int) -> None:
        """Add candidate j to the portfolio and roll the incremental state."""
        j = int(j)
        v = self.cand_scores[j]                      # (S,)
        self.own_above += (self.cand_scores < v[None, :]).astype(np.uint8)
        self.own_ties += (self.cand_scores == v[None, :]).astype(np.uint8)
        self.selected.append(j)

    def reward(self) -> float:
        """R(S): expected total gross dollars for the committed portfolio."""
        if not self.selected:
            return 0.0
        idx = np.asarray(self.selected, dtype=np.int64)
        n_ab = self.n_above_field[idx].astype(np.int64) + self.own_above[idx].astype(np.int64)
        extra = self.f_ties[idx].astype(np.int64) + self.own_ties[idx].astype(np.int64)
        per = _band_mean_vec(n_ab, extra, self.payout_cum, self.L)
        return float(per.sum(axis=0).mean())


def _band_mean_vec(n_above, width_extra, payout_cum, L):
    """Vectorised `_band_mean` over an arbitrarily shaped integer array."""
    lo = np.clip(n_above, 0, L)
    hi = np.clip(n_above + width_extra, 0, L)
    w = hi - lo
    return np.where(w > 0, (payout_cum[hi] - payout_cum[lo]) / np.maximum(w, 1), 0.0)
