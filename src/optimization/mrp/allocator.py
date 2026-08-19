"""Multi-contest greedy: one global argmax over (candidate, contest) pairs.

Production (`external_pool.allocate_contests`) walks contests in a fixed order
-- entry fee desc, then prize pool asc -- and fills each to completion before
touching the next, coupling them only through a shared removal mask. Which
contest gets the good lineups is therefore decided by a hand-set sort key.

Here it is decided by marginal dollars. Every pick is the best (candidate,
contest) pair in the whole slate, so contest ordering falls out of the
objective. Entry counts per contest stay EXOGENOUS -- they come from the
purchased DK entries file -- which makes the constraint a partition matroid,
under which greedy on a monotone submodular function keeps a 1/2 guarantee
(Nemhauser-Wolsey-Fisher; the 1-1/e figure is the cardinality case).

LAZY BY CONSTRUCTION. Committing an entry to contest c changes dR only inside
contest c: our entries in different contests never compete. So a pick costs one
`marginal_gains()` call, not one per contest.

THREE OVERLAP RULES, doing three different jobs -- see `AllocationRules`.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np


@dataclass
class AllocationRules:
    """Overlap governance.

    gamma_in
        Max shared players against entries ALREADY IN THE SAME CONTEST. This
        is the EV rule -- the same-contest set is the only set our entries
        actually compete with. Haugh & Singal swept gamma in {5,6,7} of C=9 and
        found C-3 best; for DK MLB Classic (C=10) that is 7. Their gamma=C run
        collapsed 50 entries onto ~10 unique lineups, the same failure shape as
        our own p_win cull diversity collapse.
    gamma_out
        Max shared players against entries in ANY OTHER contest. This is NOT an
        EV rule -- separate contests never compete, so overlap across them
        costs nothing in expectation. It is bankroll-variance control, and it
        is priced as such. Note 8 is a no-op against a pool that has already
        been through `_find_near_duplicate_removals` (all 9/10 pairs culled, so
        two distinct pool lineups share at most 8 by construction); the knob
        only bites at 7 or lower.
    allow_cross_contest_duplicates
        False reproduces today's shared-removal mask. Kept off by user
        decision. Under the paper's logic this costs EV -- if a lineup is the
        best marginal addition in two separate contests there is no
        competitive reason not to play it twice -- so it is a risk choice, not
        an optimisation.
    """

    gamma_in: int = 7
    gamma_out: int = 8
    allow_cross_contest_duplicates: bool = False
    roster_size: int = 10


@dataclass
class Pick:
    contest_id: object
    candidate: int
    delta: float
    step: int


@dataclass
class AllocationResult:
    picks: list[Pick] = _field(default_factory=list)
    unfilled: dict = _field(default_factory=dict)

    def by_contest(self) -> dict:
        out: dict = {}
        for p in self.picks:
            out.setdefault(p.contest_id, []).append(p.candidate)
        return out

    def reward_total(self, states) -> float:
        return float(sum(states[cid].reward() for cid in {p.contest_id for p in self.picks}))


def _overlap_vector(indicator: np.ndarray, j: int) -> np.ndarray:
    """(M,) shared-player count between every candidate and candidate j.

    `indicator` is the (P, M) 0/1 matrix `external_pool._lineup_indicator_matrix`
    builds, so this is one matvec rather than a Python set intersection loop.
    """
    return indicator.T @ indicator[:, j]


def allocate(
    states: dict,
    slots: dict,
    indicator: np.ndarray,
    rules: AllocationRules | None = None,
    progress_cb=None,
) -> AllocationResult:
    """Fill every contest's slots by global marginal-dollar greedy.

    Parameters
    ----------
    states : {contest_id: ContestDeltaState} -- one per contest, each already
        built on that contest's own world slice and payout table.
    slots : {contest_id: int} -- entries purchased for that contest.
    indicator : (P, M) float32 0/1 lineup-composition matrix, shared by all
        contests (the candidate pool is one pool).
    """
    rules = rules or AllocationRules()
    contest_ids = [cid for cid in states if slots.get(cid, 0) > 0]
    if not contest_ids:
        return AllocationResult(unfilled=dict(slots))

    M = indicator.shape[1]
    C = rules.roster_size

    gains = {cid: states[cid].marginal_gains() for cid in contest_ids}
    remaining = {cid: int(slots[cid]) for cid in contest_ids}
    # Max shared-player count against entries already in this contest / in any
    # other contest. int16 keeps these trivially small at any realistic M.
    max_in = {cid: np.zeros(M, dtype=np.int16) for cid in contest_ids}
    max_out = {cid: np.zeros(M, dtype=np.int16) for cid in contest_ids}
    used_in = {cid: np.zeros(M, dtype=bool) for cid in contest_ids}
    used_any = np.zeros(M, dtype=bool)

    result = AllocationResult()
    total_slots = sum(remaining.values())

    for step in range(total_slots):
        best = None
        for cid in contest_ids:
            if remaining[cid] <= 0:
                continue
            ok = ~used_in[cid]
            if not rules.allow_cross_contest_duplicates:
                ok &= ~used_any
            if rules.gamma_in < C:
                ok &= max_in[cid] <= rules.gamma_in
            if rules.gamma_out < C:
                ok &= max_out[cid] <= rules.gamma_out
            if not ok.any():
                continue
            g = np.where(ok, gains[cid], -np.inf)
            j = int(np.argmax(g))
            if best is None or g[j] > best[0]:
                best = (float(g[j]), cid, j)

        if best is None or not np.isfinite(best[0]):
            break

        delta, cid, j = best
        states[cid].commit(j)
        result.picks.append(Pick(contest_id=cid, candidate=j, delta=delta, step=step))
        remaining[cid] -= 1
        used_in[cid][j] = True
        used_any[j] = True

        ov = _overlap_vector(indicator, j).astype(np.int16)
        np.maximum(max_in[cid], ov, out=max_in[cid])
        for other in contest_ids:
            if other != cid:
                np.maximum(max_out[other], ov, out=max_out[other])

        # Only the contest we just touched has stale gains: entries in
        # different contests never compete, so nothing else moved.
        if remaining[cid] > 0:
            gains[cid] = states[cid].marginal_gains()

        if progress_cb is not None:
            progress_cb(step + 1, total_slots)

    result.unfilled = {cid: n for cid, n in remaining.items() if n > 0}
    return result
