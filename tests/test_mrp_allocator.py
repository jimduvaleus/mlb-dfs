"""Tests for the multi-contest marginal-dollar greedy.

Covers the three overlap rules, the exogenous-slot (partition matroid)
contract, and the two load-bearing optimisations: that the global argmax is
really the global argmax, and that gains for untouched contests are genuinely
unchanged (the lazy claim the whole runtime rests on).
"""
import numpy as np
import pytest

from src.optimization.mrp.allocator import AllocationRules, allocate
from src.optimization.mrp.delta_reward import ContestDeltaState

ROSTER = 10


def _payout(total=120):
    arr = np.zeros(total, dtype=np.float64)
    for a, b, amt in [(0, 1, 800.0), (1, 2, 300.0), (2, 5, 100.0),
                      (5, 12, 30.0), (12, 40, 4.0)]:
        arr[a:b] = amt
    return arr


def _pool(rng, M, n_players=60):
    """(M, ROSTER) player ids + the (P, M) indicator matrix."""
    ids = np.array([rng.choice(n_players, size=ROSTER, replace=False) for _ in range(M)])
    ind = np.zeros((n_players, M), dtype=np.float32)
    for j in range(M):
        ind[ids[j], j] = 1.0
    return ids, ind


def _states(rng, n_contests, M, S=50, F=70):
    return _build(_arrays(rng, n_contests, M, S, F))


def _arrays(rng, n_contests, M, S=50, F=70):
    """Raw fixture arrays, so the same contest set can be rebuilt exactly."""
    return [
        (np.sort(rng.normal(120.0, 20.0, size=(S, F)).astype(np.float32), axis=1),
         rng.normal(128.0, 20.0, size=(M, S)).astype(np.float32))
        for _ in range(n_contests)
    ]


def _build(arrays):
    return {f"c{c}": ContestDeltaState(cand, field, _payout())
            for c, (field, cand) in enumerate(arrays)}


def _max_overlap(ids, a, b):
    return len(set(ids[a].tolist()) & set(ids[b].tolist()))


def test_slots_are_filled_exactly_and_not_exceeded():
    rng = np.random.default_rng(0)
    M = 40
    ids, ind = _pool(rng, M)
    states = _states(rng, 3, M)
    slots = {"c0": 5, "c1": 3, "c2": 4}

    res = allocate(states, slots, ind, AllocationRules(gamma_in=ROSTER, gamma_out=ROSTER))
    by = res.by_contest()

    assert not res.unfilled
    for cid, n in slots.items():
        assert len(by[cid]) == n
    assert len(res.picks) == sum(slots.values())


def test_no_lineup_appears_in_two_contests_by_default():
    rng = np.random.default_rng(1)
    M = 40
    ids, ind = _pool(rng, M)
    states = _states(rng, 3, M)
    res = allocate(states, {"c0": 4, "c1": 4, "c2": 4}, ind, AllocationRules())

    picked = [p.candidate for p in res.picks]
    assert len(picked) == len(set(picked)), "a lineup was reused across contests"


def test_cross_contest_duplicates_allowed_when_enabled():
    """With the rule off, the same strong lineup SHOULD land in several contests."""
    rng = np.random.default_rng(2)
    M = 40
    ids, ind = _pool(rng, M)
    states = _states(rng, 3, M)
    res = allocate(states, {"c0": 3, "c1": 3, "c2": 3}, ind,
                   AllocationRules(gamma_in=ROSTER, gamma_out=ROSTER,
                                   allow_cross_contest_duplicates=True))
    picked = [p.candidate for p in res.picks]
    assert len(picked) > len(set(picked)), "expected at least one reuse"


@pytest.mark.parametrize("gamma_in", [4, 6, 7])
def test_gamma_in_is_respected_within_a_contest(gamma_in):
    rng = np.random.default_rng(3)
    M = 120
    ids, ind = _pool(rng, M, n_players=26)      # small pool -> forced overlap
    states = _states(rng, 2, M)
    res = allocate(states, {"c0": 5, "c1": 5}, ind,
                   AllocationRules(gamma_in=gamma_in, gamma_out=ROSTER))

    for cid, members in res.by_contest().items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ov = _max_overlap(ids, members[a], members[b])
                assert ov <= gamma_in, f"{cid}: overlap {ov} > gamma_in {gamma_in}"


@pytest.mark.parametrize("gamma_out", [4, 6])
def test_gamma_out_is_respected_across_contests(gamma_out):
    rng = np.random.default_rng(4)
    M = 120
    ids, ind = _pool(rng, M, n_players=26)
    states = _states(rng, 3, M)
    res = allocate(states, {"c0": 4, "c1": 4, "c2": 4}, ind,
                   AllocationRules(gamma_in=ROSTER, gamma_out=gamma_out))

    by = res.by_contest()
    for ca in by:
        for cb in by:
            if ca >= cb:
                continue
            for a in by[ca]:
                for b in by[cb]:
                    ov = _max_overlap(ids, a, b)
                    assert ov <= gamma_out, f"{ca}/{cb}: overlap {ov} > gamma_out {gamma_out}"


def test_tighter_gamma_in_lowers_realised_intra_contest_overlap():
    """The knob has to actually move portfolio shape, not just pass a bound."""
    rng = np.random.default_rng(5)
    M = 400
    # 20 players over 10-player rosters => ~5 shared by chance, so gamma_in=3
    # is a constraint that genuinely binds rather than a bound that never trips.
    ids, ind = _pool(rng, M, n_players=20)
    arrays = _arrays(rng, 1, M)

    def mean_overlap(g):
        res = allocate(_build(arrays), {"c0": 5}, ind,
                       AllocationRules(gamma_in=g, gamma_out=ROSTER))
        assert not res.unfilled, f"gamma_in={g} could not fill the portfolio"
        mem = res.by_contest()["c0"]
        return np.mean([_max_overlap(ids, mem[a], mem[b])
                        for a in range(len(mem)) for b in range(len(mem)) if a < b])

    tight, loose = mean_overlap(4), mean_overlap(ROSTER)
    assert tight < loose, f"gamma_in did not reduce overlap: {tight} vs {loose}"


def test_greedy_picks_the_true_global_argmax_pair():
    """Verify the (candidate, contest) argmax against an exhaustive scan."""
    rng = np.random.default_rng(6)
    M = 25
    ids, ind = _pool(rng, M)
    arrays = _arrays(rng, 3, M)
    states = _build(arrays)
    rules = AllocationRules(gamma_in=ROSTER, gamma_out=ROSTER)

    res = allocate(states, {"c0": 2, "c1": 2, "c2": 2}, ind, rules)

    # Replay from scratch, choosing each step by brute-force scan.
    fresh = _build(arrays)                            # same arrays, fresh state
    used = np.zeros(M, dtype=bool)
    left = {"c0": 2, "c1": 2, "c2": 2}
    for pick in res.picks:
        best = None
        for cid in fresh:
            if left[cid] <= 0:
                continue
            g = np.where(~used, fresh[cid].marginal_gains(), -np.inf)
            j = int(np.argmax(g))
            if best is None or g[j] > best[0]:
                best = (float(g[j]), cid, j)
        assert (pick.contest_id, pick.candidate) == (best[1], best[2])
        assert pick.delta == pytest.approx(best[0])
        fresh[best[1]].commit(best[2])
        left[best[1]] -= 1
        used[best[2]] = True


def test_gains_of_untouched_contests_are_unchanged():
    """The lazy-refresh claim: committing in c cannot move dR in c' != c."""
    rng = np.random.default_rng(7)
    M = 30
    _, ind = _pool(rng, M)
    states = _states(rng, 2, M)

    before = states["c1"].marginal_gains().copy()
    states["c0"].commit(int(np.argmax(states["c0"].marginal_gains())))
    after = states["c1"].marginal_gains()

    np.testing.assert_array_equal(before, after)


def test_marginal_gains_are_non_increasing_across_picks():
    """Submodularity, observed end to end: the paper's saturation effect."""
    rng = np.random.default_rng(8)
    M = 60
    _, ind = _pool(rng, M)
    states = _states(rng, 1, M, S=200)
    res = allocate(states, {"c0": 10}, ind,
                   AllocationRules(gamma_in=ROSTER, gamma_out=ROSTER))

    deltas = [p.delta for p in res.picks]
    assert all(b <= a + 1e-9 for a, b in zip(deltas, deltas[1:])), deltas


def test_unfilled_reported_when_constraints_exhaust_the_pool():
    rng = np.random.default_rng(9)
    M = 6
    _, ind = _pool(rng, M)
    states = _states(rng, 1, M)
    res = allocate(states, {"c0": 20}, ind, AllocationRules())
    assert res.unfilled.get("c0", 0) > 0
    assert len(res.picks) == M


# ---------------------------------------------------------------------------
# Relaxation: a purchased slot is money already spent
# ---------------------------------------------------------------------------

def test_relaxes_rather_than_leaving_a_purchased_slot_empty():
    """A tight gamma_in on a thin pool must loosen, not silently drop entries.
    An unfilled entry is an entry fee paid for nothing."""
    rng = np.random.default_rng(20)
    M = 60
    # 14 players over 10-player rosters => every pair shares >= 6, so gamma_in=3
    # is unsatisfiable for a second lineup and MUST relax.
    ids, ind = _pool(rng, M, n_players=14)
    arrays = _arrays(rng, 1, M)

    res = allocate(_build(arrays), {"c0": 5}, ind,
                   AllocationRules(gamma_in=3, gamma_out=ROSTER))

    assert not res.unfilled, f"left {res.unfilled} unfilled instead of relaxing"
    assert len(res.by_contest()["c0"]) == 5
    assert res.relaxations, "relaxed silently -- the loosening must be recorded"
    assert all(r.contest_id == "c0" for r in res.relaxations)


def test_gamma_out_is_surrendered_before_gamma_in():
    """gamma_out is bankroll-variance control; gamma_in is the actual
    competition rule. Give up the free constraint first."""
    rng = np.random.default_rng(21)
    M = 60
    ids, ind = _pool(rng, M, n_players=14)
    arrays = _arrays(rng, 2, M)

    res = allocate(_build(arrays), {"c0": 4, "c1": 4}, ind,
                   AllocationRules(gamma_in=3, gamma_out=3))

    assert res.relaxations, "expected relaxations on this pool"
    # Caps are PER CONTEST, so the global sequence interleaves contests. The
    # ordering claim is within each contest.
    for cid in {r.contest_id for r in res.relaxations}:
        seq = [r.rule for r in res.relaxations if r.contest_id == cid]
        first_in = next((i for i, r in enumerate(seq) if r == "gamma_in"), len(seq))
        assert all(r == "gamma_out" for r in seq[:first_in]), (
            f"{cid}: gamma_in relaxed before gamma_out was exhausted: {seq}")
        # And gamma_out must be fully open by the time gamma_in moves.
        if first_in < len(seq):
            out_steps = sum(1 for r in seq[:first_in] if r == "gamma_out")
            assert out_steps == ROSTER - 3, (
                f"{cid}: only {out_steps} gamma_out steps before gamma_in moved")


def test_relaxation_is_scoped_to_the_starved_contest():
    rng = np.random.default_rng(22)
    M = 400
    # Wide pool: c0 is starved only because we give it an impossible cap.
    ids, ind = _pool(rng, M, n_players=14)
    arrays = _arrays(rng, 2, M)
    res = allocate(_build(arrays), {"c0": 6, "c1": 6}, ind,
                   AllocationRules(gamma_in=3, gamma_out=ROSTER))
    touched = {r.contest_id for r in res.relaxations}
    assert touched <= {"c0", "c1"}
    # Whichever contests relaxed, every slot still got filled.
    assert not res.unfilled


def test_genuine_exhaustion_still_reports_unfilled():
    """Relaxation cannot invent lineups: with fewer distinct candidates than
    slots, the shortfall must still be reported rather than hidden."""
    rng = np.random.default_rng(23)
    M = 6
    _, ind = _pool(rng, M)
    states = _states(rng, 1, M)
    res = allocate(states, {"c0": 20}, ind, AllocationRules())
    assert res.unfilled.get("c0", 0) == 14
    assert len(res.picks) == 6


def test_no_relaxations_when_the_pool_is_comfortable():
    rng = np.random.default_rng(24)
    M = 200
    _, ind = _pool(rng, M, n_players=60)
    arrays = _arrays(rng, 1, M)
    res = allocate(_build(arrays), {"c0": 5}, ind,
                   AllocationRules(gamma_in=7, gamma_out=8))
    assert res.relaxations == [], "relaxed with plenty of headroom"
