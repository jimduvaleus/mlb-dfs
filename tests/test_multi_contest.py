"""Per-contest selection: fill order, disjointness, and the contest-outer sweep.

The invariants here are the ones that were silent when broken: a lineup entered
twice, an arm constraining another arm, a field width that does not match the
ladder it is scored against, and the dupe model applied unscaled across
contests two orders of magnitude apart in size.
"""
from pathlib import Path

import numpy as np
import pytest

from src.optimization.gpp_portfolio import DUPE_REF_FIELD_SIZE
from src.optimization.multi_contest import (
    SharedFieldPool,
    assign_per_contest,
    contest_payout_matrix,
    describe_slots,
    resolve_contest_slots,
    scale_dupes_for_field,
    select_per_contest,
    select_per_contest_multi_arm,
)


class _Rec:
    """Minimal stand-in for EntryRecord — resolve_contest_slots uses getattr."""

    def __init__(self, contest_name, contest_id, fee_cents, pool_cents, entry_id=0):
        self.contest_name = contest_name
        self.contest_id = contest_id
        self.entry_fee_cents = fee_cents
        self.prize_pool_cents = pool_cents
        self.entry_fee_raw = f"${fee_cents / 100:.2f}"
        self.entry_id = entry_id


def _entries(spec):
    """spec: [(name, contest_id, fee_$, pool_$, n_entries)] -> all_file_entries."""
    recs = []
    eid = 0
    for name, cid, fee, pool, n in spec:
        for _ in range(n):
            recs.append(_Rec(name, cid, int(fee * 100), int(pool * 100), eid))
            eid += 1
    return [(Path("DKEntries.csv"), recs)]


REAL_SPEC = [
    # name,                                  id,    fee,  pool,   entries
    ("MLB $175K Bat Flip [$50K to 1st]", "c-flip", 18.0, 175_000.0, 20),
    ("MLB $3 Hot Corner", "c-corner", 3.0, 1_500.0, 12),
    ("MLB $25 Skipper", "c-skip", 25.0, 5_000.0, 1),
]


# --------------------------------------------------------------------------
# Grouping and fill order
# --------------------------------------------------------------------------

def test_groups_by_contest_id_and_counts_entries():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    by_id = {s.contest_id: s for s in slots}
    assert set(by_id) == {"c-flip", "c-corner", "c-skip"}
    assert by_id["c-flip"].n_entries == 20
    assert by_id["c-corner"].n_entries == 12
    assert by_id["c-skip"].n_entries == 1
    assert sum(s.n_entries for s in slots) == 33


def test_blank_contest_names_are_skipped_not_made_into_a_phantom_slot():
    files = _entries(REAL_SPEC)
    files[0][1].append(_Rec("", "", 0, 0))
    files[0][1].append(_Rec("   ", None, 0, 0))
    slots = resolve_contest_slots(files)
    assert len(slots) == 3
    assert all(s.contest_name.strip() for s in slots)


def test_fill_order_leads_with_top_prize():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    prizes = [s.top_prize for s in slots]
    assert prizes == sorted(prizes, reverse=True)
    # The $50K-to-1st contest gets first refusal on the pool, not the one with
    # the most entries in the file.
    assert slots[0].contest_id == "c-flip"


def test_fill_order_is_stable_for_identical_contests():
    spec = [
        ("MLB $3 Pickoff", "a", 3.0, 2_000.0, 4),
        ("MLB $3 Pickoff", "b", 3.0, 2_000.0, 4),
    ]
    slots = resolve_contest_slots(_entries(spec))
    assert [s.contest_id for s in slots] == ["a", "b"]


def test_resolved_ladder_matches_the_advertised_pool_not_backsolved_entries():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    flip = next(s for s in slots if s.contest_id == "c-flip")
    # A $175K pool at an $18 fee back-solves to 9,722 entries, which would
    # match the 9,803-entry Bat Flip variant; the advertised pool identifies the
    # 11,437 one exactly. The two differ by $25K of prize money.
    assert flip.field_size == 11_437
    assert flip.prize_pool == pytest.approx(175_000.0, rel=0.01)
    assert flip.top_prize == pytest.approx(50_000.0)
    assert not flip.is_approximate


def test_describe_slots_lists_every_contest():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    text = describe_slots(slots)
    assert len(text.splitlines()) == len(slots) + 1
    for s in slots:
        assert s.contest_name[:40] in text


# --------------------------------------------------------------------------
# Selection: disjointness and arm independence
# --------------------------------------------------------------------------

def _take_first(ctx, avail, k, slot):
    return list(avail[:k])


def test_select_per_contest_slices_are_disjoint():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    M = 200
    cands = list(range(M))
    picks, diag = select_per_contest(
        slots, cands, np.zeros((M, 4), dtype=np.float32),
        make_field=lambda f: None,
        select_fn=lambda field, payout, avail, k, slot: _take_first(None, avail, k, slot),
    )
    flat = [i for v in picks.values() for i in v]
    assert len(flat) == len(set(flat)) == 33
    assert diag[slots[0].contest_id]["n_entries"] == slots[0].n_entries


def test_exclude_used_false_lets_contests_reuse_lineups():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    M = 200
    picks, _ = select_per_contest(
        slots, list(range(M)), np.zeros((M, 4), dtype=np.float32),
        make_field=lambda f: None,
        select_fn=lambda field, payout, avail, k, slot: _take_first(None, avail, k, slot),
        exclude_used=False,
    )
    flat = [i for v in picks.values() for i in v]
    assert len(flat) == 33
    assert len(set(flat)) < 33  # every contest started from index 0 again


def test_shortlist_too_small_raises_rather_than_silently_underfilling():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    with pytest.raises(RuntimeError, match="remain unused"):
        select_per_contest(
            slots, list(range(25)), np.zeros((25, 4), dtype=np.float32),
            make_field=lambda f: None,
            select_fn=lambda field, payout, avail, k, slot: list(avail[:k]),
        )


def test_prepare_runs_once_per_contest_not_once_per_arm():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    calls = []
    picks, _ = select_per_contest_multi_arm(
        slots, 200,
        make_field=lambda f: f,
        prepare_fn=lambda field, slot: calls.append(slot.contest_id) or field,
        arm_fns={f"arm{i}": _take_first for i in range(6)},
    )
    # Six arms, three contests: the expensive per-contest work happens 3 times.
    assert calls == [s.contest_id for s in slots]
    assert set(picks) == {f"arm{i}" for i in range(6)}


def test_arms_do_not_constrain_each_other():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    picks, _ = select_per_contest_multi_arm(
        slots, 200,
        make_field=lambda f: None,
        prepare_fn=lambda field, slot: None,
        arm_fns={"a": _take_first, "b": _take_first},
    )
    # Same deterministic rule, so the two arms agree exactly -- which they only
    # can if each keeps its own `used` set.
    assert picks["a"] == picks["b"]
    for arm in ("a", "b"):
        flat = [i for v in picks[arm].values() for i in v]
        assert len(flat) == len(set(flat)) == 33


def test_multi_arm_pool_consumption_diagnostic_tracks_the_greediest_arm():
    slots = resolve_contest_slots(_entries(REAL_SPEC))
    _, diag = select_per_contest_multi_arm(
        slots, 100,
        make_field=lambda f: None,
        prepare_fn=lambda field, slot: None,
        arm_fns={"a": _take_first},
    )
    last = diag[slots[-1].contest_id]
    assert last["pool_consumed_pct"] == pytest.approx(33.0)
    assert last["table_name"]
    assert last["entry_fee"] > 0


# --------------------------------------------------------------------------
# Assignment back onto entries
# --------------------------------------------------------------------------

def test_assign_per_contest_puts_each_contests_lineups_on_its_own_entries():
    files = _entries(REAL_SPEC)
    slots = resolve_contest_slots(files)
    cands = [f"lu{i}" for i in range(200)]
    picks, _ = select_per_contest(
        slots, cands, np.zeros((200, 4), dtype=np.float32),
        make_field=lambda f: None,
        select_fn=lambda field, payout, avail, k, slot: list(avail[:k]),
    )
    out = assign_per_contest(slots, picks, cands)
    pairs = out[Path("DKEntries.csv")]
    assert len(pairs) == 33
    by_contest: dict = {}
    for rec, lu in pairs:
        by_contest.setdefault(rec.contest_id, []).append(lu)
    for slot in slots:
        expected = [cands[i] for i in picks[slot.contest_id]]
        assert by_contest[slot.contest_id] == expected


# --------------------------------------------------------------------------
# Shared field pool
# --------------------------------------------------------------------------

def _pool(n_raw=40, n_samples=2, n_sims=5, seed=0):
    rng = np.random.default_rng(seed)
    raws = [rng.integers(0, 50, size=(n_raw, 10)) for _ in range(n_samples)]
    # sort_fn stands in for score-and-sort: one row per sim world, ascending.
    def sort_fn(sub):
        base = sub.sum(axis=1).astype(np.float32)
        out = np.tile(base, (n_sims, 1))
        return np.ascontiguousarray(np.sort(out, axis=1))
    return SharedFieldPool(raws, sort_fn, seed=7), raws


def test_field_pool_subsamples_to_the_requested_size():
    pool, _ = _pool(n_raw=40, n_samples=3)
    fields = list(pool.fields(12))
    assert len(fields) == 3 == pool.n_samples
    assert all(f.shape == (5, 12) for f in fields)
    assert pool.capacity == 40


def test_field_pool_caps_at_capacity_rather_than_oversampling():
    pool, _ = _pool(n_raw=40)
    assert all(f.shape[1] == 40 for f in pool.fields(10_000))


def test_field_pool_subsets_are_nested_across_contest_sizes():
    """A small contest's field must be a subset of a large one's — that is what
    makes the ownership spread between them a ladder effect, not field noise."""
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 1000, size=(60, 10))
    pool = SharedFieldPool([raw], sort_fn=lambda sub: sub, seed=11)
    small = {tuple(r) for r in list(pool.fields(10))[0]}
    large = {tuple(r) for r in list(pool.fields(35))[0]}
    assert len(small) == 10 and len(large) == 35
    assert small <= large


def test_field_pool_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 1000, size=(60, 10))
    a = list(SharedFieldPool([raw], lambda s: s, seed=11).fields(20))[0]
    b = list(SharedFieldPool([raw], lambda s: s, seed=11).fields(20))[0]
    c = list(SharedFieldPool([raw], lambda s: s, seed=12).fields(20))[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_empty_field_pool_raises():
    pool = SharedFieldPool([], lambda s: s)
    with pytest.raises(ValueError, match="empty"):
        list(pool.fields(10))


# --------------------------------------------------------------------------
# Dupe scaling across contest sizes
# --------------------------------------------------------------------------

def test_dupe_scale_is_linear_in_field_size():
    e = np.array([2.0, 0.0, 10.0])
    at_ref = scale_dupes_for_field(e, int(DUPE_REF_FIELD_SIZE))
    assert at_ref == pytest.approx(1.0 / (1.0 + e), rel=1e-5)
    # A 234-entry contest duplicates ~64x less than the reference field.
    small = scale_dupes_for_field(e, 234)
    assert small[0] > at_ref[0]
    assert small[0] == pytest.approx(
        1.0 / (1.0 + 2.0 * 234 / DUPE_REF_FIELD_SIZE), rel=1e-5,
    )
    assert np.all(small <= 1.0)


def test_dupe_scale_never_exceeds_one_for_zero_dupes():
    assert scale_dupes_for_field(np.array([0.0]), 20_000)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Per-contest payout matrix
# --------------------------------------------------------------------------

def _flat_ladder(n_paid, prize, total):
    arr = np.zeros(total, dtype=np.float32)
    arr[:n_paid] = prize
    return arr


def test_payout_matrix_subtracts_the_entry_fee():
    # One candidate that beats the entire field in every world.
    cand = np.full((1, 4), 500.0, dtype=np.float32)
    field = np.zeros((4, 10), dtype=np.float32)
    ladder = _flat_ladder(3, 10.0, 10)          # top 3 of 10 ranks pay
    out = contest_payout_matrix(cand, [field], ladder, entry_fee=2.0)
    assert out.shape == (1, 4)
    assert np.all(out > 0)
    # A candidate that beats nobody finishes out of the money in every world,
    # so its payout is exactly the entry fee it burned.
    loser = contest_payout_matrix(
        np.full((1, 4), -500.0, dtype=np.float32), [field], ladder, entry_fee=2.0,
    )
    assert np.allclose(loser, -2.0)


def test_payout_matrix_averages_over_field_samples():
    cand = np.array([[100.0, 100.0]], dtype=np.float32)
    win = np.zeros((2, 8), dtype=np.float32)          # candidate beats all
    lose = np.full((2, 8), 1_000.0, dtype=np.float32)  # candidate beats none
    ladder = _flat_ladder(4, 20.0, 8)
    only_win = contest_payout_matrix(cand, [win], ladder, entry_fee=1.0)
    only_lose = contest_payout_matrix(cand, [lose], ladder, entry_fee=1.0)
    both = contest_payout_matrix(cand, [win, lose], ladder, entry_fee=1.0)
    assert both == pytest.approx((only_win + only_lose) / 2.0, rel=1e-4)


def test_payout_matrix_is_chunk_invariant():
    rng = np.random.default_rng(5)
    cand = rng.normal(120, 25, size=(37, 6)).astype(np.float32)
    field = np.ascontiguousarray(
        np.sort(rng.normal(120, 25, size=(6, 50)).astype(np.float32), axis=1)
    )
    ladder = _flat_ladder(12, 30.0, 50)
    a = contest_payout_matrix(cand, [field], ladder, 2.0, cand_chunk=4)
    b = contest_payout_matrix(cand, [field], ladder, 2.0, cand_chunk=1_000)
    assert np.allclose(a, b)


def test_payout_matrix_rejects_a_field_width_change_mid_contest():
    cand = np.zeros((2, 3), dtype=np.float32)
    f1 = np.zeros((3, 10), dtype=np.float32)
    f2 = np.zeros((3, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="field width changed"):
        contest_payout_matrix(cand, [f1, f2], _flat_ladder(4, 5.0, 12), 1.0)


def test_payout_matrix_requires_at_least_one_field():
    with pytest.raises(ValueError, match="no field samples"):
        contest_payout_matrix(
            np.zeros((2, 3), dtype=np.float32), iter([]),
            _flat_ladder(4, 5.0, 12), 1.0,
        )


def test_dupe_penalty_only_reduces_the_winning_bands():
    rng = np.random.default_rng(9)
    cand = rng.normal(120, 25, size=(20, 8)).astype(np.float32)
    field = np.ascontiguousarray(
        np.sort(rng.normal(120, 25, size=(8, 60)).astype(np.float32), axis=1)
    )
    ladder = np.zeros(60, dtype=np.float32)
    ladder[:5] = 400.0
    ladder[5:20] = 20.0
    clean = contest_payout_matrix(cand, [field], ladder, 2.0)
    duped = contest_payout_matrix(
        cand, [field], ladder, 2.0,
        e_dupes=np.full(20, 5.0), dupe_min_gross_payout=100.0,
    )
    assert np.all(duped <= clean + 1e-4)
    assert duped.mean() < clean.mean()
