"""The pool-wide floor cull on the MRP path.

MRP shipped without this cull while every other ev_type applied it, so a
lineup the rest of the funnel considers unplayable was still reachable by
`dR` (2 of 79 entries in the 08/22 live portfolio came from below the
configured cutoff). These tests pin the two things that make the wiring
trustworthy: MRP culls the SAME lineups `allocate_contests` would from the
same basis, and an A/B's preassigned incumbents are exempt from it -- those
entries are already bought, so culling them would silently delete the other
arm rather than the candidate.
"""
import numpy as np
import pytest

from src.api.external_pool import compute_proj_score_floor
from src.optimization.mrp.runner import _floor_keep_indices, allocate_marginal_reward
from tests.test_mrp_runner import CFG, _fixture, _group


def _basis(n, seed=3):
    """A floor basis with a couple of non-finite cells, which the cull must
    treat as failures rather than as passes or as errors."""
    b = np.random.default_rng(seed).normal(150.0, 12.0, size=n)
    b[1] = np.nan
    b[2] = -np.inf
    return b


def test_matches_allocate_contests_mask_exactly():
    """Same basis, same percentile -> same survivors, cell for cell."""
    n = 40
    basis = _basis(n)
    keep_idx, diag = _floor_keep_indices(n, basis, 30.0, None)

    cutoff, n_culled = compute_proj_score_floor(basis, 30.0)
    expected = np.flatnonzero(np.isfinite(basis) & (basis >= cutoff))

    np.testing.assert_array_equal(keep_idx, expected)
    assert diag["cutoff"] == pytest.approx(cutoff)
    assert diag["n_culled"] == n_culled == n - len(expected)


def test_non_finite_scores_are_culled_not_kept():
    basis = _basis(40)
    keep_idx, _ = _floor_keep_indices(40, basis, 30.0, None)
    assert 1 not in keep_idx, "NaN ceiling must not survive the floor"
    assert 2 not in keep_idx, "-inf ceiling must not survive the floor"


@pytest.mark.parametrize("pct", [0.0, -5.0])
def test_disabled_percentile_keeps_everything(pct):
    keep_idx, diag = _floor_keep_indices(40, _basis(40), pct, None)
    np.testing.assert_array_equal(keep_idx, np.arange(40))
    assert diag == {}


def test_no_floor_scores_keeps_everything():
    keep_idx, diag = _floor_keep_indices(40, None, 30.0, None)
    np.testing.assert_array_equal(keep_idx, np.arange(40))
    assert diag == {}


def test_misaligned_floor_scores_raise():
    with pytest.raises(ValueError, match="align"):
        _floor_keep_indices(40, np.zeros(39), 30.0, None)


def test_preassigned_incumbents_are_exempt():
    """The other arm's bought entries stay in, however low their ceiling."""
    n = 40
    basis = _basis(n)
    cutoff, _ = compute_proj_score_floor(basis, 30.0)
    doomed = [int(i) for i in np.flatnonzero(~(np.isfinite(basis) & (basis >= cutoff)))[:3]]

    keep_idx, diag = _floor_keep_indices(n, basis, 30.0, {"c1": doomed})

    assert set(doomed).issubset(set(keep_idx.tolist()))
    assert diag["n_preassigned_exempt"] == len(doomed)
    assert diag["n_culled"] == diag["n_culled_before_exempt"] - len(doomed)


# --- end to end through the real allocator -------------------------------

def test_selection_never_returns_a_below_floor_lineup():
    df, sim, pool = _fixture(n_pool=40)
    basis = _basis(len(pool.lineups))
    cutoff, _ = compute_proj_score_floor(basis, 50.0)
    below = {
        frozenset(lu.player_ids)
        for lu, b in zip(pool.lineups, basis)
        if not (np.isfinite(b) and b >= cutoff)
    }
    assert below, "fixture must actually have culled lineups to be meaningful"

    alloc, diag = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 5)], CFG,
        floor_scores=basis, proj_score_floor_percentile=50.0,
    )

    picked = {frozenset(lu.player_ids) for lu, _ in alloc.portfolio}
    assert picked, "the cull must not starve the contest entirely"
    assert not (picked & below), "a below-floor lineup reached the portfolio"
    assert diag.floor["n_culled"] == len(below)


def test_floor_changes_the_portfolio_it_would_otherwise_have_built():
    """Guards against the cull being wired but inert -- if MRP's picks were
    all above the cutoff anyway, this test would pass vacuously and tell us
    nothing, so it asserts the two portfolios actually differ."""
    df, sim, pool = _fixture(n_pool=40)
    groups = [_group("c1", "Four-Seamer", 6)]

    uncut, _ = allocate_marginal_reward(pool, df, sim, groups, CFG)
    # Floor out precisely what the unfiltered run chose, so the floored run is
    # forced onto different lineups.
    chosen = {frozenset(lu.player_ids) for lu, _ in uncut.portfolio}
    basis = np.array(
        [0.0 if frozenset(lu.player_ids) in chosen else 100.0 for lu in pool.lineups],
        dtype=float,
    )

    floored, diag = allocate_marginal_reward(
        pool, df, sim, groups, CFG, floor_scores=basis, proj_score_floor_percentile=20.0,
    )

    assert diag.floor["n_culled"] == len(chosen)
    assert not ({frozenset(lu.player_ids) for lu, _ in floored.portfolio} & chosen)


def test_indices_remap_so_portfolio_holds_the_right_lineups():
    """The cull subsets the candidate axis, so every pick is an index into the
    SURVIVORS. An un-remapped index would still return a valid-looking Lineup
    from the uncut pool -- silently the wrong one -- which no shape assertion
    would catch."""
    df, sim, pool = _fixture(n_pool=40)
    basis = np.array([0.0] * 20 + [100.0] * 20, dtype=float)
    survivors = {frozenset(lu.player_ids) for lu in pool.lineups[20:]}

    alloc, _ = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 5)], CFG,
        floor_scores=basis, proj_score_floor_percentile=50.0,
    )

    for lu, _ in alloc.portfolio:
        assert frozenset(lu.player_ids) in survivors


def test_preassigned_survives_end_to_end_and_stays_out_of_the_picks():
    df, sim, pool = _fixture(n_pool=40)
    basis = np.array([0.0] * 20 + [100.0] * 20, dtype=float)
    pre = {"c1": [0, 1]}  # both below the cutoff

    alloc, diag = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 6)], CFG,
        preassigned=pre, floor_scores=basis, proj_score_floor_percentile=50.0,
    )

    assert diag.floor["n_preassigned_exempt"] == 2
    incumbents = {frozenset(pool.lineups[i].player_ids) for i in (0, 1)}
    picked = {frozenset(lu.player_ids) for lu, _ in alloc.portfolio}
    assert not (picked & incumbents), "an incumbent must not be re-picked"
    assert len(alloc.portfolio) + len(alloc.unfilled) == 4, "6 slots less 2 preassigned"


def test_all_non_finite_basis_disables_the_cull_rather_than_emptying_the_pool():
    """`compute_proj_score_floor` returns None when no score is finite, so
    there is no percentile to cut on and `allocate_contests` skips the cull
    entirely. MRP must do the same: a projections file that lost its "99th"
    column should fall back to no floor, not silently refuse to enter any
    contest. (Individual non-finite cells still fail the floor -- see
    test_non_finite_scores_are_culled_not_kept.)"""
    df, sim, pool = _fixture(n_pool=40)
    basis = np.full(len(pool.lineups), np.nan)

    alloc, diag = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 5)], CFG,
        floor_scores=basis, proj_score_floor_percentile=30.0,
    )

    assert diag.floor == {}, "no cutoff exists, so no cull is reported"
    assert len(alloc.portfolio) == 5, "the contest still fills"


def test_a_cull_that_starves_a_contest_leaves_the_rest_unfilled():
    """A floor tight enough to leave fewer survivors than slots must under-fill
    the contest rather than reach back below the cutoff for a warm body.

    (The survivor set can never be fully empty: the cutoff is a percentile OF
    THE FINITE SCORES, so the largest finite score always clears it. The
    runner's empty-pool guard is therefore defensive only -- an all-non-finite
    basis takes the no-cull path above instead.)"""
    df, sim, pool = _fixture(n_pool=40)
    basis = np.full(len(pool.lineups), -np.inf)
    basis[0] = 10.0

    alloc, diag = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 5)], CFG,
        floor_scores=basis, proj_score_floor_percentile=99.0,
    )

    assert diag.floor["n_culled"] == len(pool.lineups) - 1
    assert len(alloc.portfolio) == 1, "only the single survivor can be played"
    assert frozenset(alloc.portfolio[0][0].player_ids) == frozenset(pool.lineups[0].player_ids)
    assert len(alloc.unfilled) == 4
    assert diag.n_unfilled == 4
