"""Contract test: MRP's joint reward must equal bt_core.grade_portfolio exactly.

`grade_portfolio` is the repo's realized-world truth for "k of our entries in
one contest" -- self-displacement, self-tie-splitting, our-dupe prize
splitting, clipped tie band. `src/optimization/mrp/marginal_reward.py`
reimplements those semantics over simulated worlds because src/ must not
import from tests/. This file is the contract binding the two.

If a test here fails, marginal_reward.py is wrong -- not bt_core.
"""
import numpy as np
import pytest

from src.optimization.mrp.marginal_reward import (
    _payout_cumsum,
    joint_gross_world,
    joint_gross_worlds,
    portfolio_reward,
    precompute_field_ranks,
)
from tests.bt_core import grade_pick, grade_pool, grade_portfolio


def _payout_arr(n_paying: int = 40, total: int = 200) -> np.ndarray:
    """A steep top-heavy table shaped like DK's: a few big tiers then a plateau."""
    arr = np.zeros(total, dtype=np.float64)
    tiers = [(0, 1, 1000.0), (1, 2, 500.0), (2, 3, 300.0), (3, 5, 150.0),
             (5, 10, 60.0), (10, 20, 20.0), (20, n_paying, 5.0)]
    for start, end, amt in tiers:
        arr[start:end] = amt
    return arr


@pytest.mark.parametrize("seed", range(12))
def test_matches_grade_portfolio_on_continuous_scores(seed):
    rng = np.random.default_rng(seed)
    payout = _payout_arr()
    field = np.sort(rng.normal(120.0, 25.0, size=300))
    own = rng.normal(130.0, 25.0, size=rng.integers(1, 15))

    ref_gross, ref_rank = grade_portfolio(own, field, payout)
    got_gross, got_rank = joint_gross_world(own, field, _payout_cumsum(payout))

    np.testing.assert_array_equal(ref_rank, got_rank)
    np.testing.assert_allclose(ref_gross, got_gross, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(12))
def test_matches_grade_portfolio_with_heavy_ties(seed):
    """Ties are where self-tie-splitting and dupe splitting actually bite."""
    rng = np.random.default_rng(1000 + seed)
    payout = _payout_arr()
    # Coarse integer scores force many exact ties, with the field and with us.
    field = np.sort(rng.integers(100, 115, size=200).astype(np.float64))
    own = rng.integers(100, 115, size=rng.integers(2, 12)).astype(np.float64)

    ref_gross, ref_rank = grade_portfolio(own, field, payout)
    got_gross, got_rank = joint_gross_world(own, field, _payout_cumsum(payout))

    np.testing.assert_array_equal(ref_rank, got_rank)
    np.testing.assert_allclose(ref_gross, got_gross, rtol=0, atol=0)


def test_exact_duplicate_entries_split_one_prize():
    """Two identical lineups must SPLIT a prize, not each claim it."""
    payout = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float64)
    field = np.sort(np.array([50.0, 60.0, 70.0], dtype=np.float64))
    own = np.array([90.0, 90.0], dtype=np.float64)

    ref_gross, _ = grade_portfolio(own, field, payout)
    got_gross, _ = joint_gross_world(own, field, _payout_cumsum(payout))

    np.testing.assert_allclose(ref_gross, got_gross)
    assert got_gross.sum() == pytest.approx(100.0), "one prize, split two ways"
    assert got_gross[0] == pytest.approx(50.0)


def test_self_displacement_is_priced():
    """A second entry below our best must be ranked BELOW it, not beside it."""
    payout = np.array([100.0, 10.0, 0.0, 0.0], dtype=np.float64)
    field = np.sort(np.array([50.0, 60.0, 70.0], dtype=np.float64))
    own = np.array([95.0, 90.0], dtype=np.float64)

    gross, rank = joint_gross_world(own, field, _payout_cumsum(payout))
    np.testing.assert_array_equal(rank, [1, 2])
    np.testing.assert_allclose(gross, [100.0, 10.0])

    # Independent per-lineup grading -- what production does -- double-counts.
    indep, _ = grade_pool(own, field, payout)
    assert indep.sum() > gross.sum()


@pytest.mark.parametrize("seed", range(8))
def test_k1_reduces_to_grade_pick(seed):
    rng = np.random.default_rng(2000 + seed)
    payout = _payout_arr()
    field = np.sort(rng.normal(120.0, 25.0, size=250))
    v = float(rng.normal(130.0, 25.0))

    ref_gross, ref_rank = grade_pick(v, field, payout)
    got_gross, got_rank = joint_gross_world(np.array([v]), field, _payout_cumsum(payout))

    assert int(got_rank[0]) == ref_rank
    assert float(got_gross[0]) == pytest.approx(ref_gross)


def test_nan_entries_do_not_displace_others():
    payout = _payout_arr()
    field = np.sort(np.array([50.0, 60.0, 70.0], dtype=np.float64))
    own = np.array([200.0, np.nan, 65.0], dtype=np.float64)

    ref_gross, ref_rank = grade_portfolio(own, field, payout)
    got_gross, got_rank = joint_gross_world(own, field, _payout_cumsum(payout))

    np.testing.assert_array_equal(ref_rank, got_rank)
    np.testing.assert_allclose(ref_gross, got_gross, equal_nan=True)
    assert got_rank[1] == -1


@pytest.mark.parametrize("seed", range(6))
def test_joint_gross_worlds_matches_per_world_reference(seed):
    """The (k, S) driver must agree with grade_portfolio world by world."""
    rng = np.random.default_rng(3000 + seed)
    payout = _payout_arr()
    S, F, k = 37, 120, 6
    field_sorted = np.sort(rng.normal(120.0, 25.0, size=(S, F)).astype(np.float32), axis=1)
    own = rng.normal(130.0, 25.0, size=(k, S))

    got = joint_gross_worlds(own, field_sorted, payout, chunk=8)
    for s in range(S):
        ref, _ = grade_portfolio(own[:, s], field_sorted[s], payout)
        np.testing.assert_allclose(got[:, s], ref, rtol=0, atol=0)

    assert portfolio_reward(own, field_sorted, payout, chunk=8) == pytest.approx(
        float(np.nansum(got, axis=0).mean())
    )


def test_portfolio_reward_is_submodular_on_a_steep_table():
    """Adding a lineup to a LARGER portfolio must add no more than to a smaller
    one -- monotone submodularity, the property the greedy's guarantee rests on
    (paper Appendix B.2)."""
    rng = np.random.default_rng(7)
    payout = _payout_arr()
    S, F = 400, 150
    field_sorted = np.sort(rng.normal(120.0, 25.0, size=(S, F)).astype(np.float32), axis=1)
    pool = rng.normal(132.0, 22.0, size=(9, S))

    for extra in range(3, 9):
        small = pool[:2]
        large = pool[:5]
        d_small = (portfolio_reward(np.vstack([small, pool[extra:extra + 1]]), field_sorted, payout)
                   - portfolio_reward(small, field_sorted, payout))
        d_large = (portfolio_reward(np.vstack([large, pool[extra:extra + 1]]), field_sorted, payout)
                   - portfolio_reward(large, field_sorted, payout))
        assert d_large <= d_small + 1e-9, f"submodularity violated at extra={extra}"


@pytest.mark.parametrize("seed", range(6))
def test_precompute_field_ranks_matches_direct_searchsorted(seed):
    rng = np.random.default_rng(4000 + seed)
    S, F, M = 23, 90, 40
    field_sorted = np.sort(rng.integers(100, 130, size=(S, F)).astype(np.float32), axis=1)
    cand = rng.integers(100, 130, size=(M, S)).astype(np.float32)

    n_above, f_ties = precompute_field_ranks(cand, field_sorted, chunk=7)
    for s in range(S):
        right = np.searchsorted(field_sorted[s], cand[:, s], side="right")
        left = np.searchsorted(field_sorted[s], cand[:, s], side="left")
        np.testing.assert_array_equal(n_above[:, s], F - right)
        np.testing.assert_array_equal(f_ties[:, s], right - left)


def test_precompute_rank_cap_is_lossless_below_the_cap():
    """Clamping at the payout-table length must not touch any paying rank."""
    rng = np.random.default_rng(11)
    S, F, M = 15, 200, 30
    field_sorted = np.sort(rng.normal(120.0, 25.0, size=(S, F)).astype(np.float32), axis=1)
    cand = rng.normal(120.0, 25.0, size=(M, S)).astype(np.float32)

    full, _ = precompute_field_ranks(cand, field_sorted, chunk=5)
    capped, _ = precompute_field_ranks(cand, field_sorted, chunk=5, rank_cap=40)
    paying = full < 40
    np.testing.assert_array_equal(full[paying], capped[paying])
    assert (capped[~paying] == 40).all()


def test_tier_form_equals_rank_lookup_on_every_real_payout_table():
    """The bridge to the smoothed estimator.

    Formulation (2) is a weighted sum of exceedance indicators; our fast path
    is a rank lookup. Smoothing is only legitimate if those are the same
    function, so this checks it at EVERY rank of EVERY archived DK table --
    not on a synthetic example.
    """
    import glob
    import os

    from src.optimization.mrp.field_covariance import tier_boundary_ranks
    from src.optimization.mrp.marginal_reward import tier_form_payout
    from src.optimization.payout import load_payout_structure, payout_table_to_array

    names = sorted(os.path.basename(p)[:-5] for p in glob.glob("data/payout_structures/*.json"))
    assert len(names) > 20, "expected the archived payout tables to be present"

    for name in names:
        arr = payout_table_to_array(load_payout_structure(name))
        ranks = tier_boundary_ranks(arr)
        amounts = arr[ranks - 1]
        all_ranks = np.arange(1, len(arr) + 1)
        np.testing.assert_allclose(
            tier_form_payout(all_ranks, ranks, amounts), arr,
            rtol=0, atol=1e-9, err_msg=f"tier form != rank lookup for {name}",
        )
