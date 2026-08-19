"""Tests for sigma_dG -- covariance between a player and the field's cutoff.

The load-bearing test here is `test_chalk_player_covaries_with_the_cutoff`:
it is what establishes that this currency measures the thing it claims to
measure, rather than merely computing a well-formed number.
"""
import numpy as np
import pytest

from src.optimization.mrp.field_covariance import (
    assumption_52_report,
    field_order_statistics,
    lineup_sigma_scores,
    payout_weighted_sigma,
    player_field_covariance,
    tier_boundary_ranks,
)
from src.optimization.payout import load_payout_structure, payout_table_to_array


def test_tier_boundary_ranks_are_the_paying_tier_ENDS():
    """r_d is a tier's last rank -- "rank in (r_{d-1}, r_d] wins R_d"."""
    for name in ("dk_rally_cap_29411", "dk_mini_max_14268", "dk_four_seamer_2972"):
        st = load_payout_structure(name)
        ranks = tier_boundary_ranks(payout_table_to_array(st))
        paying = [t["end"] for t in st["payouts"] if t["amount"] > 0]
        np.testing.assert_array_equal(ranks, np.array(sorted(set(paying))))


def test_field_order_statistics_pick_the_right_column():
    field = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    got = field_order_statistics(field, np.array([1, 2, 5]))
    np.testing.assert_allclose(got, [[5.0, 4.0, 1.0], [50.0, 40.0, 10.0]])


@pytest.mark.parametrize("chunk", [3, 17, 10_000])
def test_covariance_matches_numpy_and_is_chunk_invariant(chunk):
    rng = np.random.default_rng(0)
    S, P, T = 200, 12, 4
    X = rng.normal(10.0, 3.0, size=(S, P))
    W = rng.normal(120.0, 8.0, size=(S, T))

    got = player_field_covariance(X, W, chunk=chunk)
    for p in range(P):
        for t in range(T):
            # population covariance (denominator S), matching the estimator
            ref = np.mean((X[:, p] - X[:, p].mean()) * (W[:, t] - W[:, t].mean()))
            assert got[p, t] == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_chalk_player_covaries_with_the_cutoff_and_a_fadeable_one_does_not():
    """The semantic contract: sigma_dG must separate a player the field is ON
    from a player the field ignores.

    Field lineups all roster player 0 and never roster player 1. Both have the
    same marginal distribution, so anything that separates them is picking up
    the FIELD, not the player -- which is the entire point of the term."""
    rng = np.random.default_rng(1)
    S, P, F = 3000, 6, 400
    sim = rng.normal(10.0, 4.0, size=(S, P))

    # Every field lineup = player 0 + two of players 2..5. Player 1 is unowned.
    others = rng.integers(2, P, size=(F, 2))
    field = sim[:, 0][:, None] + sim[:, others[:, 0]] + sim[:, others[:, 1]]
    field_sorted = np.sort(field, axis=1)

    ranks = np.array([1, 5, 20, 80])
    thr = field_order_statistics(field_sorted, ranks)
    sigma = player_field_covariance(sim, thr)

    chalk, faded = sigma[0], sigma[1]
    assert (chalk > 0).all(), "the universally-rostered player must move the cutoff"
    assert np.abs(faded).max() < 0.15 * chalk.min(), (
        f"unowned player should be near-zero: {faded} vs chalk {chalk}"
    )


def test_payout_weighted_sigma_uses_step_weights_and_is_a_convex_combination():
    payout = payout_table_to_array(load_payout_structure("dk_four_seamer_2972"))
    ranks = tier_boundary_ranks(payout)
    sigma = np.tile(np.arange(5.0)[:, None], (1, len(ranks)))   # constant across tiers

    got = payout_weighted_sigma(sigma, payout, ranks)
    np.testing.assert_allclose(got, np.arange(5.0), rtol=1e-12), "constant in => constant out"


def test_payout_weights_concentrate_on_the_top_tiers():
    """A sanity check on the objective's own weighting: (R_d - R_{d+1}) must put
    most of its mass at the very top, which is why the tail rungs are the ones
    that need smoothing."""
    payout = payout_table_to_array(load_payout_structure("dk_rally_cap_29411"))
    ranks = tier_boundary_ranks(payout)
    amounts = payout[ranks - 1]
    w = amounts - np.concatenate((amounts[1:], [0.0]))
    assert w[0] / w.sum() > 0.5, "rank-1 step should dominate a top-heavy table"


def test_lineup_sigma_scores_sum_over_the_roster():
    sigma = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    cols = np.array([[0, 1, 2], [2, 3, 4]])
    np.testing.assert_allclose(lineup_sigma_scores(sigma, cols), [7.0, 28.0])


def test_assumption_52_report_detects_constant_and_varying_cases():
    P, T = 50, 8
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, size=P)

    constant = np.tile(base[:, None], (1, T))
    rep = assumption_52_report(constant)
    assert rep["median_rel_spread"] < 1e-9
    assert rep["min_col_corr_vs_mean"] > 0.999

    varying = rng.normal(0.0, 1.0, size=(P, T))
    rep2 = assumption_52_report(varying)
    assert rep2["median_rel_spread"] > rep["median_rel_spread"]
    assert rep2["min_col_corr_vs_mean"] < 0.9
