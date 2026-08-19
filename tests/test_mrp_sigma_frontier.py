"""Tests for sigma_dG-targeted generation and, more importantly, its GATE.

The repo already has a measured negative on generated supplements, so the
burden here is on the gate: `pool_sigma_coverage` has to be able to say
"skip generation" and mean it.
"""
import numpy as np
import pytest

from src.optimization.mrp.sigma_frontier import (
    pool_sigma_coverage,
    sigma_frontier,
    sigma_objective,
)


def test_lambda_zero_is_the_plain_projection_objective():
    mu = np.array([10.0, 12.0, 8.0, 15.0])
    sig = np.array([1.0, -2.0, 3.0, 0.5])
    np.testing.assert_allclose(sigma_objective(mu, sig, 0.0), mu)


def test_lambda_penalises_high_covariance_players_monotonically():
    mu = np.full(4, 10.0)
    sig = np.array([-2.0, -1.0, 1.0, 2.0])          # ascending cutoff covariance
    prev = None
    for lam in (0.1, 0.5, 1.0):
        obj = sigma_objective(mu, sig, lam)
        assert obj[0] > obj[1] > obj[2] > obj[3], "must prefer low covariance"
        spread = obj.max() - obj.min()
        if prev is not None:
            assert spread > prev, "a larger lambda must separate them further"
        prev = spread


def test_objective_is_scale_invariant_in_sigma_units():
    """sigma is a covariance (FPTS^2) while mu is FPTS, so a raw grid would not
    be comparable across slates. Rescaling by sigma's own spread means the same
    lambda does the same job whatever units sigma arrives in."""
    rng = np.random.default_rng(0)
    mu = rng.normal(10.0, 3.0, size=30)
    sig = rng.normal(0.0, 2.0, size=30)
    a = sigma_objective(mu, sig, 0.4)
    b = sigma_objective(mu, sig * 1000.0, 0.4)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_degenerate_sigma_falls_back_to_mean():
    mu = np.array([10.0, 12.0, 8.0])
    np.testing.assert_allclose(sigma_objective(mu, np.zeros(3), 0.7), mu)


def test_gate_says_skip_when_the_pool_already_spans_the_region():
    rng = np.random.default_rng(1)
    pool = rng.normal(0.0, 1.0, size=5000)
    frontier = rng.normal(0.0, 1.0, size=50)        # same region, nothing new
    rep = pool_sigma_coverage(pool, frontier)
    assert rep["decision"] == "pool-already-spans"
    assert rep["n_pool_below_frontier_min"] > 0


def test_gate_says_generate_when_the_frontier_reaches_past_the_pool():
    rng = np.random.default_rng(2)
    pool = rng.normal(0.0, 1.0, size=5000)
    frontier = rng.normal(-8.0, 0.5, size=50)       # genuinely outside the pool
    rep = pool_sigma_coverage(pool, frontier)
    assert rep["decision"] == "generate"
    assert rep["n_pool_below_frontier_min"] == 0
    assert rep["frontier_min"] < rep["pool_min"]


def test_gate_handles_empty_inputs():
    rep = pool_sigma_coverage(np.array([]), np.array([1.0]))
    assert rep["decision"] == "insufficient-data"


def _toy_pool_df():
    """Minimal DK-legal player pool: 2 P + 8 hitters across enough teams."""
    import pandas as pd

    rows = []
    pid = 0
    for team, opp in (("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC")):
        for pos in ("P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"):
            pid += 1
            rows.append({
                "player_id": pid, "name": f"p{pid}", "position": pos,
                "eligible_positions": [pos], "team": team, "opponent": opp,
                "game": f"{team}@{opp}", "salary": 3000 + (pid % 7) * 500,
                "mean": 8.0 + (pid % 11) * 0.7, "std_dev": 5.0,
            })
    return pd.DataFrame(rows)


def test_frontier_produces_distinct_lineups_across_lambda():
    df = _toy_pool_df()
    rng = np.random.default_rng(3)
    sigma = rng.normal(0.0, 1.0, size=len(df))

    got = sigma_frontier(df, sigma, lambdas=(0.0, 0.5, 1.0), n_per_lambda=2,
                         min_uniques=1, min_stack=3)
    assert len(got) >= 2, "the sweep should yield more than one lineup"
    keys = {frozenset(l.player_ids) for l in got}
    assert len(keys) == len(got), "sweep-wide dedup should leave no repeats"
    for l in got:
        assert len(set(l.player_ids)) == 10
