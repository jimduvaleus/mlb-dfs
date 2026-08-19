"""Tests for stochastic (Dirichlet) ownership.

The invariants that matter are the ones that let this drop into the field
generator without any consumer noticing: per-group totals preserved exactly
(so the caller's scale is untouched), mean-preserving across draws (so it adds
uncertainty without moving the central estimate), and an exact
today's-behaviour arm at concentration = inf.
"""
import numpy as np
import pytest

from src.optimization.mrp.ownership_dirichlet import (
    dirichlet_ownership,
    field_ownership_draws,
    fit_concentration,
)

POSITIONS = np.array(["P", "P", "P", "C", "OF", "OF", "OF", "OF"])
OWN = np.array([0.40, 0.25, 0.05, 0.30, 0.50, 0.30, 0.15, 0.05])


def test_group_totals_are_preserved_exactly():
    rng = np.random.default_rng(0)
    for _ in range(20):
        got = dirichlet_ownership(OWN, POSITIONS, 50.0, rng)
        for g in np.unique(POSITIONS):
            m = POSITIONS == g
            assert got[m].sum() == pytest.approx(OWN[m].sum(), rel=1e-12)


def test_is_mean_preserving():
    rng = np.random.default_rng(1)
    draws = np.array([dirichlet_ownership(OWN, POSITIONS, 40.0, rng) for _ in range(6000)])
    np.testing.assert_allclose(draws.mean(axis=0), OWN, atol=0.012)


def test_infinite_concentration_reproduces_todays_behaviour():
    rng = np.random.default_rng(2)
    got = dirichlet_ownership(OWN, POSITIONS, np.inf, rng)
    np.testing.assert_array_equal(got, OWN)


def test_dispersion_decreases_as_concentration_rises():
    rng = np.random.default_rng(3)

    def spread(c):
        d = np.array([dirichlet_ownership(OWN, POSITIONS, c, rng) for _ in range(1500)])
        return float(d.std(axis=0).mean())

    tight, loose = spread(500.0), spread(10.0)
    assert tight < loose


def test_variance_matches_the_dirichlet_formula():
    """Var(p_i) = m_i (1 - m_i) / (alpha_0 + 1) on the within-group simplex --
    the identity `fit_concentration` inverts, so it has to hold here."""
    rng = np.random.default_rng(4)
    alpha0 = 60.0
    draws = np.array([dirichlet_ownership(OWN, POSITIONS, alpha0, rng) for _ in range(20000)])

    m = POSITIONS == "OF"
    total = OWN[m].sum()
    simplex = draws[:, m] / total
    mean = OWN[m] / total
    expected = mean * (1.0 - mean) / (alpha0 + 1.0)
    np.testing.assert_allclose(simplex.var(axis=0), expected, rtol=0.12)


def test_zero_ownership_players_stay_negligible_but_not_impossible():
    own = np.array([0.5, 0.5, 0.0])
    pos = np.array(["OF", "OF", "OF"])
    rng = np.random.default_rng(5)
    draws = np.array([dirichlet_ownership(own, pos, 30.0, rng) for _ in range(400)])
    assert draws[:, 2].mean() < 1e-3
    assert (draws[:, 2] >= 0).all()


def test_fit_concentration_recovers_a_known_alpha0():
    """Generate ownership FROM a known Dirichlet, then check the fit finds it."""
    rng = np.random.default_rng(6)
    n = 60
    pos = np.array(["bat"] * n)
    truth = rng.dirichlet(np.ones(n) * 2.0)
    alpha0 = 250.0

    fits = []
    for _ in range(60):
        realized = rng.dirichlet(truth * alpha0)
        fits.append(fit_concentration(truth, realized, pos)["bat"]["alpha_0"])
    est = float(np.median(fits))
    assert 0.5 * alpha0 < est < 2.0 * alpha0, f"recovered {est:.0f}, truth {alpha0}"


def test_fit_concentration_is_per_group_and_survives_degenerate_groups():
    pred = np.array([0.5, 0.5, 1.0, 0.4, 0.6])
    actual = np.array([0.6, 0.4, 1.0, 0.3, 0.7])
    pos = np.array(["bat", "bat", "P", "OF", "OF"])
    got = fit_concentration(pred, actual, pos)
    assert set(got) <= {"bat", "P", "OF"}
    assert "bat" in got and "OF" in got
    assert "P" not in got, "a single-player group carries no simplex information"
    for v in got.values():
        assert v["alpha_0"] >= 1.0


def test_field_ownership_draws_are_distinct():
    rng = np.random.default_rng(7)
    draws = field_ownership_draws(OWN, POSITIONS, 4, 40.0, rng)
    assert len(draws) == 4
    assert not np.allclose(draws[0], draws[1]), "K field draws must differ in ownership now"


def test_rejects_mismatched_shapes_and_bad_concentration():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError):
        dirichlet_ownership(OWN, POSITIONS[:-1], 10.0, rng)
    with pytest.raises(ValueError):
        dirichlet_ownership(OWN, POSITIONS, 0.0, rng)
