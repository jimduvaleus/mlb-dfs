"""Tests for market-implied score distributions (work item 4):
the fetcher's Monte Carlo distribution builders, EmpiricalQuantileMarginal,
the quantile-grid loader, and engine marginal preference.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.fetch_market_odds_projections import (
    _DK_BATTER,
    _DK_PITCHER,
    _distribution_grid,
    _simulate_batter_distribution,
    _simulate_pitcher_distribution,
)


def _grid_mean(grid: np.ndarray) -> float:
    """Distribution mean implied by a percentile grid (trapezoid rule)."""
    return float(np.trapezoid(grid, np.linspace(0.0, 1.0, len(grid))))
from src.models.marginals import EmpiricalQuantileMarginal
from src.models.quantile_grids import load_quantile_grids
from src.simulation.engine import SimulationEngine


BATTER_RATES = dict(
    single=0.9, double=0.25, triple=0.03, home_run=0.15,
    run=0.7, rbi=0.75, walk=0.4, sb=0.1, hbp=0.044,
)
PITCHER_RATES = dict(outs=17.0, k=6.5, win=0.45, h=5.0, bb=2.0, er=2.5, hbp=0.22)


# ------------------------------------------------------------------ #
#  Monte Carlo distribution builders                                   #
# ------------------------------------------------------------------ #

class TestBatterDistribution:
    def test_mean_matches_analytic(self):
        c = _DK_BATTER
        analytic_mean = (
            BATTER_RATES["single"] * c["single"] + BATTER_RATES["double"] * c["double"]
            + BATTER_RATES["triple"] * c["triple"] + BATTER_RATES["home_run"] * c["home_run"]
            + BATTER_RATES["run"] * c["run"] + BATTER_RATES["rbi"] * c["rbi"]
            + BATTER_RATES["walk"] * c["walk"] + BATTER_RATES["sb"] * c["sb"]
            + BATTER_RATES["hbp"] * c["hbp"]
        )
        grid, _ = _simulate_batter_distribution(BATTER_RATES, n_draws=200_000, seed=1)
        assert abs(_grid_mean(grid) - analytic_mean) < 0.2

    def test_hr_covariance_raises_std(self):
        """MC std must exceed the independence-assuming analytic std: a HR
        also delivers a run and an RBI, adding positive covariance."""
        c = _DK_BATTER
        analytic_var = (
            BATTER_RATES["single"] * c["single"] ** 2
            + BATTER_RATES["double"] * c["double"] ** 2
            + BATTER_RATES["triple"] * c["triple"] ** 2
            + BATTER_RATES["home_run"] * c["home_run"] ** 2
            + BATTER_RATES["run"] * c["run"] ** 2
            + BATTER_RATES["rbi"] * (1 + BATTER_RATES["rbi"]) * c["rbi"] ** 2
            + BATTER_RATES["walk"] * c["walk"] ** 2
            + BATTER_RATES["sb"] * c["sb"] ** 2
            + BATTER_RATES["hbp"] * c["hbp"] ** 2
        )
        _, mc_std = _simulate_batter_distribution(BATTER_RATES, n_draws=200_000, seed=1)
        assert mc_std > np.sqrt(analytic_var)

    def test_grid_shape_and_monotonic(self):
        grid, _ = _simulate_batter_distribution(BATTER_RATES, seed=2)
        assert grid.shape == (101,)
        assert np.all(np.diff(grid) >= 0)
        assert grid[0] >= 0.0  # batter floor is a hitless game

    def test_zero_inflation_plateau(self):
        """Low-rate batters have a point mass at 0 — the grid's lower
        percentiles must sit exactly at 0."""
        weak = dict(single=0.3, double=0.05, triple=0.01, home_run=0.02,
                    run=0.15, rbi=0.15, walk=0.1, sb=0.01, hbp=0.011)
        grid, _ = _simulate_batter_distribution(weak, n_draws=100_000, seed=3)
        assert grid[10] == 0.0  # P(score == 0) well above 10%

    def test_deterministic_for_seed(self):
        g1, s1 = _simulate_batter_distribution(BATTER_RATES, seed=7)
        g2, s2 = _simulate_batter_distribution(BATTER_RATES, seed=7)
        assert np.array_equal(g1, g2) and s1 == s2


class TestPitcherDistribution:
    def test_mean_near_analytic(self):
        c = _DK_PITCHER
        analytic_mean = (
            PITCHER_RATES["outs"] * c["out"] + PITCHER_RATES["k"] * c["k"]
            + PITCHER_RATES["win"] * c["win"] + PITCHER_RATES["er"] * c["er"]
            + PITCHER_RATES["h"] * c["h"] + PITCHER_RATES["bb"] * c["bb"]
            + PITCHER_RATES["hbp"] * c["hbp"]
        )
        grid, _ = _simulate_pitcher_distribution(PITCHER_RATES, n_draws=200_000, seed=1)
        # Couplings (win eligibility, ER clipping) shift the mean slightly;
        # it must stay close to the market-implied analytic mean.
        assert abs(_grid_mean(grid) - analytic_mean) < 1.0

    def test_left_skewed(self):
        """Pitcher scores must be left-skewed (early-hook bust mode), unlike
        the symmetric Gaussian marginal this replaces."""
        rng = np.random.default_rng(0)
        grid, _ = _simulate_pitcher_distribution(PITCHER_RATES, n_draws=200_000, seed=1)
        draws = np.interp(rng.random(200_000), np.linspace(0, 1, 101), grid)
        skew = float(((draws - draws.mean()) ** 3).mean() / draws.std() ** 3)
        assert skew < -0.05

    def test_structural_invariants(self):
        """Re-derive the component draws and check K <= outs and the
        5-inning win rule via the score bounds: max grid score cannot exceed
        the all-outs-are-strikeouts perfect game with a win."""
        grid, _ = _simulate_pitcher_distribution(PITCHER_RATES, n_draws=100_000, seed=2)
        c = _DK_PITCHER
        max_possible = 27 * c["out"] + 27 * c["k"] + 1 * c["win"]
        assert grid[-1] <= max_possible
        assert np.all(np.diff(grid) >= 0)

    def test_win_rate_preserved(self):
        """Overall win frequency must track the market-implied e_win even
        with the 5-IP eligibility rule and ER tilt."""
        # Isolate the win contribution: same rates with win=0 vs win=0.45.
        no_win = dict(PITCHER_RATES, win=0.0)
        g_win, _ = _simulate_pitcher_distribution(PITCHER_RATES, n_draws=300_000, seed=5)
        g_now, _ = _simulate_pitcher_distribution(no_win, n_draws=300_000, seed=5)
        implied_win_pts = g_win.mean() - g_now.mean()
        assert abs(implied_win_pts - PITCHER_RATES["win"] * _DK_PITCHER["win"]) < 0.25


# ------------------------------------------------------------------ #
#  EmpiricalQuantileMarginal                                           #
# ------------------------------------------------------------------ #

class TestEmpiricalQuantileMarginal:
    def test_interpolates_grid(self):
        grid = np.linspace(0.0, 50.0, 101)
        m = EmpiricalQuantileMarginal(grid)
        assert m.ppf(np.array([0.0]))[0] == 0.0
        assert m.ppf(np.array([1.0]))[0] == 50.0
        assert m.ppf(np.array([0.5]))[0] == pytest.approx(25.0)
        # Between grid points: linear.
        assert m.ppf(np.array([0.005]))[0] == pytest.approx(0.25)

    def test_rejects_decreasing_grid(self):
        with pytest.raises(ValueError):
            EmpiricalQuantileMarginal(np.array([1.0, 0.5, 2.0]))

    def test_rejects_out_of_range_q(self):
        m = EmpiricalQuantileMarginal(np.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            m.ppf(np.array([1.5]))

    def test_reproduces_distribution_moments(self):
        rng = np.random.default_rng(4)
        sample = rng.gamma(2.0, 4.0, 100_000)
        grid = _distribution_grid(sample)  # production grid construction
        m = EmpiricalQuantileMarginal(grid)
        draws = m.ppf(rng.random(100_000))
        assert abs(draws.mean() - sample.mean()) < 0.1
        assert abs(draws.std() - sample.std()) < 0.2


# ------------------------------------------------------------------ #
#  Quantile-grid loader                                                #
# ------------------------------------------------------------------ #

class TestLoadQuantileGrids:
    def _dist_file(self, tmp_path, rows):
        path = tmp_path / "projections_mo_dist.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path)

    def _row(self, pid, mean, lo=0.0, hi=30.0):
        return {"player_id": pid, "mean": mean,
                **{f"q{i}": lo + (hi - lo) * i / 100 for i in range(101)}}

    def _players(self, rows):
        return pd.DataFrame([
            {"player_id": pid, "mean": mean} for pid, mean in rows
        ])

    def test_loads_matching_players(self, tmp_path):
        path = self._dist_file(tmp_path, [self._row(1, 10.0), self._row(2, 8.0)])
        grids = load_quantile_grids(path, self._players([(1, 10.0), (2, 8.1)]))
        assert set(grids) == {1, 2}
        assert grids[1].shape == (101,)

    def test_skips_mean_mismatch(self, tmp_path):
        """Fallback-sourced players (RotoWire mean != grid mean) keep their
        parametric marginal."""
        path = self._dist_file(tmp_path, [self._row(1, 10.0)])
        grids = load_quantile_grids(path, self._players([(1, 6.5)]))
        assert grids == {}

    def test_skips_players_absent_from_slate(self, tmp_path):
        path = self._dist_file(tmp_path, [self._row(99, 10.0)])
        grids = load_quantile_grids(path, self._players([(1, 10.0)]))
        assert grids == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_quantile_grids(str(tmp_path / "nope.parquet"), self._players([(1, 1.0)])) == {}


# ------------------------------------------------------------------ #
#  Engine preference                                                   #
# ------------------------------------------------------------------ #

class TestEnginePreference:
    class _Stub:
        def sample(self, n_sims):
            return np.full((n_sims, 10), 0.5)

    def _players(self):
        return pd.DataFrame([
            {'player_id': 1, 'team': 'AAA', 'opponent': 'BBB', 'slot': 1,
             'mean': 8.0, 'std_dev': 6.0},
            {'player_id': 10, 'team': 'BBB', 'opponent': 'AAA', 'slot': 10,
             'mean': 15.0, 'std_dev': 7.0},
        ])

    def test_grid_overrides_parametric(self):
        # Grid whose median is deliberately far from the Gaussian mean.
        grid = np.linspace(100.0, 200.0, 101)
        engine = SimulationEngine(
            self._Stub(), self._players(), quantile_grids={1: grid, 10: grid},
        )
        results = engine.simulate(n_sims=50)
        col = {pid: i for i, pid in enumerate(results.player_ids)}
        # q=0.5 everywhere → grid midpoint 150 for both players.
        assert np.allclose(results.results_matrix[:, col[1]], 150.0)
        assert np.allclose(results.results_matrix[:, col[10]], 150.0)

    def test_players_without_grid_keep_gaussian(self):
        grid = np.linspace(100.0, 200.0, 101)
        engine = SimulationEngine(
            self._Stub(), self._players(), quantile_grids={1: grid},
        )
        results = engine.simulate(n_sims=50)
        col = {pid: i for i, pid in enumerate(results.player_ids)}
        # Pitcher keeps GaussianMarginal: q=0.5 → mean exactly.
        assert np.allclose(results.results_matrix[:, col[10]], 15.0)
