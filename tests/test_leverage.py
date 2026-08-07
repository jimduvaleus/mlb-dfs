"""Tests for src/optimization/leverage.py (optimal ownership / leverage,
plan doc need-a-plan-to-squishy-sutherland.md)."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.leverage import (
    OWN_FLOOR_PCT,
    compute_candidate_p_opt,
    compute_leverage,
    compute_optimal_ownership,
)
from src.optimization.lineup import Lineup
from src.simulation.results import SimulationResults


def _pool(rng, n_players=12, n_lineups=8, roster_size=4, n_sims=300, seed=0):
    ids = list(range(1, n_players + 1))
    lineups = [
        Lineup(player_ids=sorted(rng.choice(ids, size=roster_size, replace=False).tolist()))
        for _ in range(n_lineups)
    ]
    sim_results = SimulationResults(
        player_ids=ids,
        results_matrix=rng.normal(10.0, 3.0, size=(n_sims, n_players)).astype(np.float32),
    )
    return lineups, sim_results


class TestComputeCandidatePOpt:
    def test_dominant_lineup_has_higher_p_opt(self):
        rng = np.random.default_rng(0)
        lineups, sim_results = _pool(rng)
        # Force one lineup's players to always score far higher than everyone
        # else's, so it should dominate every simulated world.
        dominant_pids = lineups[0].player_ids
        col = {pid: i for i, pid in enumerate(sim_results.player_ids)}
        boosted = sim_results.results_matrix.copy()
        for pid in dominant_pids:
            boosted[:, col[pid]] += 1000.0
        sim_results = SimulationResults(sim_results.player_ids, boosted)

        out = compute_candidate_p_opt(lineups, sim_results, {"c": 5.0})
        assert out["c"].shape == (len(lineups),)
        assert out["c"][0] == out["c"].max()
        assert np.all(out["c"][0] > out["c"][1:])

    def test_resolution_cap_defaults_to_pool_size(self):
        rng = np.random.default_rng(1)
        lineups, sim_results = _pool(rng, n_lineups=6)
        M = len(lineups)
        # A field_size far beyond M should be capped to M by default, so it
        # must agree exactly with requesting field_size == M directly.
        huge = compute_candidate_p_opt(lineups, sim_results, {"huge": M * 1000.0}, sharpness=0.1)
        at_cap = compute_candidate_p_opt(lineups, sim_results, {"at_cap": float(M)}, sharpness=0.1)
        np.testing.assert_allclose(huge["huge"], at_cap["at_cap"])

    def test_explicit_resolution_cap_overrides_pool_size(self):
        rng = np.random.default_rng(2)
        lineups, sim_results = _pool(rng, n_lineups=6)
        capped = compute_candidate_p_opt(
            lineups, sim_results, {"c": 1000.0}, sharpness=0.1, resolution_cap=3.0,
        )
        at_three = compute_candidate_p_opt(lineups, sim_results, {"c": 3.0}, sharpness=0.1)
        np.testing.assert_allclose(capped["c"], at_three["c"])

    def test_one_call_serves_multiple_field_size_keys(self):
        rng = np.random.default_rng(3)
        lineups, sim_results = _pool(rng, n_lineups=6)
        combined = compute_candidate_p_opt(lineups, sim_results, {"small": 2.0, "big": 20.0})
        sep_small = compute_candidate_p_opt(lineups, sim_results, {"small": 2.0})
        sep_big = compute_candidate_p_opt(lineups, sim_results, {"big": 20.0})
        np.testing.assert_allclose(combined["small"], sep_small["small"])
        np.testing.assert_allclose(combined["big"], sep_big["big"])


class TestComputeOptimalOwnership:
    def _players_df(self, player_ids):
        return pd.DataFrame({"player_id": player_ids, "ownership": [1.0] * len(player_ids)})

    def test_player_rostered_by_whole_pool_gets_exactly_100pct(self):
        rng = np.random.default_rng(4)
        ids = list(range(1, 9))
        # Every lineup includes player 1.
        lineups = [Lineup(player_ids=[1] + sorted(
            rng.choice(ids[1:], size=3, replace=False).tolist())) for _ in range(10)]
        sim_results = SimulationResults(
            ids, rng.normal(10.0, 3.0, size=(200, len(ids))).astype(np.float32),
        )
        players_df = self._players_df(ids)
        out = compute_optimal_ownership(lineups, sim_results, players_df, {"c": 4.0})
        p1_idx = players_df.index[players_df["player_id"] == 1][0]
        assert out["c"][p1_idx] == pytest.approx(100.0)

    def test_ownership_sums_to_100_times_roster_size(self):
        rng = np.random.default_rng(5)
        lineups, sim_results = _pool(rng, n_players=14, n_lineups=10, roster_size=5)
        players_df = self._players_df(sim_results.player_ids)
        out = compute_optimal_ownership(lineups, sim_results, players_df, {"c": 3.0})
        # Every lineup contributes its weight to exactly `roster_size` players,
        # so the player-level shares must sum to 100 * roster_size regardless
        # of how p_opt weights individual lineups.
        assert out["c"].sum() == pytest.approx(500.0, rel=1e-5)

    def test_output_aligned_to_players_df_row_order(self):
        rng = np.random.default_rng(6)
        lineups, sim_results = _pool(rng, n_players=10, n_lineups=8, roster_size=4)
        shuffled_ids = list(reversed(sim_results.player_ids))
        players_df = self._players_df(shuffled_ids)
        out = compute_optimal_ownership(lineups, sim_results, players_df, {"c": 2.0})
        assert out["c"].shape == (len(shuffled_ids),)
        # Rebuild via the natural id order and confirm a simple relabeling
        # recovers the same per-player values (order-independence check).
        natural_df = self._players_df(sim_results.player_ids)
        out_natural = compute_optimal_ownership(lineups, sim_results, natural_df, {"c": 2.0})
        by_id_shuffled = dict(zip(shuffled_ids, out["c"]))
        by_id_natural = dict(zip(sim_results.player_ids, out_natural["c"]))
        for pid in sim_results.player_ids:
            assert by_id_shuffled[pid] == pytest.approx(by_id_natural[pid])


class TestComputeLeverage:
    def test_diff_and_ratio_basic(self):
        players_df = pd.DataFrame({"player_id": [1, 2, 3], "ownership": [10.0, 20.0, 0.0]})
        optimal = np.array([15.0, 15.0, 5.0])
        diff, ratio = compute_leverage(optimal, players_df)
        np.testing.assert_allclose(diff, [5.0, -5.0, 5.0])
        np.testing.assert_allclose(
            ratio, [5.0 / 10.0, -5.0 / 20.0, 5.0 / OWN_FLOOR_PCT],
        )

    def test_zero_projected_ownership_uses_floor_not_inf(self):
        players_df = pd.DataFrame({"player_id": [1], "ownership": [0.0]})
        diff, ratio = compute_leverage(np.array([2.0]), players_df)
        assert np.isfinite(ratio).all()
        assert ratio[0] == pytest.approx(2.0 / OWN_FLOOR_PCT)
