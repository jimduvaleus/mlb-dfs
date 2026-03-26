"""
Regression test: engine applies floor-at-zero only for batters, not pitchers.

Uses a mock copula that always returns q=0.5, so GaussianMarginal.ppf returns
the player's mean exactly. Players are configured with negative means to make
the expected outcome unambiguous.
"""
import numpy as np
import pandas as pd
import pytest

from src.simulation.engine import SimulationEngine


class _FixedQuantileCopula:
    """Stub copula that always returns q=0.5 for every slot."""

    def sample(self, n_sims: int) -> np.ndarray:
        return np.full((n_sims, 10), 0.5)


@pytest.fixture
def engine():
    players = pd.DataFrame([
        # Batter (slot 1) — negative mean; score should be clipped to 0
        {'player_id': 1, 'team': 'NYY', 'opponent': 'BOS', 'slot': 1,
         'mean': -5.0, 'std_dev': 2.0},
        # Pitcher (slot 10) — negative mean; score should pass through unchanged
        {'player_id': 10, 'team': 'NYY', 'opponent': 'BOS', 'slot': 10,
         'mean': -10.0, 'std_dev': 3.0},
    ])
    return SimulationEngine(_FixedQuantileCopula(), players)


def test_pitcher_negative_scores_pass_through(engine):
    results = engine.simulate(n_sims=200)
    pid_to_col = {pid: i for i, pid in enumerate(results.player_ids)}

    pitcher_scores = results.results_matrix[:, pid_to_col[10]]
    assert np.all(pitcher_scores < 0), (
        "Pitcher scores should be negative (mean=-10, q=0.5); clipping must not apply."
    )


def test_batter_scores_clipped_to_zero(engine):
    results = engine.simulate(n_sims=200)
    pid_to_col = {pid: i for i, pid in enumerate(results.player_ids)}

    batter_scores = results.results_matrix[:, pid_to_col[1]]
    assert np.all(batter_scores >= 0), (
        "Batter scores should be clipped to 0 even when mean is negative."
    )
