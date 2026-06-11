"""PPD simulation zeroing: seeded determinism and per-game semantics."""
import numpy as np
import pandas as pd

from src.api.pipeline import PipelineRunner
from src.simulation.results import SimulationResults


def _fixture():
    players = pd.DataFrame([
        {"player_id": 1, "game": "A@B", "team": "A"},
        {"player_id": 2, "game": "A@B", "team": "B"},
        {"player_id": 3, "game": "C@D", "team": "C"},
    ])
    rng = np.random.default_rng(0)
    matrix = rng.uniform(1.0, 30.0, size=(1000, 3))  # strictly positive
    return players, SimulationResults([1, 2, 3], matrix)


def test_seeded_ppd_is_reproducible():
    players, sim = _fixture()
    r1, s1 = PipelineRunner._apply_ppd_to_simulation(sim, players, {"A@B": 25.0}, rng_seed=7)
    r2, s2 = PipelineRunner._apply_ppd_to_simulation(sim, players, {"A@B": 25.0}, rng_seed=7)
    assert np.array_equal(r1.results_matrix, r2.results_matrix)
    assert s1 == s2


def test_different_seeds_zero_different_rows():
    players, sim = _fixture()
    r1, _ = PipelineRunner._apply_ppd_to_simulation(sim, players, {"A@B": 25.0}, rng_seed=7)
    r2, _ = PipelineRunner._apply_ppd_to_simulation(sim, players, {"A@B": 25.0}, rng_seed=8)
    assert not np.array_equal(r1.results_matrix, r2.results_matrix)


def test_zeroes_whole_game_at_requested_rate():
    players, sim = _fixture()
    res, stats = PipelineRunner._apply_ppd_to_simulation(sim, players, {"A@B": 25.0}, rng_seed=7)
    m = res.results_matrix
    zeroed = (m[:, 0] == 0.0)
    # 25% of 1000 sims, both A@B columns zeroed together, C@D untouched.
    assert stats["A@B"]["n_sims_zeroed"] == 250
    assert zeroed.sum() == 250
    assert np.array_equal(zeroed, m[:, 1] == 0.0)
    assert np.all(m[:, 2] > 0.0)
    # Non-zeroed rows are unchanged.
    assert np.array_equal(m[~zeroed], sim.results_matrix[~zeroed])
