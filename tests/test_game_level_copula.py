"""
Game-level copula sampling: the two units of a game — (team=A, opponent=B) and
(team=B, opponent=A) — must share a single bootstrapped historical game so that
a team's batters stay correlated with its own pitcher and the game-level run
environment. Single-row games and copulas without pairing info fall back to
independent draws.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.copula import EmpiricalCopula
from src.simulation.engine import SimulationEngine


def _write_copula(tmp_path, rows, index_tuples, index_names=('game_id', 'team_id')):
    """rows: list of scalar fill values, one (10-wide) copula row each."""
    index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    df = pd.DataFrame(
        np.array(rows)[:, None] * np.ones((len(rows), 10)),
        index=index,
        columns=range(1, 11),
    )
    path = tmp_path / "copula.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def paired_copula_path(tmp_path):
    """3 complete games (constant quantile per game) + 1 single-row game."""
    return _write_copula(
        tmp_path,
        rows=[0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.99],
        index_tuples=[
            ("G1", "AAA"), ("G1", "BBB"),
            ("G2", "CCC"), ("G2", "DDD"),
            ("G3", "EEE"), ("G3", "FFF"),
            ("G4", "GGG"),  # incomplete: partner row was dropped upstream
        ],
    )


class TestSampleGames:
    def test_rows_come_from_same_game(self, paired_copula_path):
        # Overlay off: this test asserts exact row equality to verify pairing.
        copula = EmpiricalCopula(paired_copula_path, env_overlay_gamma=0.0, env_overlay_delta=0.0)
        pairs = copula.sample_games(n_sims=500)
        assert pairs.shape == (500, 2, 10)
        # Both rows of a draw carry the same per-game constant.
        assert np.array_equal(pairs[:, 0, :], pairs[:, 1, :])

    def test_unpaired_games_excluded(self, paired_copula_path):
        copula = EmpiricalCopula(paired_copula_path)
        assert copula._paired_games.shape == (3, 2, 10)
        pairs = copula.sample_games(n_sims=500)
        assert 0.99 not in pairs

    def test_team_order_randomized(self, tmp_path):
        path = _write_copula(
            tmp_path,
            rows=[0.2, 0.8],
            index_tuples=[("G1", "AAA"), ("G1", "BBB")],
        )
        copula = EmpiricalCopula(path, env_overlay_gamma=0.0, env_overlay_delta=0.0)
        pairs = copula.sample_games(n_sims=2000)
        # Pairing intact regardless of orientation...
        assert np.array_equal(np.sort(pairs[:, :, 0], axis=1),
                              np.tile([0.2, 0.8], (2000, 1)))
        # ...and each orientation appears roughly half the time.
        frac_first = float(np.mean(pairs[:, 0, 0] == 0.2))
        assert 0.4 < frac_first < 0.6

    def test_fallback_without_game_index(self, tmp_path):
        df = pd.DataFrame(
            np.random.default_rng(0).random((6, 10)), columns=range(1, 11)
        )
        path = tmp_path / "flat.parquet"
        df.to_parquet(path)
        copula = EmpiricalCopula(str(path))
        assert copula._paired_games is None
        pairs = copula.sample_games(n_sims=100)
        assert pairs.shape == (100, 2, 10)


class TestEngineGameLevelSampling:
    def _players(self):
        return pd.DataFrame([
            {'player_id': 1, 'team': 'AAA', 'opponent': 'BBB', 'slot': 1,
             'mean': 5.0, 'std_dev': 2.0},
            {'player_id': 2, 'team': 'BBB', 'opponent': 'AAA', 'slot': 1,
             'mean': 5.0, 'std_dev': 2.0},
            {'player_id': 10, 'team': 'BBB', 'opponent': 'AAA', 'slot': 10,
             'mean': 15.0, 'std_dev': 6.0},
        ])

    def test_opposing_units_share_game_draw(self, paired_copula_path):
        # Overlay off: exact-equality assertion checks the shared game draw.
        copula = EmpiricalCopula(paired_copula_path, env_overlay_gamma=0.0, env_overlay_delta=0.0)
        engine = SimulationEngine(copula, self._players())
        results = engine.simulate(n_sims=300)
        col = {pid: i for i, pid in enumerate(results.player_ids)}
        m = results.results_matrix

        # Identical marginals + same-game (constant) quantiles → identical
        # scores for the two slot-1 batters on opposite teams.
        assert np.array_equal(m[:, col[1]], m[:, col[2]])

    def test_unit_without_partner_falls_back(self, paired_copula_path):
        copula = EmpiricalCopula(paired_copula_path)
        players = self._players().iloc[:1].reset_index(drop=True)  # AAA unit only
        engine = SimulationEngine(copula, players)
        results = engine.simulate(n_sims=50)
        assert results.results_matrix.shape == (50, 1)

    def test_stub_copula_without_sample_games(self):
        class _Stub:
            def sample(self, n_sims):
                return np.full((n_sims, 10), 0.5)

        engine = SimulationEngine(_Stub(), self._players())
        results = engine.simulate(n_sims=50)
        assert results.results_matrix.shape == (50, 3)


class TestEnvOverlay:
    """The sim-time dependence overlay must hit its correlation targets while
    leaving marginals unchanged (uniform on the rank scale)."""

    def test_overlay_preserves_uniform_marginals(self, tmp_path):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.random((4000, 10)), columns=range(1, 11))
        path = tmp_path / "flat.parquet"
        df.to_parquet(path)
        copula = EmpiricalCopula(str(path))
        rows = copula.sample(50_000)
        for c in (0, 4, 9):
            q = np.percentile(rows[:, c], [5, 50, 95])
            assert abs(q[0] - 0.05) < 0.02
            assert abs(q[1] - 0.50) < 0.02
            assert abs(q[2] - 0.95) < 0.02

    def test_overlay_raises_batter_dependence(self, tmp_path):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.random((4000, 10)), columns=range(1, 11))
        path = tmp_path / "flat.parquet"
        df.to_parquet(path)
        copula = EmpiricalCopula(str(path))  # source rows independent
        rows = copula.sample(50_000)
        u = rows[:, :9]
        cm = np.corrcoef(u.T)
        bb = cm[np.triu_indices(9, k=1)].mean()
        # gamma=0.271 over independent rows -> gamma^2/(1+gamma^2) ~ 0.068
        assert 0.04 < bb < 0.10
        bp = np.mean([np.corrcoef(u[:, i], rows[:, 9])[0, 1] for i in range(9)])
        # positive batter and pitcher loadings -> positive cross term
        assert bp > 0.05

    def test_overlay_disabled_is_identity(self, tmp_path):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.random((50, 10)), columns=range(1, 11))
        path = tmp_path / "flat.parquet"
        df.to_parquet(path)
        copula = EmpiricalCopula(str(path), env_overlay_gamma=0.0, env_overlay_delta=0.0)
        rows = copula.sample(200)
        src = df.values
        assert all(any(np.allclose(r, s) for s in src) for r in rows[:20])
