"""Tests for EV-guided candidate pool refinement:
ContestScorer.score_batch() and CandidateGenerator.generate_mutants().
"""
import numpy as np
import pandas as pd
import pytest

from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.contest import _player_meta_from_df
from src.optimization.gpp_portfolio import ContestScorer
from src.optimization.lineup import Lineup
from src.optimization.ownership import compute_heuristic_ownership
from src.simulation.results import SimulationResults


def _make_player(pid, pos, salary, team, game, mean=20.0):
    row = {
        "player_id": pid,
        "name": f"P{pid}",
        "position": pos,
        "salary": salary,
        "team": team,
        "game": game,
        "mean": mean,
        "std_dev": 5.0,
        "slot": 9 if pos == "P" else 1,
    }
    parts = game.split("@")
    row["opponent"] = parts[1] if team == parts[0] else parts[0]
    return row


@pytest.fixture
def players_df():
    rows = [
        # Game A@B
        _make_player(1,  "P",  8000, "B", "A@B", 25.0),
        _make_player(2,  "P",  7000, "B", "A@B", 22.0),
        _make_player(20, "P",  8500, "A", "A@B", 21.0),
        _make_player(5,  "C",  4000, "A", "A@B", 18.0),
        _make_player(7,  "1B", 4200, "A", "A@B", 19.0),
        _make_player(9,  "2B", 4100, "A", "A@B", 17.0),
        _make_player(11, "3B", 3900, "A", "A@B", 16.0),
        _make_player(13, "SS", 3800, "A", "A@B", 15.0),
        _make_player(15, "OF", 4000, "A", "A@B", 20.0),
        _make_player(19, "OF", 3600, "A", "A@B", 18.0),
        _make_player(16, "OF", 4000, "B", "A@B", 18.0),
        _make_player(24, "1B", 3500, "B", "A@B", 16.0),
        _make_player(25, "C",  3400, "B", "A@B", 15.0),
        # Game C@D
        _make_player(3,  "P",  7500, "D", "C@D", 24.0),
        _make_player(4,  "P",  6500, "D", "C@D", 21.0),
        _make_player(21, "P",  7500, "C", "C@D", 20.0),
        _make_player(6,  "C",  3800, "C", "C@D", 18.0),
        _make_player(8,  "1B", 3800, "C", "C@D", 17.0),
        _make_player(10, "2B", 3800, "C", "C@D", 16.0),
        _make_player(12, "3B", 3800, "C", "C@D", 15.0),
        _make_player(14, "SS", 3800, "C", "C@D", 14.0),
        _make_player(17, "OF", 3800, "C", "C@D", 19.0),
        _make_player(22, "OF", 3800, "C", "C@D", 17.0),
        _make_player(23, "OF", 3600, "C", "C@D", 16.0),
        _make_player(18, "OF", 3800, "D", "C@D", 18.0),
        _make_player(26, "2B", 3500, "D", "C@D", 16.0),
        _make_player(27, "SS", 3400, "D", "C@D", 15.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sim_results(players_df):
    rng = np.random.default_rng(42)
    pids = players_df["player_id"].tolist()
    matrix = rng.uniform(0, 40, size=(500, len(pids))).astype(np.float32)
    return SimulationResults(player_ids=pids, results_matrix=matrix)


@pytest.fixture
def ownership_vec(players_df):
    return compute_heuristic_ownership(players_df)


@pytest.fixture
def gen(players_df, ownership_vec):
    return CandidateGenerator(players_df, ownership_vec, rng_seed=0)


@pytest.fixture
def candidates(gen):
    return gen.generate(n_candidates=50)


@pytest.fixture
def test_payout_arr():
    arr = np.zeros(200, dtype=np.float32)
    arr[0] = 5000.0
    arr[1:52] = 6.0
    return arr


def _make_scorer(sim_results, players_df, ownership_vec, test_payout_arr):
    return ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=10,
    )


# ------------------------------------------------------------------ #
#  ContestScorer.score_batch                                           #
# ------------------------------------------------------------------ #

class TestScoreBatch:
    def test_matches_score_candidates_rows(
        self, sim_results, players_df, candidates, ownership_vec, test_payout_arr
    ):
        scorer = _make_scorer(sim_results, players_df, ownership_vec, test_payout_arr)
        _, full = scorer.score_candidates(candidates[:20])
        # Re-scoring a subset against the cached fields must reproduce the
        # same rows exactly (same fields, deterministic scoring).
        batch = scorer.score_batch(candidates[5:15])
        assert np.array_equal(batch, full[5:15])

    def test_requires_prior_score_candidates(
        self, sim_results, players_df, candidates, ownership_vec, test_payout_arr
    ):
        scorer = _make_scorer(sim_results, players_df, ownership_vec, test_payout_arr)
        with pytest.raises(RuntimeError, match="score_candidates"):
            scorer.score_batch(candidates[:5])

    def test_appends_col_lineups(
        self, sim_results, players_df, candidates, ownership_vec, test_payout_arr
    ):
        scorer = _make_scorer(sim_results, players_df, ownership_vec, test_payout_arr)
        scorer.score_candidates(candidates[:20])
        assert scorer.last_col_lineups.shape == (20, 10)
        scorer.score_batch(candidates[20:30])
        assert scorer.last_col_lineups.shape == (30, 10)

    def test_batch_shape_and_dtype(
        self, sim_results, players_df, candidates, ownership_vec, test_payout_arr
    ):
        scorer = _make_scorer(sim_results, players_df, ownership_vec, test_payout_arr)
        scorer.score_candidates(candidates[:10])
        batch = scorer.score_batch(candidates[10:17])
        assert batch.shape == (7, sim_results.n_sims)
        assert batch.dtype == np.float32


# ------------------------------------------------------------------ #
#  CandidateGenerator.generate_mutants                                 #
# ------------------------------------------------------------------ #

class TestGenerateMutants:
    def test_mutants_are_valid_and_stacked(self, gen, candidates, players_df):
        parents = candidates[:10]
        seen = {frozenset(int(p) for p in lu.player_ids) for lu in candidates}
        mutants = gen.generate_mutants(parents, n_per_parent=3, seen=seen, rng_seed=7)
        assert len(mutants) > 0

        meta = _player_meta_from_df(players_df)
        for mu in mutants:
            assert mu.is_valid(meta), f"Invalid mutant: {mu.player_ids}"
            assert gen._check_stack([int(p) for p in mu.player_ids])

    def test_mutants_deduped_and_added_to_seen(self, gen, candidates):
        parents = candidates[:10]
        pool_keys = {frozenset(int(p) for p in lu.player_ids) for lu in candidates}
        seen = set(pool_keys)
        mutants = gen.generate_mutants(parents, n_per_parent=3, seen=seen, rng_seed=7)

        mutant_keys = [frozenset(int(p) for p in mu.player_ids) for mu in mutants]
        # No duplicates among mutants, none colliding with the original pool.
        assert len(set(mutant_keys)) == len(mutant_keys)
        assert not (set(mutant_keys) & pool_keys)
        # All mutants were registered in seen.
        assert set(mutant_keys) <= seen

    def test_mutants_preserve_position_multiset(self, gen, candidates, players_df):
        pid_pos = dict(zip(players_df["player_id"].astype(int), players_df["position"]))
        parents = candidates[:5]
        seen = {frozenset(int(p) for p in lu.player_ids) for lu in candidates}
        mutants = gen.generate_mutants(parents, n_per_parent=4, seen=seen, rng_seed=11)
        assert len(mutants) > 0

        parent_multiset = sorted(pid_pos[int(p)] for p in parents[0].player_ids)
        for mu in mutants:
            assert sorted(pid_pos[int(p)] for p in mu.player_ids) == parent_multiset

    def test_mutants_differ_by_one_or_two_players(self, gen, candidates):
        parents = candidates[:5]
        seen = {frozenset(int(p) for p in lu.player_ids) for lu in candidates}
        mutants = gen.generate_mutants(parents, n_per_parent=4, seen=seen, rng_seed=3)
        assert len(mutants) > 0

        parent_sets = [set(int(p) for p in lu.player_ids) for lu in parents]
        for mu in mutants:
            mu_set = set(int(p) for p in mu.player_ids)
            min_diff = min(len(mu_set - ps) for ps in parent_sets)
            assert 1 <= min_diff <= 2

    def test_deterministic_with_seed(self, gen, candidates):
        parents = candidates[:8]
        pool_keys = {frozenset(int(p) for p in lu.player_ids) for lu in candidates}
        m1 = gen.generate_mutants(parents, n_per_parent=3, seen=set(pool_keys), rng_seed=42)
        m2 = gen.generate_mutants(parents, n_per_parent=3, seen=set(pool_keys), rng_seed=42)
        assert [mu.player_ids for mu in m1] == [mu.player_ids for mu in m2]

    def test_unknown_parent_players_skipped(self, gen, candidates):
        ghost = Lineup(player_ids=[9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010])
        seen = set()
        mutants = gen.generate_mutants([ghost], n_per_parent=3, seen=seen, rng_seed=1)
        assert mutants == []
