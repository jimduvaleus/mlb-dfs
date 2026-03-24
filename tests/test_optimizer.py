"""
Tests for Phase 3: Basin-Hopping Optimizer.

Covers Lineup constraint validation, score computation, and the full
BasinHoppingOptimizer (random lineup generation, mutation, local search,
and end-to-end optimize()).
"""
import numpy as np
import pandas as pd
import pytest

from src.optimization.lineup import (
    Lineup,
    ROSTER_REQUIREMENTS,
    SALARY_CAP,
    MAX_HITTERS_PER_TEAM,
)
from src.optimization.optimizer import BasinHoppingOptimizer, _build_player_meta
from src.simulation.results import SimulationResults


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

def _make_player(pid, pos, salary, team, game="A@B"):
    return {'player_id': pid, 'name': f'P{pid}', 'position': pos,
            'salary': salary, 'team': team, 'game': game}


@pytest.fixture
def players_df():
    """
    Two-game slate with 3-5 candidates per position.
    Game "A@B": teams A (hitters) and B (pitcher + some hitters)
    Game "C@D": teams C and D
    Salary per player is low enough that any 10-man lineup fits the $50k cap.
    """
    rows = [
        # Pitchers (need 2) – one from each game
        _make_player(1,  'P',  8000, 'B', 'A@B'),
        _make_player(2,  'P',  7500, 'D', 'C@D'),
        _make_player(3,  'P',  7000, 'B', 'A@B'),
        _make_player(4,  'P',  6500, 'D', 'C@D'),
        # Catchers (need 1)
        _make_player(5,  'C',  4000, 'A', 'A@B'),
        _make_player(6,  'C',  3800, 'C', 'C@D'),
        # 1B (need 1)
        _make_player(7,  '1B', 4000, 'A', 'A@B'),
        _make_player(8,  '1B', 3800, 'C', 'C@D'),
        # 2B (need 1)
        _make_player(9,  '2B', 4000, 'A', 'A@B'),
        _make_player(10, '2B', 3800, 'C', 'C@D'),
        # 3B (need 1)
        _make_player(11, '3B', 4000, 'A', 'A@B'),
        _make_player(12, '3B', 3800, 'C', 'C@D'),
        # SS (need 1)
        _make_player(13, 'SS', 4000, 'A', 'A@B'),
        _make_player(14, 'SS', 3800, 'C', 'C@D'),
        # OF (need 3)
        _make_player(15, 'OF', 4000, 'A', 'A@B'),
        _make_player(16, 'OF', 4000, 'B', 'A@B'),
        _make_player(17, 'OF', 3800, 'C', 'C@D'),
        _make_player(18, 'OF', 3800, 'D', 'C@D'),
        _make_player(19, 'OF', 3600, 'A', 'A@B'),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sim_results(players_df):
    rng = np.random.default_rng(0)
    pids = players_df['player_id'].tolist()
    matrix = rng.uniform(0, 40, size=(500, len(pids))).astype(np.float64)
    return SimulationResults(player_ids=pids, results_matrix=matrix)


@pytest.fixture
def player_meta(players_df):
    return _build_player_meta(players_df)


# ------------------------------------------------------------------ #
#  Lineup.is_valid                                                     #
# ------------------------------------------------------------------ #

# Valid baseline lineup: P1(B,A@B), P2(D,C@D) | C5(A) | 1B7(A) | 2B9(A) | 3B11(A) | SS13(A)
# | OF16(B,A@B) | OF17(C,C@D) | OF18(D,C@D)
# Hitters from A = 5 (C,1B,2B,3B,SS) ≤ MAX; 2 games (A@B, C@D) ✓
VALID_IDS = [1, 2, 5, 7, 9, 11, 13, 16, 17, 18]


def test_lineup_is_valid_baseline(player_meta):
    assert Lineup(VALID_IDS).is_valid(player_meta)


def test_lineup_wrong_number_of_players(player_meta):
    assert not Lineup(VALID_IDS[:9]).is_valid(player_meta)


def test_lineup_salary_over_cap(player_meta):
    # Inflate salaries to exceed cap
    inflated = {pid: {**m, 'salary': 10_000.0} for pid, m in player_meta.items()}
    assert not Lineup(VALID_IDS).is_valid(inflated)


def test_lineup_wrong_position_mix(player_meta):
    # Replace both pitchers with OF → missing 2 P, extra 2 OF
    bad_ids = [15, 19, 5, 7, 9, 11, 13, 16, 17, 18]
    assert not Lineup(bad_ids).is_valid(player_meta)


def test_lineup_too_many_hitters_one_team(player_meta):
    # P1(B,A@B), P2(D,C@D), C5(A), 1B7(A), 2B9(A), 3B11(A), SS13(A), OF15(A), OF17(C), OF18(D)
    # Hitters from A: C5, 1B7, 2B9, 3B11, SS13, OF15 = 6 > 5 → invalid
    too_many_a = [1, 2, 5, 7, 9, 11, 13, 15, 17, 18]
    assert not Lineup(too_many_a).is_valid(player_meta)


def test_lineup_only_one_game():
    # Build custom meta: 10 different teams (team constraint passes) but all one game.
    # The only failing constraint is: 1 game < MIN_GAMES=2.
    positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    meta = {
        i + 1: {'position': positions[i], 'salary': 4000.0,
                'team': f'T{i}', 'game': 'A@B'}
        for i in range(10)
    }
    ids = list(range(1, 11))
    assert not Lineup(ids).is_valid(meta)


def test_lineup_game_check_skipped_when_no_game_info():
    # Same single-game lineup but with game = '' → constraint skipped → valid
    positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    meta = {
        i + 1: {'position': positions[i], 'salary': 4000.0,
                'team': f'T{i}', 'game': ''}
        for i in range(10)
    }
    ids = list(range(1, 11))
    assert Lineup(ids).is_valid(meta)


# ------------------------------------------------------------------ #
#  Lineup.score                                                        #
# ------------------------------------------------------------------ #

def test_lineup_score_all_above_target(sim_results):
    """When all simulations beat the target the score should be 1.0."""
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    # Set every cell to 100 so any 10-player sum exceeds target=50
    matrix = np.full_like(sim_results.results_matrix, 100.0)
    results = SimulationResults(sim_results.player_ids, matrix)
    lineup = Lineup(sim_results.player_ids[:10])
    assert lineup.score(results.results_matrix, col_map, target=50.0) == pytest.approx(1.0)


def test_lineup_score_none_above_target(sim_results):
    """When no simulation beats the target the score should be 0.0."""
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    matrix = np.zeros_like(sim_results.results_matrix)
    results = SimulationResults(sim_results.player_ids, matrix)
    lineup = Lineup(sim_results.player_ids[:10])
    assert lineup.score(results.results_matrix, col_map, target=1.0) == pytest.approx(0.0)


def test_lineup_score_fraction(sim_results):
    """Half the simulations beat target → score ≈ 0.5."""
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    n = sim_results.n_sims
    matrix = np.zeros((n, sim_results.n_players))
    # First half: every player scores 10 → 100 total; second half: 0
    matrix[: n // 2, :] = 10.0
    results = SimulationResults(sim_results.player_ids, matrix)
    lineup = Lineup(sim_results.player_ids[:10])
    s = lineup.score(results.results_matrix, col_map, target=50.0)
    assert s == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------------------------ #
#  BasinHoppingOptimizer internals                                     #
# ------------------------------------------------------------------ #

def make_optimizer(sim_results, players_df, target=100.0, n_chains=3, n_steps=10):
    return BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        n_chains=n_chains,
        temperature=0.05,
        n_steps=n_steps,
        rng_seed=42,
    )


def test_random_valid_lineup(sim_results, players_df, player_meta):
    opt = make_optimizer(sim_results, players_df)
    rng = np.random.default_rng(0)
    for _ in range(10):
        lineup = opt._runner._random_valid_lineup(rng)
        assert isinstance(lineup, Lineup)
        assert len(lineup.player_ids) == 10
        assert lineup.is_valid(player_meta)


def test_mutate_preserves_validity(sim_results, players_df, player_meta):
    opt = make_optimizer(sim_results, players_df)
    rng = np.random.default_rng(1)
    lineup = opt._runner._random_valid_lineup(rng)
    for _ in range(20):
        mutated = opt._runner._mutate(lineup, rng)
        assert mutated.is_valid(player_meta), f"Mutated lineup invalid: {mutated.player_ids}"


def test_local_search_does_not_decrease_score(sim_results, players_df, player_meta):
    opt = make_optimizer(sim_results, players_df, target=150.0)
    rng = np.random.default_rng(2)
    lineup = opt._runner._random_valid_lineup(rng)
    cols = [opt.col_map[pid] for pid in lineup.player_ids]
    totals = opt.sim_matrix[:, cols].sum(axis=1)
    score_before = float((totals >= opt.target).mean())

    improved_lineup, improved_totals = opt._runner._local_search(lineup, totals, rng)
    score_after = float((improved_totals >= opt.target).mean())

    assert score_after >= score_before - 1e-9
    assert improved_lineup.is_valid(player_meta)


# ------------------------------------------------------------------ #
#  End-to-end optimize()                                               #
# ------------------------------------------------------------------ #

def test_optimize_returns_valid_lineup(sim_results, players_df, player_meta):
    opt = make_optimizer(sim_results, players_df, target=150.0, n_chains=5, n_steps=20)
    lineup, score = opt.optimize()

    assert isinstance(lineup, Lineup)
    assert lineup.is_valid(player_meta)
    assert 0.0 <= score <= 1.0


def test_optimize_score_is_reproducible(sim_results, players_df):
    """Same seed should produce the same result."""
    opt1 = make_optimizer(sim_results, players_df, target=150.0, n_chains=3, n_steps=10)
    opt2 = make_optimizer(sim_results, players_df, target=150.0, n_chains=3, n_steps=10)
    lineup1, score1 = opt1.optimize()
    lineup2, score2 = opt2.optimize()
    assert score1 == pytest.approx(score2)
    assert sorted(lineup1.player_ids) == sorted(lineup2.player_ids)


def test_optimize_high_value_player_selected():
    """
    When one player has dramatically higher simulated scores the optimizer
    should prefer them in the final lineup.
    """
    rng = np.random.default_rng(99)

    # Two pitchers (P), one catcher, one 1B, one 2B, one 3B, one SS, three OF
    # per game – two games to satisfy the multi-game constraint.
    rows = [
        _make_player(1,  'P',  8000, 'B', 'A@B'),
        _make_player(2,  'P',  7000, 'D', 'C@D'),
        _make_player(3,  'C',  4000, 'A', 'A@B'),
        _make_player(4,  '1B', 4000, 'A', 'A@B'),
        _make_player(5,  '2B', 4000, 'A', 'A@B'),
        _make_player(6,  '3B', 4000, 'C', 'C@D'),
        _make_player(7,  'SS', 4000, 'C', 'C@D'),
        # OF – player 8 is the "stud", players 9-11 are average
        _make_player(8,  'OF', 4000, 'C', 'C@D'),   # stud
        _make_player(9,  'OF', 4000, 'A', 'A@B'),
        _make_player(10, 'OF', 4000, 'D', 'C@D'),
        _make_player(11, 'OF', 3500, 'B', 'A@B'),
    ]
    df = pd.DataFrame(rows)
    pids = df['player_id'].tolist()
    n_sims, n_players = 1000, len(pids)
    matrix = rng.uniform(0, 10, size=(n_sims, n_players)).astype(np.float64)
    # Make player 8 (index 7) score extremely high
    matrix[:, 7] = 200.0

    results = SimulationResults(player_ids=pids, results_matrix=matrix)
    opt = BasinHoppingOptimizer(
        sim_results=results,
        players_df=df,
        target=250.0,
        n_chains=5,
        n_steps=30,
        rng_seed=0,
    )
    lineup, _ = opt.optimize()
    assert 8 in lineup.player_ids, "High-value player should be in the optimized lineup"
