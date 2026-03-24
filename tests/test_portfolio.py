"""
Tests for Phase 5: Portfolio Construction.

Covers PortfolioConstructor — iterative greedy lineup selection with simulation
row consumption.
"""
import numpy as np
import pandas as pd
import pytest

from src.optimization.lineup import Lineup
from src.optimization.optimizer import _build_player_meta
from src.optimization.portfolio import PortfolioConstructor
from src.simulation.results import SimulationResults


# ------------------------------------------------------------------ #
#  Shared fixtures (mirrors test_optimizer.py slate)                   #
# ------------------------------------------------------------------ #

def _make_player(pid, pos, salary, team, game="A@B"):
    return {'player_id': pid, 'name': f'P{pid}', 'position': pos,
            'salary': salary, 'team': team, 'game': game}


@pytest.fixture
def players_df():
    """
    Two-game slate sufficient to produce valid DK Classic lineups.
    Every salary is low so any 10-player subset fits the $50k cap.
    """
    rows = [
        _make_player(1,  'P',  8000, 'B', 'A@B'),
        _make_player(2,  'P',  7500, 'D', 'C@D'),
        _make_player(3,  'P',  7000, 'B', 'A@B'),
        _make_player(4,  'P',  6500, 'D', 'C@D'),
        _make_player(5,  'C',  4000, 'A', 'A@B'),
        _make_player(6,  'C',  3800, 'C', 'C@D'),
        _make_player(7,  '1B', 4000, 'A', 'A@B'),
        _make_player(8,  '1B', 3800, 'C', 'C@D'),
        _make_player(9,  '2B', 4000, 'A', 'A@B'),
        _make_player(10, '2B', 3800, 'C', 'C@D'),
        _make_player(11, '3B', 4000, 'A', 'A@B'),
        _make_player(12, '3B', 3800, 'C', 'C@D'),
        _make_player(13, 'SS', 4000, 'A', 'A@B'),
        _make_player(14, 'SS', 3800, 'C', 'C@D'),
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


def _make_constructor(sim_results, players_df, portfolio_size=3, target=150.0, seed=42):
    return PortfolioConstructor(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        portfolio_size=portfolio_size,
        n_chains=3,
        n_steps=10,
        rng_seed=seed,
    )


# ------------------------------------------------------------------ #
#  Basic construction                                                  #
# ------------------------------------------------------------------ #

def test_construct_returns_requested_size(sim_results, players_df):
    pc = _make_constructor(sim_results, players_df, portfolio_size=3)
    portfolio = pc.construct()
    assert len(portfolio) == 3


def test_construct_returns_lineup_score_tuples(sim_results, players_df):
    pc = _make_constructor(sim_results, players_df, portfolio_size=2)
    for lineup, score in pc.construct():
        assert isinstance(lineup, Lineup)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_all_lineups_are_valid(sim_results, players_df):
    player_meta = _build_player_meta(players_df)
    pc = _make_constructor(sim_results, players_df, portfolio_size=3)
    for lineup, _ in pc.construct():
        assert lineup.is_valid(player_meta), (
            f"Invalid lineup in portfolio: {lineup.player_ids}"
        )


def test_each_lineup_has_ten_players(sim_results, players_df):
    pc = _make_constructor(sim_results, players_df, portfolio_size=3)
    for lineup, _ in pc.construct():
        assert len(lineup.player_ids) == 10


# ------------------------------------------------------------------ #
#  Consumption mechanics                                               #
# ------------------------------------------------------------------ #

def test_consumption_produces_diverse_lineups():
    """
    Two groups of simulation rows are each dominated by a completely disjoint
    set of 8 non-pitcher players (pitchers are shared).  The first optimized
    lineup clears the target on rows 0-499 only; after consumption the second
    lineup must use the alternate group to clear rows 500-999.  The two lineups
    therefore differ in all 8 non-pitcher positions.

    Design: target = 270, each good player scores 30 → 10-player sum = 300 ≥ 270.
    In the off-half, only the 2 pitchers score → 60 < 270, so that half is NOT
    consumed by the wrong lineup.
    """
    # Pitchers (both lineups must include P1 from A@B and P2 from C@D)
    # Group A (rows 0-499): C3, 1B4, 2B5, 3B6, SS7, OF8, OF9, OF10
    # Group B (rows 500-999): C11, 1B12, 2B13, 3B14, SS15, OF16, OF17, OF18
    rows = [
        _make_player(1,  'P',  8000, 'B', 'A@B'),
        _make_player(2,  'P',  7000, 'D', 'C@D'),
        # Group A non-pitchers (all team A from game A@B — max 5 hitters per team so
        # we spread across two teams to stay within constraints)
        _make_player(3,  'C',  4000, 'A', 'A@B'),
        _make_player(4,  '1B', 4000, 'A', 'A@B'),
        _make_player(5,  '2B', 4000, 'A', 'A@B'),
        _make_player(6,  '3B', 4000, 'C', 'C@D'),
        _make_player(7,  'SS', 4000, 'C', 'C@D'),
        _make_player(8,  'OF', 4000, 'C', 'C@D'),
        _make_player(9,  'OF', 4000, 'D', 'C@D'),
        _make_player(10, 'OF', 4000, 'B', 'A@B'),
        # Group B non-pitchers (alternate pool, same position mix, different teams)
        _make_player(11, 'C',  4000, 'C', 'C@D'),
        _make_player(12, '1B', 4000, 'C', 'C@D'),
        _make_player(13, '2B', 4000, 'C', 'C@D'),
        _make_player(14, '3B', 4000, 'A', 'A@B'),
        _make_player(15, 'SS', 4000, 'A', 'A@B'),
        _make_player(16, 'OF', 4000, 'A', 'A@B'),
        _make_player(17, 'OF', 4000, 'D', 'C@D'),
        _make_player(18, 'OF', 4000, 'B', 'A@B'),
    ]
    df = pd.DataFrame(rows)
    pids = df['player_id'].tolist()
    col_map = {pid: i for i, pid in enumerate(pids)}
    n_sims, n_players = 1000, len(pids)

    matrix = np.zeros((n_sims, n_players))
    # Pitchers score 30 in all rows
    for pid in [1, 2]:
        matrix[:, col_map[pid]] = 30.0
    # Group A: rows 0-499 only
    for pid in [3, 4, 5, 6, 7, 8, 9, 10]:
        matrix[:500, col_map[pid]] = 30.0
    # Group B: rows 500-999 only
    for pid in [11, 12, 13, 14, 15, 16, 17, 18]:
        matrix[500:, col_map[pid]] = 30.0
    # target=270: need 9+ scoring players (each scores 30).
    # Group-A lineup (P1,P2 + 8 from A): 10*30=300 ≥ 270 in rows 0-499; 2*30=60 < 270 in rows 500-999.
    # Group-B lineup (P1,P2 + 8 from B): 10*30=300 ≥ 270 in rows 500-999; 60 < 270 in rows 0-499.

    results = SimulationResults(player_ids=pids, results_matrix=matrix)
    pc = PortfolioConstructor(
        sim_results=results,
        players_df=df,
        target=270.0,
        portfolio_size=2,
        n_chains=10,
        n_steps=30,
        rng_seed=0,
    )
    portfolio = pc.construct()
    assert len(portfolio) == 2, (
        f"Expected 2 lineups but got {len(portfolio)}; "
        "consumption may have wiped all rows after the first lineup"
    )
    lu1, lu2 = portfolio[0][0], portfolio[1][0]
    # The two lineups share the 2 pitchers but differ in the 8 non-pitcher slots
    non_pitcher_ids_1 = set(lu1.player_ids) - {1, 2}
    non_pitcher_ids_2 = set(lu2.player_ids) - {1, 2}
    assert non_pitcher_ids_1 != non_pitcher_ids_2, (
        "Non-pitcher players should differ after consumption"
    )


def test_stops_early_when_all_rows_consumed(players_df):
    """
    If a single lineup already hits the target on every simulation row,
    the constructor should stop after 1 lineup rather than producing
    portfolio_size lineups.
    """
    pids = players_df['player_id'].tolist()
    # Every cell = 100 → any 10-player sum = 1000, always beats target=10
    matrix = np.full((200, len(pids)), 100.0)
    results = SimulationResults(player_ids=pids, results_matrix=matrix)

    pc = PortfolioConstructor(
        sim_results=results,
        players_df=players_df,
        target=10.0,
        portfolio_size=5,
        n_chains=3,
        n_steps=5,
        rng_seed=0,
    )
    portfolio = pc.construct()
    # All rows consumed after first lineup → at most 1 lineup returned
    assert len(portfolio) <= 1


def test_full_score_computed_against_full_matrix(sim_results, players_df):
    """
    The reported score for each lineup must be P(total >= target) over the
    *complete* simulation matrix, not just the active subset used during
    optimisation.
    """
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    full_matrix = sim_results.results_matrix

    pc = _make_constructor(sim_results, players_df, portfolio_size=2, target=150.0)
    portfolio = pc.construct()

    for lineup, reported_score in portfolio:
        cols = [col_map[pid] for pid in lineup.player_ids]
        expected_score = float(
            (full_matrix[:, cols].sum(axis=1) >= 150.0).mean()
        )
        assert reported_score == pytest.approx(expected_score, abs=1e-9)


# ------------------------------------------------------------------ #
#  Reproducibility                                                     #
# ------------------------------------------------------------------ #

def test_construct_is_reproducible(sim_results, players_df):
    """Same seed → same portfolio scores and player sets."""
    pc1 = _make_constructor(sim_results, players_df, portfolio_size=2, seed=7)
    pc2 = _make_constructor(sim_results, players_df, portfolio_size=2, seed=7)
    port1 = pc1.construct()
    port2 = pc2.construct()
    assert len(port1) == len(port2)
    for (lu1, s1), (lu2, s2) in zip(port1, port2):
        assert s1 == pytest.approx(s2)
        assert sorted(lu1.player_ids) == sorted(lu2.player_ids)


def test_different_seeds_may_differ(sim_results, players_df):
    """Two different seeds should not always produce the identical portfolio."""
    pc1 = _make_constructor(sim_results, players_df, portfolio_size=2, seed=1)
    pc2 = _make_constructor(sim_results, players_df, portfolio_size=2, seed=999)
    port1 = pc1.construct()
    port2 = pc2.construct()
    # At least one lineup should differ (extremely unlikely to be identical)
    any_differ = any(
        sorted(lu1.player_ids) != sorted(lu2.player_ids)
        for (lu1, _), (lu2, _) in zip(port1, port2)
    )
    assert any_differ


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #

def test_portfolio_size_one(sim_results, players_df):
    """A portfolio of size 1 should behave identically to a single optimizer run."""
    player_meta = _build_player_meta(players_df)
    pc = _make_constructor(sim_results, players_df, portfolio_size=1)
    portfolio = pc.construct()
    assert len(portfolio) == 1
    lineup, score = portfolio[0]
    assert lineup.is_valid(player_meta)
    assert 0.0 <= score <= 1.0


def test_zero_score_target_never_consumed(players_df):
    """
    If target is impossibly high (no lineup ever hits it), no rows are
    consumed and we still get portfolio_size lineups — each with score 0.
    """
    pids = players_df['player_id'].tolist()
    matrix = np.zeros((100, len(pids)))  # all zeros → totals always 0
    results = SimulationResults(player_ids=pids, results_matrix=matrix)

    pc = PortfolioConstructor(
        sim_results=results,
        players_df=players_df,
        target=1e9,   # impossible
        portfolio_size=3,
        n_chains=3,
        n_steps=5,
        rng_seed=0,
    )
    portfolio = pc.construct()
    assert len(portfolio) == 3
    for _, score in portfolio:
        assert score == pytest.approx(0.0)
