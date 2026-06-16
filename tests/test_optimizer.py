"""
Tests for lineup validation, score computation, and player metadata helpers.
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
from src.optimization.optimizer import _build_player_meta
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
        # Extra pitchers from teams A and C (oppose B and D respectively),
        # needed for lineups where the opposing team's batters are not stacked.
        _make_player(20, 'P',  8000, 'A', 'A@B'),
        _make_player(21, 'P',  7500, 'C', 'C@D'),
        # Extra OFs from C so VALID_IDS can field 3 OFs without exceeding the
        # 5-hitter-per-team cap on team A.
        _make_player(22, 'OF', 3800, 'C', 'C@D'),
        _make_player(23, 'OF', 3600, 'C', 'C@D'),
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

# Valid baseline lineup: P20(A,A@B,oppB), P21(C,C@D,oppD) | C5(A) | 1B7(A) | 2B9(A)
# | 3B11(A) | SS13(A) | OF17(C) | OF22(C) | OF23(C)
# Hitters from A = 5 (C,1B,2B,3B,SS) ≤ MAX; 2 games ✓; no pitcher opposes any batter ✓
VALID_IDS = [20, 21, 5, 7, 9, 11, 13, 17, 22, 23]


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
    # P20(A,oppB), P21(C,oppD), C5(A), 1B7(A), 2B9(A), 3B11(A), SS13(A), OF15(A), OF17(C), OF19(A)
    # Hitters from A: C5, 1B7, 2B9, 3B11, SS13, OF15, OF19 = 7 > 5 → invalid
    # (pitchers P20/P21 oppose B/D, none of which appear as batters → pitcher check passes)
    too_many_a = [20, 21, 5, 7, 9, 11, 13, 15, 17, 19]
    assert not Lineup(too_many_a).is_valid(player_meta)


def test_lineup_pitcher_opposing_batter_invalid(player_meta):
    # P1(B,A@B) opposes team A; lineup includes C5/1B7/2B9/3B11/SS13 all from A → invalid
    ids = [1, 2, 5, 7, 9, 11, 13, 16, 17, 18]
    assert not Lineup(ids).is_valid(player_meta)


def test_lineup_two_pitchers_same_team_invalid(player_meta):
    # P1(team B) and P3(team B) are both from team B → invalid
    # Batters chosen from teams that neither pitcher opposes (C, D) to isolate the new constraint.
    ids = [1, 3, 6, 8, 10, 12, 14, 17, 18, 22]
    assert not Lineup(ids).is_valid(player_meta)


def test_lineup_pitcher_opposing_constraint_skipped_without_opponent_info():
    # When opponent key is absent from meta the check is skipped (analogous to game check)
    positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    meta = {
        i + 1: {'position': positions[i], 'salary': 4000.0,
                'team': f'T{i}', 'game': f'G{i // 2}@G{i // 2 + 1}'}
        for i in range(10)
    }
    assert Lineup(list(range(1, 11))).is_valid(meta)


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
#  Salary floor                                                        #
# ------------------------------------------------------------------ #

# VALID_IDS total salary: P20(8000)+P21(7500)+C5(4000)+1B7(4000)+2B9(4000)
#   +3B11(4000)+SS13(4000)+OF17(3800)+OF22(3800)+OF23(3600) = 46_700

def test_lineup_salary_below_floor(player_meta):
    # VALID_IDS sums to 46_700 which is below floor=48_000 → invalid
    assert not Lineup(VALID_IDS).is_valid(player_meta, salary_floor=48_000.0)


def test_lineup_salary_at_floor(player_meta):
    # Exactly at floor → valid
    assert Lineup(VALID_IDS).is_valid(player_meta, salary_floor=46_700.0)


def test_lineup_floor_none_no_effect(player_meta):
    # salary_floor=None should leave existing behaviour unchanged
    assert Lineup(VALID_IDS).is_valid(player_meta, salary_floor=None)


# ------------------------------------------------------------------ #
#  _build_player_meta                                                  #
# ------------------------------------------------------------------ #

def test_build_player_meta_stores_eligible_positions():
    rows = [_make_player(1, '3B', 4000, 'A')]
    df = pd.DataFrame(rows)
    df['eligible_positions'] = [['3B', 'SS']]
    meta = _build_player_meta(df)
    assert meta[1]['eligible_positions'] == ['3B', 'SS']
    assert meta[1]['position'] == '3B'


def test_build_player_meta_falls_back_without_column():
    rows = [_make_player(1, 'SS', 4000, 'A')]
    df = pd.DataFrame(rows)
    # No eligible_positions column — should fall back to [position]
    meta = _build_player_meta(df)
    assert meta[1]['eligible_positions'] == ['SS']


# ------------------------------------------------------------------ #
#  Multi-position eligibility                                          #
# ------------------------------------------------------------------ #

def test_is_valid_multipos_assignment_required():
    # Lineup with no pure SS: player 20 is 3B/SS and must fill SS for the
    # lineup to be valid; player 21 fills 3B.
    positions = ['P', 'P', 'C', '1B', '2B', 'OF', 'OF', 'OF']
    meta = {
        i + 1: {'position': positions[i], 'eligible_positions': [positions[i]],
                'salary': 4000.0, 'team': f'T{i}', 'game': 'A@B' if i < 5 else 'C@D'}
        for i in range(len(positions))
    }
    # Player 20: primary 3B, also eligible SS
    meta[20] = {'position': '3B', 'eligible_positions': ['3B', 'SS'],
                'salary': 4000.0, 'team': 'T8', 'game': 'A@B'}
    # Player 21: pure 3B
    meta[21] = {'position': '3B', 'eligible_positions': ['3B'],
                'salary': 4000.0, 'team': 'T9', 'game': 'C@D'}

    # Lineup: standard 8 + player 20 (fills SS) + player 21 (fills 3B)
    ids = list(range(1, 9)) + [20, 21]
    assert Lineup(ids).is_valid(meta)


def test_is_valid_rejects_impossible_assignment():
    # Two 3B/SS players and no pure 3B or pure SS → can only fill one of each
    # but we have two 3B/SS players and need exactly one 3B + one SS.
    # That's actually fine (one fills 3B, other fills SS).
    # Make it impossible: three 3B/SS players competing for one 3B + one SS slot.
    positions = ['P', 'P', 'C', '1B', '2B', 'OF', 'OF', 'OF']
    meta = {
        i + 1: {'position': positions[i], 'eligible_positions': [positions[i]],
                'salary': 4000.0, 'team': f'T{i}', 'game': 'A@B' if i < 5 else 'C@D'}
        for i in range(len(positions))
    }
    # Players 20, 21, 22 all only eligible for 3B/SS — but we need exactly 1 3B + 1 SS
    # and have 3 candidates for those 2 slots: one will be left unmatched,
    # meaning the lineup has 10 players but only 9 can be assigned → invalid.
    for pid, game in [(20, 'A@B'), (21, 'C@D'), (22, 'A@B')]:
        meta[pid] = {'position': '3B', 'eligible_positions': ['3B', 'SS'],
                     'salary': 4000.0, 'team': f'TX{pid}', 'game': game}

    ids = list(range(1, 9)) + [20, 21, 22]  # 11 players → fails size check first
    assert not Lineup(ids).is_valid(meta)

    # Correct size but impossible: replace one of the 8 standard players with a third 3B/SS
    # so we have 2 "pure" slots occupied by 3B/SS players but still need the 8th standard slot
    # The cleaner impossible case: swap out an OF for a third 3B/SS player
    ids_10 = list(range(1, 8)) + [20, 21, 22]  # 10 players: positions P,P,C,1B,2B,OF,OF + 3x(3B/SS)
    # slots needed: 3B×1 + SS×1 but only 2 can be filled by the three 3B/SS players
    # (one is left out) — wait, 3 players for 2 slots is fine (2 get matched).
    # The real impossibility: we have positions P,P,C,1B,2B,OF,OF,3B/SS,3B/SS,3B/SS
    # ROSTER needs: P×2,C×1,1B×1,2B×1,3B×1,SS×1,OF×3 — we have 3B/SS covering 3B+SS,
    # but only 2 OF players for 3 OF slots → impossible.
    assert not Lineup(ids_10).is_valid(meta)


# ------------------------------------------------------------------ #
#  Pipeline _build_players_df multi-position passthrough               #
# ------------------------------------------------------------------ #

def _make_slate_df(rows, include_eligible=True):
    """Build a minimal slate DataFrame resembling DraftKingsSlateIngestor output."""
    df = pd.DataFrame(rows)
    if include_eligible and 'eligible_positions' not in df.columns:
        df['eligible_positions'] = df['position'].apply(lambda p: [p])
    df['game'] = df.get('game', 'A@B')
    return df


def _make_pipeline_runner():
    """Return a PipelineRunner instance; __init__ only stores config_path, no I/O."""
    from src.api.pipeline import PipelineRunner
    return PipelineRunner.__new__(PipelineRunner)


def test_build_players_df_preserves_eligible_positions():
    """eligible_positions from slate_df must survive _build_players_df."""
    from src.api.pipeline import PipelineRunner
    runner = _make_pipeline_runner()
    rows = [
        {'player_id': 1, 'name': 'A', 'position': '3B', 'eligible_positions': ['3B', 'SS'],
         'salary': 5000, 'team': 'NYY', 'game': 'NYY@BOS'},
        {'player_id': 2, 'name': 'B', 'position': 'P', 'eligible_positions': ['P'],
         'salary': 8000, 'team': 'BOS', 'game': 'NYY@BOS'},
    ]
    slate_df = pd.DataFrame(rows)
    players_df = runner._build_players_df(slate_df, proj_df=None)
    assert 'eligible_positions' in players_df.columns
    ep = players_df.loc[players_df['player_id'] == 1, 'eligible_positions'].iloc[0]
    assert list(ep) == ['3B', 'SS']


def test_build_players_df_without_eligible_positions_omits_column():
    """Slates without eligible_positions column must not crash and omit the column."""
    runner = _make_pipeline_runner()
    rows = [
        {'player_id': 1, 'name': 'A', 'position': 'SS', 'salary': 5000,
         'team': 'NYY', 'game': 'NYY@BOS'},
    ]
    slate_df = pd.DataFrame(rows)
    players_df = runner._build_players_df(slate_df, proj_df=None)
    assert 'eligible_positions' not in players_df.columns
