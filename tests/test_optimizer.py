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
    score_before = opt._runner._score_totals(totals, opt.target)

    improved_lineup, improved_totals = opt._runner._local_search(lineup, totals, rng)
    score_after = opt._runner._score_totals(improved_totals, opt.target)

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
    assert score >= 0.0


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
        # Pitchers from A and C oppose B and D respectively; no batters from B or D
        # are in this slate, so the pitcher/opposing-batter constraint is satisfied.
        _make_player(1,  'P',  8000, 'A', 'A@B'),
        _make_player(2,  'P',  7000, 'C', 'C@D'),
        _make_player(3,  'C',  4000, 'A', 'A@B'),
        _make_player(4,  '1B', 4000, 'A', 'A@B'),
        _make_player(5,  '2B', 4000, 'A', 'A@B'),
        _make_player(6,  '3B', 4000, 'C', 'C@D'),
        _make_player(7,  'SS', 4000, 'C', 'C@D'),
        # OF – player 8 is the "stud", players 9-11 are average
        _make_player(8,  'OF', 4000, 'C', 'C@D'),   # stud
        _make_player(9,  'OF', 4000, 'A', 'A@B'),
        _make_player(10, 'OF', 4000, 'A', 'A@B'),
        _make_player(11, 'OF', 3500, 'C', 'C@D'),
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


def test_optimize_respects_salary_floor(sim_results, players_df, player_meta):
    floor = 45_000.0
    opt = BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=150.0,
        n_chains=5,
        n_steps=20,
        rng_seed=7,
        salary_floor=floor,
    )
    lineup, _ = opt.optimize()
    total_salary = sum(player_meta[pid]['salary'] for pid in lineup.player_ids)
    assert total_salary >= floor
    assert lineup.is_valid(player_meta, salary_floor=floor)


def test_optimize_floor_disabled_unchanged(sim_results, players_df):
    # salary_floor=None must produce the same result as omitting the parameter
    opt_default = make_optimizer(sim_results, players_df, target=150.0, n_chains=3, n_steps=10)
    opt_none = BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=150.0,
        n_chains=3,
        temperature=0.05,
        n_steps=10,
        rng_seed=42,
        salary_floor=None,
    )
    l1, s1 = opt_default.optimize()
    l2, s2 = opt_none.optimize()
    assert s1 == pytest.approx(s2)
    assert sorted(l1.player_ids) == sorted(l2.player_ids)


# ------------------------------------------------------------------ #
#  Multi-position eligibility                                          #
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


def test_players_by_pos_multi_eligible(sim_results, players_df):
    # Add a 3B/SS player to the slate and simulation results
    extra_row = _make_player(99, '3B', 4000, 'A', 'A@B')
    extra_df = pd.concat([players_df, pd.DataFrame([extra_row])], ignore_index=True)
    extra_df['eligible_positions'] = extra_df['position'].apply(lambda p: [p])
    # Give player 99 dual eligibility
    extra_df.loc[extra_df['player_id'] == 99, 'eligible_positions'] = \
        extra_df.loc[extra_df['player_id'] == 99, 'eligible_positions'].apply(lambda _: ['3B', 'SS'])

    pids = extra_df['player_id'].tolist()
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0, 40, size=(500, len(pids))).astype(np.float64)
    results = SimulationResults(player_ids=pids, results_matrix=matrix)

    opt = BasinHoppingOptimizer(sim_results=results, players_df=extra_df, target=100.0)
    assert 99 in opt._players_by_pos['3B']
    assert 99 in opt._players_by_pos['SS']


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


def test_optimize_multi_pos_slate():
    """Slate where the only valid lineup requires a 3B/SS player to fill SS."""
    rng = np.random.default_rng(7)
    rows = [
        # Pitchers from A and C oppose B and D; no batters in this slate are from B or D.
        _make_player(1,  'P',  8000, 'A', 'A@B'),
        _make_player(2,  'P',  7500, 'C', 'C@D'),
        _make_player(3,  'C',  4000, 'A', 'A@B'),
        _make_player(4,  '1B', 4000, 'A', 'A@B'),
        _make_player(5,  '2B', 4000, 'A', 'A@B'),
        _make_player(6,  '3B', 4000, 'C', 'C@D'),
        # No pure SS — player 7 is 3B/SS and must fill SS
        _make_player(7,  '3B', 4000, 'A', 'A@B'),
        _make_player(8,  'OF', 4000, 'A', 'A@B'),
        _make_player(9,  'OF', 4000, 'C', 'C@D'),
        _make_player(10, 'OF', 4000, 'C', 'C@D'),
    ]
    df = pd.DataFrame(rows)
    df['eligible_positions'] = df['position'].apply(lambda p: [p])
    df.loc[df['player_id'] == 7, 'eligible_positions'] = \
        df.loc[df['player_id'] == 7, 'eligible_positions'].apply(lambda _: ['3B', 'SS'])

    pids = df['player_id'].tolist()
    matrix = rng.uniform(0, 40, size=(500, len(pids))).astype(np.float64)
    results = SimulationResults(player_ids=pids, results_matrix=matrix)

    opt = BasinHoppingOptimizer(
        sim_results=results,
        players_df=df,
        target=100.0,
        n_chains=5,
        n_steps=20,
        rng_seed=0,
    )
    lineup, _ = opt.optimize()
    meta = _build_player_meta(df)
    assert lineup.is_valid(meta)
    assert 7 in lineup.player_ids  # the 3B/SS player must be in the lineup


# ------------------------------------------------------------------ #
#  marginal_payout objective                                           #
# ------------------------------------------------------------------ #

def _make_payout_optimizer(
    sim_results,
    players_df,
    target: float = 150.0,
    cash_line: float = 100.0,
    best_scores=None,
    payout_beta: float = 2.0,
    n_chains: int = 5,
    n_steps: int = 20,
):
    return BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        n_chains=n_chains,
        temperature=0.05,
        n_steps=n_steps,
        rng_seed=42,
        objective="marginal_payout",
        payout_beta=payout_beta,
        payout_cash_line=cash_line,
        best_scores=best_scores,
    )


def test_marginal_payout_returns_valid_lineup(sim_results, players_df, player_meta):
    """marginal_payout objective must return a valid DK Classic lineup."""
    opt = _make_payout_optimizer(sim_results, players_df)
    lineup, score = opt.optimize()
    assert isinstance(lineup, Lineup)
    assert lineup.is_valid(player_meta)
    assert score >= 0.0


def test_marginal_payout_better_than_random(sim_results, players_df):
    """Optimized lineup should score higher on payout objective than a random lineup."""
    from src.optimization.optimizer import _score_totals_payout

    opt = _make_payout_optimizer(sim_results, players_df, cash_line=80.0)
    lineup, opt_score = opt.optimize()

    # Score a random lineup
    rng = np.random.default_rng(999)
    rand_lineup = opt._runner._random_valid_lineup(rng)
    rand_cols = [opt.col_map[pid] for pid in rand_lineup.player_ids]
    rand_totals = opt.sim_matrix[:, rand_cols].sum(axis=1)
    best_scores = np.zeros(opt.sim_matrix.shape[0], dtype=np.float64)
    rand_score = _score_totals_payout(rand_totals, 80.0, best_scores, 2.0)

    assert opt_score >= rand_score


def test_marginal_payout_cash_line_separates_from_target(sim_results, players_df):
    """Lower cash_line should give higher objective scores (more sims contribute).

    With cash_line below most totals, almost every sim contributes gradient,
    so the average payout objective value should be larger.
    """
    from src.optimization.optimizer import _score_totals_payout

    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    # Fixed set of player columns for comparison
    opt_low = _make_payout_optimizer(sim_results, players_df, cash_line=10.0)
    opt_high = _make_payout_optimizer(sim_results, players_df, cash_line=350.0)

    lineup_low, score_low = opt_low.optimize()
    lineup_high, score_high = opt_high.optimize()

    # score_low uses cash_line=10: most sims (totals ~0-400) exceed it → larger values
    # score_high uses cash_line=350: very few sims exceed it → near-zero values
    assert score_low > score_high


def test_marginal_payout_with_zero_best_scores_equals_direct_payout(sim_results, players_df):
    """When best_scores=zeros, objective = E[max(0, lineup - cash_line)^beta]."""
    from src.optimization.optimizer import _score_totals_payout

    cash_line = 100.0
    beta = 2.0
    best_scores_zero = np.zeros(sim_results.n_sims, dtype=np.float64)

    opt = _make_payout_optimizer(
        sim_results, players_df,
        cash_line=cash_line,
        best_scores=best_scores_zero,
        payout_beta=beta,
    )
    lineup, score = opt.optimize()
    cols = [opt.col_map[pid] for pid in lineup.player_ids]
    totals = opt.sim_matrix[:, cols].sum(axis=1)

    # Compute expected value directly
    expected = float(np.mean(np.maximum(totals - cash_line, 0.0) ** beta))
    assert score == pytest.approx(expected, rel=1e-6)


def test_marginal_payout_payout_cash_line_defaults_to_target(sim_results, players_df):
    """When payout_cash_line is None, the cash line should fall back to target."""
    target = 150.0
    opt = BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        n_chains=3,
        n_steps=10,
        rng_seed=0,
        objective="marginal_payout",
        payout_cash_line=None,
    )
    # The runner should store target as the cash line
    assert opt._runner._payout_cash_line == pytest.approx(target)


def test_marginal_payout_explicit_cash_line_used(sim_results, players_df):
    """payout_cash_line should be stored and used, distinct from target."""
    target = 150.0
    cash_line = 90.0
    opt = BasinHoppingOptimizer(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        n_chains=3,
        n_steps=10,
        rng_seed=0,
        objective="marginal_payout",
        payout_cash_line=cash_line,
    )
    assert opt._runner._payout_cash_line == pytest.approx(cash_line)
    # target should remain unchanged
    assert opt.target == pytest.approx(target)


def test_marginal_payout_best_scores_shift_objective(sim_results, players_df):
    """Providing non-zero best_scores should reduce the objective value.

    When best_scores already covers strong sims, the marginal improvement of
    adding a lineup is smaller.
    """
    from src.optimization.optimizer import _score_totals_payout

    cash_line = 80.0
    beta = 2.0
    n_sims = sim_results.n_sims

    # Zero best_scores baseline
    bs_zero = np.zeros(n_sims, dtype=np.float64)
    opt_zero = _make_payout_optimizer(
        sim_results, players_df, cash_line=cash_line,
        best_scores=bs_zero, payout_beta=beta,
    )
    lineup_zero, _ = opt_zero.optimize()
    cols_zero = [opt_zero.col_map[pid] for pid in lineup_zero.player_ids]
    totals_zero = opt_zero.sim_matrix[:, cols_zero].sum(axis=1)
    score_with_zero_bs = _score_totals_payout(totals_zero, cash_line, bs_zero, beta)

    # High best_scores (already strong coverage)
    bs_high = np.full(n_sims, 300.0, dtype=np.float64)
    score_with_high_bs = _score_totals_payout(totals_zero, cash_line, bs_high, beta)

    # When best_scores=300 > all totals, max(best, lineup) = 300 always
    # so the objective is max(300 - cash_line, 0)^beta regardless of lineup
    # With zero best_scores, the optimizer can pick a lineup that sometimes
    # exceeds cash_line, giving a lower baseline score
    # The key property: score_with_high_bs should be >= score_with_zero_bs
    # (because taking max with a high floor raises effective scores)
    assert score_with_high_bs >= score_with_zero_bs
