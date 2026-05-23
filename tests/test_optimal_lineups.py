"""Tests for generate_optimal_lineups (shared ILP solver)."""
import pytest
import pandas as pd

from src.optimization.lineup import Lineup
from src.optimization.optimal_lineups import generate_optimal_lineups


# ------------------------------------------------------------------ #
#  Fixture                                                             #
# ------------------------------------------------------------------ #

def _make_player(pid, pos, salary, team, game, mean=20.0, eligible_positions=None):
    opp = game.split("@")[1] if team == game.split("@")[0] else game.split("@")[0]
    return {
        "player_id": pid,
        "name": f"P{pid}",
        "position": pos,
        "eligible_positions": eligible_positions or [pos],
        "salary": salary,
        "team": team,
        "game": game,
        "mean": mean,
        "std_dev": 5.0,
        "slot": 10 if pos == "P" else 1,
        "opponent": opp,
    }


@pytest.fixture
def players_df():
    """
    Three-game slate with enough depth to produce 20+ valid optimal lineups.

    Games: A@B, C@D, E@F
    Stack candidates: teams A, C, E (each has 7-8 batters).
    """
    rows = [
        # --- Game A@B ---
        _make_player(1,  "P",  8000, "B", "A@B", mean=25.0),
        _make_player(3,  "P",  7000, "B", "A@B", mean=22.0),
        _make_player(20, "P",  8500, "A", "A@B", mean=21.0),
        _make_player(5,  "C",  4000, "A", "A@B", mean=18.0),
        _make_player(7,  "1B", 4200, "A", "A@B", mean=19.0),
        _make_player(9,  "2B", 4100, "A", "A@B", mean=17.0),
        _make_player(11, "3B", 3900, "A", "A@B", mean=16.0),
        _make_player(13, "SS", 3800, "A", "A@B", mean=15.0),
        _make_player(15, "OF", 4000, "A", "A@B", mean=20.0),
        _make_player(19, "OF", 3600, "A", "A@B", mean=18.0),
        _make_player(16, "OF", 4000, "B", "A@B", mean=18.0),
        _make_player(30, "1B", 3800, "B", "A@B", mean=17.0),
        _make_player(31, "3B", 3700, "B", "A@B", mean=16.0),

        # --- Game C@D ---
        _make_player(2,  "P",  7500, "D", "C@D", mean=24.0),
        _make_player(4,  "P",  6500, "D", "C@D", mean=21.0),
        _make_player(21, "P",  7500, "C", "C@D", mean=20.0),
        _make_player(6,  "C",  3800, "C", "C@D", mean=18.0),
        _make_player(8,  "1B", 3800, "C", "C@D", mean=17.0),
        _make_player(10, "2B", 3800, "C", "C@D", mean=16.0),
        _make_player(12, "3B", 3800, "C", "C@D", mean=15.0),
        _make_player(14, "SS", 3800, "C", "C@D", mean=14.0),
        _make_player(17, "OF", 3800, "C", "C@D", mean=19.0),
        _make_player(22, "OF", 3800, "C", "C@D", mean=17.0),
        _make_player(23, "OF", 3600, "C", "C@D", mean=16.0),
        _make_player(18, "OF", 3800, "D", "C@D", mean=18.0),
        _make_player(32, "2B", 3700, "D", "C@D", mean=16.0),
        _make_player(33, "SS", 3600, "D", "C@D", mean=15.0),

        # --- Game E@F ---
        _make_player(40, "P",  8000, "F", "E@F", mean=23.0),
        _make_player(41, "P",  7200, "F", "E@F", mean=20.0),
        _make_player(42, "P",  7000, "E", "E@F", mean=19.0),
        _make_player(50, "C",  4000, "E", "E@F", mean=17.0),
        _make_player(51, "1B", 3900, "E", "E@F", mean=16.0),
        _make_player(52, "2B", 3800, "E", "E@F", mean=15.0),
        _make_player(53, "3B", 3700, "E", "E@F", mean=14.0),
        _make_player(54, "SS", 3600, "E", "E@F", mean=13.0),
        _make_player(55, "OF", 3800, "E", "E@F", mean=17.0),
        _make_player(56, "OF", 3700, "E", "E@F", mean=16.0),
        _make_player(57, "OF", 3600, "F", "E@F", mean=16.0),
        _make_player(58, "1B", 3600, "F", "E@F", mean=15.0),
    ]
    return pd.DataFrame(rows)


def _build_meta(df: pd.DataFrame) -> dict:
    return {
        int(r.player_id): {
            "position": r.position,
            "eligible_positions": list(r.eligible_positions),
            "salary": float(r.salary),
            "team": r.team,
            "opponent": r.opponent,
            "game": r.game,
        }
        for r in df.itertuples(index=False)
    }


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

def test_top5_returns_correct_count(players_df):
    lineups = generate_optimal_lineups(players_df, n=5, min_uniques=3)
    assert len(lineups) == 5


def test_lineups_are_valid(players_df):
    meta = _build_meta(players_df)
    lineups = generate_optimal_lineups(players_df, n=5, min_uniques=3)
    for lu in lineups:
        assert lu.is_valid(meta), f"Invalid lineup: {lu.player_ids}"


def test_scores_nonincreasing(players_df):
    mean_map = {int(r.player_id): float(r.mean) for r in players_df.itertuples(index=False)}
    lineups = generate_optimal_lineups(players_df, n=5, min_uniques=3)
    scores = [sum(mean_map[pid] for pid in lu.player_ids) for lu in lineups]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"


def test_min_uniques_respected(players_df):
    min_u = 3
    lineups = generate_optimal_lineups(players_df, n=5, min_uniques=min_u)
    for i in range(len(lineups) - 1):
        for j in range(i + 1, len(lineups)):
            shared = set(lineups[i].player_ids) & set(lineups[j].player_ids)
            diff = 10 - len(shared)
            assert diff >= min_u, (
                f"Lineups {i} and {j} share {len(shared)} players "
                f"(only {diff} unique, need {min_u})"
            )


def test_deterministic(players_df):
    """Same input produces identical lineups on repeated calls."""
    r1 = generate_optimal_lineups(players_df, n=5, min_uniques=3)
    r2 = generate_optimal_lineups(players_df, n=5, min_uniques=3)
    assert [lu.player_ids for lu in r1] == [lu.player_ids for lu in r2]


def test_progress_cb(players_df):
    counts = []
    generate_optimal_lineups(players_df, n=5, min_uniques=3, progress_cb=counts.append)
    assert counts == [1, 2, 3, 4, 5]


def test_salary_floor_respected(players_df):
    lineups = generate_optimal_lineups(players_df, n=5, min_uniques=3, salary_floor=45000)
    salary_map = {int(r.player_id): float(r.salary) for r in players_df.itertuples(index=False)}
    for lu in lineups:
        total = sum(salary_map[pid] for pid in lu.player_ids)
        assert total >= 45000, f"Lineup salary {total} below floor 45000"
