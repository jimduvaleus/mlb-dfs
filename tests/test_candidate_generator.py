"""Tests for Phase 1: CandidateGenerator."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.lineup import Lineup
from src.optimization.ownership import compute_heuristic_ownership


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

def _make_player(pid, pos, salary, team, game, mean=20.0):
    return {
        "player_id": pid,
        "name": f"P{pid}",
        "position": pos,
        "salary": salary,
        "team": team,
        "game": game,
        "mean": mean,
        "std_dev": 5.0,
        "slot": 9 if pos == "P" else 1,
        "opponent": game.split("@")[1] if team == game.split("@")[0] else game.split("@")[0],
    }


@pytest.fixture
def players_df():
    """
    Three-game slate with enough depth to produce 100+ unique stacked lineups.

    Games: "A@B", "C@D", "E@F"
    Away teams (A, C, E): 7-8 batters across all positions.
    Home teams (B, D, F): 2 pitchers + 2-3 batters.
    This gives multiple valid primary-stack teams and ample pitcher / fill options.
    """
    rows = [
        # --- Game A@B ---
        # Pitchers for B and A
        _make_player(1,  "P",  8000, "B", "A@B", mean=25.0),
        _make_player(3,  "P",  7000, "B", "A@B", mean=22.0),
        _make_player(20, "P",  8500, "A", "A@B", mean=21.0),
        # Team A batters (7)
        _make_player(5,  "C",  4000, "A", "A@B", mean=18.0),
        _make_player(7,  "1B", 4200, "A", "A@B", mean=19.0),
        _make_player(9,  "2B", 4100, "A", "A@B", mean=17.0),
        _make_player(11, "3B", 3900, "A", "A@B", mean=16.0),
        _make_player(13, "SS", 3800, "A", "A@B", mean=15.0),
        _make_player(15, "OF", 4000, "A", "A@B", mean=20.0),
        _make_player(19, "OF", 3600, "A", "A@B", mean=18.0),
        # Team B batters (3)
        _make_player(16, "OF", 4000, "B", "A@B", mean=18.0),
        _make_player(30, "1B", 3800, "B", "A@B", mean=17.0),
        _make_player(31, "3B", 3700, "B", "A@B", mean=16.0),

        # --- Game C@D ---
        # Pitchers for D and C
        _make_player(2,  "P",  7500, "D", "C@D", mean=24.0),
        _make_player(4,  "P",  6500, "D", "C@D", mean=21.0),
        _make_player(21, "P",  7500, "C", "C@D", mean=20.0),
        # Team C batters (8)
        _make_player(6,  "C",  3800, "C", "C@D", mean=18.0),
        _make_player(8,  "1B", 3800, "C", "C@D", mean=17.0),
        _make_player(10, "2B", 3800, "C", "C@D", mean=16.0),
        _make_player(12, "3B", 3800, "C", "C@D", mean=15.0),
        _make_player(14, "SS", 3800, "C", "C@D", mean=14.0),
        _make_player(17, "OF", 3800, "C", "C@D", mean=19.0),
        _make_player(22, "OF", 3800, "C", "C@D", mean=17.0),
        _make_player(23, "OF", 3600, "C", "C@D", mean=16.0),
        # Team D batters (3)
        _make_player(18, "OF", 3800, "D", "C@D", mean=18.0),
        _make_player(32, "2B", 3700, "D", "C@D", mean=16.0),
        _make_player(33, "SS", 3600, "D", "C@D", mean=15.0),

        # --- Game E@F ---
        # Pitchers for F and E
        _make_player(40, "P",  8000, "F", "E@F", mean=23.0),
        _make_player(41, "P",  7000, "F", "E@F", mean=21.0),
        _make_player(42, "P",  7500, "E", "E@F", mean=20.0),
        # Team E batters (7)
        _make_player(50, "C",  4100, "E", "E@F", mean=18.0),
        _make_player(51, "1B", 4000, "E", "E@F", mean=19.0),
        _make_player(52, "2B", 3900, "E", "E@F", mean=17.0),
        _make_player(53, "3B", 3800, "E", "E@F", mean=16.0),
        _make_player(54, "SS", 3700, "E", "E@F", mean=15.0),
        _make_player(55, "OF", 4000, "E", "E@F", mean=20.0),
        _make_player(56, "OF", 3800, "E", "E@F", mean=18.0),
        # Team F batters (3)
        _make_player(57, "OF", 3900, "F", "E@F", mean=19.0),
        _make_player(58, "C",  3700, "F", "E@F", mean=16.0),
        _make_player(59, "SS", 3600, "F", "E@F", mean=15.0),
    ]
    df = pd.DataFrame(rows)

    def _opponent(row):
        parts = row["game"].split("@")
        return parts[1] if row["team"] == parts[0] else parts[0]

    df["opponent"] = df.apply(_opponent, axis=1)
    return df


@pytest.fixture
def ownership_vec(players_df):
    return compute_heuristic_ownership(players_df)


@pytest.fixture
def generator(players_df, ownership_vec):
    return CandidateGenerator(players_df, ownership_vec, rng_seed=42)


# ------------------------------------------------------------------ #
#  Basic generation tests                                             #
# ------------------------------------------------------------------ #

def test_generate_returns_correct_count(generator):
    results = generator.generate(n_candidates=100)
    assert len(results) == 100


def test_generate_returns_lineup_objects(generator):
    results = generator.generate(n_candidates=10)
    assert all(isinstance(lu, Lineup) for lu in results)
    assert all(len(lu.player_ids) == 10 for lu in results)


def test_generate_deterministic_with_seed(players_df, ownership_vec):
    gen1 = CandidateGenerator(players_df, ownership_vec, rng_seed=99)
    gen2 = CandidateGenerator(players_df, ownership_vec, rng_seed=99)
    r1 = gen1.generate(n_candidates=20)
    r2 = gen2.generate(n_candidates=20)
    assert [lu.player_ids for lu in r1] == [lu.player_ids for lu in r2]


def test_generate_different_seeds_differ(players_df, ownership_vec):
    gen1 = CandidateGenerator(players_df, ownership_vec, rng_seed=1)
    gen2 = CandidateGenerator(players_df, ownership_vec, rng_seed=2)
    r1 = {frozenset(lu.player_ids) for lu in gen1.generate(n_candidates=50)}
    r2 = {frozenset(lu.player_ids) for lu in gen2.generate(n_candidates=50)}
    # Extremely unlikely to be identical
    assert r1 != r2


# ------------------------------------------------------------------ #
#  Validity tests                                                     #
# ------------------------------------------------------------------ #

def test_all_lineups_pass_is_valid(generator, players_df):
    from src.optimization.optimizer import _build_player_meta
    pmeta = _build_player_meta(players_df)
    results = generator.generate(n_candidates=50)
    for i, lu in enumerate(results):
        assert lu.is_valid(pmeta), f"Lineup {i} failed is_valid: {lu.player_ids}"


def test_all_lineups_unique_players(generator):
    results = generator.generate(n_candidates=50)
    for lu in results:
        assert len(set(lu.player_ids)) == 10, f"Duplicate player in {lu.player_ids}"


def test_salary_cap_respected(generator, players_df):
    pmeta = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=50)
    for lu in results:
        total = sum(pmeta[pid]["salary"] for pid in lu.player_ids)
        assert total <= 50_000, f"Salary {total} exceeds cap: {lu.player_ids}"


def test_min_two_games(generator, players_df):
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=50)
    for lu in results:
        games = {df_dict[pid]["game"] for pid in lu.player_ids}
        assert len(games) >= 2, f"Only {len(games)} game(s) in lineup {lu.player_ids}"


def test_no_pitcher_batter_conflict(generator, players_df):
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=50)
    for lu in results:
        pitcher_opps = {
            df_dict[pid]["opponent"]
            for pid in lu.player_ids
            if df_dict[pid]["position"] == "P"
        }
        batter_teams = {
            df_dict[pid]["team"]
            for pid in lu.player_ids
            if df_dict[pid]["position"] != "P"
        }
        assert not pitcher_opps & batter_teams, (
            f"Pitcher-batter conflict: pitchers oppose {pitcher_opps}, "
            f"batters from {batter_teams}; lineup {lu.player_ids}"
        )


def test_exactly_two_pitchers(generator, players_df):
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=50)
    for lu in results:
        n_pitchers = sum(1 for pid in lu.player_ids if df_dict[pid]["position"] == "P")
        assert n_pitchers == 2, f"Expected 2 pitchers, got {n_pitchers}"


# ------------------------------------------------------------------ #
#  Stacking tests                                                     #
# ------------------------------------------------------------------ #

def test_stacking_requirement_met(generator, players_df):
    """All lineups must satisfy the GPP stacking rule."""
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=100)
    for i, lu in enumerate(results):
        team_counts: dict[str, int] = {}
        for pid in lu.player_ids:
            if df_dict[pid]["position"] != "P":
                t = df_dict[pid]["team"]
                team_counts[t] = team_counts.get(t, 0) + 1
        sorted_counts = sorted(team_counts.values(), reverse=True)
        top = sorted_counts[0]
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0
        ok = top >= 5 or (top + second >= 6 and second >= 2)
        assert ok, (
            f"Lineup {i} fails stacking rule: team_counts={team_counts}, "
            f"player_ids={lu.player_ids}"
        )


def test_max_five_hitters_per_team(generator, players_df):
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = generator.generate(n_candidates=100)
    for lu in results:
        team_counts: dict[str, int] = {}
        for pid in lu.player_ids:
            if df_dict[pid]["position"] != "P":
                t = df_dict[pid]["team"]
                team_counts[t] = team_counts.get(t, 0) + 1
        assert max(team_counts.values()) <= 5, (
            f"Team hitter cap violated: {team_counts}"
        )


# ------------------------------------------------------------------ #
#  Salary floor test                                                  #
# ------------------------------------------------------------------ #

def test_salary_floor_enforced(players_df, ownership_vec):
    gen = CandidateGenerator(
        players_df, ownership_vec, rng_seed=7, salary_floor=44_000.0
    )
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = gen.generate(n_candidates=30)
    for lu in results:
        total = sum(df_dict[pid]["salary"] for pid in lu.player_ids)
        assert total >= 44_000.0, f"Salary {total} below floor"


# ------------------------------------------------------------------ #
#  Error handling                                                     #
# ------------------------------------------------------------------ #

def test_single_game_raises(players_df, ownership_vec):
    single_game_df = players_df[players_df["game"] == "A@B"].copy()
    ow = compute_heuristic_ownership(single_game_df)
    gen = CandidateGenerator(single_game_df, ow, rng_seed=0)
    with pytest.raises(RuntimeError, match="2 games"):
        gen.generate(n_candidates=10)


def test_progress_callback_called(players_df, ownership_vec):
    gen = CandidateGenerator(players_df, ownership_vec, rng_seed=42)
    calls = []
    gen.generate(n_candidates=600, max_attempts_multiplier=20,
                 progress_cb=lambda n: calls.append(n))
    # Should have been called at least once (every 500 lineups)
    assert len(calls) >= 1
    assert calls[0] == 500
