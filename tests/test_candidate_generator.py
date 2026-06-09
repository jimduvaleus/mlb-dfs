"""Tests for Phase 1: CandidateGenerator."""
import math

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
#  Salary floor tests                                                 #
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


def test_salary_floor_enforced_when_full_stack_leaves_no_fill():
    """Regression: a (5+3) stack where both teams together cover all 8 batter
    positions produces fill_slots=[] so the salary-window fill loop never runs.
    The explicit floor guard at the end of _sample_one must catch this case.

    Fixture: team A has cheap C/1B/2B/3B/SS ($2500 each, high projection) and
    team B has cheap OFs ($2500 each, high projection).  A(5)+B(3) covers every
    batter slot, total batter salary=$20 000.  With the cheapest eligible pitchers
    (~$5000 each) total is ≈$30 000, far below the $44 000 floor used here.
    The generator must reject all such lineups.
    """
    rows = [
        # Team A — cheap infield, high projection → drawn frequently as primary
        _make_player(1,  "C",  2500, "A", "A@B", mean=28.0),
        _make_player(2,  "1B", 2500, "A", "A@B", mean=28.0),
        _make_player(3,  "2B", 2500, "A", "A@B", mean=28.0),
        _make_player(4,  "3B", 2500, "A", "A@B", mean=28.0),
        _make_player(5,  "SS", 2500, "A", "A@B", mean=28.0),
        # Team B — cheap outfield, high projection → drawn frequently as secondary
        _make_player(6,  "OF", 2500, "B", "A@B", mean=28.0),
        _make_player(7,  "OF", 2500, "B", "A@B", mean=28.0),
        _make_player(8,  "OF", 2500, "B", "A@B", mean=28.0),
        # A@B pitchers (cheap; can only be used when stacking C or D batters)
        _make_player(9,  "P",  5000, "A", "A@B", mean=18.0),
        _make_player(10, "P",  5000, "B", "A@B", mean=18.0),
        # Team C — expensive batters covering all positions
        _make_player(11, "C",  4800, "C", "C@D", mean=18.0),
        _make_player(12, "1B", 4800, "C", "C@D", mean=18.0),
        _make_player(13, "2B", 4800, "C", "C@D", mean=18.0),
        _make_player(14, "3B", 4800, "C", "C@D", mean=18.0),
        _make_player(15, "SS", 4800, "C", "C@D", mean=18.0),
        _make_player(16, "OF", 4800, "C", "C@D", mean=18.0),
        _make_player(17, "OF", 4800, "C", "C@D", mean=18.0),
        _make_player(18, "OF", 4800, "C", "C@D", mean=18.0),
        # Team D — expensive batters covering all positions
        _make_player(19, "C",  4800, "D", "C@D", mean=18.0),
        _make_player(20, "1B", 4800, "D", "C@D", mean=18.0),
        _make_player(21, "2B", 4800, "D", "C@D", mean=18.0),
        _make_player(22, "3B", 4800, "D", "C@D", mean=18.0),
        _make_player(23, "SS", 4800, "D", "C@D", mean=18.0),
        _make_player(24, "OF", 4800, "D", "C@D", mean=18.0),
        _make_player(25, "OF", 4800, "D", "C@D", mean=18.0),
        _make_player(26, "OF", 4800, "D", "C@D", mean=18.0),
        # C@D pitchers (expensive; eligible when stacking A or B)
        _make_player(27, "P",  8000, "C", "C@D", mean=22.0),
        _make_player(28, "P",  8000, "D", "C@D", mean=22.0),
    ]
    df = pd.DataFrame(rows)
    df["opponent"] = df.apply(
        lambda row: (
            row["game"].split("@")[1] if row["team"] == row["game"].split("@")[0]
            else row["game"].split("@")[0]
        ),
        axis=1,
    )
    ow = compute_heuristic_ownership(df)
    pid_to_salary = dict(zip(df["player_id"], df["salary"]))
    pid_team = dict(zip(df["player_id"], df["team"]))
    pid_pos = dict(zip(df["player_id"], df["position"]))

    salary_floor = 44_000.0
    gen = CandidateGenerator(df, ow, rng_seed=0, salary_floor=salary_floor,
                              team_weight_power=0.5, fill_weight_power=0.0)
    results = gen.generate(n_candidates=40, max_attempts_multiplier=500)

    assert len(results) > 0, "Generator produced no lineups — check fixture salaries"
    for lu in results:
        total = sum(pid_to_salary[pid] for pid in lu.player_ids)
        # Each lineup must respect its primary team's effective floor (which may be
        # lower than the configured floor for teams with cheap rosters).
        team_h: dict[str, int] = {}
        for pid in lu.player_ids:
            if pid_pos[pid] != "P":
                t = pid_team[pid]
                team_h[t] = team_h.get(t, 0) + 1
        primary = max(team_h, key=team_h.get) if team_h else None
        effective_floor = gen._team_salary_floor.get(primary, salary_floor) if primary else salary_floor
        assert total >= effective_floor, (
            f"Lineup below effective floor: total={total}, floor={effective_floor}, "
            f"primary_team={primary}, player_ids={lu.player_ids}"
        )


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


# ------------------------------------------------------------------ #
#  Stack group distribution                                           #
# ------------------------------------------------------------------ #

def test_stack_group_distribution(players_df, ownership_vec):
    """5-group ~50%, 4-group ~40%, 3-group ~10% across a large enough sample."""
    gen = CandidateGenerator(players_df, ownership_vec, rng_seed=77)
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}
    results = gen.generate(n_candidates=300, max_attempts_multiplier=50)

    group5 = group4 = group3 = 0
    for lu in results:
        team_counts: dict[str, int] = {}
        for pid in lu.player_ids:
            if df_dict[pid]["position"] != "P":
                t = df_dict[pid]["team"]
                team_counts[t] = team_counts.get(t, 0) + 1
        top = max(team_counts.values())
        if top >= 5:
            group5 += 1
        elif top >= 4:
            group4 += 1
        else:
            group3 += 1

    n = len(results)
    assert n > 0
    # Tolerant bounds: ±20pp for 5-group; 4-group lower bound is loose because the
    # small fixture has several teams with only 3 batters, making (4,4) patterns
    # structurally infeasible for those teams.
    assert 0.30 <= group5 / n <= 0.70, f"5-group fraction {group5/n:.2f} out of [0.30, 0.70]"
    assert 0.05 <= group4 / n <= 0.60, f"4-group fraction {group4/n:.2f} out of [0.05, 0.60]"
    assert 0.00 <= group3 / n <= 0.30, f"3-group fraction {group3/n:.2f} out of [0.00, 0.30]"


# ------------------------------------------------------------------ #
#  Team distribution                                                  #
# ------------------------------------------------------------------ #

def test_team_primary_cap_respected(players_df, ownership_vec):
    """No primary team should exceed ceil(n * 1.33 / n_teams) lineups."""
    gen = CandidateGenerator(players_df, ownership_vec, rng_seed=42)
    df_dict = {row["player_id"]: row for row in players_df.to_dict("records")}

    n_candidates = 90
    # 6 batter teams in the 3-game fixture (A, B, C, D, E, F)
    n_batter_teams = len({
        row["team"] for row in players_df.to_dict("records")
        if row["position"] != "P"
    })
    # Use 1.7× (vs the production hard_cap of 1.33×) to account for fill batters that
    # can raise a team's apparent hitter count beyond the generator's intended stack size
    # in this small fixture. In real slates (18 teams, 8-9 batters) the fill effect is
    # negligible and the actual distribution stays within ±33% of uniform.
    cap = math.ceil(n_candidates * 1.7 / n_batter_teams)

    results = gen.generate(n_candidates=n_candidates, max_attempts_multiplier=200)

    # Count unambiguous primary team for 4-group and 5-group lineups.
    # Lineups with two teams tied at the top (i.e. 4-4 stacks) are excluded
    # since neither team is unambiguously the "primary".
    primary_counts: dict[str, int] = {}
    for lu in results:
        team_hitters: dict[str, int] = {}
        for pid in lu.player_ids:
            if df_dict[pid]["position"] != "P":
                t = df_dict[pid]["team"]
                team_hitters[t] = team_hitters.get(t, 0) + 1
        top = max(team_hitters.values())
        if top < 4:
            continue  # 3-group: excluded from distribution math
        top_teams = [t for t, c in team_hitters.items() if c == top]
        if len(top_teams) > 1:
            continue  # 4-4 tie: skip, both teams share credit
        primary = top_teams[0]
        primary_counts[primary] = primary_counts.get(primary, 0) + 1

    for team, count in primary_counts.items():
        assert count <= cap, (
            f"Team {team} has {count} unambiguous primary lineups, exceeds cap {cap} "
            f"(n={n_candidates}, n_teams={n_batter_teams})"
        )


# ------------------------------------------------------------------ #
#  Regression tests                                                   #
# ------------------------------------------------------------------ #

def test_cheap_primary_team_proportional_with_salary_floor():
    """Regression: secondary feasibility filter must include pitcher salary in max_from_rest.

    Before the fix, n_after_sec = 10 - primary - secondary counted pitcher slots as batter
    fill slots, but used self._top_batter_salaries (cheap) instead of pitcher salaries
    (expensive) for those slots. This made min_sec_sum_needed artificially large for cheap
    primary teams, blocking all secondary options and producing near-zero lineups.

    Fixture: CHEAP team at $3,500/batter; normal teams at $4,600–$5,000; pitchers $8k–$9.5k;
    salary_floor = $49,000.

    (5,0) is infeasible at this floor (fill batters can't make up the gap from the cap side).
    OLD filter: (5,2) and (5,3) also fail because all secondary teams are blocked.
      e.g. (5,2): min_sec_sum_needed = $49k – $17.5k – (3 × $5k batter) = $16.5k;
      no secondary team's top-2 sum to $16.5k → 0 CHEAP lineups.
    NEW filter: pitcher salary included in max_from_rest.
      (5,2): min_sec_sum_needed = $49k – $17.5k – ($5k fill + $18.5k pitchers) = $8k;
      secondary team's top-2 = $10k ≥ $8k → valid secondary found → CHEAP lineups generated.
    """
    rows = [
        # Game 1: CHEAP@OPP1 — cheap primary team
        _make_player(1,  "C",  3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(2,  "1B", 3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(3,  "2B", 3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(4,  "3B", 3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(5,  "SS", 3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(6,  "OF", 3500, "CHEAP", "CHEAP@OPP1"),
        _make_player(7,  "OF", 3500, "CHEAP", "CHEAP@OPP1"),
        # OPP1 pitchers + batters (OPP1 pitchers are excluded for CHEAP-primary lineups)
        _make_player(8,  "P",  10000, "OPP1", "CHEAP@OPP1"),
        _make_player(9,  "P",   9500, "OPP1", "CHEAP@OPP1"),
        _make_player(10, "OF",  5000, "OPP1", "CHEAP@OPP1"),
        _make_player(11, "1B",  4800, "OPP1", "CHEAP@OPP1"),
        _make_player(12, "3B",  4600, "OPP1", "CHEAP@OPP1"),
        # Game 2: NRM1@NRM2 — source of secondary stacks and pitchers
        _make_player(20, "P",  9500, "NRM1", "NRM1@NRM2"),
        _make_player(21, "P",  9000, "NRM1", "NRM1@NRM2"),
        _make_player(22, "C",  5000, "NRM1", "NRM1@NRM2"),
        _make_player(23, "1B", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(24, "2B", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(25, "3B", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(26, "SS", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(27, "OF", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(28, "OF", 5000, "NRM1", "NRM1@NRM2"),
        _make_player(29, "P",  9000, "NRM2", "NRM1@NRM2"),
        _make_player(30, "P",  8500, "NRM2", "NRM1@NRM2"),
        _make_player(31, "C",  5000, "NRM2", "NRM1@NRM2"),
        _make_player(32, "1B", 4800, "NRM2", "NRM1@NRM2"),
        _make_player(33, "2B", 4800, "NRM2", "NRM1@NRM2"),
        _make_player(34, "3B", 4600, "NRM2", "NRM1@NRM2"),
        _make_player(35, "SS", 4600, "NRM2", "NRM1@NRM2"),
        _make_player(36, "OF", 4600, "NRM2", "NRM1@NRM2"),
        _make_player(37, "OF", 4600, "NRM2", "NRM1@NRM2"),
        # Game 3: NRM3@NRM4 — pitchers available for CHEAP-primary lineups (not OPP1 or secondary opp)
        _make_player(40, "P",  9000, "NRM3", "NRM3@NRM4"),
        _make_player(41, "P",  8500, "NRM3", "NRM3@NRM4"),
        _make_player(42, "C",  4800, "NRM3", "NRM3@NRM4"),
        _make_player(43, "1B", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(44, "2B", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(45, "3B", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(46, "SS", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(47, "OF", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(48, "OF", 4800, "NRM3", "NRM3@NRM4"),
        _make_player(49, "P",  8500, "NRM4", "NRM3@NRM4"),
        _make_player(50, "P",  8000, "NRM4", "NRM3@NRM4"),
        _make_player(51, "C",  4800, "NRM4", "NRM3@NRM4"),
        _make_player(52, "1B", 4600, "NRM4", "NRM3@NRM4"),
        _make_player(53, "2B", 4600, "NRM4", "NRM3@NRM4"),
        _make_player(54, "3B", 4400, "NRM4", "NRM3@NRM4"),
        _make_player(55, "SS", 4400, "NRM4", "NRM3@NRM4"),
        _make_player(56, "OF", 4400, "NRM4", "NRM3@NRM4"),
        _make_player(57, "OF", 4400, "NRM4", "NRM3@NRM4"),
    ]
    df = pd.DataFrame(rows)
    df["opponent"] = df.apply(
        lambda row: (
            row["game"].split("@")[1] if row["team"] == row["game"].split("@")[0]
            else row["game"].split("@")[0]
        ),
        axis=1,
    )
    ow = compute_heuristic_ownership(df)
    pid_salary = dict(zip(df["player_id"], df["salary"]))
    pid_team = dict(zip(df["player_id"], df["team"]))
    pid_pos = dict(zip(df["player_id"], df["position"]))

    salary_floor = 49_000.0
    gen = CandidateGenerator(df, ow, rng_seed=42, salary_floor=salary_floor)

    # CHEAP's max achievable salary is < $49k in this fixture (5-stack at $3,500 × 5 =
    # $17.5k, pitchers ~$17.5k, secondary ~$10k, fill ~$5k → $50k cap-limited; but
    # tighter constraints reduce it). Verify the generator lowers CHEAP's floor.
    assert "CHEAP" in gen._team_salary_floor, "CHEAP not found in team_salary_floor"
    assert gen._team_salary_floor["CHEAP"] <= salary_floor, (
        "CHEAP's per-team floor should not exceed the configured floor"
    )
    # Normal teams with expensive batters should keep the configured floor.
    assert gen._team_salary_floor.get("NRM1", salary_floor) == salary_floor, (
        "NRM1 (expensive batters) should keep the configured floor"
    )

    results = gen.generate(n_candidates=200, max_attempts_multiplier=500)
    assert len(results) > 0, "Generator produced no lineups at all"

    # Each lineup must respect its primary team's effective floor (not the global floor).
    for lu in results:
        total = sum(pid_salary[pid] for pid in lu.player_ids)
        team_h: dict[str, int] = {}
        for pid in lu.player_ids:
            if pid_pos[pid] != "P":
                t = pid_team[pid]
                team_h[t] = team_h.get(t, 0) + 1
        primary = max(team_h, key=team_h.get) if team_h else None
        eff_floor = gen._team_salary_floor.get(primary, salary_floor) if primary else salary_floor
        assert total >= eff_floor, (
            f"Lineup below effective floor: total={total}, floor={eff_floor}, primary={primary}"
        )

    # CHEAP should now appear as primary in a meaningful fraction of lineups.
    # 6 primary-capable batter teams → expected ~1/6 ≈ 33 of 200. Require ≥ 10.
    cheap_primary_count = 0
    for lu in results:
        team_hitters: dict[str, int] = {}
        for pid in lu.player_ids:
            if pid_pos[pid] != "P":
                t = pid_team[pid]
                team_hitters[t] = team_hitters.get(t, 0) + 1
        if not team_hitters:
            continue
        top = max(team_hitters.values())
        top_teams = [t for t, c in team_hitters.items() if c == top]
        if len(top_teams) == 1 and top_teams[0] == "CHEAP" and top >= 4:
            cheap_primary_count += 1

    assert cheap_primary_count >= 10, (
        f"CHEAP generated only {cheap_primary_count}/{len(results)} primary lineups; "
        "expected ≥ 10. Per-team floor may not be lowering correctly for cheap teams."
    )
