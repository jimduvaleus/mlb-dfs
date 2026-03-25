import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.process_historical import assign_slots


@pytest.fixture
def mock_batting_df():
    """
    Batting rows as produced by _prep_batting_df + process_batting_stats:
    columns include player_id, game_id, team_id, slot, dk_points.
    Single game ATL202306010: NYY is the away team, ATL is the home team.
    """
    return pd.DataFrame({
        "player_id": ["nyy_bat1", "nyy_bat2", "atl_bat1", "atl_bat2"],
        "game_id":   ["ATL202306010"] * 4,
        "team_id":   ["NYY", "NYY", "ATL", "ATL"],
        "slot":      [1, 2, 1, 2],
        "dk_points": [5.0, 2.0, 7.0, 14.0],
    })


@pytest.fixture
def mock_pitching_df():
    """
    Pitching rows as produced by _prep_pitching_df + process_pitching_stats:
    columns include player_id, game_id, pit_team_id, GS, dk_points.
    """
    return pd.DataFrame({
        "player_id":   ["atl_pit1", "nyy_pit1"],
        "game_id":     ["ATL202306010", "ATL202306010"],
        "pit_team_id": ["ATL", "NYY"],
        "GS":          [1, 1],
        "dk_points":   [30.15, 8.0],
    })


def test_assign_slots_row_count(mock_batting_df, mock_pitching_df):
    """4 batters + 2 pitcher-slot rows = 6 total rows."""
    result = assign_slots(mock_batting_df, mock_pitching_df)
    assert len(result) == 6


def test_assign_slots_columns(mock_batting_df, mock_pitching_df):
    result = assign_slots(mock_batting_df, mock_pitching_df)
    assert set(result.columns) == {"game_id", "team_id", "player_id", "slot", "dk_points"}


def test_assign_slots_batter_slots(mock_batting_df, mock_pitching_df):
    """Each batter should carry their batting order slot (1 or 2)."""
    result = assign_slots(mock_batting_df, mock_pitching_df)
    batters = result[result["slot"] != 10]

    nyy_bat1 = batters[batters["player_id"] == "nyy_bat1"].iloc[0]
    assert nyy_bat1["slot"] == 1
    assert nyy_bat1["team_id"] == "NYY"

    atl_bat2 = batters[batters["player_id"] == "atl_bat2"].iloc[0]
    assert atl_bat2["slot"] == 2
    assert atl_bat2["team_id"] == "ATL"


def test_assign_slots_pitcher_assigned_to_opposing_team(mock_batting_df, mock_pitching_df):
    """
    Slot 10 should contain the *opposing* starter.
    - NYY batting group → faces ATL starter (atl_pit1), dk_points = 30.15
    - ATL batting group → faces NYY starter (nyy_pit1), dk_points = 8.0
    """
    result = assign_slots(mock_batting_df, mock_pitching_df)
    pitchers = result[result["slot"] == 10]
    assert len(pitchers) == 2

    nyy_side = pitchers[pitchers["team_id"] == "NYY"].iloc[0]
    assert nyy_side["player_id"] == "atl_pit1"
    assert nyy_side["dk_points"] == pytest.approx(30.15)

    atl_side = pitchers[pitchers["team_id"] == "ATL"].iloc[0]
    assert atl_side["player_id"] == "nyy_pit1"
    assert atl_side["dk_points"] == pytest.approx(8.0)


def test_assign_slots_no_pitcher_on_own_team(mock_batting_df, mock_pitching_df):
    """A pitcher must never appear in slot 10 for their own team's batting group."""
    result = assign_slots(mock_batting_df, mock_pitching_df)
    pitchers = result[result["slot"] == 10]

    # atl_pit1 should only appear under NYY (not ATL)
    assert pitchers[
        (pitchers["player_id"] == "atl_pit1") & (pitchers["team_id"] == "ATL")
    ].empty

    # nyy_pit1 should only appear under ATL (not NYY)
    assert pitchers[
        (pitchers["player_id"] == "nyy_pit1") & (pitchers["team_id"] == "NYY")
    ].empty
