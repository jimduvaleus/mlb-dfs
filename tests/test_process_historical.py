import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.process_historical import assign_slots


@pytest.fixture
def mock_event_file(tmp_path):
    """
    Minimal cwevent CSV for a single game: ATL (home) vs NYY (away).

    Retrosheet game ID format: HHHyyyymmddG — first three chars are the home team.
    game_id "ATL202306010" → home team = ATL, away team = NYY (from AWAY_TEAM_ID).

    BAT_HOME_ID: 0 = visitor is batting (NYY batters, ATL pitcher)
                 1 = home is batting   (ATL batters, NYY pitcher)
    """
    csv_content = (
        "GAME_ID,AWAY_TEAM_ID,BAT_HOME_ID,BAT_ID,BAT_LINEUP_ID,PIT_ID\n"
        "ATL202306010,NYY,0,nyy_bat1,1,atl_pit1\n"
        "ATL202306010,NYY,0,nyy_bat2,2,atl_pit1\n"
        "ATL202306010,NYY,1,atl_bat1,1,nyy_pit1\n"
        "ATL202306010,NYY,1,atl_bat2,2,nyy_pit1\n"
    )
    event_file = tmp_path / "events_2023.csv"
    event_file.write_text(csv_content)
    return str(event_file)


@pytest.fixture
def mock_batting_df():
    """
    Batting stats in cwbox column names, with dk_points already calculated.
    """
    return pd.DataFrame({
        "playerid": ["nyy_bat1", "nyy_bat2", "atl_bat1", "atl_bat2"],
        "gameid":   ["ATL202306010"] * 4,
        "dk_points": [5.0, 2.0, 7.0, 14.0],
    })


@pytest.fixture
def mock_pitching_df():
    """
    Pitching stats in cwbox column names, starters only (GS=1), dk_points calculated.
    pit_team_id is NOT present here — it is derived from the event file inside assign_slots.
    """
    return pd.DataFrame({
        "playerid":  ["atl_pit1", "nyy_pit1"],
        "gameid":    ["ATL202306010", "ATL202306010"],
        "GS":        [1, 1],
        "dk_points": [30.15, 8.0],
    })


def test_assign_slots_row_count(mock_batting_df, mock_pitching_df, mock_event_file):
    """4 batters + 2 pitcher-slot rows = 6 total rows."""
    result = assign_slots(mock_batting_df, mock_pitching_df, mock_event_file)
    assert len(result) == 6


def test_assign_slots_columns(mock_batting_df, mock_pitching_df, mock_event_file):
    result = assign_slots(mock_batting_df, mock_pitching_df, mock_event_file)
    assert set(result.columns) == {"game_id", "team_id", "player_id", "slot", "dk_points"}


def test_assign_slots_batter_slots(mock_batting_df, mock_pitching_df, mock_event_file):
    """Each batter should carry their batting order slot (1 or 2)."""
    result = assign_slots(mock_batting_df, mock_pitching_df, mock_event_file)
    batters = result[result["slot"] != 10]

    nyy_bat1 = batters[batters["player_id"] == "nyy_bat1"].iloc[0]
    assert nyy_bat1["slot"] == 1
    assert nyy_bat1["team_id"] == "NYY"

    atl_bat2 = batters[batters["player_id"] == "atl_bat2"].iloc[0]
    assert atl_bat2["slot"] == 2
    assert atl_bat2["team_id"] == "ATL"


def test_assign_slots_pitcher_assigned_to_opposing_team(mock_batting_df, mock_pitching_df, mock_event_file):
    """
    Slot 10 should contain the *opposing* starter.
    - NYY batting group → faces ATL starter (atl_pit1), dk_points = 30.15
    - ATL batting group → faces NYY starter (nyy_pit1), dk_points = 8.0
    """
    result = assign_slots(mock_batting_df, mock_pitching_df, mock_event_file)
    pitchers = result[result["slot"] == 10]
    assert len(pitchers) == 2

    nyy_side = pitchers[pitchers["team_id"] == "NYY"].iloc[0]
    assert nyy_side["player_id"] == "atl_pit1"
    assert nyy_side["dk_points"] == pytest.approx(30.15)

    atl_side = pitchers[pitchers["team_id"] == "ATL"].iloc[0]
    assert atl_side["player_id"] == "nyy_pit1"
    assert atl_side["dk_points"] == pytest.approx(8.0)


def test_assign_slots_no_pitcher_on_own_team(mock_batting_df, mock_pitching_df, mock_event_file):
    """A pitcher must never appear in slot 10 for their own team's batting group."""
    result = assign_slots(mock_batting_df, mock_pitching_df, mock_event_file)
    pitchers = result[result["slot"] == 10]

    # atl_pit1 should only appear under NYY (not ATL)
    assert pitchers[
        (pitchers["player_id"] == "atl_pit1") & (pitchers["team_id"] == "ATL")
    ].empty

    # nyy_pit1 should only appear under ATL (not NYY)
    assert pitchers[
        (pitchers["player_id"] == "nyy_pit1") & (pitchers["team_id"] == "NYY")
    ].empty
