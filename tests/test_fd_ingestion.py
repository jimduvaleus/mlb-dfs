"""
Tests for FanDuel slate ingestion (Phase 3).

Coverage:
- FanDuelSlateIngestor: column renaming, salary parsing, position parsing
  (single and compound positions), game field, team/opponent, player_id
  extraction, fd_player_id preservation, roster_position passthrough
- BaseSlateIngestor ABC enforcement
- factory.get_ingestor dispatches correctly
- factory.find_fd_slate: date-based selection, latest-wins, no-file case
"""

import csv
import io
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.dk_slate import BaseSlateIngestor, DraftKingsSlateIngestor, Player
from src.ingestion.factory import find_fd_slate, get_ingestor
from src.ingestion.fd_slate import FanDuelSlateIngestor
from src.platforms.base import Platform


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_fd_csv(players: list[dict], entries: list[dict] | None = None) -> str:
    """
    Build a minimal but structurally accurate FanDuel salary CSV string.

    Layout mirrors the real FD export:
      Row 0  : upload-template header (entry_id … UTIL, empty, Instructions)
      Row 1-N: entry rows (or padding) in cols 0-12; instructions in col 14
      Row 6  : player-pool header in col 14+  (always at index 6)
      Row 7+ : player data in col 14+

    The left section (cols 0-12) and right section (col 14+) are written into
    the same CSV side-by-side, separated by an empty column 13.
    """
    PLAYER_COL_OFFSET = 14  # player section starts at column index 14

    # Player-pool column headers (col 14+)
    player_headers = [
        "Player ID + Player Name", "Id", "Position",
        "First Name", "Nickname", "Last Name",
        "FPPG", "Played", "Salary", "Game",
        "Team", "Opponent",
        "Injury Indicator", "Injury Details", "Tier",
        "Probable Pitcher", "Batting Order", "Roster Position",
    ]

    # Upload-template column headers (cols 0-12)
    upload_headers = [
        "entry_id", "contest_id", "contest_name", "entry_fee",
        "P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL",
    ]

    # Pad left section to 13 cols + 1 empty separator = 14 cols before player data
    def _left_pad(cols: list, width: int = PLAYER_COL_OFFSET) -> list:
        return (cols + [""] * width)[:width]

    rows: list[list] = []

    # Row 0: upload header + empty + "Instructions"
    rows.append(_left_pad(upload_headers) + ["Instructions"])

    # Rows 1-5: instruction text in col 14; entries (if any) in cols 0-12
    instructions = [
        "1) Edit your lineup",
        "2) Paste Player ID + Name or ID",
        "3) Don't leave blank rows",
        "4) You can not create new entries",
        "",  # row 5 is blank instruction
    ]
    for i, instr in enumerate(instructions):
        left = []
        if entries and i < len(entries):
            e = entries[i]
            left = [
                e.get("entry_id", ""), e.get("contest_id", ""),
                e.get("contest_name", ""), e.get("entry_fee", ""),
                e.get("P", ""), e.get("C/1B", ""), e.get("2B", ""),
                e.get("3B", ""), e.get("SS", ""),
                e.get("OF1", ""), e.get("OF2", ""), e.get("OF3", ""),
                e.get("UTIL", ""),
            ]
        rows.append(_left_pad(left) + [instr])

    # Row 6: player-pool header (this is where FanDuelSlateIngestor looks)
    rows.append(_left_pad([]) + player_headers)

    # Rows 7+: one row per player
    for p in players:
        fd_id = p["fd_player_id"]
        name = p["name"]
        pid_name = f"{fd_id}:{name}"
        player_row = [
            pid_name,
            fd_id,
            p["position"],
            p.get("first_name", name.split()[0]),
            name,
            p.get("last_name", name.split()[-1]),
            str(p.get("fppg", "20.0")),
            str(p.get("played", "10")),
            str(p["salary"]),
            p["game"],
            p["team"],
            p["opponent"],
            p.get("injury_indicator", ""),
            p.get("injury_details", ""),
            "",  # Tier
            "",  # Probable Pitcher
            str(p.get("batting_order", "")),
            p["roster_position"],
        ]
        rows.append(_left_pad([]) + player_row)

    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _write_fd_csv(tmp_path: Path, filename: str, players: list[dict],
                  entries: list[dict] | None = None) -> Path:
    p = tmp_path / filename
    p.write_text(_make_fd_csv(players, entries))
    return p


# Minimal player pool: pitcher + several batters covering all FD position types
SAMPLE_PLAYERS = [
    {
        "fd_player_id": "128874-10001",
        "name": "Zack Wheeler",
        "position": "P",
        "salary": 11000,
        "game": "ARI@PHI",
        "team": "PHI",
        "opponent": "ARI",
        "roster_position": "P",
    },
    {
        "fd_player_id": "128874-20001",
        "name": "Aaron Judge",
        "position": "OF",
        "salary": 4300,
        "game": "NYY@TB",
        "team": "NYY",
        "opponent": "TB",
        "roster_position": "OF/UTIL",
    },
    {
        "fd_player_id": "128874-30001",
        "name": "CJ Abrams",
        "position": "SS",
        "salary": 4200,
        "game": "WSH@MIL",
        "team": "WSH",
        "opponent": "MIL",
        "roster_position": "SS/UTIL",
    },
    {
        "fd_player_id": "128874-40001",
        "name": "Ben Rice",
        "position": "C/1B",      # compound position
        "salary": 4000,
        "game": "NYY@TB",
        "team": "NYY",
        "opponent": "TB",
        "roster_position": "C/1B/UTIL",
    },
    {
        "fd_player_id": "128874-50001",
        "name": "Yandy Diaz",
        "position": "1B",
        "salary": 3900,
        "game": "NYY@TB",
        "team": "TB",
        "opponent": "NYY",
        "roster_position": "C/1B/UTIL",
    },
    {
        "fd_player_id": "128874-60001",
        "name": "Brice Turang",
        "position": "2B",
        "salary": 3900,
        "game": "WSH@MIL",
        "team": "MIL",
        "opponent": "WSH",
        "roster_position": "2B/UTIL",
    },
    {
        "fd_player_id": "128874-70001",
        "name": "Jose Ramirez",
        "position": "3B",
        "salary": 4500,
        "game": "CLE@MIN",
        "team": "CLE",
        "opponent": "MIN",
        "roster_position": "3B/UTIL",
    },
]


# ---------------------------------------------------------------------------
# BaseSlateIngestor ABC
# ---------------------------------------------------------------------------

class TestBaseSlateIngestorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSlateIngestor()

    def test_concrete_subclass_without_abstract_methods_raises(self):
        class Incomplete(BaseSlateIngestor):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_with_all_methods_is_valid(self):
        class Minimal(BaseSlateIngestor):
            def get_slate_dataframe(self):
                return pd.DataFrame()
            def get_players(self):
                return []
        assert isinstance(Minimal(), BaseSlateIngestor)


# ---------------------------------------------------------------------------
# FanDuelSlateIngestor — parsing
# ---------------------------------------------------------------------------

@pytest.fixture()
def fd_csv_path(tmp_path):
    return _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", SAMPLE_PLAYERS)


@pytest.fixture()
def fd_df(fd_csv_path):
    return FanDuelSlateIngestor(str(fd_csv_path)).get_slate_dataframe()


class TestFDIngestorSchema:
    def test_required_columns_present(self, fd_df):
        required = {"player_id", "fd_player_id", "name", "position",
                    "eligible_positions", "roster_position",
                    "salary", "team", "opponent", "game"}
        assert required.issubset(set(fd_df.columns))

    def test_row_count(self, fd_df):
        assert len(fd_df) == len(SAMPLE_PLAYERS)

    def test_player_id_is_int(self, fd_df):
        assert fd_df["player_id"].dtype == int

    def test_salary_is_numeric(self, fd_df):
        assert pd.api.types.is_numeric_dtype(fd_df["salary"])


class TestFDIngestorPlayerIdExtraction:
    def test_player_id_is_numeric_suffix(self, fd_df):
        # "128874-10001" → 10001
        pitcher = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert pitcher["player_id"] == 10001

    def test_fd_player_id_preserved(self, fd_df):
        pitcher = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert pitcher["fd_player_id"] == "128874-10001"


class TestFDIngestorPositionParsing:
    def test_single_position(self, fd_df):
        row = fd_df[fd_df["name"] == "Aaron Judge"].iloc[0]
        assert row["position"] == "OF"
        assert row["eligible_positions"] == ["OF"]

    def test_compound_position_c1b(self, fd_df):
        """Player listed as 'C/1B' should have both positions in eligible_positions."""
        row = fd_df[fd_df["name"] == "Ben Rice"].iloc[0]
        assert row["position"] == "C"
        assert "1B" in row["eligible_positions"]
        assert "C" in row["eligible_positions"]

    def test_pitcher_position(self, fd_df):
        row = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert row["position"] == "P"
        assert row["eligible_positions"] == ["P"]

    def test_all_positions_valid(self, fd_df):
        valid = {"P", "C", "1B", "2B", "3B", "SS", "OF"}
        assert set(fd_df["position"].unique()).issubset(valid)


class TestFDIngestorRosterPosition:
    def test_pitcher_roster_position(self, fd_df):
        row = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert row["roster_position"] == "P"

    def test_of_roster_position(self, fd_df):
        row = fd_df[fd_df["name"] == "Aaron Judge"].iloc[0]
        assert row["roster_position"] == "OF/UTIL"

    def test_c1b_roster_position(self, fd_df):
        row = fd_df[fd_df["name"] == "Ben Rice"].iloc[0]
        assert row["roster_position"] == "C/1B/UTIL"

    def test_1b_fills_c1b_slot(self, fd_df):
        """A 1B player should also have a C/1B roster_position."""
        row = fd_df[fd_df["name"] == "Yandy Diaz"].iloc[0]
        assert row["roster_position"] == "C/1B/UTIL"


class TestFDIngestorSalaryParsing:
    def test_salary_values_correct(self, fd_df):
        pitcher = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert pitcher["salary"] == 11000.0

    def test_invalid_salary_raises(self, tmp_path):
        bad_players = [{**SAMPLE_PLAYERS[0], "salary": "N/A"}]
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", bad_players)
        with pytest.raises(ValueError, match="salaries"):
            FanDuelSlateIngestor(str(path))


class TestFDIngestorGameTeamOpponent:
    def test_game_field(self, fd_df):
        row = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert row["game"] == "ARI@PHI"

    def test_team_field(self, fd_df):
        row = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert row["team"] == "PHI"

    def test_opponent_field(self, fd_df):
        row = fd_df[fd_df["name"] == "Zack Wheeler"].iloc[0]
        assert row["opponent"] == "ARI"


class TestFDIngestorGetPlayers:
    def test_returns_player_objects(self, fd_csv_path):
        players = FanDuelSlateIngestor(str(fd_csv_path)).get_players()
        assert all(isinstance(p, Player) for p in players)

    def test_player_count(self, fd_csv_path):
        players = FanDuelSlateIngestor(str(fd_csv_path)).get_players()
        assert len(players) == len(SAMPLE_PLAYERS)

    def test_player_fd_player_id(self, fd_csv_path):
        players = FanDuelSlateIngestor(str(fd_csv_path)).get_players()
        pitcher = next(p for p in players if p.name == "Zack Wheeler")
        assert pitcher.fd_player_id == "128874-10001"

    def test_player_opponent(self, fd_csv_path):
        players = FanDuelSlateIngestor(str(fd_csv_path)).get_players()
        pitcher = next(p for p in players if p.name == "Zack Wheeler")
        assert pitcher.opponent == "ARI"


class TestFDIngestorBadFile:
    def test_missing_player_header_raises(self, tmp_path):
        """A CSV with no 'Player ID + Player Name' sentinel should raise ValueError."""
        path = tmp_path / "bad.csv"
        path.write_text("col1,col2\nval1,val2\n")
        with pytest.raises(ValueError, match="player-pool header"):
            FanDuelSlateIngestor(str(path))


# ---------------------------------------------------------------------------
# factory.find_fd_slate
# ---------------------------------------------------------------------------

class TestFindFdSlate:
    def test_returns_none_when_no_files(self, tmp_path):
        assert find_fd_slate(str(tmp_path)) is None

    def test_returns_single_file(self, tmp_path):
        fname = "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        (tmp_path / fname).write_text("x")
        result = find_fd_slate(str(tmp_path))
        assert result == str(tmp_path / fname)

    def test_returns_most_recent_when_multiple(self, tmp_path):
        files = [
            "FanDuel-MLB-2026-04-10-111111-entries-upload-template.csv",
            "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv",
            "FanDuel-MLB-2026-04-11-222222-entries-upload-template.csv",
        ]
        for f in files:
            (tmp_path / f).write_text("x")
        result = find_fd_slate(str(tmp_path))
        assert os.path.basename(result) == "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "DKSalaries.csv").write_text("x")
        (tmp_path / "other.txt").write_text("x")
        assert find_fd_slate(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# factory.get_ingestor dispatch
# ---------------------------------------------------------------------------

class TestGetIngestor:
    def test_draftkings_returns_dk_ingestor(self, tmp_path):
        # Just check the type; DK ingestor won't be constructed with a fake path
        # so we verify the dispatch only via isinstance check on a real DK file.
        # Use a minimal but structurally valid DK CSV.
        dk_csv = textwrap.dedent("""\
            Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame
            SP,Zack Wheeler (12345),Zack Wheeler,12345,P,10000,PHI@ARI 04/12/2026 01:00PM ET,PHI,30.0
        """)
        path = tmp_path / "DKSalaries.csv"
        path.write_text(dk_csv)
        ingestor = get_ingestor(Platform.DRAFTKINGS, str(path))
        assert isinstance(ingestor, DraftKingsSlateIngestor)

    def test_fanduel_returns_fd_ingestor(self, tmp_path):
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", SAMPLE_PLAYERS)
        ingestor = get_ingestor(Platform.FANDUEL, str(path))
        assert isinstance(ingestor, FanDuelSlateIngestor)

    def test_fanduel_auto_discovers_when_no_path(self, tmp_path):
        fd_path = _write_fd_csv(
            tmp_path,
            "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv",
            SAMPLE_PLAYERS,
        )
        # Patch find_fd_slate to return our temp file when called with no args.
        with patch("src.ingestion.factory.find_fd_slate", return_value=str(fd_path)):
            ingestor = get_ingestor(Platform.FANDUEL, "")
        assert isinstance(ingestor, FanDuelSlateIngestor)

    def test_fanduel_raises_when_no_file_and_no_path(self, tmp_path):
        with patch("src.ingestion.factory.find_fd_slate", return_value=None):
            with pytest.raises(FileNotFoundError, match="FanDuel"):
                get_ingestor(Platform.FANDUEL, "")

    def test_unknown_platform_raises(self, tmp_path):
        with pytest.raises((ValueError, AttributeError)):
            get_ingestor("unknown_platform", "")


# ---------------------------------------------------------------------------
# Integration: parse the real FD CSV from data/raw/ (skipped if absent)
# ---------------------------------------------------------------------------

_REAL_FD_DIR = "data/raw"


@pytest.mark.skipif(
    find_fd_slate(_REAL_FD_DIR) is None,
    reason="No real FanDuel CSV present in data/raw/ — skipping integration test",
)
class TestRealFDCSVIntegration:
    @pytest.fixture(scope="class")
    def real_df(self):
        path = find_fd_slate(_REAL_FD_DIR)
        return FanDuelSlateIngestor(path).get_slate_dataframe()

    def test_has_players(self, real_df):
        assert len(real_df) > 0

    def test_no_null_player_ids(self, real_df):
        assert real_df["player_id"].notna().all()

    def test_no_null_salaries(self, real_df):
        assert real_df["salary"].notna().all()

    def test_positions_are_valid(self, real_df):
        valid = {"P", "C", "1B", "2B", "3B", "SS", "OF"}
        assert set(real_df["position"].unique()).issubset(valid)

    def test_games_not_empty(self, real_df):
        assert (real_df["game"] != "").all()
