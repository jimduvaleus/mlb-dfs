"""
Tests for FanDuel entry file handling (Phase 5).

Coverage:
- scan_fd_entry_files: returns most-recent file, empty list when none found
- parse_fd_entry_file: extracts entries from top portion, skips instruction rows,
  parses fee correctly
- assign_players_to_fd_slots: bipartite matching for all 9 FD slots including
  compound slots (C/1B, UTIL); error on unresolvable assignment
- assign_fd_lineups_to_entries: descending fee order; partial portfolio warning
- write_fd_upload_files: correct 13-column header; fd_player_id values in slots
- get_entry_handlers factory: dispatches to DK and FD handlers correctly
"""

import csv
import io
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.api.fd_entries import (
    FD_UPLOAD_HEADER,
    FD_UPLOAD_SLOTS,
    FDEntryRecord,
    assign_fd_lineups_to_entries,
    assign_players_to_fd_slots,
    parse_fd_entry_file,
    scan_fd_entry_files,
    write_fd_upload_files,
)
from src.api.entries_factory import get_entry_handlers
from src.platforms.base import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fd_csv_with_entries(entries: list[dict]) -> str:
    """
    Build a minimal FanDuel upload-template CSV string containing the given
    entries in the top portion (rows 1-N) and a stub player-pool section.

    entries: list of dicts with keys entry_id, contest_id, contest_name,
             entry_fee, P, C/1B, 2B, 3B, SS, OF1, OF2, OF3, UTIL.
    """
    PLAYER_COL_OFFSET = 14

    def _left_pad(cols: list, width: int = PLAYER_COL_OFFSET) -> list:
        return (cols + [""] * width)[:width]

    rows: list[list] = []

    # Row 0: upload header
    upload_headers = [
        "entry_id", "contest_id", "contest_name", "entry_fee",
        "P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL",
    ]
    rows.append(_left_pad(upload_headers) + ["Instructions"])

    # Rows 1-5: entries (if any) + instruction text in col 14
    instructions = [
        "1) Edit your lineup",
        "2) Paste Player ID + Name or ID",
        "3) Don't leave blank rows",
        "4) You can not create new entries",
        "",
    ]
    for i, instr in enumerate(instructions):
        left = []
        if i < len(entries):
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

    # Row 6: minimal player-pool header (required by FanDuelSlateIngestor sentinel)
    player_headers = [
        "Player ID + Player Name", "Id", "Position",
        "First Name", "Nickname", "Last Name",
        "FPPG", "Played", "Salary", "Game",
        "Team", "Opponent",
        "Injury Indicator", "Injury Details", "Tier",
        "Probable Pitcher", "Batting Order", "Roster Position",
    ]
    rows.append(_left_pad([]) + player_headers)

    # Row 7: one stub player row so the file is non-trivially valid
    rows.append(
        _left_pad([]) + [
            "128874-99999:Stub Player", "128874-99999", "P",
            "Stub", "Stub Player", "Player",
            "20.0", "3", "8000", "TEA@TEB", "TEA", "TEB",
            "", "", "", "", "", "P",
        ]
    )

    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _write_fd_csv(tmp_path: Path, filename: str, entries: list[dict]) -> Path:
    p = tmp_path / filename
    p.write_text(_make_fd_csv_with_entries(entries))
    return p


# A single sample entry
SAMPLE_ENTRY = {
    "entry_id": "3679070205",
    "contest_id": "128874-279459600",
    "contest_name": "$2K Sun MLB Wiffle Ball (150 Entries Max)",
    "entry_fee": "0.05",
    "P":    "128874-10001",
    "C/1B": "128874-40001",
    "2B":   "128874-60001",
    "3B":   "128874-70001",
    "SS":   "128874-30001",
    "OF1":  "128874-20001",
    "OF2":  "128874-20002",
    "OF3":  "128874-20003",
    "UTIL": "128874-50001",
}


# ---------------------------------------------------------------------------
# Minimal slate_df for slot-assignment tests
# ---------------------------------------------------------------------------

def _make_slate_df(player_specs: list[dict]) -> pd.DataFrame:
    """
    Build a minimal slate DataFrame matching FanDuelSlateIngestor output schema.

    Each spec must have: player_id (int), fd_player_id (str),
    position (str, primary), eligible_positions (list[str]).
    """
    rows = []
    for spec in player_specs:
        rows.append({
            "player_id":          int(spec["player_id"]),
            "fd_player_id":       spec["fd_player_id"],
            "name":               spec.get("name", f"Player{spec['player_id']}"),
            "position":           spec["position"],
            "eligible_positions": spec["eligible_positions"],
            "roster_position":    spec.get("roster_position", spec["position"]),
            "salary":             spec.get("salary", 3500),
            "team":               spec.get("team", "TEA"),
            "opponent":           spec.get("opponent", "TEB"),
            "game":               spec.get("game", "TEA@TEB"),
        })
    return pd.DataFrame(rows)


# A 9-player set covering all FD slots (P, C/1B, 2B, 3B, SS, OF×3, UTIL)
NINE_PLAYER_SPECS = [
    {"player_id": 10001, "fd_player_id": "128874-10001", "position": "P",  "eligible_positions": ["P"]},
    {"player_id": 40001, "fd_player_id": "128874-40001", "position": "C",  "eligible_positions": ["C", "1B"]},
    {"player_id": 60001, "fd_player_id": "128874-60001", "position": "2B", "eligible_positions": ["2B"]},
    {"player_id": 70001, "fd_player_id": "128874-70001", "position": "3B", "eligible_positions": ["3B"]},
    {"player_id": 30001, "fd_player_id": "128874-30001", "position": "SS", "eligible_positions": ["SS"]},
    {"player_id": 20001, "fd_player_id": "128874-20001", "position": "OF", "eligible_positions": ["OF"]},
    {"player_id": 20002, "fd_player_id": "128874-20002", "position": "OF", "eligible_positions": ["OF"]},
    {"player_id": 20003, "fd_player_id": "128874-20003", "position": "OF", "eligible_positions": ["OF"]},
    {"player_id": 50001, "fd_player_id": "128874-50001", "position": "1B", "eligible_positions": ["1B"]},
]

NINE_PLAYER_IDS = [s["player_id"] for s in NINE_PLAYER_SPECS]


# ---------------------------------------------------------------------------
# scan_fd_entry_files
# ---------------------------------------------------------------------------

class TestScanFdEntryFiles:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert scan_fd_entry_files(str(tmp_path)) == []

    def test_non_fd_files_ignored(self, tmp_path):
        (tmp_path / "DKSalaries.csv").write_text("x")
        (tmp_path / "something.txt").write_text("x")
        assert scan_fd_entry_files(str(tmp_path)) == []

    def test_single_fd_file_returned(self, tmp_path):
        fname = "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        _write_fd_csv(tmp_path, fname, [])
        result = scan_fd_entry_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == fname

    def test_most_recent_returned_when_multiple(self, tmp_path):
        files = [
            "FanDuel-MLB-2026-04-10-111111-entries-upload-template.csv",
            "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv",
            "FanDuel-MLB-2026-04-11-222222-entries-upload-template.csv",
        ]
        for f in files:
            _write_fd_csv(tmp_path, f, [])
        result = scan_fd_entry_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"

    def test_returns_list_not_path(self, tmp_path):
        _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [])
        result = scan_fd_entry_files(str(tmp_path))
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)


# ---------------------------------------------------------------------------
# parse_fd_entry_file
# ---------------------------------------------------------------------------

class TestParseFdEntryFile:
    def test_parses_single_entry(self, tmp_path):
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [SAMPLE_ENTRY])
        records = parse_fd_entry_file(path)
        assert len(records) == 1
        rec = records[0]
        assert rec.entry_id == "3679070205"
        assert rec.contest_id == "128874-279459600"
        assert rec.contest_name == "$2K Sun MLB Wiffle Ball (150 Entries Max)"
        assert rec.entry_fee_raw == "0.05"

    def test_fee_cents_parsed_correctly(self, tmp_path):
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [SAMPLE_ENTRY])
        records = parse_fd_entry_file(path)
        assert records[0].entry_fee_cents == 5  # $0.05 → 5 cents

    def test_instruction_rows_skipped(self, tmp_path):
        # _make_fd_csv_with_entries adds instruction rows with empty entry_id
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [SAMPLE_ENTRY])
        records = parse_fd_entry_file(path)
        assert len(records) == 1  # only the one real entry

    def test_no_entries_returns_empty(self, tmp_path):
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [])
        records = parse_fd_entry_file(path)
        assert records == []

    def test_multiple_entries(self, tmp_path):
        entries = [
            {**SAMPLE_ENTRY, "entry_id": "111", "entry_fee": "5"},
            {**SAMPLE_ENTRY, "entry_id": "222", "entry_fee": "1"},
            {**SAMPLE_ENTRY, "entry_id": "333", "entry_fee": "0.25"},
        ]
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", entries)
        records = parse_fd_entry_file(path)
        assert len(records) == 3
        ids = [r.entry_id for r in records]
        assert "111" in ids and "222" in ids and "333" in ids

    def test_wrong_header_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Wrong,Header,Format,Here\n1,2,3,4\n")
        with pytest.raises(ValueError, match="[Uu]nexpected header"):
            parse_fd_entry_file(path)

    def test_returns_fd_entry_records(self, tmp_path):
        path = _write_fd_csv(tmp_path, "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv", [SAMPLE_ENTRY])
        records = parse_fd_entry_file(path)
        assert all(isinstance(r, FDEntryRecord) for r in records)


# ---------------------------------------------------------------------------
# assign_players_to_fd_slots
# ---------------------------------------------------------------------------

class TestAssignPlayersToFdSlots:
    @pytest.fixture()
    def slate_df(self):
        return _make_slate_df(NINE_PLAYER_SPECS)

    def test_valid_9_player_lineup_assigned(self, slate_df):
        result = assign_players_to_fd_slots(NINE_PLAYER_IDS, slate_df)
        assert len(result) == 9

    def test_returns_fd_player_id_strings(self, slate_df):
        result = assign_players_to_fd_slots(NINE_PLAYER_IDS, slate_df)
        # All values should be fd_player_id format (e.g. "128874-XXXXX")
        assert all(isinstance(v, str) for v in result)
        assert all("-" in v for v in result), "Expected fd_player_id format with hyphen"

    def test_slot_order_matches_fd_slots(self, slate_df):
        """First slot must be filled by the pitcher."""
        result = assign_players_to_fd_slots(NINE_PLAYER_IDS, slate_df)
        assert FD_UPLOAD_SLOTS[0] == "P"
        # The pitcher's fd_player_id is "128874-10001"
        assert result[0] == "128874-10001"

    def test_c1b_slot_filled_when_c_player_present(self):
        """A lineup with a pure-C player and a pure-1B player should resolve all slots."""
        specs = [
            {"player_id": 10001, "fd_player_id": "128874-10001", "position": "P",  "eligible_positions": ["P"]},
            {"player_id": 40001, "fd_player_id": "128874-40001", "position": "C",  "eligible_positions": ["C"]},
            {"player_id": 60001, "fd_player_id": "128874-60001", "position": "2B", "eligible_positions": ["2B"]},
            {"player_id": 70001, "fd_player_id": "128874-70001", "position": "3B", "eligible_positions": ["3B"]},
            {"player_id": 30001, "fd_player_id": "128874-30001", "position": "SS", "eligible_positions": ["SS"]},
            {"player_id": 20001, "fd_player_id": "128874-20001", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20002, "fd_player_id": "128874-20002", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20003, "fd_player_id": "128874-20003", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 50001, "fd_player_id": "128874-50001", "position": "1B", "eligible_positions": ["1B"]},
        ]
        df = _make_slate_df(specs)
        player_ids = [s["player_id"] for s in specs]
        result = assign_players_to_fd_slots(player_ids, df)
        # Both C and 1B are eligible for C/1B; either assignment is valid.
        # What matters is that all 9 slots are filled and C/1B goes to one of them.
        assert len(result) == 9
        c1b_idx = FD_UPLOAD_SLOTS.index("C/1B")
        assert result[c1b_idx] in {"128874-40001", "128874-50001"}
        util_idx = FD_UPLOAD_SLOTS.index("UTIL")
        assert result[util_idx] in {"128874-40001", "128874-50001"}
        # They must be assigned to different slots
        assert result[c1b_idx] != result[util_idx]

    def test_c1b_slot_filled_by_1b_player(self):
        """A 1B player should fill the C/1B slot when no C is in lineup."""
        specs = [
            {"player_id": 10001, "fd_player_id": "128874-10001", "position": "P",  "eligible_positions": ["P"]},
            {"player_id": 40001, "fd_player_id": "128874-40001", "position": "1B", "eligible_positions": ["1B"]},
            {"player_id": 60001, "fd_player_id": "128874-60001", "position": "2B", "eligible_positions": ["2B"]},
            {"player_id": 70001, "fd_player_id": "128874-70001", "position": "3B", "eligible_positions": ["3B"]},
            {"player_id": 30001, "fd_player_id": "128874-30001", "position": "SS", "eligible_positions": ["SS"]},
            {"player_id": 20001, "fd_player_id": "128874-20001", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20002, "fd_player_id": "128874-20002", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20003, "fd_player_id": "128874-20003", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 50001, "fd_player_id": "128874-50001", "position": "OF", "eligible_positions": ["OF"]},
        ]
        df = _make_slate_df(specs)
        player_ids = [s["player_id"] for s in specs]
        result = assign_players_to_fd_slots(player_ids, df)
        c1b_idx = FD_UPLOAD_SLOTS.index("C/1B")
        assert result[c1b_idx] == "128874-40001"

    def test_util_slot_filled_by_non_pitcher(self, slate_df):
        """UTIL slot must be filled by a non-pitcher."""
        result = assign_players_to_fd_slots(NINE_PLAYER_IDS, slate_df)
        util_idx = FD_UPLOAD_SLOTS.index("UTIL")
        util_fd_id = result[util_idx]
        # The pitcher is 128874-10001; UTIL must not be the pitcher
        assert util_fd_id != "128874-10001"

    def test_util_accepts_of_player(self):
        """An OF player should be assignable to UTIL if all OF slots are taken."""
        specs = [
            {"player_id": 10001, "fd_player_id": "128874-10001", "position": "P",  "eligible_positions": ["P"]},
            {"player_id": 40001, "fd_player_id": "128874-40001", "position": "C",  "eligible_positions": ["C", "1B"]},
            {"player_id": 60001, "fd_player_id": "128874-60001", "position": "2B", "eligible_positions": ["2B"]},
            {"player_id": 70001, "fd_player_id": "128874-70001", "position": "3B", "eligible_positions": ["3B"]},
            {"player_id": 30001, "fd_player_id": "128874-30001", "position": "SS", "eligible_positions": ["SS"]},
            {"player_id": 20001, "fd_player_id": "128874-20001", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20002, "fd_player_id": "128874-20002", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20003, "fd_player_id": "128874-20003", "position": "OF", "eligible_positions": ["OF"]},
            {"player_id": 20004, "fd_player_id": "128874-20004", "position": "OF", "eligible_positions": ["OF"]},
        ]
        df = _make_slate_df(specs)
        player_ids = [s["player_id"] for s in specs]
        # Should not raise — 4th OF goes to UTIL
        result = assign_players_to_fd_slots(player_ids, df)
        assert len(result) == 9

    def test_player_not_in_slate_raises(self, slate_df):
        bad_ids = [99999] + NINE_PLAYER_IDS[1:]
        with pytest.raises(ValueError, match="not found in FD slate"):
            assign_players_to_fd_slots(bad_ids, slate_df)

    def test_impossible_assignment_raises(self):
        """9 pitchers cannot fill the 9 FD slots."""
        specs = [
            {"player_id": i, "fd_player_id": f"128874-{i}", "position": "P", "eligible_positions": ["P"]}
            for i in range(1, 10)
        ]
        df = _make_slate_df(specs)
        with pytest.raises(ValueError, match="[Uu]nmatched slots"):
            assign_players_to_fd_slots([s["player_id"] for s in specs], df)

    def test_all_slots_assigned(self, slate_df):
        result = assign_players_to_fd_slots(NINE_PLAYER_IDS, slate_df)
        assert len(result) == len(FD_UPLOAD_SLOTS)
        assert all(v for v in result)  # no empty strings


# ---------------------------------------------------------------------------
# assign_fd_lineups_to_entries
# ---------------------------------------------------------------------------

def _make_mock_lineup(player_ids: list[int]):
    lu = MagicMock()
    lu.player_ids = player_ids
    return lu


class TestAssignFdLineupsToEntries:
    def _make_entries(self, fees: list[str]) -> list[tuple[Path, list[FDEntryRecord]]]:
        path = Path("fake.csv")
        records = [
            FDEntryRecord(
                entry_id=str(i),
                contest_id="c1",
                contest_name="Contest",
                entry_fee_cents=round(float(f) * 100),
                entry_fee_raw=f,
            )
            for i, f in enumerate(fees)
        ]
        return [(path, records)]

    def test_highest_fee_gets_first_lineup(self):
        entries = self._make_entries(["1", "5", "0.25"])
        portfolio = [(_make_mock_lineup([i] * 9), 0.0) for i in range(3)]
        result = assign_fd_lineups_to_entries(entries, portfolio)
        path = list(result.keys())[0]
        assigned = result[path]
        # After sorting by fee: $5, $1, $0.25 → lineups 0, 1, 2
        fees = [float(rec.entry_fee_raw) for rec, _ in assigned]
        assert fees == sorted(fees, reverse=True)

    def test_fewer_lineups_than_entries_warns(self, caplog):
        import logging
        entries = self._make_entries(["1", "1", "1"])
        portfolio = [(_make_mock_lineup([0] * 9), 0.0)]  # only 1 lineup
        with caplog.at_level(logging.WARNING, logger="src.api.fd_entries"):
            result = assign_fd_lineups_to_entries(entries, portfolio)
        path = list(result.keys())[0]
        assert len(result[path]) == 1
        assert "2 entries will not receive a lineup" in caplog.text

    def test_result_is_indexed_by_path(self):
        path = Path("test.csv")
        rec = FDEntryRecord("1", "c1", "Contest", 100, "1")
        entries = [(path, [rec])]
        portfolio = [(_make_mock_lineup([0] * 9), 0.0)]
        result = assign_fd_lineups_to_entries(entries, portfolio)
        assert path in result


# ---------------------------------------------------------------------------
# write_fd_upload_files
# ---------------------------------------------------------------------------

class TestWriteFdUploadFiles:
    @pytest.fixture()
    def slate_df(self):
        return _make_slate_df(NINE_PLAYER_SPECS)

    def _make_fd_entry(self, entry_id: str = "999") -> FDEntryRecord:
        return FDEntryRecord(
            entry_id=entry_id,
            contest_id="c1",
            contest_name="Contest",
            entry_fee_cents=500,
            entry_fee_raw="5",
        )

    def test_upload_file_created(self, tmp_path, slate_df):
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")  # placeholder
        entry = self._make_fd_entry()
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        written = write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        assert len(written) == 1
        assert Path(written[0]).exists()

    def test_upload_header_is_correct(self, tmp_path, slate_df):
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")
        entry = self._make_fd_entry()
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        upload_path = tmp_path / f"upload_{path.name}"
        with open(upload_path, newline="") as f:
            header = next(csv.reader(f))
        assert header == FD_UPLOAD_HEADER

    def test_upload_row_has_13_columns(self, tmp_path, slate_df):
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")
        entry = self._make_fd_entry()
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        upload_path = tmp_path / f"upload_{path.name}"
        with open(upload_path, newline="") as f:
            rows = list(csv.reader(f))
        # header + 1 data row
        assert len(rows) == 2
        assert len(rows[1]) == 13

    def test_slot_columns_contain_fd_player_ids(self, tmp_path, slate_df):
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")
        entry = self._make_fd_entry()
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        upload_path = tmp_path / f"upload_{path.name}"
        with open(upload_path, newline="") as f:
            rows = list(csv.reader(f))
        data_row = rows[1]
        slot_values = data_row[4:]  # columns 4-12 are the 9 slot values
        assert len(slot_values) == 9
        # All should be fd_player_id format
        assert all("-" in v for v in slot_values)

    def test_entry_metadata_written(self, tmp_path, slate_df):
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")
        entry = self._make_fd_entry("12345")
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        upload_path = tmp_path / f"upload_{path.name}"
        with open(upload_path, newline="") as f:
            rows = list(csv.reader(f))
        data_row = rows[1]
        assert data_row[0] == "12345"   # entry_id
        assert data_row[3] == "5"       # entry_fee_raw

    def test_no_assignments_skips_file(self, tmp_path, slate_df, caplog):
        import logging
        path = tmp_path / "FanDuel-MLB-2026-04-12-128874.csv"
        path.write_text("")
        entries = [(path, [])]
        with caplog.at_level(logging.WARNING, logger="src.api.fd_entries"):
            written = write_fd_upload_files(entries, {}, slate_df, str(tmp_path))
        assert written == []

    def test_upload_filename_prefixed(self, tmp_path, slate_df):
        fname = "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        path = tmp_path / fname
        path.write_text("")
        entry = self._make_fd_entry()
        lineup = _make_mock_lineup(NINE_PLAYER_IDS)
        entries = [(path, [entry])]
        assignments = {path: [(entry, lineup)]}
        written = write_fd_upload_files(entries, assignments, slate_df, str(tmp_path))
        assert Path(written[0]).name == f"upload_{fname}"


# ---------------------------------------------------------------------------
# entries_factory.get_entry_handlers
# ---------------------------------------------------------------------------

class TestGetEntryHandlers:
    def test_dk_handlers_returned(self):
        from src.api.dk_entries import (
            scan_entry_files, parse_entry_file,
            assign_lineups_to_entries, write_upload_files,
        )
        handlers = get_entry_handlers(Platform.DRAFTKINGS)
        assert handlers["scan"] is scan_entry_files
        assert handlers["parse"] is parse_entry_file
        assert handlers["assign"] is assign_lineups_to_entries
        assert handlers["write"] is write_upload_files

    def test_fd_handlers_returned(self):
        handlers = get_entry_handlers(Platform.FANDUEL)
        assert handlers["scan"] is scan_fd_entry_files
        assert handlers["parse"] is parse_fd_entry_file
        assert handlers["assign"] is assign_fd_lineups_to_entries
        assert handlers["write"] is write_fd_upload_files

    def test_all_four_keys_present(self):
        for platform in (Platform.DRAFTKINGS, Platform.FANDUEL):
            handlers = get_entry_handlers(platform)
            assert set(handlers.keys()) == {"scan", "parse", "assign", "write"}

    def test_unknown_platform_raises(self):
        with pytest.raises(ValueError, match="[Nn]o entry handlers"):
            get_entry_handlers("unknown")
