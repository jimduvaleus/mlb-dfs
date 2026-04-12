"""
FanDuel entry file parser and upload file writer.

FanDuel uses a single CSV file as both salary list and upload template.
Its layout is unconventional:

  Row 0:       Column headers: entry_id, contest_id, contest_name, entry_fee,
               P, C/1B, 2B, 3B, SS, OF, OF, OF, UTIL, "", Instructions
  Rows 1-N:    Contest entries (entry_id non-empty); cols 0-12 only are used.
               Col 13 is empty; col 14 holds instruction text.
  Rows N+1-M:  Rows with empty entry_id (instruction rows) — skipped.
  Row M+1:     Player-pool header row (col 14 == "Player ID + Player Name").
  Rows M+2+:   Player pool data — handled by FanDuelSlateIngestor.

Responsibilities:
  - Scan data/raw/ for FanDuel-MLB-*.csv files; return the most recent by date.
  - Parse entry rows from the top portion of the file.
  - Assign portfolio lineups to entries in descending entry-fee order.
  - Write upload_<filename>.csv files ready for FanDuel submission (cols 0-12
    only; slot columns contain the FD player ID string, e.g. "128874-16961").
"""
import csv
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.platforms.fanduel import FD_SLOT_ELIGIBILITY, FD_ROSTER

logger = logging.getLogger(__name__)

# Slot order for FD upload: matches the column order in the template header.
FD_UPLOAD_SLOTS = list(FD_ROSTER.slots)  # ['P','C/1B','2B','3B','SS','OF','OF','OF','UTIL']

# The 13 column headers written to the upload file (cols 0-12 of the FD CSV).
FD_UPLOAD_HEADER = [
    "entry_id", "contest_id", "contest_name", "entry_fee",
    "P", "C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL",
]


@dataclass
class FDEntryRecord:
    entry_id: str
    contest_id: str
    contest_name: str
    entry_fee_cents: int   # "$0.05" -> 5; numeric sort key
    entry_fee_raw: str     # written verbatim to upload file


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

_FD_DATE_RE = re.compile(r"FanDuel-MLB-(\d{4}-\d{2}-\d{2})-")


def _extract_date(path: Path) -> date:
    """Extract YYYY-MM-DD from a FanDuel filename; return date.min on failure."""
    m = _FD_DATE_RE.search(path.name)
    if not m:
        return date.min
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return date.min


def scan_fd_entry_files(raw_dir: str) -> list[Path]:
    """
    Return the most-recent FanDuel-MLB-*.csv file in raw_dir as a one-element
    list, or an empty list if none are found.

    When multiple files are present the one with the latest date in its
    filename takes precedence.
    """
    d = Path(raw_dir)
    candidates = sorted(d.glob("FanDuel-MLB-*.csv"), key=_extract_date, reverse=True)
    if not candidates:
        return []
    if len(candidates) > 1:
        logger.info(
            "Multiple FanDuel CSV files found; using most recent: %s",
            candidates[0].name,
        )
    return [candidates[0]]


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------

def _parse_fee_cents(fee_str: str) -> int:
    """Convert "$0.05" or "$4" to integer cents (5, 400). Returns 0 on failure."""
    cleaned = fee_str.strip().lstrip("$").strip()
    try:
        return round(float(cleaned) * 100)
    except ValueError:
        return 0


def parse_fd_entry_file(path: Path) -> list[FDEntryRecord]:
    """
    Parse entry rows from a FanDuel MLB salary/upload-template CSV.

    Only rows where column 0 (entry_id) is non-empty are treated as entries.
    Columns 14+ (player pool / instructions) are ignored.

    # TODO: verify column order against additional real FD entry files.
    """
    records: list[FDEntryRecord] = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return records

    # Validate the first 4 header tokens.
    header = [c.strip() for c in rows[0][:4]]
    expected = ["entry_id", "contest_id", "contest_name", "entry_fee"]
    if header != expected:
        raise ValueError(
            f"Unexpected header in {path.name}: {header!r} "
            f"(expected {expected!r})"
        )

    for row in rows[1:]:
        if len(row) < 4:
            continue
        entry_id = row[0].strip()
        if not entry_id:
            continue  # instruction / player-pool row
        fee_raw = row[3].strip()
        records.append(FDEntryRecord(
            entry_id=entry_id,
            contest_id=row[1].strip(),
            contest_name=row[2].strip(),
            entry_fee_cents=_parse_fee_cents(fee_raw),
            entry_fee_raw=fee_raw,
        ))

    return records


# ---------------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------------

def assign_players_to_fd_slots(
    player_ids: list[int],
    slate_df: pd.DataFrame,
) -> list[str]:
    """
    Map 9 internal player IDs to FD upload slot order using bipartite matching.

    Slot order: P, C/1B, 2B, 3B, SS, OF, OF, OF, UTIL

    FD slot eligibility (compound slots expand to multiple positions):
      C/1B -> {C, 1B}
      UTIL -> {C, 1B, 2B, 3B, SS, OF}

    Parameters
    ----------
    player_ids : list of 9 internal player IDs (int)
    slate_df   : DataFrame with columns player_id (int), fd_player_id (str),
                 position (str), and eligible_positions (list[str])

    Returns
    -------
    list of 9 FD player ID strings (e.g. "128874-16961") in FD slot order

    Raises
    ------
    ValueError if no valid assignment exists or a player is not in slate_df
    """
    pid_set = set(player_ids)
    sub = slate_df[slate_df["player_id"].isin(pid_set)]

    # Build eligibility map: int player_id -> set of positions
    id_to_elig: dict[int, set[str]] = {}
    id_to_fd_id: dict[int, str] = {}
    for _, row in sub.iterrows():
        pid = int(row["player_id"])
        ep = row.get("eligible_positions")
        if ep and isinstance(ep, list):
            id_to_elig[pid] = set(ep)
        else:
            id_to_elig[pid] = {str(row["position"])}
        id_to_fd_id[pid] = str(row["fd_player_id"])

    missing = pid_set - set(id_to_elig)
    if missing:
        raise ValueError(
            f"Player IDs not found in FD slate: {missing}"
        )

    players = list(player_ids)
    n_slots = len(FD_UPLOAD_SLOTS)

    # match_slot[slot_index] = index into `players` list (-1 = unmatched)
    match_slot = [-1] * n_slots

    def _try_assign(player_idx: int, visited: set) -> bool:
        elig = id_to_elig.get(players[player_idx], set())
        for j, slot_label in enumerate(FD_UPLOAD_SLOTS):
            slot_positions = FD_SLOT_ELIGIBILITY.get(slot_label, {slot_label})
            if elig & slot_positions and j not in visited:
                visited.add(j)
                if match_slot[j] == -1 or _try_assign(match_slot[j], visited):
                    match_slot[j] = player_idx
                    return True
        return False

    for i in range(len(players)):
        _try_assign(i, set())

    if -1 in match_slot:
        unmatched = [FD_UPLOAD_SLOTS[j] for j, v in enumerate(match_slot) if v == -1]
        raise ValueError(
            f"Could not assign all players to FD slots. "
            f"Unmatched slots: {unmatched}. Player IDs: {player_ids}"
        )

    return [id_to_fd_id[players[match_slot[j]]] for j in range(n_slots)]


# ---------------------------------------------------------------------------
# Entry assignment (same logic as DK: highest-fee entries get first lineups)
# ---------------------------------------------------------------------------

def assign_fd_lineups_to_entries(
    all_file_entries: list[tuple[Path, list[FDEntryRecord]]],
    portfolio: list,  # list of (Lineup, float)
) -> dict[Path, list[tuple[FDEntryRecord, object]]]:
    """
    Assign portfolio lineups to FD entries in descending entry-fee order.

    Returns
    -------
    dict mapping each file Path to its list of (FDEntryRecord, Lineup) pairs,
    in file-row order.
    """
    flat: list[tuple[int, int, Path, FDEntryRecord]] = []
    idx = 0
    for file_path, records in all_file_entries:
        for rec in records:
            flat.append((rec.entry_fee_cents, idx, file_path, rec))
            idx += 1

    flat.sort(key=lambda x: x[0], reverse=True)

    n_lineups = len(portfolio)
    if len(flat) > n_lineups:
        logger.warning(
            "%d FD entries but only %d lineups — %d entries will not receive a lineup.",
            len(flat), n_lineups, len(flat) - n_lineups,
        )

    result: dict[Path, list[tuple[FDEntryRecord, object]]] = {}
    for i, (_, _, file_path, entry) in enumerate(flat):
        if i >= n_lineups:
            break
        lineup, _ = portfolio[i]
        result.setdefault(file_path, []).append((entry, lineup))

    return result


# ---------------------------------------------------------------------------
# Upload file writer
# ---------------------------------------------------------------------------

def write_fd_upload_files(
    all_file_entries: list[tuple[Path, list[FDEntryRecord]]],
    assignments: dict[Path, list[tuple[FDEntryRecord, object]]],
    slate_df: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """
    Write one upload_<filename>.csv per source FD entry file.

    Each upload file has the 13-column FD header (entry_id … UTIL) with slot
    columns containing FD player ID strings (e.g. "128874-16961").

    Returns list of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for file_path, _ in all_file_entries:
        file_assignments = assignments.get(file_path)
        if not file_assignments:
            logger.warning(
                "No lineups assigned to %s — skipping upload file.", file_path.name
            )
            continue

        upload_path = out / f"upload_{file_path.name}"
        rows = []
        for entry, lineup in file_assignments:
            try:
                slot_fd_ids = assign_players_to_fd_slots(lineup.player_ids, slate_df)
            except ValueError as exc:
                logger.error(
                    "FD slot assignment failed for entry %s: %s", entry.entry_id, exc
                )
                continue

            rows.append([
                entry.entry_id,
                entry.contest_id,
                entry.contest_name,
                entry.entry_fee_raw,
            ] + slot_fd_ids)

        with open(upload_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FD_UPLOAD_HEADER)
            writer.writerows(rows)

        logger.info("Wrote %d FD entries to %s", len(rows), upload_path)
        written.append(str(upload_path))

    return written
