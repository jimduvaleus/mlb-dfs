"""
DraftKings entry file parser and upload file writer.

Responsibilities:
  - Scan data/raw/ for *Entries.csv files
  - Parse DK entry files (which have player reference data appended in extra columns)
  - Assign portfolio lineups to entries in descending entry-fee order
  - Write upload_<filename>.csv files ready for DraftKings submission
"""
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.optimization.lineup import SLOTS

logger = logging.getLogger(__name__)

UPLOAD_HEADER = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee",
                 "P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]


@dataclass
class EntryRecord:
    entry_id: str
    contest_name: str
    contest_id: str
    entry_fee_cents: int   # "$4" -> 400; used as numeric sort key
    entry_fee_raw: str     # "$4"; written verbatim to upload file


def scan_entry_files(raw_dir: str) -> list[Path]:
    """Return all *Entries.csv paths found in raw_dir."""
    d = Path(raw_dir)
    return sorted(d.glob("*Entries.csv"))


def _parse_fee_cents(fee_str: str) -> int:
    """Convert "$4" or "$1.50" to integer cents (400, 150). Returns 0 on failure."""
    cleaned = fee_str.strip().lstrip("$").strip()
    try:
        return round(float(cleaned) * 100)
    except ValueError:
        return 0


def parse_entry_file(path: Path) -> list[EntryRecord]:
    """
    Parse a DraftKings entry CSV and return the list of contest entries.

    DK entry files have an unusual structure:
      - Row 0: header (Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, ...)
      - Rows 1-N: contest entries in columns 0-13; columns 14+ are ignored
      - Further rows: blank Entry ID; only columns 14+ have content (player reference pool)

    We read every row via csv.reader (avoiding pandas column-width inference),
    take only the first 14 columns, and skip rows with a blank Entry ID.
    """
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return records

    # Validate header
    header = rows[0]
    expected = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    if header[:4] != expected:
        raise ValueError(
            f"Unexpected header in {path.name}: {header[:4]!r} "
            f"(expected {expected!r})"
        )

    for row in rows[1:]:
        if len(row) < 4:
            continue
        entry_id = row[0].strip()
        if not entry_id:
            continue  # reference-data-only row
        fee_raw = row[3].strip()
        records.append(EntryRecord(
            entry_id=entry_id,
            contest_name=row[1].strip(),
            contest_id=row[2].strip(),
            entry_fee_cents=_parse_fee_cents(fee_raw),
            entry_fee_raw=fee_raw,
        ))

    return records


def assign_players_to_slots(
    player_ids: list[int],
    slate_df: pd.DataFrame,
) -> list[int]:
    """
    Map 10 player IDs to the DK upload slot order: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF.

    Uses bipartite matching (same algorithm as Lineup.is_valid()) so multi-eligible
    players (e.g. 1B/OF) are handled correctly.

    Parameters
    ----------
    player_ids : list of 10 player IDs
    slate_df : DataFrame with columns player_id, position, and optionally eligible_positions

    Returns
    -------
    list of 10 player IDs in SLOTS order

    Raises
    ------
    ValueError if no valid assignment exists
    """
    pid_set = set(player_ids)
    sub = slate_df[slate_df["player_id"].isin(pid_set)]

    # Build eligibility map: player_id -> set of positions
    id_to_elig: dict[int, set[str]] = {}
    for _, row in sub.iterrows():
        pid = int(row["player_id"])
        ep = row.get("eligible_positions")
        if ep and isinstance(ep, list):
            id_to_elig[pid] = set(ep)
        else:
            id_to_elig[pid] = {str(row["position"])}

    players = list(player_ids)
    n_slots = len(SLOTS)

    # match_slot[slot_index] = index into `players` list (-1 = unmatched)
    match_slot = [-1] * n_slots

    def _try_assign(player_idx: int, visited: set) -> bool:
        elig = id_to_elig.get(players[player_idx], set())
        for j, slot_pos in enumerate(SLOTS):
            if slot_pos in elig and j not in visited:
                visited.add(j)
                if match_slot[j] == -1 or _try_assign(match_slot[j], visited):
                    match_slot[j] = player_idx
                    return True
        return False

    for i in range(len(players)):
        _try_assign(i, set())

    if -1 in match_slot:
        unmatched_slots = [SLOTS[j] for j, v in enumerate(match_slot) if v == -1]
        raise ValueError(
            f"Could not assign all players to slots. "
            f"Unmatched slots: {unmatched_slots}. Player IDs: {player_ids}"
        )

    return [players[match_slot[j]] for j in range(n_slots)]


def assign_lineups_to_entries(
    all_file_entries: list[tuple[Path, list[EntryRecord]]],
    portfolio: list,  # list of (Lineup, float)
) -> dict[Path, list[tuple[EntryRecord, object]]]:
    """
    Assign portfolio lineups to entries across all files in descending fee order.

    The highest-fee entry (across all files) receives portfolio[0], the next
    highest receives portfolio[1], and so on. Within a fee tier, the original
    file/row order is preserved (stable sort).

    Returns
    -------
    dict mapping each file Path to its list of (EntryRecord, Lineup) pairs,
    in the order they appear in the file.
    """
    # Flatten: (fee_cents, original_index, file_path, entry_record)
    flat: list[tuple[int, int, Path, EntryRecord]] = []
    idx = 0
    for file_path, records in all_file_entries:
        for rec in records:
            flat.append((rec.entry_fee_cents, idx, file_path, rec))
            idx += 1

    # Sort descending by fee; stable on idx preserves original order within ties
    flat.sort(key=lambda x: x[0], reverse=True)

    n_lineups = len(portfolio)
    if len(flat) > n_lineups:
        logger.warning(
            "%d entries but only %d lineups — %d entries will not receive a lineup.",
            len(flat), n_lineups, len(flat) - n_lineups,
        )

    result: dict[Path, list[tuple[EntryRecord, object]]] = {}
    for i, (_, _, file_path, entry) in enumerate(flat):
        if i >= n_lineups:
            break
        lineup, _ = portfolio[i]
        result.setdefault(file_path, []).append((entry, lineup))

    return result


def write_upload_files(
    all_file_entries: list[tuple[Path, list[EntryRecord]]],
    assignments: dict[Path, list[tuple[EntryRecord, object]]],
    slate_df: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """
    Write one upload_<filename>.csv per source entry file.

    Each upload file has the same 14-column header as the source entry file,
    with player slot columns containing integer player IDs (not "Name (ID)" format).

    Returns list of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for file_path, _ in all_file_entries:
        file_assignments = assignments.get(file_path)
        if not file_assignments:
            logger.warning("No lineups assigned to %s — skipping upload file.", file_path.name)
            continue

        upload_path = out / f"upload_{file_path.name}"
        rows = []
        for entry, lineup in file_assignments:
            try:
                slot_ids = assign_players_to_slots(lineup.player_ids, slate_df)
            except ValueError as exc:
                logger.error(
                    "Slot assignment failed for entry %s: %s", entry.entry_id, exc
                )
                continue

            rows.append([
                entry.entry_id,
                entry.contest_name,
                entry.contest_id,
                entry.entry_fee_raw,
            ] + [str(pid) for pid in slot_ids])

        with open(upload_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(UPLOAD_HEADER)
            writer.writerows(rows)

        logger.info("Wrote %d entries to %s", len(rows), upload_path)
        written.append(str(upload_path))

    return written
