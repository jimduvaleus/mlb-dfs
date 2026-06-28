"""
DraftKings entry file parser and upload file writer.

Responsibilities:
  - Scan data/raw/ for *Entries.csv files
  - Parse DK entry files (which have player reference data appended in extra columns)
  - Assign portfolio lineups to entries in ascending (prize_pool / entry_fee) order
  - Write upload_<filename>.csv files ready for DraftKings submission
"""
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.optimization.lineup import SLOTS

logger = logging.getLogger(__name__)

UPLOAD_HEADER = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee",
                 "P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def name_sort_key(name: str) -> tuple[str, str]:
    """Sort key mirroring DK's own echo-back convention for duplicate-position
    roster slots (P,P and OF,OF,OF): alphabetical by last name, first name as
    tiebreak. Confirmed empirically by diffing an uploaded vs. downloaded
    entries CSV — DK reorders these slots on its end regardless of upload
    column order, so we match that order rather than fight it."""
    parts = name.strip().split()
    if not parts:
        return ("", "")
    last = parts[-1].rstrip(".").lower()
    if last in _NAME_SUFFIXES and len(parts) > 1:
        last = parts[-2].rstrip(".").lower()
    first = parts[0].rstrip(".").lower()
    return (last, first)


def alphabetize_duplicate_slots(
    ordered_ids: list[Optional[int]], id_to_name: dict[int, str]
) -> list[Optional[int]]:
    """Within each group of slots sharing the same SLOTS label (P,P and
    OF,OF,OF), reorder the assigned players alphabetically by last name."""
    groups: dict[str, list[int]] = {}
    for j, pos in enumerate(SLOTS):
        groups.setdefault(pos, []).append(j)
    out = list(ordered_ids)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        group_ids = sorted(
            (ordered_ids[j] for j in idxs),
            key=lambda pid: name_sort_key(str(id_to_name.get(pid, ""))),
        )
        for slot_j, pid in zip(idxs, group_ids):
            out[slot_j] = pid
    return out


@dataclass
class EntrySlotPlayer:
    name: str
    player_id: Optional[int]   # None if the "(ID)" suffix is absent/unparseable


@dataclass
class EntryRecord:
    entry_id: str
    contest_name: str
    contest_id: str
    entry_fee_cents: int        # "$4" -> 400
    entry_fee_raw: str          # "$4"; written verbatim to upload file
    prize_pool_cents: Optional[int] = None  # "$5K" -> 500000; None if not parseable
    # Columns 4-13 of the entry row (slots P,P,C,1B,2B,3B,SS,OF,OF,OF).
    # None element = empty cell (unfilled reservation).
    slot_players: list = field(default_factory=list)


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


_PRIZE_POOL_RE = re.compile(r"\$(\d+(?:\.\d+)?)(K|M)?", re.IGNORECASE)

# DK runs ~180 parallel "Qualifier" contests feeding a single $100K final; the
# advertised prize pool is the *final's* pool, not the qualifier's. Approximate
# the qualifier's own pool by dividing it out until DK exposes a real number.
_QUALIFIER_RE = re.compile(r"qualifier", re.IGNORECASE)
_QUALIFIER_DIVISOR = 180

_PLAYER_CELL_RE = re.compile(r"^\s*(.*?)\s*\((\d+)\)\s*$")
_BARE_ID_RE = re.compile(r"^\s*(\d+)\s*$")


def _parse_slot_players(row: list[str]) -> list:
    """Parse entry-row columns 4-13 into 10 Optional[EntrySlotPlayer] slots.

    Cells come in two formats: "Name (ID)" (DK entry downloads) or a bare
    integer ID (upload_*.csv files written by write_upload_files). Bare-ID
    players get an empty name — callers resolve display names via the slate.
    """
    cells = row[4:14]
    cells += [""] * (10 - len(cells))
    slots: list = []
    for cell in cells:
        cell = cell.strip()
        if not cell:
            slots.append(None)
            continue
        m = _PLAYER_CELL_RE.match(cell)
        if m:
            slots.append(EntrySlotPlayer(name=m.group(1), player_id=int(m.group(2))))
            continue
        m = _BARE_ID_RE.match(cell)
        if m:
            slots.append(EntrySlotPlayer(name="", player_id=int(m.group(1))))
        else:
            slots.append(EntrySlotPlayer(name=cell, player_id=None))
    return slots

def _parse_prize_pool_cents(contest_name: str) -> Optional[int]:
    """
    Extract the prize pool from a DK contest name and return it in cents.

    Looks for the first "$<number>[K|M]" token, e.g.:
      "MLB $5K Chin Music [Single Entry]"  -> 500_000_00 (= $5,000 * 100)
      "MLB $1.5K Pickoff"                  -> 150_000_00 (= $1,500 * 100)
      "MLB $20K Four-Seamer"               -> 2_000_000_00 (= $20,000 * 100)

    If "Qualifier" appears in the contest name, the parsed amount is the
    pool of the $100K final the qualifier feeds into, not the qualifier's
    own pool, so it's divided by `_QUALIFIER_DIVISOR` (~180 parallel
    qualifiers per final), e.g.:
      "MLB $100K Baseball Pocket Cup Qualifier #144" -> 555_55 (= $100,000 / 180 * 100)

    Returns None if no such token is found.
    """
    m = _PRIZE_POOL_RE.search(contest_name)
    if not m:
        return None
    amount = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    if suffix == "K":
        amount *= 1_000
    elif suffix == "M":
        amount *= 1_000_000
    if _QUALIFIER_RE.search(contest_name):
        amount /= _QUALIFIER_DIVISOR
    return round(amount * 100)


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
        contest_name = row[1].strip()
        records.append(EntryRecord(
            entry_id=entry_id,
            contest_name=contest_name,
            contest_id=row[2].strip(),
            entry_fee_cents=_parse_fee_cents(fee_raw),
            entry_fee_raw=fee_raw,
            prize_pool_cents=_parse_prize_pool_cents(contest_name),
            slot_players=_parse_slot_players(row),
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

    ordered_ids = [players[match_slot[j]] for j in range(n_slots)]
    id_to_name = dict(zip(slate_df["player_id"], slate_df.get("name", slate_df["player_id"])))
    return alphabetize_duplicate_slots(ordered_ids, id_to_name)


def _sort_ratio(rec: EntryRecord) -> float:
    """
    Sort key: prize_pool / entry_fee (ascending = fewest implied entries first).

    When prize_pool is unknown, returns infinity so those entries sort last.
    """
    if rec.prize_pool_cents and rec.entry_fee_cents:
        return rec.prize_pool_cents / rec.entry_fee_cents
    return float("inf")


def assign_lineups_to_entries(
    all_file_entries: list[tuple[Path, list[EntryRecord]]],
    portfolio: list,  # list of (Lineup, float)
) -> dict[Path, list[tuple[EntryRecord, object]]]:
    """
    Assign portfolio lineups to entries across all files in ascending
    (prize_pool / entry_fee) ratio order.

    The entry with the smallest ratio (fewest implied opponents) receives
    portfolio[0] (the strongest lineup). Within a ratio tier, original
    file/row order is preserved (stable sort). Entries whose prize pool
    cannot be parsed from the contest name sort last.

    Returns
    -------
    dict mapping each file Path to its list of (EntryRecord, Lineup) pairs,
    in the order they appear in the file.
    """
    # Flatten: (sort_ratio, original_index, file_path, entry_record)
    flat: list[tuple[float, int, Path, EntryRecord]] = []
    idx = 0
    for file_path, records in all_file_entries:
        for rec in records:
            flat.append((_sort_ratio(rec), idx, file_path, rec))
            idx += 1

    # Sort ascending by ratio; tie-break descending by fee (higher fee first), then original order
    flat.sort(key=lambda x: (x[0], -x[3].entry_fee_cents, x[1]))

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
