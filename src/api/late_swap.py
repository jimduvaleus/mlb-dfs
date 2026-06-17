"""
Late swap engine for DraftKings Classic entries.

DraftKings accepts re-uploaded entry CSVs with changed players as long as
players whose games have already started are untouched. This module:

  - Determines locked/swappable slots for already-submitted entries
    (a player locks once their game's scheduled start time has passed)
  - Builds the eligible-replacement pool (projection-merged slate minus
    slate/player exclusions and non-starters)
  - Heuristically picks replacements for marked players (projection mean
    + same-team stack bonus - small cross-entry diversity penalty)
  - Writes swap_<OriginalName>.csv upload files (changed entries only)
  - Persists swap results so the UI survives a server restart

Slot semantics: entry CSV columns 4-13 are fixed slots P,P,C,1B,2B,3B,SS,
OF,OF,OF. Locked and kept players stay in their column; a replacement must
be eligible for the vacated column's slot (no bipartite re-shuffling —
DK's late-swap rule is stricter than Lineup.is_valid's matching).

All functions take `now` as a parameter; only the server layer reads the
clock. Slate start times are naive Eastern-time ISO strings, so callers
must pass a naive Eastern `now`.
"""
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple

import pandas as pd

from src.optimization.lineup import (
    Lineup,
    MAX_HITTERS_PER_TEAM,
    MIN_GAMES,
    SALARY_CAP,
    SLOTS,
)
from .dk_entries import EntryRecord, EntrySlotPlayer, UPLOAD_HEADER

logger = logging.getLogger(__name__)

STATE_FILENAME = "late_swap_draftkings.json"


def scan_swap_entry_files(output_dir: str) -> List[Path]:
    """Entry files to late-swap: outputs/*Entries*.csv, typically the
    upload_*.csv files written at portfolio completion (they reflect what was
    actually submitted to DK). Our own swap_* outputs are excluded."""
    d = Path(output_dir)
    return sorted(
        p for p in d.glob("*Entries*.csv") if not p.name.startswith("swap_")
    )


def swap_file_name(source_file: str) -> str:
    """swap_<OriginalName>.csv, with any upload_ prefix stripped
    (upload_MEDKEntries.csv -> swap_MEDKEntries.csv)."""
    base = source_file
    if base.startswith("upload_"):
        base = base[len("upload_"):]
    return f"swap_{base}"


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def is_game_started(game_start_time, now: datetime) -> bool:
    """True when the game has started (or its start time is unknown — safe default)."""
    if not game_start_time or pd.isna(game_start_time):
        return True
    try:
        return now >= datetime.fromisoformat(str(game_start_time))
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Player pools
# ---------------------------------------------------------------------------

def _heuristic_mean(salary) -> float:
    try:
        return round(float(salary) / 600.0, 2)
    except (TypeError, ValueError):
        return 0.0


def build_swap_pools(
    slate_df: pd.DataFrame,
    proj_df: Optional[pd.DataFrame],
    exclusions: Optional[dict] = None,
    confirmed_team_lineups: Optional[Dict[str, Dict[int, int]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (lookup_df, candidates_df) from the slate and projections.

    lookup_df: every slate player with a `mean` column (projection, or the
    salary/600 heuristic when no projection exists). Used to display and
    value players already rostered in entries.

    candidates_df: the swap-in pool — projected starters (rows with a
    non-null lineup_slot when projections carry that column, mirroring
    PipelineRunner._build_players_df) plus any player in a currently
    confirmed Twitter lineup ({team: {pid: slot}}), minus slate/player
    exclusions (both "both" and "candidates" scopes — excluded players were
    kept out of optimization, so they must not enter via swap either).

    Manual per-player projection overrides (`exclusions["player_projection_overrides"]`,
    set via the Projections tab) take precedence over both the merged
    projection and the salary heuristic, mirroring the override applied in
    GET /api/projections/players — otherwise the swap heuristic scores a
    manually-corrected player using a stale or heuristic mean.

    Confirmed lineups also scratch: batters on a confirmed team who are NOT
    in that team's lineup are dropped from the pool (same rule the pipeline
    applies via _apply_twitter_overrides). Confirmed players absent from the
    projections pool get the salary/600 heuristic mean and are flagged
    `newly_confirmed` — a late lineup announcement (e.g. a post-lock
    replacement) whose heuristic projection understates a guaranteed
    starter, so the scorer boosts them.
    """
    confirmed_pids: set = {
        int(pid)
        for slots in (confirmed_team_lineups or {}).values()
        for pid in slots
    }
    confirmed_teams = set((confirmed_team_lineups or {}).keys())

    lookup_df = slate_df.copy()
    starters_pids: Optional[set] = None
    if proj_df is not None and not proj_df.empty:
        proj = proj_df.copy().rename(columns={"mu": "mean", "sigma": "std_dev"})
        proj = proj.drop_duplicates("player_id")
        lookup_df = lookup_df.merge(
            proj[["player_id", "mean"]], on="player_id", how="left"
        )
        if "lineup_slot" in proj.columns:
            starters = proj[proj["lineup_slot"].notna()]
            starters_pids = set(
                pd.to_numeric(starters["player_id"], errors="coerce").dropna().astype(int)
            )
    else:
        lookup_df["mean"] = pd.NA

    no_proj = lookup_df["mean"].isna()
    lookup_df.loc[no_proj, "mean"] = lookup_df.loc[no_proj, "salary"].map(_heuristic_mean)
    lookup_df["mean"] = lookup_df["mean"].astype(float)

    raw_overrides = (exclusions or {}).get("player_projection_overrides", {}) or {}
    overrides = {int(k): float(v) for k, v in raw_overrides.items()}
    if overrides:
        override_series = lookup_df["player_id"].map(overrides)
        lookup_df["mean"] = override_series.combine_first(lookup_df["mean"])

    candidates_df = lookup_df
    if starters_pids is not None:
        candidates_df = candidates_df[
            candidates_df["player_id"].isin(starters_pids | confirmed_pids)
        ]
    if confirmed_teams:
        scratched = (
            (candidates_df["position"] != "P")
            & candidates_df["team"].isin(confirmed_teams)
            & ~candidates_df["player_id"].isin(confirmed_pids)
        )
        candidates_df = candidates_df[~scratched]

    candidates_df = candidates_df.copy()
    candidates_df["newly_confirmed"] = candidates_df["player_id"].isin(
        confirmed_pids - (starters_pids or set())
    )

    if exclusions:
        excl_teams = set(exclusions.get("excluded_teams", [])) | set(
            exclusions.get("candidate_excluded_teams", [])
        )
        excl_games = set(exclusions.get("excluded_games", [])) | set(
            exclusions.get("candidate_excluded_games", [])
        )
        excl_pids = set(exclusions.get("excluded_player_ids", [])) | set(
            exclusions.get("candidate_excluded_player_ids", [])
        )
        candidates_df = candidates_df[
            ~candidates_df["team"].isin(excl_teams)
            & ~candidates_df["game"].isin(excl_games)
            & ~candidates_df["player_id"].isin(excl_pids)
        ]

    return lookup_df, candidates_df.copy()


def _player_record(row) -> dict:
    ep = row.get("eligible_positions")
    if not isinstance(ep, list) or not ep:
        ep = [str(row["position"])] if row.get("position") else []
    return {
        "player_id": int(row["player_id"]),
        "name": str(row.get("name", row["player_id"])),
        "team": str(row.get("team", "")),
        "opponent": str(row.get("opponent", "")),
        "position": str(row.get("position", "")),
        "eligible_positions": ep,
        "salary": int(row["salary"]) if pd.notna(row.get("salary")) else None,
        "mean": float(row["mean"]) if pd.notna(row.get("mean")) else None,
        "game": str(row.get("game", "")),
        "game_start_time": str(row["game_start_time"]) if row.get("game_start_time") else "",
        "missing_from_slate": False,
        "newly_confirmed": bool(row.get("newly_confirmed", False)),
    }


def build_player_lookup(lookup_df: pd.DataFrame) -> Dict[int, dict]:
    return {int(r["player_id"]): _player_record(r) for _, r in lookup_df.iterrows()}


def load_raw_salaries(slate_path) -> Dict[int, int]:
    """Recover {player_id: salary} straight from the raw DK CSV.

    Covers entry players the ingestor dropped (e.g. a PPD game) but that are
    still present in the salary file — their salary is needed for cap math.
    """
    try:
        sl = pd.read_csv(slate_path, usecols=["ID", "Salary"])
        sl["_pid"] = pd.to_numeric(sl["ID"], errors="coerce")
        sl = sl.dropna(subset=["_pid", "Salary"])
        return {int(r["_pid"]): int(r["Salary"]) for _, r in sl.iterrows()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Entry state
# ---------------------------------------------------------------------------

@dataclass
class SlotState:
    slot_index: int                       # 0..9
    slot_position: str                    # SLOTS[slot_index]
    original: Optional[EntrySlotPlayer]   # None = empty cell (reservation)
    locked: bool
    missing_from_slate: bool
    swapped_in_id: Optional[int] = None
    swap_source: Optional[str] = None     # "auto" | "manual"

    @property
    def current_player_id(self) -> Optional[int]:
        if self.swapped_in_id is not None:
            return self.swapped_in_id
        return self.original.player_id if self.original else None

    @property
    def changed(self) -> bool:
        orig_id = self.original.player_id if self.original else None
        return self.swapped_in_id is not None and self.swapped_in_id != orig_id


@dataclass
class EntrySwapState:
    entry_id: str
    source_file: str
    contest_name: str
    contest_id: str
    entry_fee_raw: str
    slots: List[SlotState]
    warnings: List[dict] = field(default_factory=list)

    @property
    def n_swappable(self) -> int:
        return sum(1 for s in self.slots if not s.locked)

    @property
    def changed(self) -> bool:
        return any(s.changed for s in self.slots)

    def has_unknown_player_data(self) -> bool:
        return any(w.get("reason") == "unknown_player_data" for w in self.warnings)


def build_entry_states(
    all_file_entries: List[Tuple[Path, List[EntryRecord]]],
    lookup: Dict[int, dict],
    raw_salaries: Dict[int, int],
    now: datetime,
) -> List[EntrySwapState]:
    """Build per-entry swap state, resolving each slot against the slate lookup.

    Players missing from the slate get a locked placeholder lookup record
    (salary recovered from raw_salaries when possible). Entries containing a
    player with no recoverable salary are flagged `unknown_player_data` —
    the salary cap can't be verified, so auto-swap skips them.
    """
    states: List[EntrySwapState] = []
    for file_path, records in all_file_entries:
        for rec in records:
            slot_players = list(rec.slot_players) + [None] * (10 - len(rec.slot_players))
            slots: List[SlotState] = []
            warnings: List[dict] = []
            for i, sp in enumerate(slot_players[:10]):
                if sp is None:
                    slots.append(SlotState(i, SLOTS[i], None, locked=False,
                                           missing_from_slate=False))
                    continue
                pid = sp.player_id
                meta = lookup.get(pid) if pid is not None else None
                if meta is None:
                    salary = raw_salaries.get(pid) if pid is not None else None
                    placeholder = {
                        "player_id": pid,
                        "name": sp.name or (f"#{pid}" if pid is not None else ""),
                        "team": "",
                        "opponent": "", "position": "", "eligible_positions": [],
                        "salary": salary, "mean": None, "game": "",
                        "game_start_time": "", "missing_from_slate": True,
                    }
                    if pid is not None:
                        lookup[pid] = placeholder
                    slots.append(SlotState(i, SLOTS[i], sp, locked=True,
                                           missing_from_slate=True))
                    if salary is None:
                        warnings.append({"slot_index": i, "reason": "unknown_player_data"})
                else:
                    locked = is_game_started(meta.get("game_start_time"), now)
                    slots.append(SlotState(i, SLOTS[i], sp, locked=locked,
                                           missing_from_slate=False))
            states.append(EntrySwapState(
                entry_id=rec.entry_id,
                source_file=file_path.name,
                contest_name=rec.contest_name,
                contest_id=rec.contest_id,
                entry_fee_raw=rec.entry_fee_raw,
                slots=slots,
                warnings=warnings,
            ))
    return states


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def eligible_candidates(
    entry: EntrySwapState,
    slot_index: int,
    candidates: List[dict],
    lookup: Dict[int, dict],
    per_entry_excluded: Set[int],
    now: datetime,
    vacant_slot_indices: Optional[Set[int]] = None,
    salary_reserve: float = 0.0,
) -> List[dict]:
    """
    Return the candidates eligible to fill `slot_index`, checked against the
    entry's other 9 slots (vacant slots contribute nothing; their cost is
    covered by `salary_reserve`).

    Mirrors Lineup.is_valid's constraint set for pinned columns: slot
    position eligibility, game not started, no duplicates, salary cap,
    max hitters per team, pitcher-vs-opposing-batter (both directions),
    two pitchers from different teams, and min distinct games (only
    enforced once no vacancies remain to fix it later).
    """
    vacant = set(vacant_slot_indices or set())
    vacant.discard(slot_index)
    slot_position = entry.slots[slot_index].slot_position

    others: List[dict] = []
    for s in entry.slots:
        if s.slot_index == slot_index or s.slot_index in vacant:
            continue
        pid = s.current_player_id
        if pid is None:
            vacant.add(s.slot_index)  # empty cell with no fill yet
            continue
        meta = lookup.get(pid)
        if meta is None or meta.get("salary") is None:
            return []  # cap unverifiable — no eligible candidates
        others.append(meta)

    others_salary = sum(m["salary"] for m in others)
    budget = SALARY_CAP - others_salary - salary_reserve
    other_ids = {m["player_id"] for m in others}
    batter_teams = Counter(m["team"] for m in others if m["position"] != "P" and m["team"])
    pitcher_opponents = {m["opponent"] for m in others if m["position"] == "P" and m["opponent"]}
    pitcher_teams = [m["team"] for m in others if m["position"] == "P" and m["team"]]
    games = {m["game"] for m in others if m["game"]}
    has_vacancies = bool(vacant)

    out: List[dict] = []
    for c in candidates:
        if slot_position not in c["eligible_positions"]:
            continue
        if is_game_started(c.get("game_start_time"), now):
            continue
        if c["player_id"] in other_ids or c["player_id"] in per_entry_excluded:
            continue
        if c["salary"] is None or c["salary"] > budget:
            continue
        if c["position"] != "P":
            if batter_teams.get(c["team"], 0) >= MAX_HITTERS_PER_TEAM:
                continue
            if c["team"] in pitcher_opponents:
                continue
        else:
            if c["opponent"] and c["opponent"] in batter_teams:
                continue
            if c["team"] in pitcher_teams:
                continue
        if not has_vacancies and c["game"]:
            if len(games | {c["game"]}) < MIN_GAMES:
                continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class CandidateScorer(Protocol):
    def score(self, cand: dict, entry: EntrySwapState,
              kept_batter_teams: Counter, usage: Counter) -> float: ...


class HeuristicScorer:
    """projection mean + same-team stack bonus + newly-confirmed boost
    - capped diversity penalty.

    The diversity penalty only reorders valid candidates — capped at 1.2
    FPTS it can never leave a slot unfilled or force a clearly worse pick
    (failing to swap costs far more EV than concentrated swaps).

    The newly-confirmed boost prioritizes players announced into a lineup
    after projections were fetched (post-lock replacements): their
    salary/600 heuristic mean understates a guaranteed starter.
    """
    W_STACK = 0.5            # FPTS per kept same-team batter, capped at 4 (+2.0 max)
    W_DIVERSITY = 0.3        # per prior auto swap-in of this player this run
    W_NEWLY_CONFIRMED = 2.0  # confirmed in a lineup but absent from the projections pool

    def score(self, cand: dict, entry: EntrySwapState,
              kept_batter_teams: Counter, usage: Counter) -> float:
        s = float(cand["mean"] or 0.0)
        if cand["position"] != "P":
            s += self.W_STACK * min(kept_batter_teams.get(cand["team"], 0), 4)
        if cand.get("newly_confirmed"):
            s += self.W_NEWLY_CONFIRMED
        s -= min(self.W_DIVERSITY * usage.get(cand["player_id"], 0), 1.2)
        return s


def _sort_key(scored: Tuple[float, dict]):
    score, c = scored
    return (-score, -(c["mean"] or 0.0), c["salary"] or 0, c["player_id"])


# ---------------------------------------------------------------------------
# Swap engine
# ---------------------------------------------------------------------------

def _effective_marks(
    entry: EntrySwapState,
    entry_marks: Dict[str, Set[int]],
    bulk_player_ids: Set[int],
    bulk_teams: Set[str],
    lookup: Dict[int, dict],
) -> Set[int]:
    """Slot indices to vacate: per-entry marks, bulk marks, and empty cells
    (implicitly marked). Locked slots are never vacated."""
    marked_ids = set(entry_marks.get(entry.entry_id, set())) | bulk_player_ids
    out: Set[int] = set()
    for s in entry.slots:
        if s.locked:
            continue
        if s.original is None:
            out.add(s.slot_index)
            continue
        pid = s.original.player_id
        if pid in marked_ids:
            out.add(s.slot_index)
        elif pid is not None and lookup.get(pid, {}).get("team") in bulk_teams:
            out.add(s.slot_index)
    return out


def run_swap(
    entry_states: List[EntrySwapState],
    entry_marks: Dict[str, Set[int]],
    bulk_player_ids: Set[int],
    bulk_teams: Set[str],
    candidates_df: pd.DataFrame,
    lookup: Dict[int, dict],
    scorer: CandidateScorer,
    now: datetime,
) -> None:
    """Fill marked slots across all entries in place (greedy, scarcest slot
    first within each entry). Bulk-marked players/teams are globally excluded
    from the candidate pool; per-entry marks only exclude within their entry."""
    base_candidates = [
        c for c in (_player_record(r) for _, r in candidates_df.iterrows())
        if c["player_id"] not in bulk_player_ids
        and c["team"] not in bulk_teams
        and not is_game_started(c.get("game_start_time"), now)
    ]
    # Position-only scarcity counts and per-position minimum salary (for reserves)
    pos_counts = {pos: sum(1 for c in base_candidates if pos in c["eligible_positions"])
                  for pos in set(SLOTS)}
    pos_min_salary = {
        pos: min((c["salary"] for c in base_candidates
                  if pos in c["eligible_positions"] and c["salary"] is not None),
                 default=0)
        for pos in set(SLOTS)
    }
    usage: Counter = Counter()

    def slot_reserve(entry: EntrySwapState, slot_index: int) -> float:
        """Worst-case cost of a still-vacated slot: an unfillable slot keeps
        its original player, so reserving the original's salary guarantees the
        cap is never busted by later keeps. Empty cells reserve the cheapest
        position-eligible candidate."""
        s = entry.slots[slot_index]
        if s.original is not None and s.original.player_id is not None:
            meta = lookup.get(s.original.player_id)
            if meta and meta.get("salary") is not None:
                return meta["salary"]
        return pos_min_salary.get(s.slot_position, 0)

    def fill_entry(entry: EntrySwapState, vacated: Set[int],
                   per_entry_excluded: Set[int]) -> Set[int]:
        """Greedy-fill the vacated slots; returns the unfillable slot indices."""
        order = sorted(vacated, key=lambda i: pos_counts.get(entry.slots[i].slot_position, 0))
        unfilled = set(order)
        unfillable: Set[int] = set()
        for slot_index in order:
            unfilled.discard(slot_index)
            reserve = sum(slot_reserve(entry, i) for i in unfilled)
            elig = eligible_candidates(
                entry, slot_index, base_candidates, lookup, per_entry_excluded,
                now, vacant_slot_indices=unfilled, salary_reserve=reserve,
            )
            if not elig:
                entry.warnings.append({"slot_index": slot_index, "reason": "no_valid_candidate"})
                unfillable.add(slot_index)
                continue
            kept_batter_teams = _kept_batter_teams(entry, lookup, exclude={slot_index} | unfilled)
            scored = [(scorer.score(c, entry, kept_batter_teams, usage), c) for c in elig]
            scored.sort(key=_sort_key)
            chosen = scored[0][1]
            slot = entry.slots[slot_index]
            slot.swapped_in_id = chosen["player_id"]
            slot.swap_source = "auto"
            usage[chosen["player_id"]] += 1
        return unfillable

    for entry in entry_states:
        if entry.has_unknown_player_data():
            continue
        vacated = _effective_marks(entry, entry_marks, bulk_player_ids, bulk_teams, lookup)
        if not vacated:
            continue
        per_entry_excluded = set(entry_marks.get(entry.entry_id, set()))

        unfillable = fill_entry(entry, vacated, per_entry_excluded)
        if not entry.changed or _entry_is_valid(entry, lookup):
            continue
        # An unfillable slot kept its original player, but earlier fills in
        # this entry treated it as vacant and never checked constraints
        # against that original (e.g. a new pitcher now opposes a kept
        # batter). Revert and retry with the unfillable slots treated as
        # kept from the start.
        _revert_auto_swaps(entry, usage)
        retry = vacated - unfillable
        if retry and retry != vacated:
            fill_entry(entry, retry, per_entry_excluded)
            if entry.changed and not _entry_is_valid(entry, lookup):
                _revert_auto_swaps(entry, usage)
                entry.warnings.append({"slot_index": None, "reason": "validation_failed_reverted"})
        else:
            entry.warnings.append({"slot_index": None, "reason": "validation_failed_reverted"})


def _kept_batter_teams(entry: EntrySwapState, lookup: Dict[int, dict],
                       exclude: Set[int]) -> Counter:
    teams: Counter = Counter()
    for s in entry.slots:
        if s.slot_index in exclude:
            continue
        pid = s.current_player_id
        if pid is None:
            continue
        meta = lookup.get(pid)
        if meta and meta["position"] != "P" and meta["team"]:
            teams[meta["team"]] += 1
    return teams


def _entry_is_valid(entry: EntrySwapState, lookup: Dict[int, dict]) -> bool:
    """Belt-and-suspenders Lineup.is_valid on the final 10. Returns True when
    the check passes or cannot apply (empty slot / incomplete metadata)."""
    final_ids = [s.current_player_id for s in entry.slots]
    if any(pid is None for pid in final_ids):
        return True  # still-empty slot — is_valid can't apply
    metas = {pid: lookup.get(pid) for pid in final_ids}
    if any(m is None or m.get("missing_from_slate") or m.get("salary") is None
           for m in metas.values()):
        return True
    player_meta = {
        pid: {
            "position": m["position"],
            "eligible_positions": m["eligible_positions"],
            "salary": m["salary"],
            "team": m["team"],
            "opponent": m["opponent"],
            "game": m["game"],
        }
        for pid, m in metas.items()
    }
    return Lineup(player_ids=list(final_ids)).is_valid(player_meta)


def _revert_auto_swaps(entry: EntrySwapState, usage: Counter) -> None:
    for s in entry.slots:
        if s.swap_source == "auto":
            if s.swapped_in_id is not None and usage.get(s.swapped_in_id, 0) > 0:
                usage[s.swapped_in_id] -= 1
            s.swapped_in_id = None
            s.swap_source = None


def apply_override(
    entry: EntrySwapState,
    slot_index: int,
    player_id: int,
    candidates_df: pd.DataFrame,
    lookup: Dict[int, dict],
    bulk_player_ids: Set[int],
    bulk_teams: Set[str],
    now: datetime,
) -> Optional[str]:
    """Manually set a slot's swap-in. Passing the slot's original player id
    reverts the swap. Returns an error reason, or None on success."""
    if not (0 <= slot_index < len(entry.slots)):
        return "invalid_slot"
    slot = entry.slots[slot_index]
    if slot.locked:
        return "slot_locked"
    if slot.swapped_in_id is not None and is_game_started(
        lookup.get(slot.swapped_in_id, {}).get("game_start_time"), now
    ):
        return "slot_locked"
    orig_id = slot.original.player_id if slot.original else None
    if player_id == orig_id and slot.swapped_in_id is not None:
        # Revert to the original player. Other swaps in the entry may have
        # spent the salary this swap freed, so re-validate before committing.
        prev_id, prev_source = slot.swapped_in_id, slot.swap_source
        slot.swapped_in_id = None
        slot.swap_source = None
        if not _entry_is_valid(entry, lookup):
            slot.swapped_in_id, slot.swap_source = prev_id, prev_source
            return "not_eligible"
        return None
    cands = [
        c for c in (_player_record(r) for _, r in candidates_df.iterrows())
        if c["player_id"] == player_id
        and c["player_id"] not in bulk_player_ids
        and c["team"] not in bulk_teams
    ]
    elig = eligible_candidates(entry, slot_index, cands, lookup,
                               per_entry_excluded=set(), now=now)
    if not elig:
        return "not_eligible"
    slot.swapped_in_id = player_id
    slot.swap_source = "manual"
    return None


def candidates_for_slot(
    entry: EntrySwapState,
    slot_index: int,
    candidates_df: pd.DataFrame,
    lookup: Dict[int, dict],
    bulk_player_ids: Set[int],
    bulk_teams: Set[str],
    now: datetime,
) -> List[dict]:
    """Eligible candidates for a slot, scored (usage zeroed — overrides are
    user judgment) and sorted best-first. For the candidates dropdown."""
    cands = [
        c for c in (_player_record(r) for _, r in candidates_df.iterrows())
        if c["player_id"] not in bulk_player_ids and c["team"] not in bulk_teams
    ]
    elig = eligible_candidates(entry, slot_index, cands, lookup,
                               per_entry_excluded=set(), now=now)
    kept = _kept_batter_teams(entry, lookup, exclude={slot_index})
    scorer = HeuristicScorer()
    out = []
    for c in elig:
        c = dict(c)
        c["score"] = round(scorer.score(c, entry, kept, Counter()), 2)
        out.append(c)
    out.sort(key=lambda c: (-c["score"], -(c["mean"] or 0.0),
                            c["salary"] or 0, c["player_id"]))
    return out


def slot_max_salary(entry: EntrySwapState, slot_index: int,
                    lookup: Dict[int, dict]) -> Optional[float]:
    """Remaining cap room for a slot given the other 9 current players."""
    total = 0
    for s in entry.slots:
        if s.slot_index == slot_index:
            continue
        pid = s.current_player_id
        if pid is None:
            continue
        meta = lookup.get(pid)
        if meta is None or meta.get("salary") is None:
            return None
        total += meta["salary"]
    return SALARY_CAP - total


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_swap_files(entry_states: List[EntrySwapState], output_dir: str) -> List[str]:
    """Write one swap_<OriginalName>.csv per source file, containing only the
    entries where at least one player changed. Sources with zero changed
    entries get no file (and a stale swap_ file from a prior run is removed)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_file: Dict[str, List[EntrySwapState]] = {}
    for e in entry_states:
        by_file.setdefault(e.source_file, []).append(e)

    written: List[str] = []
    for source_file, entries in by_file.items():
        swap_path = out / swap_file_name(source_file)
        changed = [e for e in entries if e.changed]
        if not changed:
            if swap_path.exists():
                swap_path.unlink()
            continue
        rows = []
        for e in changed:
            ids = [s.current_player_id for s in e.slots]
            rows.append(
                [e.entry_id, e.contest_name, e.contest_id, e.entry_fee_raw]
                + ["" if pid is None else str(pid) for pid in ids]
            )
        with open(swap_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(UPLOAD_HEADER)
            writer.writerows(rows)
        logger.info("Wrote %d swapped entries to %s", len(rows), swap_path)
        written.append(str(swap_path))
    return written


def delete_swap_files(output_dir: str) -> None:
    for p in Path(output_dir).glob("swap_*Entries*.csv"):
        p.unlink()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_state(
    output_dir: str,
    slate_fingerprint: str,
    run_at: str,
    bulk_player_ids: Set[int],
    bulk_teams: Set[str],
    entry_states: List[EntrySwapState],
    written_files: List[str],
) -> None:
    data = {
        "slate_fingerprint": slate_fingerprint,
        "run_at": run_at,
        "bulk_marked_player_ids": sorted(bulk_player_ids),
        "bulk_marked_teams": sorted(bulk_teams),
        "entries": {
            e.entry_id: {
                "source_file": e.source_file,
                "swaps": [
                    {
                        "slot_index": s.slot_index,
                        "out_player_id": s.original.player_id if s.original else None,
                        "in_player_id": s.swapped_in_id,
                        "source": s.swap_source,
                    }
                    for s in e.slots if s.swapped_in_id is not None
                ],
                "warnings": e.warnings,
            }
            for e in entry_states
            if e.warnings or any(s.swapped_in_id is not None for s in e.slots)
        },
        "written_files": written_files,
    }
    path = Path(output_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(output_dir: str, slate_fingerprint: str) -> Optional[dict]:
    """Load persisted swap state; None when absent or stale (slate changed)."""
    path = Path(output_dir) / STATE_FILENAME
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("slate_fingerprint") != slate_fingerprint:
        return None
    return data


def clear_state(output_dir: str) -> None:
    path = Path(output_dir) / STATE_FILENAME
    if path.exists():
        path.unlink()


def apply_saved_state(entry_states: List[EntrySwapState], saved: dict) -> None:
    """Re-apply persisted swaps onto freshly parsed entry states. A swap is
    only applied when the slot's current original matches the recorded
    out_player_id (guards against a re-downloaded entries file)."""
    saved_entries = saved.get("entries", {})
    for entry in entry_states:
        rec = saved_entries.get(entry.entry_id)
        if not rec:
            continue
        for sw in rec.get("swaps", []):
            i = sw.get("slot_index")
            if i is None or not (0 <= i < len(entry.slots)):
                continue
            slot = entry.slots[i]
            orig_id = slot.original.player_id if slot.original else None
            if orig_id != sw.get("out_player_id"):
                continue
            slot.swapped_in_id = sw.get("in_player_id")
            slot.swap_source = sw.get("source")
        for w in rec.get("warnings", []):
            if w not in entry.warnings:
                entry.warnings.append(w)


# ---------------------------------------------------------------------------
# API serialization
# ---------------------------------------------------------------------------

def _player_dict(meta: Optional[dict]) -> Optional[dict]:
    if meta is None:
        return None
    return {
        "player_id": meta["player_id"],
        "name": meta["name"],
        "team": meta["team"],
        "position": meta["position"],
        "eligible_positions": meta["eligible_positions"],
        "salary": meta["salary"],
        "mean": meta["mean"],
        "game": meta["game"],
        "game_start_time": meta["game_start_time"],
    }


def entry_to_dict(entry: EntrySwapState, lookup: Dict[int, dict], now: datetime) -> dict:
    slots = []
    for s in entry.slots:
        orig_meta = None
        if s.original is not None:
            if s.original.player_id is not None:
                orig_meta = lookup.get(s.original.player_id)
            if orig_meta is None:
                pid = s.original.player_id
                orig_meta = {
                    "player_id": pid,
                    "name": s.original.name or (f"#{pid}" if pid is not None else ""),
                    "team": "", "position": "", "eligible_positions": [],
                    "salary": None, "mean": None, "game": "", "game_start_time": "",
                }
        swapped_meta = lookup.get(s.swapped_in_id) if s.swapped_in_id is not None else None
        swapped_dict = _player_dict(swapped_meta)
        if swapped_dict is not None:
            swapped_dict["locked"] = is_game_started(
                swapped_meta.get("game_start_time"), now
            )
        slots.append({
            "slot_index": s.slot_index,
            "slot_position": s.slot_position,
            "player": _player_dict(orig_meta),
            "locked": s.locked,
            "missing_from_slate": s.missing_from_slate,
            "swapped_in": swapped_dict,
            "swap_source": s.swap_source,
        })
    return {
        "entry_id": entry.entry_id,
        "source_file": entry.source_file,
        "contest_name": entry.contest_name,
        "contest_id": entry.contest_id,
        "entry_fee": entry.entry_fee_raw,
        "n_swappable": entry.n_swappable,
        "warnings": entry.warnings,
        "slots": slots,
    }
