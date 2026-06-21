"""
Tests for the Late Swap feature.

Coverage:
- parse_entry_file slot-player extension: ids/names parsed, empty cells,
  name-without-id, reference-pool rows skipped, metadata regression
- Locking: started/future/empty start time, missing-from-slate
- build_swap_pools: projection merge, salary/600 fallback, lineup_slot
  starter filtering, exclusion removal
- eligible_candidates: each constraint rejected individually (position,
  started, duplicate, salary, hitter cap, batter-vs-opposing-pitcher,
  pitcher-vs-opposing-batter, same-team second pitcher, min games,
  min-games deferral with vacancies)
- HeuristicScorer: stack bonus monotone + capped, diversity capped
- run_swap: top candidate chosen, bulk team marks, per-entry marks local,
  diversity flip, empty-cell implicit fill, no-candidate warning, locked
  slots untouched, finals pass Lineup.is_valid
- write_swap_files: only changed entries, UPLOAD_HEADER, fixed columns,
  stale file deletion
- Persistence: save/load round trip, fingerprint mismatch, apply guards
- apply_override: success, not_eligible, slot_locked
"""
import csv
import io
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.api.dk_entries import UPLOAD_HEADER, EntrySlotPlayer, parse_entry_file
from src.api.late_swap import (
    EntrySwapState,
    HeuristicScorer,
    SlotState,
    apply_override,
    apply_saved_state,
    build_entry_states,
    build_player_lookup,
    build_swap_pools,
    eligible_candidates,
    entry_to_dict,
    is_game_started,
    load_state,
    recompute_locks,
    run_swap,
    save_state,
    scan_swap_entry_files,
    write_swap_files,
)
from src.optimization.lineup import Lineup, SLOTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 6, 11, 19, 30)       # 7:30 PM ET
T_LOCK = "2026-06-11T19:05:00"            # started
T_OPEN1 = "2026-06-11T21:05:00"           # not started
T_OPEN2 = "2026-06-11T22:10:00"           # not started

# pid, name, position, eligible_positions, salary, team, opponent, game, start, mean
_PLAYERS = [
    (1,  "P One",         "P",  ["P"],        8000, "AAA", "BBB", "AAA@BBB", T_LOCK,  15.0),
    (2,  "C Two",         "C",  ["C"],        3500, "AAA", "BBB", "AAA@BBB", T_LOCK,   8.0),
    (3,  "B Three",       "1B", ["1B"],       4500, "AAA", "BBB", "AAA@BBB", T_LOCK,   9.0),
    (4,  "B Four",        "2B", ["2B"],       4200, "BBB", "AAA", "AAA@BBB", T_LOCK,   8.4),
    (5,  "B Five",        "3B", ["3B"],       4300, "BBB", "AAA", "AAA@BBB", T_LOCK,   8.6),
    (6,  "S Six",         "SS", ["SS"],       4400, "BBB", "AAA", "AAA@BBB", T_LOCK,   8.8),
    (7,  "O Seven",       "OF", ["OF"],       4000, "AAA", "BBB", "AAA@BBB", T_LOCK,   8.0),
    (8,  "O Eight",       "OF", ["OF"],       5000, "AAA", "BBB", "AAA@BBB", T_LOCK,  10.0),
    (9,  "O Nine",        "OF", ["OF"],       5200, "BBB", "AAA", "AAA@BBB", T_LOCK,  10.4),
    (10, "P Ten",         "P",  ["P"],        6800, "CCC", "DDD", "CCC@DDD", T_OPEN1, 14.0),
    (11, "P Eleven",      "P",  ["P"],        6500, "DDD", "CCC", "CCC@DDD", T_OPEN1, 13.5),
    (12, "C Twelve",      "C",  ["C"],        3800, "CCC", "DDD", "CCC@DDD", T_OPEN1,  7.6),
    (13, "B Thirteen",    "1B", ["1B"],       4100, "CCC", "DDD", "CCC@DDD", T_OPEN1,  8.2),
    (14, "B Fourteen",    "2B", ["2B"],       3900, "CCC", "DDD", "CCC@DDD", T_OPEN1,  7.8),
    (15, "B Fifteen",     "3B", ["3B"],       4600, "DDD", "CCC", "CCC@DDD", T_OPEN1,  9.2),
    (16, "S Sixteen",     "SS", ["SS"],       4800, "DDD", "CCC", "CCC@DDD", T_OPEN1,  9.6),
    (17, "O Seventeen",   "OF", ["OF"],       4700, "CCC", "DDD", "CCC@DDD", T_OPEN1,  9.4),
    (18, "O Eighteen",    "OF", ["OF"],       5100, "CCC", "DDD", "CCC@DDD", T_OPEN1, 10.2),
    (19, "O Nineteen",    "OF", ["OF"],       5300, "DDD", "CCC", "CCC@DDD", T_OPEN1, 10.6),
    (20, "O Twenty",      "OF", ["OF"],       3500, "DDD", "CCC", "CCC@DDD", T_OPEN1,  7.0),
    (21, "C TwentyOne",   "C",  ["C"],        3000, "DDD", "CCC", "CCC@DDD", T_OPEN1,  6.0),
    (22, "B TwentyTwo",   "2B", ["2B"],       5000, "DDD", "CCC", "CCC@DDD", T_OPEN1, 10.0),
    (23, "F TwentyThree", "1B", ["1B", "OF"], 4400, "DDD", "CCC", "CCC@DDD", T_OPEN1,  8.8),
    (24, "O TwentyFour",  "OF", ["OF"],       4000, "CCC", "DDD", "CCC@DDD", T_OPEN1,  8.0),
    (30, "P Thirty",      "P",  ["P"],        7000, "EEE", "FFF", "EEE@FFF", T_OPEN2, 14.5),
    (31, "C ThirtyOne",   "C",  ["C"],        3600, "EEE", "FFF", "EEE@FFF", T_OPEN2,  7.2),
    (32, "B ThirtyTwo",   "1B", ["1B"],       4000, "FFF", "EEE", "EEE@FFF", T_OPEN2,  8.0),
    (33, "O ThirtyThree", "OF", ["OF"],       4500, "EEE", "FFF", "EEE@FFF", T_OPEN2,  9.0),
    (34, "O ThirtyFour",  "OF", ["OF"],       4900, "FFF", "EEE", "EEE@FFF", T_OPEN2,  9.8),
    (35, "S ThirtyFive",  "SS", ["SS"],       4100, "EEE", "FFF", "EEE@FFF", T_OPEN2,  8.2),
    (36, "B ThirtySix",   "2B", ["2B"],       3700, "FFF", "EEE", "EEE@FFF", T_OPEN2,  7.4),
    (37, "B ThirtySeven", "3B", ["3B"],       4200, "EEE", "FFF", "EEE@FFF", T_OPEN2,  8.4),
    (38, "O ThirtyEight", "OF", ["OF"],       6000, "EEE", "FFF", "EEE@FFF", T_OPEN2, 12.0),
    (39, "P ThirtyNine",  "P",  ["P"],        6000, "FFF", "EEE", "EEE@FFF", T_OPEN2, 12.5),
    (46, "P FortySix",    "P",  ["P"],        5800, "EEE", "FFF", "EEE@FFF", T_OPEN2, 11.6),
    (47, "P FortySeven",  "P",  ["P"],        6000, "CCC", "DDD", "CCC@DDD", T_OPEN1, 12.0),
    (48, "O FortyEight",  "OF", ["OF"],       4100, "BBB", "AAA", "AAA@BBB", T_LOCK,   8.2),
    (49, "O FortyNine",   "OF", ["OF"],       4500, "EEE", "FFF", "EEE@FFF", T_OPEN2,  8.9),
]

NAMES = {p[0]: p[1] for p in _PLAYERS}

# Entry 1: mixed locked/unlocked, total salary 49,900, passes Lineup.is_valid
ENTRY1_PIDS = [1, 30, 2, 13, 14, 15, 16, 7, 17, 19]


def _slate_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": pid, "name": name, "position": pos,
                "eligible_positions": elig, "salary": sal, "team": team,
                "opponent": opp, "game": game, "game_start_time": start,
                "mean": mean,
            }
            for pid, name, pos, elig, sal, team, opp, game, start, mean in _PLAYERS
        ]
    )


@pytest.fixture
def slate_df():
    return _slate_df()


@pytest.fixture
def lookup(slate_df):
    return build_player_lookup(slate_df)


def _player_cell(pid) -> str:
    if pid is None:
        return ""
    if isinstance(pid, str):
        return pid  # raw cell (e.g. name without id)
    return f"{NAMES[pid]} ({pid})"


def _entry_csv_text(entries: list[dict], with_reference_pool: bool = True) -> str:
    """entries: dicts with entry_id, contest_name, contest_id, fee, players (len 10)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(UPLOAD_HEADER + ["", "Name", "ID"])
    for e in entries:
        writer.writerow(
            [e["entry_id"], e["contest_name"], e["contest_id"], e["fee"]]
            + [_player_cell(p) for p in e["players"]]
            + ["", "Ref Player", "999999"]
        )
    if with_reference_pool:
        writer.writerow([""] * 14 + ["", "Pool Player", "888888"])
    return buf.getvalue()


def _write_entry_file(tmp_path: Path, filename: str, entries: list[dict]) -> Path:
    p = tmp_path / filename
    p.write_text(_entry_csv_text(entries))
    return p


def _entry(entry_id: str, players: list, contest="MLB $5K Test", cid="111", fee="$4") -> dict:
    return {"entry_id": entry_id, "contest_name": contest, "contest_id": cid,
            "fee": fee, "players": players}


def _states_for(tmp_path, lookup, file_entries: dict[str, list[dict]],
                raw_salaries=None) -> list[EntrySwapState]:
    """file_entries: {filename: [entry dicts]} -> parsed EntrySwapStates."""
    all_file_entries = []
    for filename, entries in file_entries.items():
        path = _write_entry_file(tmp_path, filename, entries)
        all_file_entries.append((path, parse_entry_file(path)))
    return build_entry_states(all_file_entries, lookup, raw_salaries or {}, NOW)


def _slot_by_pid(entry: EntrySwapState, pid: int) -> SlotState:
    return next(s for s in entry.slots if s.original and s.original.player_id == pid)


# ---------------------------------------------------------------------------
# parse_entry_file slot players
# ---------------------------------------------------------------------------

class TestParseEntryFileSlotPlayers:
    def test_slot_players_parsed(self, tmp_path):
        path = _write_entry_file(tmp_path, "MEDKEntries.csv", [_entry("100", ENTRY1_PIDS)])
        records = parse_entry_file(path)
        assert len(records) == 1
        rec = records[0]
        assert len(rec.slot_players) == 10
        assert [sp.player_id for sp in rec.slot_players] == ENTRY1_PIDS
        assert rec.slot_players[0].name == "P One"

    def test_empty_cell_is_none(self, tmp_path):
        players = ENTRY1_PIDS[:9] + [None]
        path = _write_entry_file(tmp_path, "MEDKEntries.csv", [_entry("100", players)])
        rec = parse_entry_file(path)[0]
        assert rec.slot_players[9] is None
        assert rec.slot_players[8] is not None

    def test_name_without_id(self, tmp_path):
        players = ENTRY1_PIDS[:9] + ["Mystery Man"]
        path = _write_entry_file(tmp_path, "MEDKEntries.csv", [_entry("100", players)])
        rec = parse_entry_file(path)[0]
        assert rec.slot_players[9] == EntrySlotPlayer(name="Mystery Man", player_id=None)

    def test_bare_id_cells(self, tmp_path):
        # upload_*.csv files written by write_upload_files carry bare IDs
        players = [str(pid) for pid in ENTRY1_PIDS]
        path = _write_entry_file(tmp_path, "upload_MEDKEntries.csv", [_entry("100", players)])
        rec = parse_entry_file(path)[0]
        assert [sp.player_id for sp in rec.slot_players] == ENTRY1_PIDS
        assert all(sp.name == "" for sp in rec.slot_players)

    def test_reference_pool_rows_skipped(self, tmp_path):
        path = _write_entry_file(tmp_path, "MEDKEntries.csv", [_entry("100", ENTRY1_PIDS)])
        assert len(parse_entry_file(path)) == 1

    def test_metadata_regression(self, tmp_path):
        path = _write_entry_file(tmp_path, "MEDKEntries.csv", [_entry("100", ENTRY1_PIDS)])
        rec = parse_entry_file(path)[0]
        assert rec.entry_id == "100"
        assert rec.contest_name == "MLB $5K Test"
        assert rec.contest_id == "111"
        assert rec.entry_fee_cents == 400
        assert rec.entry_fee_raw == "$4"
        assert rec.prize_pool_cents == 500_000  # $5K in cents


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class TestLocking:
    def test_is_game_started(self):
        assert is_game_started(T_LOCK, NOW) is True
        assert is_game_started(T_OPEN1, NOW) is False
        assert is_game_started("", NOW) is True
        assert is_game_started(None, NOW) is True
        assert is_game_started("not-a-date", NOW) is True

    def test_locked_flags(self, tmp_path, lookup):
        states = _states_for(tmp_path, lookup, {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        entry = states[0]
        locked = {s.original.player_id for s in entry.slots if s.locked}
        assert locked == {1, 2, 7}
        assert entry.n_swappable == 7

    def test_missing_from_slate(self, tmp_path, lookup):
        players = ENTRY1_PIDS[:9] + [999]
        NAMES[999] = "Ghost Player"
        try:
            states = _states_for(
                tmp_path, lookup, {"MEDKEntries.csv": [_entry("100", players)]},
                raw_salaries={999: 5300},
            )
        finally:
            del NAMES[999]
        slot = states[0].slots[9]
        assert slot.locked is True
        assert slot.missing_from_slate is True
        assert not states[0].has_unknown_player_data()  # salary recovered

    def test_missing_salary_flags_entry(self, tmp_path, lookup):
        players = ENTRY1_PIDS[:9] + [999]
        NAMES[999] = "Ghost Player"
        try:
            states = _states_for(tmp_path, lookup, {"MEDKEntries.csv": [_entry("100", players)]})
        finally:
            del NAMES[999]
        assert states[0].has_unknown_player_data()

    def test_all_locked_entry_zero_swappable(self, tmp_path, lookup):
        players = [1, 2, 3, 4, 5, 6, 7, 8, 9, 48]
        states = _states_for(tmp_path, lookup, {"MEDKEntries.csv": [_entry("100", players)]})
        assert states[0].n_swappable == 0


# ---------------------------------------------------------------------------
# build_swap_pools
# ---------------------------------------------------------------------------

class TestBuildSwapPools:
    def test_projection_merge_and_fallback(self, slate_df):
        slate = slate_df.drop(columns=["mean"])
        proj = pd.DataFrame({"player_id": [13, 17], "mean": [9.9, 11.1],
                             "lineup_slot": [3.0, 5.0]})
        lookup_df, candidates_df = build_swap_pools(slate, proj)
        lk = lookup_df.set_index("player_id")
        assert lk.loc[13, "mean"] == 9.9
        assert lk.loc[20, "mean"] == round(3500 / 600.0, 2)  # heuristic fallback
        assert set(candidates_df["player_id"]) == {13, 17}   # starters only

    def test_no_projections(self, slate_df):
        slate = slate_df.drop(columns=["mean"])
        lookup_df, candidates_df = build_swap_pools(slate, None)
        assert len(candidates_df) == len(slate)
        assert (lookup_df["mean"] > 0).all()

    def test_exclusions_removed(self, slate_df):
        exclusions = {
            "excluded_teams": ["DDD"],
            "candidate_excluded_player_ids": [33],
            "excluded_games": ["AAA@BBB"],
        }
        _, candidates_df = build_swap_pools(slate_df.drop(columns=["mean"]), None, exclusions)
        assert not (candidates_df["team"] == "DDD").any()
        assert 33 not in set(candidates_df["player_id"])
        assert not (candidates_df["game"] == "AAA@BBB").any()

    def test_projection_override_takes_precedence(self, slate_df):
        slate = slate_df.drop(columns=["mean"])
        proj = pd.DataFrame({"player_id": [13, 17], "mean": [4.8, 11.1],
                             "lineup_slot": [3.0, 5.0]})
        exclusions = {"player_projection_overrides": {"13": 6.5}}
        lookup_df, candidates_df = build_swap_pools(slate, proj, exclusions)
        lk = lookup_df.set_index("player_id")
        assert lk.loc[13, "mean"] == 6.5
        assert lk.loc[17, "mean"] == 11.1
        assert candidates_df.set_index("player_id").loc[13, "mean"] == 6.5

    def test_projection_override_on_heuristic_fallback(self, slate_df):
        slate = slate_df.drop(columns=["mean"])
        exclusions = {"player_projection_overrides": {"20": 12.3}}
        lookup_df, _ = build_swap_pools(slate, None, exclusions)
        assert lookup_df.set_index("player_id").loc[20, "mean"] == 12.3


# ---------------------------------------------------------------------------
# eligible_candidates
# ---------------------------------------------------------------------------

def _all_candidates(lookup):
    return [dict(m) for m in lookup.values()]


class TestEligibility:
    def _entry1(self, tmp_path, lookup) -> EntrySwapState:
        return _states_for(tmp_path, lookup,
                           {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})[0]

    def _eligible_ids(self, entry, slot_index, lookup, excluded=frozenset(), **kw):
        elig = eligible_candidates(entry, slot_index, _all_candidates(lookup),
                                   lookup, set(excluded), NOW, **kw)
        return {c["player_id"] for c in elig}

    def test_vacated_of_slot(self, tmp_path, lookup):
        entry = self._entry1(tmp_path, lookup)
        ids = self._eligible_ids(entry, 9, lookup, excluded={19})
        # 18/20/23/24/33/49 in; 8 (started), 17 (duplicate), 34 (vs opposing
        # pitcher 30), 38 (salary > 5400 budget), 12 (position) all out
        assert ids == {18, 20, 23, 24, 33, 49}

    def test_salary_infeasible_slot(self, tmp_path, lookup):
        entry = self._entry1(tmp_path, lookup)
        # vacate 1B (pid 13, budget $4,200): 23 too pricey, 32 vs opposing
        # pitcher, 3 started -> nothing
        assert self._eligible_ids(entry, 3, lookup, excluded={13}) == set()

    def test_pitcher_vs_kept_opposing_batters(self, tmp_path, lookup):
        entry = self._entry1(tmp_path, lookup)
        ids = self._eligible_ids(entry, 1, lookup, excluded={30})
        # 10 opposes kept DDD batters, 11 opposes kept CCC batters,
        # 46 (EEE) and 39 (FFF, no kept EEE batters... 15/16/19 are DDD) ok
        assert 10 not in ids and 11 not in ids
        assert 39 in ids and 46 in ids

    def test_same_team_second_pitcher(self, tmp_path, lookup):
        # Entry 4: P10 (CCC) + P30 (EEE), batters AAA/CCC/EEE
        players = [10, 30, 2, 3, 14, 37, 35, 7, 33, 17]
        entry = _states_for(tmp_path, lookup,
                            {"MEDKEntries.csv": [_entry("400", players)]})[0]
        ids = self._eligible_ids(entry, 1, lookup, excluded={30})
        assert 47 not in ids   # second CCC pitcher
        assert 39 not in ids   # FFF pitcher opposes kept EEE batters
        assert ids == {46}

    def test_hitter_cap(self, tmp_path, lookup):
        # 5 CCC batters kept (12,13,14,17,24); vacated OF can't add a 6th
        players = [1, 30, 12, 13, 14, 15, 16, 17, 24, 20]
        entry = _states_for(tmp_path, lookup,
                            {"MEDKEntries.csv": [_entry("500", players)]})[0]
        ids = self._eligible_ids(entry, 9, lookup, excluded={20})
        assert 18 not in ids  # CCC batter -> would be 6th
        assert 33 in ids

    def test_min_games(self, tmp_path, lookup):
        # 9 kept players all from CCC@DDD: candidate from the same game fails
        players = [12, 13, 14, 15, 16, 17, 18, 19, 22, 20]
        entry = _states_for(tmp_path, lookup,
                            {"MEDKEntries.csv": [_entry("600", players)]})[0]
        ids = self._eligible_ids(entry, 9, lookup, excluded={20})
        assert 23 not in ids  # same game as all 9 kept players
        assert 33 in ids

    def test_min_games_deferred_with_vacancies(self, tmp_path, lookup):
        players = [12, 13, 14, 15, 16, 17, 18, 19, 22, 20]
        entry = _states_for(tmp_path, lookup,
                            {"MEDKEntries.csv": [_entry("600", players)]})[0]
        ids = self._eligible_ids(entry, 9, lookup, excluded={20},
                                 vacant_slot_indices={8})
        assert 23 in ids  # another vacancy can still satisfy min games

    def test_salary_reserve(self, tmp_path, lookup):
        entry = self._entry1(tmp_path, lookup)
        # budget 5400; reserve 1000 shrinks it to 4400
        ids = self._eligible_ids(entry, 9, lookup, excluded={19}, salary_reserve=1000.0)
        assert 18 not in ids and 33 not in ids  # 5100/4500 > 4400
        assert 20 in ids and 24 in ids and 23 in ids


# ---------------------------------------------------------------------------
# HeuristicScorer
# ---------------------------------------------------------------------------

class TestHeuristicScorer:
    def _cand(self, pid=99, pos="OF", team="CCC", mean=10.0):
        return {"player_id": pid, "position": pos, "team": team, "mean": mean}

    def _entry(self):
        return EntrySwapState("1", "f.csv", "c", "1", "$1", slots=[])

    def test_stack_bonus_monotone_and_capped(self):
        scorer = HeuristicScorer()
        e = self._entry()
        scores = [scorer.score(self._cand(), e, Counter({"CCC": n}), Counter())
                  for n in range(6)]
        assert scores == sorted(scores)
        assert scores[1] - scores[0] == pytest.approx(scorer.W_STACK)
        assert scores[5] == scores[4]  # capped at 4 teammates

    def test_pitcher_gets_no_stack_bonus(self):
        scorer = HeuristicScorer()
        s = scorer.score(self._cand(pos="P"), self._entry(), Counter({"CCC": 3}), Counter())
        assert s == 10.0

    def test_diversity_penalty_capped(self):
        scorer = HeuristicScorer()
        e = self._entry()
        s2 = scorer.score(self._cand(), e, Counter(), Counter({99: 2}))
        s10 = scorer.score(self._cand(), e, Counter(), Counter({99: 10}))
        assert s2 == pytest.approx(10.0 - 0.6)
        assert s10 == pytest.approx(10.0 - 1.2)  # capped


# ---------------------------------------------------------------------------
# run_swap
# ---------------------------------------------------------------------------

class TestRunSwap:
    def _run(self, tmp_path, lookup, slate_df, file_entries, entry_marks=None,
             bulk_pids=frozenset(), bulk_teams=frozenset(), candidates_df=None):
        states = _states_for(tmp_path, lookup, file_entries)
        run_swap(
            states,
            {k: set(v) for k, v in (entry_marks or {}).items()},
            set(bulk_pids), set(bulk_teams),
            candidates_df if candidates_df is not None else slate_df,
            lookup, HeuristicScorer(), NOW,
        )
        return states

    def test_marked_player_replaced_by_top_score(self, tmp_path, lookup, slate_df):
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]},
                           entry_marks={"100": [19]})
        slot = states[0].slots[9]
        # 18 wins: mean 10.2 + 1.5 stack (3 kept CCC batters)
        assert slot.swapped_in_id == 18
        assert slot.swap_source == "auto"

    def test_final_lineup_is_valid(self, tmp_path, lookup, slate_df):
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]},
                           entry_marks={"100": [19, 30]})
        entry = states[0]
        final_ids = [s.current_player_id for s in entry.slots]
        assert all(pid is not None for pid in final_ids)
        meta = {pid: {k: lookup[pid][k] for k in
                      ("position", "eligible_positions", "salary", "team", "opponent", "game")}
                for pid in final_ids}
        assert Lineup(player_ids=final_ids).is_valid(meta)
        assert not any(w["reason"] == "validation_failed_reverted" for w in entry.warnings)

    def test_no_candidate_warning_keeps_original(self, tmp_path, lookup, slate_df):
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]},
                           entry_marks={"100": [13]})
        entry = states[0]
        assert entry.slots[3].swapped_in_id is None
        assert {"slot_index": 3, "reason": "no_valid_candidate"} in entry.warnings

    def test_bulk_team_across_entries(self, tmp_path, lookup, slate_df):
        entry2_pids = [1, 30, 2, 13, 14, 15, 16, 7, 24, 19]
        states = self._run(
            tmp_path, lookup, slate_df,
            {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS), _entry("200", entry2_pids)]},
            bulk_teams={"DDD"},
        )
        for entry in states:
            for s in entry.slots:
                pid = s.current_player_id
                if s.original and lookup[s.original.player_id]["team"] == "DDD" and not s.locked:
                    assert s.swapped_in_id is not None, f"DDD player not swapped in {entry.entry_id}"
                assert lookup[pid]["team"] != "DDD" or s.locked

    def test_diversity_never_blocks_only_candidate(self, tmp_path, lookup, slate_df):
        # 3B/SS each have exactly one candidate (37/35) — both entries must
        # still receive them despite the usage penalty.
        entry2_pids = [1, 30, 2, 13, 14, 15, 16, 7, 24, 19]
        states = self._run(
            tmp_path, lookup, slate_df,
            {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS), _entry("200", entry2_pids)]},
            bulk_teams={"DDD"},
        )
        for entry in states:
            assert entry.slots[5].swapped_in_id == 37
            assert entry.slots[6].swapped_in_id == 35

    def test_diversity_spreads_close_candidates(self, tmp_path, lookup, slate_df):
        # Pool restricted to two near-equal EEE OFs: 33 (9.0) and 49 (8.9).
        # Entry 1 takes 33; the 0.3 usage penalty flips entry 2 to 49.
        candidates_df = slate_df[slate_df["player_id"].isin([33, 49])]
        states = self._run(
            tmp_path, lookup, slate_df,
            {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS), _entry("200", ENTRY1_PIDS)]},
            entry_marks={"100": [19], "200": [19]},
            candidates_df=candidates_df,
        )
        assert states[0].slots[9].swapped_in_id == 33
        assert states[1].slots[9].swapped_in_id == 49

    def test_empty_cell_implicitly_filled(self, tmp_path, lookup, slate_df):
        players = ENTRY1_PIDS[:9] + [None]
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("100", players)]})
        slot = states[0].slots[9]
        assert slot.swapped_in_id == 18
        assert states[0].changed

    def test_per_entry_mark_is_local(self, tmp_path, lookup, slate_df):
        # 18 marked out of entry 100 only; entry 200's vacated OF still gets 18.
        states = self._run(
            tmp_path, lookup, slate_df,
            {"MEDKEntries.csv": [
                _entry("100", [1, 30, 2, 13, 14, 15, 16, 7, 18, 19]),
                _entry("200", ENTRY1_PIDS),
            ]},
            entry_marks={"100": [18], "200": [19]},
        )
        assert states[0].slots[8].swapped_in_id != 18
        assert states[1].slots[9].swapped_in_id == 18

    def test_bulk_marked_player_excluded_everywhere(self, tmp_path, lookup, slate_df):
        states = self._run(
            tmp_path, lookup, slate_df,
            {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]},
            entry_marks={"100": [19]},
            bulk_pids={18},
        )
        # 18 is globally excluded; next best is 23 (8.8 + 1.0 DDD stack)
        assert states[0].slots[9].swapped_in_id == 23

    def test_locked_slots_never_vacated(self, tmp_path, lookup, slate_df):
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]},
                           bulk_teams={"AAA"})
        entry = states[0]
        assert all(s.swapped_in_id is None for s in entry.slots if s.locked)

    def test_unfillable_slot_triggers_consistent_retry(self, tmp_path, lookup, slate_df):
        # Mark P30 (EEE) and OF33 (EEE). Pool = {39: P FFF opposing EEE,
        # 38: OF EEE}. Pass 1 fills P with 39 (slot 9 presumed vacated),
        # then slot 9 is unfillable (38 opposes the new pitcher) and keeps
        # 33 (EEE) — making 39 oppose a kept batter. The retry treats slot 9
        # as kept from the start, so 39 is rejected and nothing swaps.
        players = [1, 30, 2, 13, 14, 15, 16, 7, 17, 33]
        candidates_df = slate_df[slate_df["player_id"].isin([39, 38])]
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("700", players)]},
                           entry_marks={"700": [30, 33]},
                           candidates_df=candidates_df)
        entry = states[0]
        assert not entry.changed
        warned_slots = {w["slot_index"] for w in entry.warnings}
        assert warned_slots == {1, 9}
        assert all(w["reason"] == "no_valid_candidate" for w in entry.warnings)

    def test_originals_reserve_prevents_cap_bust(self, tmp_path, lookup, slate_df):
        # Mark 2B 14 ($3.9k) and OF 17 ($4.7k) on the $49.9k entry. Pool:
        # OF = {38 ($6k)}, 2B = {36 (blocked by opposing pitcher 30), 22 ($5k)}.
        # OF fills first (scarcer). Without originals-based reserves, 38
        # would be affordable only by spending the budget slot 4 needs when
        # it ends up keeping its original — busting the $50k cap. With them,
        # 38 is rejected up front and the entry simply keeps its originals.
        candidates_df = slate_df[slate_df["player_id"].isin([38, 36, 22])]
        states = self._run(tmp_path, lookup, slate_df,
                           {"MEDKEntries.csv": [_entry("800", ENTRY1_PIDS)]},
                           entry_marks={"800": [14, 17]},
                           candidates_df=candidates_df)
        entry = states[0]
        total = sum(lookup[s.current_player_id]["salary"] for s in entry.slots)
        assert total <= 50_000
        assert not any(w["reason"] == "validation_failed_reverted" for w in entry.warnings)
        assert {w["slot_index"] for w in entry.warnings} == {4, 8}

    def test_unknown_player_data_entry_skipped(self, tmp_path, lookup, slate_df):
        players = ENTRY1_PIDS[:9] + [999]
        NAMES[999] = "Ghost Player"
        try:
            states = _states_for(tmp_path, lookup,
                                 {"MEDKEntries.csv": [_entry("100", players)]})
        finally:
            del NAMES[999]
        run_swap(states, {"100": {13}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        assert not states[0].changed


# ---------------------------------------------------------------------------
# write_swap_files
# ---------------------------------------------------------------------------

class TestWriteSwapFiles:
    def test_only_changed_entries_written(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup, {
            "MEDKEntries.csv": [_entry("100", ENTRY1_PIDS), _entry("200", ENTRY1_PIDS)],
            "GEDKEntries.csv": [_entry("300", ENTRY1_PIDS)],
        })
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        out_dir = tmp_path / "outputs"
        written = write_swap_files(states, str(out_dir), lookup)
        # The OF trio (7="O Seven", 17="O Seventeen", 18="O Eighteen" after the
        # swap) reorders alphabetically by last name in the primary file
        # ("Eighteen" < "Seven" < "Seventeen"), which differs from the
        # original pinned columns -- so the swap_reversed_ hedge file is
        # also written.
        assert written == [
            str(out_dir / "swap_MEDKEntries.csv"),
            str(out_dir / "swap_reversed_MEDKEntries.csv"),
        ]
        with open(out_dir / "swap_MEDKEntries.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == UPLOAD_HEADER
        assert len(rows) == 2  # header + entry 100 only
        expected = [str(p) for p in ENTRY1_PIDS[:7]] + ["18", "7", "17"]
        assert rows[1] == ["100", "MLB $5K Test", "111", "$4"] + expected

        with open(out_dir / "swap_reversed_MEDKEntries.csv") as f:
            reversed_rows = list(csv.reader(f))
        # locked/kept players keep their original columns; swapped slot has new id
        reversed_expected = [str(p) for p in ENTRY1_PIDS[:9]] + ["18"]
        assert reversed_rows[1] == ["100", "MLB $5K Test", "111", "$4"] + reversed_expected

    def test_stale_swap_file_deleted(self, tmp_path, lookup):
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        stale = out_dir / "swap_MEDKEntries.csv"
        stale_reversed = out_dir / "swap_reversed_MEDKEntries.csv"
        stale.write_text("stale")
        stale_reversed.write_text("stale")
        written = write_swap_files(states, str(out_dir), lookup)  # nothing changed
        assert written == []
        assert not stale.exists()
        assert not stale_reversed.exists()

    def test_upload_prefix_stripped_from_swap_name(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup,
                             {"upload_MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        out_dir = tmp_path / "outputs"
        written = write_swap_files(states, str(out_dir), lookup)
        assert written == [
            str(out_dir / "swap_MEDKEntries.csv"),
            str(out_dir / "swap_reversed_MEDKEntries.csv"),
        ]


class TestScanSwapEntryFiles:
    def test_scan_includes_uploads_excludes_swaps(self, tmp_path):
        for name in ["upload_MEDKEntries.csv", "upload_GEDKEntries.csv",
                     "GEDKEntries.csv", "swap_MEDKEntries.csv",
                     "upload_FanDuel.csv", "portfolio_draftkings.csv"]:
            (tmp_path / name).write_text("x")
        found = [p.name for p in scan_swap_entry_files(str(tmp_path))]
        assert found == ["GEDKEntries.csv", "upload_GEDKEntries.csv", "upload_MEDKEntries.csv"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def _swapped_states(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        return states

    def test_round_trip(self, tmp_path, lookup, slate_df):
        states = self._swapped_states(tmp_path, lookup, slate_df)
        out_dir = str(tmp_path / "outputs")
        save_state(out_dir, "fp1", "2026-06-11T19:35:00", {5}, {"DDD"}, states, ["x.csv"])
        saved = load_state(out_dir, "fp1")
        assert saved is not None
        assert saved["bulk_marked_player_ids"] == [5]
        assert saved["bulk_marked_teams"] == ["DDD"]
        assert saved["entries"]["100"]["swaps"] == [
            {"slot_index": 9, "out_player_id": 19, "in_player_id": 18, "source": "auto"}
        ]

        fresh = _states_for(tmp_path, lookup,
                            {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        apply_saved_state(fresh, saved)
        assert fresh[0].slots[9].swapped_in_id == 18
        assert fresh[0].slots[9].swap_source == "auto"

    def test_fingerprint_mismatch_discards(self, tmp_path, lookup, slate_df):
        states = self._swapped_states(tmp_path, lookup, slate_df)
        out_dir = str(tmp_path / "outputs")
        save_state(out_dir, "fp1", "t", set(), set(), states, [])
        assert load_state(out_dir, "other-fp") is None

    def test_apply_guards_on_changed_original(self, tmp_path, lookup, slate_df):
        states = self._swapped_states(tmp_path, lookup, slate_df)
        out_dir = str(tmp_path / "outputs")
        save_state(out_dir, "fp1", "t", set(), set(), states, [])
        saved = load_state(out_dir, "fp1")
        # entries file re-downloaded with a different player in slot 9
        changed = _states_for(tmp_path, lookup, {
            "MEDKEntries.csv": [_entry("100", ENTRY1_PIDS[:9] + [20])]})
        apply_saved_state(changed, saved)
        assert changed[0].slots[9].swapped_in_id is None

    def test_recompute_locks_uses_current_occupant(self, tmp_path, lookup, slate_df):
        """A slot's lock must track whoever is actually rostered there now,
        not the player a prior swap replaced — otherwise a still-open
        swapped-in player gets stuck behind their (already-started)
        predecessor's lock."""
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        # pid 19 (CCC/DDD game, T_OPEN1=21:05) swaps to an EEE/FFF OF
        # (T_OPEN2=22:10) by excluding the other CCC/DDD OF candidates.
        run_swap(states, {"100": {19}}, {18, 17, 24}, set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        repl = states[0].slots[9].swapped_in_id
        assert lookup[repl]["game_start_time"] == "2026-06-11T22:10:00"
        out_dir = str(tmp_path / "outputs")
        save_state(out_dir, "fp1", "t", {18, 17, 24}, set(), states, [])
        saved = load_state(out_dir, "fp1")

        # 21:30: the original (19) has started; the swapped-in replacement
        # hasn't.
        later = datetime(2026, 6, 11, 21, 30)
        path = tmp_path / "MEDKEntries.csv"
        fresh = build_entry_states([(path, parse_entry_file(path))], lookup, {}, later)
        assert fresh[0].slots[9].locked is True  # built fresh, pre-swap: original (19) has started
        apply_saved_state(fresh, saved)
        recompute_locks(fresh, lookup, later)
        assert fresh[0].slots[9].locked is False
        assert fresh[0].slots[9].swapped_in_id == repl


# ---------------------------------------------------------------------------
# apply_override
# ---------------------------------------------------------------------------

class TestApplyOverride:
    def _swapped_entry(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        return states[0]

    def test_override_success(self, tmp_path, lookup, slate_df):
        entry = self._swapped_entry(tmp_path, lookup, slate_df)
        err = apply_override(entry, 9, 33, slate_df, lookup, set(), set(), NOW)
        assert err is None
        assert entry.slots[9].swapped_in_id == 33
        assert entry.slots[9].swap_source == "manual"

    def test_override_not_eligible(self, tmp_path, lookup, slate_df):
        entry = self._swapped_entry(tmp_path, lookup, slate_df)
        assert apply_override(entry, 9, 38, slate_df, lookup, set(), set(), NOW) == "not_eligible"
        assert entry.slots[9].swapped_in_id == 18  # unchanged

    def test_override_locked_slot(self, tmp_path, lookup, slate_df):
        entry = self._swapped_entry(tmp_path, lookup, slate_df)
        assert apply_override(entry, 0, 39, slate_df, lookup, set(), set(), NOW) == "slot_locked"

    def test_override_bulk_excluded(self, tmp_path, lookup, slate_df):
        entry = self._swapped_entry(tmp_path, lookup, slate_df)
        assert apply_override(entry, 9, 33, slate_df, lookup, {33}, set(), NOW) == "not_eligible"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestEntryToDict:
    def test_shape(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        d = entry_to_dict(states[0], lookup, NOW)
        assert d["entry_id"] == "100"
        assert d["source_file"] == "MEDKEntries.csv"
        assert d["n_swappable"] == 7
        assert len(d["slots"]) == 10
        s0 = d["slots"][0]
        assert s0["locked"] is True
        assert s0["player"]["player_id"] == 1
        s9 = d["slots"][9]
        assert s9["swapped_in"]["player_id"] == 18
        assert s9["swapped_in"]["locked"] is False
        assert s9["swap_source"] == "auto"


# ---------------------------------------------------------------------------
# Portfolio display order harmonization
# ---------------------------------------------------------------------------

class TestUploadDisplayOrder:
    def test_serialized_players_match_upload_columns(self, slate_df):
        from src.api.dk_entries import assign_players_to_slots
        from src.api.pipeline import PipelineRunner

        shuffled = [19, 14, 30, 16, 2, 1, 13, 17, 7, 15]  # entry1, scrambled
        ordered = PipelineRunner._upload_display_order(shuffled, slate_df)
        assert ordered == assign_players_to_slots(shuffled, slate_df)
        # pitchers land in the two P columns in input order: 30 before 1
        assert [pid for pid in ordered if pid in (1, 30)] == ordered[:2]
        assert sorted(ordered) == sorted(shuffled)

    def test_fallback_on_unassignable_roster(self, slate_df):
        from src.api.pipeline import PipelineRunner

        nine = ENTRY1_PIDS[:9]  # FD-sized roster can't fill 10 DK slots
        assert PipelineRunner._upload_display_order(nine, slate_df) == nine


# ---------------------------------------------------------------------------
# Confirmed-lineup pool inclusion and newly-confirmed priority
# ---------------------------------------------------------------------------

class TestConfirmedLineupPool:
    def test_confirmed_player_joins_pool_with_heuristic_mean(self, slate_df):
        slate = slate_df.drop(columns=["mean"])
        # Projections know starters 13/17 only; pid 20 (DDD) is announced in
        # a confirmed lineup after the fetch (the Chad Stevens case).
        proj = pd.DataFrame({"player_id": [13, 17], "mean": [9.9, 11.1],
                             "lineup_slot": [3.0, 5.0]})
        confirmed = {"DDD": {20: 7, 15: 5}}
        _, candidates_df = build_swap_pools(slate, proj, confirmed_team_lineups=confirmed)
        cands = candidates_df.set_index("player_id")
        assert 20 in cands.index
        assert cands.loc[20, "mean"] == round(3500 / 600.0, 2)  # heuristic
        assert bool(cands.loc[20, "newly_confirmed"]) is True
        assert bool(cands.loc[13, "newly_confirmed"]) is False

    def test_scratched_batters_dropped_from_confirmed_team(self, slate_df):
        # DDD has a confirmed lineup without 19/22 — they're scratched.
        confirmed = {"DDD": {20: 7, 15: 5, 16: 6, 23: 4, 21: 2}}
        _, candidates_df = build_swap_pools(slate_df.drop(columns=["mean"]), None,
                                            confirmed_team_lineups=confirmed)
        pids = set(candidates_df["player_id"])
        assert 19 not in pids and 22 not in pids  # scratched DDD batters
        assert 11 in pids                          # DDD pitcher unaffected
        assert 17 in pids                          # other teams unaffected

    def test_newly_confirmed_scorer_boost(self):
        scorer = HeuristicScorer()
        entry = EntrySwapState("1", "f.csv", "c", "1", "$1", slots=[])
        base = {"player_id": 99, "position": "OF", "team": "CCC", "mean": 6.0}
        plain = scorer.score(dict(base), entry, Counter(), Counter())
        boosted = scorer.score(dict(base, newly_confirmed=True), entry, Counter(), Counter())
        assert boosted == pytest.approx(plain + scorer.W_NEWLY_CONFIRMED)

    def test_newly_confirmed_wins_swap(self, tmp_path, lookup, slate_df):
        # 20 (mean 7.0) loses to 33 (9.0) normally; with the +2.0 boost and
        # +1.0 DDD stack bonus (kept 15/16) it wins: 7.0+2.0+1.0 > 9.0.
        candidates_df = slate_df[slate_df["player_id"].isin([20, 33])].copy()
        candidates_df["newly_confirmed"] = candidates_df["player_id"] == 20
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), candidates_df, lookup,
                 HeuristicScorer(), NOW)
        assert states[0].slots[9].swapped_in_id == 20


class TestOverrideRevert:
    def _swapped_entry(self, tmp_path, lookup, slate_df):
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        run_swap(states, {"100": {19}}, set(), set(), slate_df, lookup,
                 HeuristicScorer(), NOW)
        return states[0]

    def test_revert_to_original(self, tmp_path, lookup, slate_df):
        entry = self._swapped_entry(tmp_path, lookup, slate_df)
        assert entry.slots[9].swapped_in_id == 18
        err = apply_override(entry, 9, 19, slate_df, lookup, set(), set(), NOW)
        assert err is None
        assert entry.slots[9].swapped_in_id is None
        assert entry.slots[9].swap_source is None
        assert not entry.changed

    def test_revert_rejected_when_entry_would_be_invalid(self, tmp_path, lookup, slate_df):
        # Swap OF 19 ($5300) down to 20 ($3500), then upgrade 1B 13 ($4100)
        # to 23 ($4400) using the freed salary. Reverting the OF swap would
        # push the total over the cap, so it must be rejected.
        states = _states_for(tmp_path, lookup,
                             {"MEDKEntries.csv": [_entry("100", ENTRY1_PIDS)]})
        entry = states[0]
        assert apply_override(entry, 9, 20, slate_df, lookup, set(), set(), NOW) is None
        assert apply_override(entry, 3, 23, slate_df, lookup, set(), set(), NOW) is None
        err = apply_override(entry, 9, 19, slate_df, lookup, set(), set(), NOW)
        assert err == "not_eligible"
        assert entry.slots[9].swapped_in_id == 20  # unchanged
