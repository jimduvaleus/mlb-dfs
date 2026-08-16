"""External candidate pool mode (SaberSim-style import).

Parses an externally produced lineup pool CSV (slot-ordered DK player IDs +
per-contest ROI blocks) and its companion per-player projections CSV, and
allocates per-contest portfolios with the existing DeterminantPortfolioSelector:
EV currency = the contest's ROI column, diversity = correlation of simulated
lineup scores (player-level sim reuse — the opponent-field/contest simulation
is bypassed entirely in this mode).

Contest ROI blocks are identified by a column ending " ROI" that has a
" Sim Dupes" sibling for the same prefix — the export also carries nine
generic "…Slate | …" bucket score columns whose headers do not have siblings,
and those are not contests. ROI StDev / Win Rate / Cash Rate / Sim Dupes
columns are deliberately unused (preserved via raw-file archiving only).
"""
from __future__ import annotations

import csv
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.api.dk_entries import EntryRecord, _parse_prize_pool_cents
from src.optimization.lineup import Lineup

logger = logging.getLogger(__name__)

_N_SLOT_COLS = 10
_SINGLE_ENTRY_RE = re.compile(r"\[\s*single\s+entry\s*\]", re.IGNORECASE)
_LINEUPS_GLOB = "lineups_*.csv"
# 'lineups_dk_mlb_classic_7-17-2026_705pm.csv' -> ('dk_mlb_classic', '7-17-2026', '705pm')
_LINEUPS_TOKEN_RE = re.compile(r"lineups_(.*?)_(\d{1,2}-\d{1,2}-\d{4})_(\d{3,4}[ap]m)", re.IGNORECASE)


def _lineup_slate_signature(name: str) -> Optional[tuple[str, str, str]]:
    """(format_prefix, normalized_date, time) identifying which slate a
    lineups_*.csv export belongs to. Two files with the same signature are
    treated as separate exports of the same slate (e.g. a browser
    re-download saved as '... (1).csv', or two separate optimizer runs); a
    different format, date, or time is a different slate. None when the
    filename doesn't carry the expected token."""
    m = _LINEUPS_TOKEN_RE.search(name)
    if not m:
        return None
    fmt, date_s, time_s = m.groups()
    mo, dy, yr = date_s.split("-")
    return (fmt.casefold(), f"{yr}-{int(mo):02d}-{int(dy):02d}", time_s.lower())


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ExternalContest:
    raw_name: str                 # exact header prefix, e.g. "MLB $20K mini-MAX [150 Entry Max]"
    norm_name: str
    roi: np.ndarray               # (M,) float64, NaN where blank
    prize_pool_cents: Optional[int]
    single_entry: bool
    # (M,) float64, NaN where blank, or None for exports predating this
    # column. Saber's raw "ROI StDev" cell is already in the same
    # percentage-point scale as `roi * 100` (verified against a real
    # archived slate: raw roi_stddev/100 divided by raw roi gave a
    # coefficient of variation of ~0.28, in line with the lineup's own
    # points-space CV of ~0.29; treating both columns as needing the same
    # x100 gave an implausible ~28x ratio) — so this is stored /100 to sit
    # on the same *unscaled fraction* footing as `roi` itself.
    roi_stddev: Optional[np.ndarray] = None


@dataclass
class ExternalPool:
    lineups: list                 # list[Lineup], slot-ordered player_ids
    contests: dict                # norm_name -> ExternalContest
    n_dropped_unknown_players: int
    n_dropped_duplicates: int
    n_dropped_near_duplicates: int  # 9/10-player-overlap lineups (see _find_near_duplicate_removals)
    source_paths: list            # list[Path], one or more lineups_*.csv for one slate


@dataclass
class ContestGroup:
    contest_id: str
    contest_name: str
    entry_fee_cents: int
    prize_pool_cents: Optional[int]
    single_entry_tag: bool
    entries: list = field(default_factory=list)  # [(Path, EntryRecord)] file order
    roi_key: str = ""             # norm_name of matched ExternalContest
    roi_fallback: bool = False


@dataclass
class ExternalAllocation:
    portfolio: list               # [(Lineup, roi)] flat, per-contest fill order
    entry_plan: list              # [(Path, EntryRecord)] parallel to portfolio
    unfilled: list                # [(Path, EntryRecord)] pool exhausted


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_external_files(raw_dir: str) -> dict:
    """All lineups_*.csv in raw_dir sharing the newest file's slate signature
    (see _lineup_slate_signature) + their companion projections CSV.

    Multiple exports of the same slate — a browser re-download saved as
    '... (1).csv', or separate optimizer runs — are returned together as
    ``lineups_paths`` so the caller can build one combined candidate pool.
    A lineups_*.csv with a different format, date, or time token is a
    different slate and is excluded. When the newest file's name doesn't
    carry a recognizable token, it's returned alone (no grouping).

    Pairing: the slate token from the reference lineups filename
    ('7-17-2026', '705pm') must appear in the companion as
    'YYYY-MM-DD-<time>' ('2026-07-17-705pm'); falls back to the newest
    MLB_*_DK_*.csv with paired_by_token=False.
    """
    d = Path(raw_dir)
    out = {"lineups_paths": [], "projections_path": None, "paired_by_token": False}
    lineup_files = sorted(d.glob(_LINEUPS_GLOB), key=lambda p: p.stat().st_mtime)
    if not lineup_files:
        return out
    newest = lineup_files[-1]
    sig = _lineup_slate_signature(newest.name)
    if sig is None:
        out["lineups_paths"] = [newest]
    else:
        matched = [p for p in lineup_files if _lineup_slate_signature(p.name) == sig]
        out["lineups_paths"] = sorted(matched, key=lambda p: p.name)
        _, date_norm, time_s = sig
        token = f"{date_norm}-{time_s}"
        for cand in sorted(d.glob("MLB_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            if token in cand.name.lower():
                out["projections_path"] = cand
                out["paired_by_token"] = True
                return out
    fallback = sorted(d.glob("MLB_*_DK_*.csv"), key=lambda p: p.stat().st_mtime)
    if fallback:
        out["projections_path"] = fallback[-1]
        logger.warning(
            "External pool: no token-matched companion for %s — falling back to newest %s",
            newest.name, out["projections_path"].name,
        )
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_contest_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def _parse_lineup_header(path: Path, rows: list, require_contest_blocks: bool = True) -> tuple:
    """Identify a lineup file's contest ROI blocks. Returns
    (contest_cols, stddev_cols, proj_score_idx): norm_name -> (raw prefix,
    col idx) / col idx / the file's "Proj Score" column index (None if
    absent) — the near-duplicate tie-break fallback when there's no ROI
    block to rank by (see parse_lineup_pool).

    ROI blocks are only structurally required for ev_type="roi" (contest
    identity/sizing for "prj_own"/"p_win" comes entirely from the DK entries
    file — see group_and_match_contests) — `require_contest_blocks=False`
    lets a file with none through with empty (contest_cols, stddev_cols)."""
    header = rows[0]
    header_set = set(header)
    contest_cols: dict[str, tuple[str, int]] = {}  # norm_name -> (raw prefix, col idx)
    stddev_cols: dict[str, int] = {}  # norm_name -> col idx of "<prefix> ROI StDev", if present
    for idx, col in enumerate(header):
        if not col.endswith(" ROI"):
            continue
        prefix = col[: -len(" ROI")]
        if f"{prefix} Sim Dupes" not in header_set:
            continue  # generic bucket column, not a contest block
        norm = normalize_contest_name(prefix)
        if norm in contest_cols:
            logger.warning("External pool: duplicate contest block %r in %s — keeping first.", prefix, path.name)
            continue
        contest_cols[norm] = (prefix, idx)
        std_col = f"{prefix} ROI StDev"
        if std_col in header_set:
            stddev_cols[norm] = header.index(std_col)
    if not contest_cols and require_contest_blocks:
        raise ValueError(
            f"External lineup file has no contest ROI blocks "
            f"(no '<name> ROI' column with a '<name> Sim Dupes' sibling): {path}"
        )
    proj_score_idx = header.index("Proj Score") if "Proj Score" in header_set else None
    return contest_cols, stddev_cols, proj_score_idx


def _pick_primary_contest_index(contest_order: list, contest_meta: dict) -> int:
    """Index into contest_order/roi_rows of the largest contest by parsed
    prize pool -- the same size proxy group_and_match_contests uses when a
    contest has no exact ROI-block match -- used as the single reference
    ROI column for near-duplicate tie-breaking (see
    _find_near_duplicate_removals). Falls back to the first contest in
    file order when none has a parseable prize pool."""
    def _size(j: int) -> int:
        prize = contest_meta[contest_order[j]][1]
        return prize if prize is not None else -1
    return max(range(len(contest_order)), key=_size)


def _find_near_duplicate_removals(player_ids_list: list, primary_roi: np.ndarray) -> set:
    """Indices to drop so no two surviving lineups' 10-player sets
    intersect in exactly 9 players (a single swapped player). Takes plain
    per-lineup player-id sequences (not Lineup objects) so callers outside
    the live pipeline — e.g. scripts/analyze_external_pool.py, flagging
    which archived lineups this pass *would* have removed — can reuse it
    directly against a DataFrame column of id lists.

    This overlap relation isn't transitive across different 9-player
    cores -- e.g. lineups A and B can each 9/10-overlap a shared lineup C
    (via two different 9-player cores) without overlapping each other --
    so a simple equivalence-class dedup (like the exact-duplicate pass)
    isn't well-defined here. Processing lineups by primary_roi descending
    and dropping one only when it conflicts with an already-kept lineup
    guarantees: every dropped lineup lost a real head-to-head to a
    higher-or-equal-ROI survivor (the pairwise rule asked for), and the
    final kept set has no conflicting pair left at all -- a violating pair
    would always be caught when its later-processed member is checked,
    unless both were already dropped for other conflicts, in which case
    neither survives to violate anything.
    """
    order = sorted(
        range(len(player_ids_list)),
        key=lambda i: primary_roi[i] if np.isfinite(primary_roi[i]) else float("-inf"),
        reverse=True,
    )
    kept_cores: dict = {}  # frozenset(9 player ids) -> kept lineup index
    removed: set = set()
    for i in order:
        ids = player_ids_list[i]
        cores = [frozenset(ids[:k] + ids[k + 1:]) for k in range(len(ids))]
        if any(c in kept_cores for c in cores):
            removed.add(i)
            continue
        for c in cores:
            kept_cores[c] = i
    return removed


def parse_pool_p99(paths, valid_ids: set[int]) -> dict:
    """{frozenset(player_ids): p99} from the lineup pool file(s)' own "99th"
    column -- SaberSim's per-lineup 99th-percentile simulated score. Used as
    a cheap ceiling signal (no simulation of ours required) for promoting
    external candidates into self-play's precision-refinement pool, see
    src.optimization.self_play. A separate parse from parse_lineup_pool
    (re-reads the same file(s)) since ExternalPool/Lineup carry no room for
    per-lineup metadata beyond ROI -- keyed by player-id set so callers can
    match it against already-deduped Lineup objects directly. Missing/
    unparseable cells and files with no "99th" column are silently skipped
    (this is a promotion signal, not a required field)."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    out: dict = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows or "99th" not in rows[0]:
            continue
        p99_idx = rows[0].index("99th")
        for r in rows[1:]:
            if len(r) < _N_SLOT_COLS:
                continue
            try:
                pids = frozenset(int(r[i]) for i in range(_N_SLOT_COLS))
            except ValueError:
                continue
            if not pids <= valid_ids:
                continue
            cell = r[p99_idx] if p99_idx < len(r) else ""
            try:
                val = float(cell)
            except ValueError:
                continue
            out.setdefault(pids, val)  # first occurrence wins, matches parse_lineup_pool's dedup
    return out


def parse_lineup_pool(paths, valid_ids: set[int], require_roi_blocks: bool = True) -> ExternalPool:
    """Parse one or more lineup exports for the same slate (see
    discover_external_files) with csv.reader on each raw header (duplicate
    'P'/'OF' slot headers and any duplicate contest names must be seen
    verbatim — never pandas' '.1' mangling).

    `require_roi_blocks=False` (pass when the configured external_pool_ev_type
    is "prj_own"/"p_win", which never read contest.roi — see allocate_contests)
    lets a file through with zero contest ROI blocks; the resulting pool just
    has an empty `contests` dict, which group_and_match_contests and the
    non-roi allocation branches already tolerate.

    Lineups are deduplicated by player-id set across *all* files combined —
    a lineup appearing more than once (within one file or across files) is
    kept only once, using the roi/roi_stddev from its first occurrence in
    file order. A second pass then also drops near-duplicates: lineups
    whose 10-player set overlaps another surviving lineup's in exactly 9
    players (a single swapped player). Conflicts are resolved by ROI in
    the largest contest by parsed prize pool (see
    _pick_primary_contest_index) — the lower-ROI lineup of any conflicting
    pair is dropped (see _find_near_duplicate_removals for why this needs
    a priority pass rather than simple equivalence-class grouping). When
    the pool has no ROI blocks at all (require_roi_blocks=False), the same
    conflict resolution instead ranks by each file's own "Proj Score"
    column (falling back further to file order wherever that's also
    missing/unparseable) rather than an arbitrary first-in-file keep.

    Contest ROI blocks are matched across files by normalized name (first
    file to define a given contest wins its raw name / prize pool / single
    entry metadata). A contest present in one file but not another leaves
    NaN roi (and roi_stddev) for the lineups sourced from file(s) missing
    that column — no different from any other blank ROI cell.

    PRE-LOCK SNAPSHOT LIMITATION (relevant to backtesting, not live use):
    this file is captured once, at/near the SLATE's first-game lock (its
    filename embeds that capture time, e.g. `lineups_..._7-22-2026_640pm.csv`
    for a slate whose first game locked around then) — it can only reflect
    confirmed-lineup/scratch information known as of that moment. Games
    later in the same slate lock individually, well after this snapshot, and
    real DK entrants (including our own accounts) can and do make post-lock
    swaps as those later games' confirmed lineups/scratches become known —
    visible directly in the archive as edits to our own real submitted
    entries after the slate-wide capture time. A backtest that grades THIS
    frozen pool against the real field is therefore comparing a pre-lock
    guess to entrants who had strictly more information for the back half
    of the slate — a real, structural handicap on this pool's apparent
    performance that has nothing to do with the selection strategy built on
    top of it. NOT external-specific, despite living in this docstring: a
    backtest's `generated` candidates are built from the archived projections
    file, which is captured at essentially this SAME pre-lock moment (verified
    2026-08-08, 07222026 archive: ~1s apart) -- so a since-scratched player's
    still-nonzero projection/ownership can seed him into generated lineups
    too. See src.optimization.self_play's SelfPlaySlateContext for the full
    note on why external-vs-generated is not the confound to worry about
    here; both-vs-the-real-field is."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    per_file: list[tuple] = []  # (path, data_rows, contest_cols, stddev_cols, proj_score_idx)
    contest_order: list[str] = []  # norm names, first-seen order across files
    contest_meta: dict[str, tuple] = {}  # norm -> (raw_prefix, prize_pool_cents, single_entry)
    any_stddev: set[str] = set()

    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise ValueError(f"External lineup file is empty: {path}")
        contest_cols, stddev_cols, proj_score_idx = _parse_lineup_header(
            path, rows, require_contest_blocks=require_roi_blocks,
        )
        for norm, (prefix, _) in contest_cols.items():
            if norm not in contest_meta:
                contest_order.append(norm)
                contest_meta[norm] = (
                    prefix, _parse_prize_pool_cents(prefix), bool(_SINGLE_ENTRY_RE.search(prefix)),
                )
        any_stddev.update(stddev_cols.keys())
        per_file.append((path, rows[1:], contest_cols, stddev_cols, proj_score_idx))

    lineups: list[Lineup] = []
    roi_rows: list[list[float]] = []
    stddev_rows: list[list[float]] = []
    proj_score_rows: list[float] = []
    seen: set[frozenset[int]] = set()
    n_unknown = 0
    n_dup = 0
    for path, data_rows, contest_cols, stddev_cols, proj_score_idx in per_file:
        for r in data_rows:
            if len(r) < _N_SLOT_COLS:
                continue
            try:
                pids = [int(r[i]) for i in range(_N_SLOT_COLS)]
            except ValueError:
                continue
            key = frozenset(pids)
            if not key <= valid_ids:
                n_unknown += 1
                continue
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            lineups.append(Lineup(player_ids=pids))
            vals = []
            std_vals = []
            for norm in contest_order:
                col = contest_cols.get(norm)
                if col is None:
                    vals.append(np.nan)
                else:
                    _, idx = col
                    cell = r[idx] if idx < len(r) else ""
                    try:
                        vals.append(float(cell))
                    except ValueError:
                        vals.append(np.nan)
                std_idx = stddev_cols.get(norm)
                cell = r[std_idx] if std_idx is not None and std_idx < len(r) else ""
                try:
                    std_vals.append(float(cell))
                except ValueError:
                    std_vals.append(np.nan)
            roi_rows.append(vals)
            stddev_rows.append(std_vals)
            proj_cell = r[proj_score_idx] if proj_score_idx is not None and proj_score_idx < len(r) else ""
            try:
                proj_score_rows.append(float(proj_cell))
            except ValueError:
                proj_score_rows.append(np.nan)

    # --- Near-duplicate removal (9/10 player overlap) --------------------
    n_near_dup = 0
    if len(lineups) > 1:
        if contest_order:
            primary_j = _pick_primary_contest_index(contest_order, contest_meta)
            primary_roi = np.array([row[primary_j] for row in roi_rows], dtype=np.float64)
        else:
            # No ROI blocks at all (require_roi_blocks=False, prj_own/p_win
            # ev_type) -- nothing to rank conflicts by from a contest block,
            # so fall back to the file's own "Proj Score" column instead of
            # an arbitrary first-in-file keep. Still NaN (-> file order via
            # _find_near_duplicate_removals's stable sort) wherever "Proj
            # Score" is itself absent/unparseable.
            primary_roi = np.array(proj_score_rows, dtype=np.float64)
        removed = _find_near_duplicate_removals([lu.player_ids for lu in lineups], primary_roi)
        if removed:
            keep = [i not in removed for i in range(len(lineups))]
            lineups = [lu for lu, k in zip(lineups, keep) if k]
            roi_rows = [row for row, k in zip(roi_rows, keep) if k]
            stddev_rows = [row for row, k in zip(stddev_rows, keep) if k]
            n_near_dup = len(removed)

    roi_mat = np.array(roi_rows, dtype=np.float64) if roi_rows else np.zeros((0, len(contest_order)))
    # See ExternalContest.roi_stddev: Saber's raw cell is already pct-pt
    # scaled like `roi * 100`, so /100 puts it on roi's own fraction scale.
    stddev_mat = (
        np.array(stddev_rows, dtype=np.float64) / 100.0
        if stddev_rows else np.zeros((0, len(contest_order)))
    )
    contests = {}
    for j, norm in enumerate(contest_order):
        raw, prize_pool_cents, single_entry = contest_meta[norm]
        contests[norm] = ExternalContest(
            raw_name=raw,
            norm_name=norm,
            roi=roi_mat[:, j],
            prize_pool_cents=prize_pool_cents,
            single_entry=single_entry,
            roi_stddev=stddev_mat[:, j] if norm in any_stddev else None,
        )
    logger.info(
        "External pool: %d lineups from %d file(s) (%d dropped unknown-player, %d duplicate, "
        "%d near-duplicate [9/10 overlap]), %d contests: %s",
        len(lineups), len(paths), n_unknown, n_dup, n_near_dup, len(contests),
        ", ".join(c.raw_name for c in contests.values()),
    )
    return ExternalPool(
        lineups=lineups, contests=contests,
        n_dropped_unknown_players=n_unknown, n_dropped_duplicates=n_dup,
        n_dropped_near_duplicates=n_near_dup,
        source_paths=paths,
    )


def parse_sabersim_projections(path: Path, platform: str = "draftkings") -> pd.DataFrame:
    """SaberSim per-player export -> canonical projections.csv frame, for use
    as a first-class ``projections_source`` (as opposed to full external-pool
    bypass mode, see ``parse_player_projections``/``build_external_players_df``
    above). Columns: player_id, name, mean, std_dev, lineup_slot,
    slot_confirmed, ownership (fraction, from "Adj Own").

    Unlike hitters, a pitcher row's Status column is ignored entirely — the
    file lists the whole rotation/bullpen, and Status on those rows isn't a
    reliable confirmed-starter signal. Instead, the pitcher with the highest
    projected mean on each team's roster is kept as that team's starter,
    and always treated as confirmed (slot_confirmed=True) for downstream
    checks — including the portfolio panel's unconfirmed-slot count, which
    must not flag an external-projection pitcher as unconfirmed. A batter
    row is kept only when it carries an Order (its projected batting slot);
    Status == "Confirmed" then distinguishes an officially confirmed slot
    from a merely projected one (slot_confirmed drives the UI's green/amber
    bubble + the team lock icon exactly as it does for the
    RotoWire/DFF/Market-Odds sources, so no separate UI plumbing is needed
    for SaberSim).

    ``status_confirmed`` separately preserves the real (non-overridden)
    Status=="Confirmed" reading for every kept row, pitchers included. Consumers
    that need to distinguish "we assumed this pitcher is starting because he's
    the highest-projected arm" from "SaberSim has actually confirmed this
    pitcher" (e.g. the Twitter/Underdog notification diff, which shouldn't flag
    a contradiction against a pitcher we only assumed) should use this column
    instead of ``slot_confirmed`` for pitcher rows.
    """
    df = pd.read_csv(path)
    mean_col = "fd_points" if platform == "fanduel" else "dk_points"
    std_col = "fd_std" if platform == "fanduel" else "dk_std"
    # SaberSim's Pos column is inconsistent across export files -- some days
    # it uses "SP"/"RP", others plain "P" (unlike the DK slate's ingested
    # `position`, which DraftKingsSlateIngestor always normalizes to "P" --
    # see twitter_lineups._PITCHER_POSITIONS for the same {SP, RP, P} set).
    is_pitcher = df["Pos"].astype(str).isin(["P", "SP", "RP"])
    confirmed = df["Status"].astype(str) == "Confirmed"
    order = pd.to_numeric(df.get("Order"), errors="coerce")
    mean_raw = pd.to_numeric(df[mean_col], errors="coerce")
    lineup_slot = pd.Series(np.nan, index=df.index, dtype=float)
    lineup_slot.loc[is_pitcher] = 10
    lineup_slot.loc[~is_pitcher] = order.loc[~is_pitcher]

    pitchers = pd.DataFrame({"team": df["Team"], "mean": mean_raw})[is_pitcher]
    keep_pitcher = pd.Series(False, index=df.index)
    starter_candidates = pitchers[pitchers["mean"].notna()]
    if not starter_candidates.empty:
        starter_idx = starter_candidates.groupby("team")["mean"].idxmax()
        keep_pitcher.loc[starter_idx] = True

    slot_confirmed = confirmed.copy()
    slot_confirmed.loc[is_pitcher] = True

    out = pd.DataFrame({
        "player_id": pd.to_numeric(df["DFS ID"], errors="coerce"),
        "name": df["Name"],
        "mean": mean_raw,
        "std_dev": pd.to_numeric(df[std_col], errors="coerce"),
        "lineup_slot": lineup_slot,
        "slot_confirmed": slot_confirmed,
        "status_confirmed": confirmed,
        "ownership": pd.to_numeric(df.get("Adj Own"), errors="coerce") / 100.0,
    })
    keep = keep_pitcher | (~is_pitcher & lineup_slot.notna())
    out = out[keep].dropna(subset=["player_id", "mean", "std_dev"]).copy()
    out["player_id"] = out["player_id"].astype(int)
    out["lineup_slot"] = out["lineup_slot"].astype(int)
    return out.drop_duplicates(subset=["player_id"], keep="first")


def parse_player_projections(path: Path) -> pd.DataFrame:
    """Companion per-player CSV -> normalized frame."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "player_id": pd.to_numeric(df["DFS ID"], errors="coerce"),
        "name": df["Name"],
        "position": df["Pos"].astype(str),
        "order": pd.to_numeric(df.get("Order"), errors="coerce"),
        "team": df["Team"].astype(str),
        "salary": pd.to_numeric(df.get("Salary"), errors="coerce"),
        "mean": pd.to_numeric(df["My Proj"], errors="coerce"),
        "std_dev": pd.to_numeric(df["dk_std"], errors="coerce"),
        "ownership": pd.to_numeric(df.get("My Own"), errors="coerce"),
        "p25": pd.to_numeric(df.get("dk_25_percentile"), errors="coerce"),
        "p50": pd.to_numeric(df.get("dk_50_percentile"), errors="coerce"),
        "p75": pd.to_numeric(df.get("dk_75_percentile"), errors="coerce"),
        "p85": pd.to_numeric(df.get("dk_85_percentile"), errors="coerce"),
        "p95": pd.to_numeric(df.get("dk_95_percentile"), errors="coerce"),
        "p99": pd.to_numeric(df.get("dk_99_percentile"), errors="coerce"),
        # Projected per-game rate stats, used only by build_quantile_grids'
        # zero-inflation to derive a per-batter blank probability
        # ((1 - OBP)^PA). NaN when the export omits them — the grid builder
        # then falls back to the population default.
        "pa": pd.to_numeric(df.get("PA"), errors="coerce"),
        "h": pd.to_numeric(df.get("H"), errors="coerce"),
        "bb": pd.to_numeric(df.get("BB"), errors="coerce"),
    })
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)
    return out.drop_duplicates(subset=["player_id"], keep="first")


# ---------------------------------------------------------------------------
# players_df synthesis + quantile grids (for the covariance sim)
# ---------------------------------------------------------------------------

def build_external_players_df(
    slate_df: pd.DataFrame,
    proj_ext: pd.DataFrame,
    pool_pids: set[int],
    derive_opponent: Callable[[str, str], str],
) -> pd.DataFrame:
    """Synthesize the SimulationEngine players_df contract
    (player_id, team, opponent, slot, mean, std_dev, position, salary, game)
    from the DK slate + external projections. Kept players: pitchers known to
    either source, batters with a batting Order, and every pool player."""
    df = slate_df.copy()
    df["player_id"] = df["player_id"].astype(int)
    df["opponent"] = df.apply(lambda r: derive_opponent(r["team"], r["game"]), axis=1)

    ext = proj_ext.set_index("player_id")
    df["ext_order"] = df["player_id"].map(ext["order"])
    df["ext_mean"] = df["player_id"].map(ext["mean"])
    df["ext_std"] = df["player_id"].map(ext["std_dev"])
    df["ext_own"] = df["player_id"].map(ext["ownership"])
    df["in_ext"] = df["player_id"].isin(ext.index)
    df["in_pool"] = df["player_id"].isin(pool_pids)

    is_pitcher = df["position"] == "P"
    keep = (is_pitcher & (df["in_ext"] | df["in_pool"])) | (~is_pitcher & df["ext_order"].notna()) | df["in_pool"]
    df = df[keep].copy()
    is_pitcher = df["position"] == "P"

    df["slot"] = 0
    df.loc[is_pitcher, "slot"] = 10
    df.loc[~is_pitcher, "slot"] = df.loc[~is_pitcher, "ext_order"]
    # Pool batters without a batting order: assign leftover slots per unit
    # (duplicates tolerated — the engine just reuses the shared copula
    # column — but logged, since they inflate intra-team correlation).
    for (_, _), grp in df[~is_pitcher].groupby(["team", "opponent"]):
        missing = grp.index[grp["slot"].isna() | (grp["slot"] <= 0)]
        if len(missing) == 0:
            continue
        used = set(int(s) for s in grp["slot"].dropna() if 1 <= s <= 9)
        free = [s for s in range(1, 10) if s not in used]
        for i, idx in enumerate(missing):
            df.loc[idx, "slot"] = free[i] if i < len(free) else ((i - len(free)) % 9) + 1
        if len(missing) > len(free):
            logger.warning(
                "External pool: %d batters without Order share slots on %s.",
                len(missing) - len(free), grp["team"].iloc[0],
            )
    df["slot"] = df["slot"].astype(int)

    # Means/stds: external file first; salary heuristic for pool players the
    # companion file doesn't know (logged).
    n_fallback = int((df["in_pool"] & df["ext_mean"].isna()).sum())
    if n_fallback:
        logger.warning(
            "External pool: %d pool players missing from the projections CSV "
            "— using salary-heuristic Gaussian marginals.", n_fallback,
        )
    df["mean"] = df["ext_mean"].fillna(df["salary"].astype(float) / 400.0).clip(lower=0.1)
    # GaussianMarginal requires sigma > 0; zero-variance rows (injured or
    # zeroed-out players in the export) get a nominal floor.
    df["std_dev"] = df["ext_std"].fillna(0.85 * df["mean"]).clip(lower=0.1)
    # Projected ownership, in percentage points (the file's "My Own" scale —
    # see parse_player_projections, which deliberately does not /100 it).
    # Pool players the companion file doesn't know get a small positive
    # floor rather than 0.0: ContestSimulator._build_pos_pools normalizes
    # ownership into a per-position sampling weight, so a hard 0 makes a
    # player mathematically impossible to draw into a simulated opponent
    # field (see generate_field / the p_win EV currency) — silently wrong
    # rather than merely imprecise, unlike the analogous mean fallback above.
    df["ownership"] = df["ext_own"].fillna(0.1).clip(lower=0.01)

    cols = ["player_id", "team", "opponent", "slot", "mean", "std_dev",
            "position", "salary", "game", "name", "eligible_positions",
            "ownership"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


_DEFAULT_SCRATCH_PROB = 0.02
"""Flat, projection-independent probability that a confirmed batter does not
play at all (late scratch after lineup confirmation). Measured 2026-07-30 across
10 archived slates: an upper bound of 3.0% of confirmed starters, via the
heuristic "confirmed batter scores 0 while an unconfirmed team-mate posts >= 5
points". Deliberately kept separate from the per-player blank probability below
because scratch risk is INDEPENDENT of projection — it takes out studs at the
same rate as punts, and a p99 world requires the best bat in the lineup to
actually play. Folding it into the projection-correlated term would
under-penalise exactly the players a ceiling depends on."""

_BLANK_PROB_FLOOR = 0.01
_BLANK_PROB_CEIL = 0.60
_DEFAULT_BLANK_PROB = 0.19
"""Population fallback when the export omits PA/H/BB (mean of the mechanistic
estimate across the 10-slate sample)."""

_MEAN_CALIB_BATTER_SS = 0.88
"""Empirical mean calibration for SaberSim BATTER projections, fitted
2026-07-30 over 10 archived slates (rostered players only, usage-weighted, PPD
games excluded): realized/grid-mean = 0.878 per-slate mean (sd 0.116, pooled
0.858), t=-3.32 vs 1.0, p=0.009. Convergent check: the market-odds fetcher's
independent `_MEAN_CALIB_BATTER` is 0.867 — two unrelated projection sources
both running ~13% hot on hitters.

Deliberately SEPARATE from the zero-inflation above. Zero-inflation is a pure
shape fix and holds the projected mean; this is the location fix. Keeping them
apart means each can be re-fitted or disabled on its own, and neither silently
absorbs the other's error."""

_MEAN_CALIB_PITCHER_SS = 1.0
"""No calibration for SaberSim PITCHER projections: the same fit gives 0.935
(sd 0.185) but t=-1.11, p=0.30 — not distinguishable from 1.0, so applying a
haircut would be fitting noise. Re-check as the archive grows. (Note an earlier
0.953 estimate was contaminated: it averaged over ALL export pitchers including
bullpen arms who never appear and score 0. Rostered pitchers are starters.)"""


def batter_blank_probability(
    pa: float, h: float, bb: float, scratch_prob: float = _DEFAULT_SCRATCH_PROB,
) -> float:
    """P(a rostered batter scores exactly 0 DK points), as a two-component
    mixture:  scratch_prob + (1 - scratch_prob) * (1 - OBP)^PA.

    A DK hitter scores nothing only if he never reaches base and drives nobody
    in — reaching at all already pays (a single is 3 pts). So the natural
    estimate is P(no times on base over his projected PA), with OBP proxied by
    the export's projected (H + BB) / PA.

    Validated 2026-07-30 on 10 archived slates (505,920 usage-weighted rostered
    batter-slates, PPD games excluded, all batters `Status == "Confirmed"`):
    the mechanistic term alone predicts 19.3% against a realized 20.6%, and
    0.02 + 0.98 * 0.193 = 20.9% — i.e. the two-component form lands on the
    observed rate with NO fitted multiplier. It is also monotone across
    predicted-blank sextiles (predicted 14.1% -> 28.5%, actual 15.8% -> 36.6%),
    though the realized slope is somewhat steeper than predicted, so the very
    highest-blank batters are still under-penalised.

    Why this matters: the simulation assigns only ~2.19% probability to a blank
    game, so a p99 lineup world (all 8 batters producing) is modelled at
    0.98^8 = 85% when reality is nearer 0.80^8 = 17%. That ~5x overstatement of
    the ceiling's precondition compounds multiplicatively and is why per-player
    upper tails calibrate almost exactly while the lineup aggregate does not.
    """
    if not all(np.isfinite([pa, h, bb])) or pa <= 0:
        play_blank = _DEFAULT_BLANK_PROB
    else:
        obp = float(np.clip((h + bb) / pa, 0.01, 0.70))
        play_blank = float((1.0 - obp) ** pa)
    p = scratch_prob + (1.0 - scratch_prob) * play_blank
    return float(np.clip(p, _BLANK_PROB_FLOOR, _BLANK_PROB_CEIL))


def _zero_inflate_grid(
    grid: np.ndarray, grid_q: np.ndarray, p_zero: float, eps: float = 1e-9,
) -> np.ndarray:
    """Insert `p_zero` total probability mass at exactly 0, preserving the
    grid's mean by rescaling the surviving mass.

    The grid already carries a little near-zero mass, so only the shortfall is
    added: `p_add = (p_zero - p_existing) / (1 - p_existing)`. The old
    distribution is then compressed into the upper `1 - p_add` of quantile
    space and multiplied by `1 / (1 - p_add)` so the mean is unchanged — the
    projected mean is SaberSim's and is not ours to move here. (Any residual
    mean bias is a separate calibration, deliberately not smuggled in.)

    Scaling the survivors up is what the data shows: conditional on not
    blanking, realized/simulated mean is 1.145 across the sample.
    """
    p_exist = float((grid <= eps).mean())
    if p_zero <= p_exist:
        return grid
    p_add = (p_zero - p_exist) / (1.0 - p_exist)
    if p_add >= 1.0 - 1e-6:
        return grid
    mean_before = float(grid.mean())
    remapped = np.where(
        grid_q < p_add, 0.0,
        np.interp(np.clip((grid_q - p_add) / (1.0 - p_add), 0.0, 1.0), grid_q, grid),
    )
    mean_after = float(remapped.mean())
    if mean_after > eps:
        remapped *= mean_before / mean_after
    return np.maximum.accumulate(remapped)


def build_quantile_grids(
    proj_ext: pd.DataFrame,
    n_points: int = 101,
    zero_inflate: bool = False,
    scratch_prob: float = _DEFAULT_SCRATCH_PROB,
    mean_calib_batter: float = 1.0,
    mean_calib_pitcher: float = 1.0,
) -> dict[int, np.ndarray]:
    """Per-player evenly spaced quantile grids for EmpiricalQuantileMarginal,
    resampled from the file's irregular percentiles. Skips a player (engine
    falls back to Gaussian) on missing/non-monotone knots or a >20% mismatch
    between the grid-implied mean and the file mean.

    `zero_inflate` adds a point mass at 0 for **batters** (see
    batter_blank_probability / _zero_inflate_grid), correcting a measured ~9x
    understatement of blank games. Off by default so existing callers and
    replay scripts are byte-identical unless they opt in. Pitchers are left
    alone: rostered starters blank only 1.1% of the time, which the raw grids
    already price about right (a 17% figure over *all* export pitchers is a
    bullpen-population artifact — relievers who never appear).

    `mean_calib_batter` / `mean_calib_pitcher` scale the finished grid — the
    location fix, applied AFTER (and independently of) the shape fix above.
    Defaults are 1.0 here so nothing changes for callers that don't opt in; the
    fitted values live in `_MEAN_CALIB_BATTER_SS` / `_MEAN_CALIB_PITCHER_SS`
    and the pipeline passes them from config. Applied to the grids only, not to
    `players_df["mean"]`: that column feeds the projected-score percentile floor,
    which is a relative ranking a uniform scalar barely moves, and leaving it
    alone keeps reported projections matching the SaberSim export. Players
    without a grid (Gaussian fallback) are therefore uncalibrated — a small
    minority whose means are salary-heuristic anyway.

    Order is load-bearing: the +-20% grid-vs-file-mean sanity check runs against
    the RAW grid, before either correction, so a calibration constant can never
    push a player into or out of the Gaussian fallback.
    """
    q_levels = np.array([0.25, 0.50, 0.75, 0.85, 0.95, 0.99])
    grid_q = np.linspace(0.0, 1.0, n_points)
    grids: dict[int, np.ndarray] = {}
    n_inflated = n_calibrated = 0
    for r in proj_ext.itertuples(index=False):
        knots = np.array([r.p25, r.p50, r.p75, r.p85, r.p95, r.p99], dtype=np.float64)
        if np.any(~np.isfinite(knots)) or not np.isfinite(r.mean) or r.mean <= 0:
            continue
        knots = np.maximum.accumulate(knots)
        # Tail extrapolation: (25,50) slope down to p0; (95,99) slope up to p100.
        p0 = knots[0] - (knots[1] - knots[0])
        if str(r.position) != "P":
            p0 = max(p0, 0.0)  # batters cannot score below 0
        p100 = knots[5] + 0.25 * (knots[5] - knots[4])
        levels = np.concatenate([[0.0], q_levels, [1.0]])
        values = np.concatenate([[p0], knots, [p100]])
        grid = np.interp(grid_q, levels, values)
        grid = np.maximum.accumulate(grid)
        grid_mean = float(grid.mean())
        if abs(grid_mean - float(r.mean)) > 0.2 * float(r.mean):
            logger.debug(
                "External pool: grid mean %.2f vs file mean %.2f for %s — Gaussian fallback.",
                grid_mean, r.mean, r.name,
            )
            continue
        is_pitcher = str(r.position) == "P"
        if zero_inflate and not is_pitcher:
            p_zero = batter_blank_probability(
                getattr(r, "pa", np.nan), getattr(r, "h", np.nan),
                getattr(r, "bb", np.nan), scratch_prob,
            )
            inflated = _zero_inflate_grid(grid, grid_q, p_zero)
            if inflated is not grid:
                n_inflated += 1
            grid = inflated
        calib = mean_calib_pitcher if is_pitcher else mean_calib_batter
        if calib != 1.0:
            grid = grid * float(calib)
            n_calibrated += 1
        grids[int(r.player_id)] = grid
    if zero_inflate or n_calibrated:
        logger.info(
            "External pool: quantile grids built for %d/%d players "
            "(%d batters zero-inflated at scratch_prob=%.3f; %d mean-calibrated, "
            "batter x%.3f / pitcher x%.3f).",
            len(grids), len(proj_ext), n_inflated, scratch_prob, n_calibrated,
            mean_calib_batter, mean_calib_pitcher,
        )
    else:
        logger.info(
            "External pool: quantile grids built for %d/%d players.", len(grids), len(proj_ext),
        )
    return grids


# ---------------------------------------------------------------------------
# Contest grouping + ROI matching
# ---------------------------------------------------------------------------

def group_and_match_contests(
    all_file_entries: list,          # [(Path, list[EntryRecord])]
    pool: ExternalPool,
) -> list[ContestGroup]:
    """Group entries by contest, match each to a pool ROI block, and order
    for allocation: entry fee desc, then assumed size (prize pool) asc
    (None last), then contest_id for determinism."""
    groups: dict[str, ContestGroup] = {}
    for file_path, records in all_file_entries:
        for rec in records:
            g = groups.get(rec.contest_id)
            if g is None:
                g = ContestGroup(
                    contest_id=rec.contest_id,
                    contest_name=rec.contest_name,
                    entry_fee_cents=rec.entry_fee_cents,
                    prize_pool_cents=rec.prize_pool_cents,
                    single_entry_tag=bool(_SINGLE_ENTRY_RE.search(rec.contest_name)),
                )
                groups[rec.contest_id] = g
            g.entries.append((file_path, rec))

    covered = list(pool.contests.values())
    for g in groups.values():
        norm = normalize_contest_name(g.contest_name)
        if norm in pool.contests:
            g.roi_key = norm
            g.roi_fallback = False
            continue
        if not covered:
            # No ROI blocks at all in the pool (ev_type prj_own/p_win with
            # require_roi_blocks=False) -- roi_key stays "" so the roi
            # allocation branch's pool.contests.get(g.roi_key) misses and
            # leaves the group unfilled, which only matters if ev_type=="roi".
            g.roi_fallback = True
            continue
        # Nearest assumed size by prize pool; sides without a parseable pool
        # sort last; ties prefer the same single/multi-entry tag, then the
        # larger covered pool.
        def _rank(c: ExternalContest):
            if g.prize_pool_cents is None or c.prize_pool_cents is None:
                diff = float("inf")
            else:
                diff = abs(c.prize_pool_cents - g.prize_pool_cents)
            tag_mismatch = 0 if c.single_entry == g.single_entry_tag else 1
            size = -(c.prize_pool_cents or 0)
            return (diff, tag_mismatch, size, c.norm_name)
        best = min(covered, key=_rank)
        g.roi_key = best.norm_name
        g.roi_fallback = True
        logger.info(
            "External pool: contest %r has no ROI block — borrowing %r (nearest assumed size).",
            g.contest_name, best.raw_name,
        )

    return sorted(
        groups.values(),
        key=lambda g: (
            -g.entry_fee_cents,
            g.prize_pool_cents if g.prize_pool_cents is not None else float("inf"),
            g.contest_id,
        ),
    )


# ---------------------------------------------------------------------------
# Correlation + allocation
# ---------------------------------------------------------------------------

_MIN_CEILING_FIT_N = 30


def compute_ceiling_ev(
    roi: np.ndarray, roi_stddev: Optional[np.ndarray], weight: float,
) -> Optional[np.ndarray]:
    """roi + weight * (residual of roi_stddev after regressing out roi),
    z-scored then rescaled to roi's own spread.

    roi_stddev is highly correlated with roi itself (0.83 measured against
    a real archived slate's mini-MAX tier) — most of what it says about a
    lineup's upside is already implied by roi. Regressing roi_stddev on roi
    and using the *residual* isolates the part that's genuinely new
    information (more upside variance than roi alone predicts) instead of
    rewarding high-roi lineups a second time for being high-roi. The
    residual is z-scored, then rescaled by roi's own stddev so `weight` is
    a unitless multiplier comparable across contests of very different
    scale (coefficient of variation ranged from -1.07 to +0.28 across
    contest tiers in one archived slate).

    Returns None (caller should fall back to plain roi) when roi_stddev is
    unavailable, weight is 0, or the pool is too small/degenerate to fit a
    meaningful residual.
    """
    if roi_stddev is None or weight == 0.0:
        return None
    finite = np.isfinite(roi) & np.isfinite(roi_stddev)
    if int(finite.sum()) < _MIN_CEILING_FIT_N:
        return None
    roi_std = float(roi[finite].std())
    if roi_std < 1e-12:
        return None

    slope, intercept = np.polyfit(roi[finite], roi_stddev[finite], 1)
    predicted = intercept + slope * roi
    resid = roi_stddev - predicted
    resid_std = float(np.nanstd(resid[finite]))
    if resid_std < 1e-12:
        return None
    resid_z = resid / resid_std
    ceiling = roi + weight * resid_z * roi_std
    # Non-finite inputs (missing per-lineup StDev) fall back to plain roi.
    return np.where(np.isfinite(ceiling), ceiling, roi)


def _lineup_indicator_matrix(lineups: list, player_ids: list) -> np.ndarray:
    """(P, M) float32 indicator: I[p, j] = 1.0 if player p is in lineup j.
    Every pool player is guaranteed present in player_ids (players_df includes
    all pool players), so no -1/missing handling is needed."""
    col_map = {int(p): i for i, p in enumerate(player_ids)}
    P = len(col_map)
    M = len(lineups)
    I = np.zeros((P, M), dtype=np.float32)
    for j, lu in enumerate(lineups):
        for pid in lu.player_ids:
            I[col_map[int(pid)], j] = 1.0
    return I


def _pava(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool-adjacent-violators: least-squares monotone (non-decreasing) fit
    to y, optionally weighted. Pure numpy/list implementation — sklearn is
    not a project dependency (mirrors scripts/evaluate_ownership.py's _pava)."""
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    blocks: list[list[float]] = []  # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            tw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / tw, tw, c1 + c2])
    if not blocks:
        return np.array([], dtype=float)
    return np.concatenate([np.full(int(c), v) for v, _, c in blocks])


def _fit_percentile_curve(
    x: np.ndarray, y: np.ndarray, min_points: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """PAVA-fit y (non-decreasing) against x, returning np.interp-ready knots
    (x, y) with duplicate x collapsed to their mean y. None when fewer than
    min_points finite (x, y) pairs are available to fit."""
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < min_points:
        return None
    order = np.argsort(x[finite], kind="stable")
    xs = x[finite][order]
    ys = _pava(y[finite][order])
    knots = pd.DataFrame({"x": xs, "y": ys}).groupby("x", as_index=False)["y"].mean()
    return knots["x"].to_numpy(), knots["y"].to_numpy()


def _interp_extrapolated(x: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    """np.interp, but linearly extrapolated past the knot range using the
    boundary segment's slope instead of flat-clamping.

    A PPD hit can push a lineup's score below the pool's own observed
    minimum (or, symmetrically, above its max) — most plausible for a
    lineup that was already the pool's weakest by raw score before the
    zeroing was applied. np.interp's default flat clamp would then report
    zero further ROI change no matter how much worse the score got, since
    there's no data past the boundary knot to interpolate against. Extending
    the nearest fitted segment's slope keeps the adjustment responsive to
    exposure magnitude even at the tails, at the cost of a linear (not
    PAVA-monotone-guaranteed) extrapolation outside the fitted range.
    """
    y = np.interp(x, knot_x, knot_y)
    if len(knot_x) < 2:
        return y
    below = x < knot_x[0]
    if np.any(below):
        dx = knot_x[1] - knot_x[0]
        slope = (knot_y[1] - knot_y[0]) / dx if dx > 0 else 0.0
        y = np.where(below, knot_y[0] + slope * (x - knot_x[0]), y)
    above = x > knot_x[-1]
    if np.any(above):
        dx = knot_x[-1] - knot_x[-2]
        slope = (knot_y[-1] - knot_y[-2]) / dx if dx > 0 else 0.0
        y = np.where(above, knot_y[-1] + slope * (x - knot_x[-1]), y)
    return y


def compute_ppd_roi_adjustment(
    pool: "ExternalPool", sim_results, sim_results_ppd, min_fit_points: int = _MIN_CEILING_FIT_N,
) -> None:
    """Mutates each ExternalContest.roi (and roi_stddev, when present) in
    pool.contests in place to reflect PPD risk. sim_results is the plain
    player-level sim (no PPD applied); sim_results_ppd is the same sim after
    PipelineRunner._apply_ppd_to_simulation has zeroed at-risk games' player
    columns in a random fraction of sim rows — the identical mechanism the
    internal pipeline uses, so the joint-risk structure (multiple at-risk
    games, shared per-game row-zeroing) is consistent between modes.

    Method: project every lineup's own score onto both sim matrices (the same
    player-indicator trick compute_pool_corr uses), reduce each to a
    per-lineup mean, and convert both into percentiles via a Normal-CDF
    z-score against the *raw* (no-PPD) score distribution's own mean/std — a
    fixed yardstick for "how good is this lineup, in the pool's own terms,"
    before and after the PPD hit. A lineup untouched by any at-risk game gets
    identical raw/ppd scores and therefore an identical percentile — an exact
    no-op. A rank-based percentile (e.g. np.searchsorted into the sorted raw
    scores) was tried and rejected: it saturates at 0/1 for any lineup that's
    already the pool's best/worst by raw score, so an already-weak, heavily
    PPD-exposed lineup could show *zero* delta despite a large absolute score
    drop, simply because it had nowhere lower to rank to. The z-score CDF has
    no such floor/ceiling — a further score drop always maps to a further
    (if shrinking) percentile decrease.

    Per contest, fit an empirical monotonic percentile -> value curve (PAVA)
    from the pool's own (raw_percentile, roi) pairs. This is what makes the
    adjustment convexity-aware without guessing a payout shape: the curve
    *is* Saber's own simulated payout structure for that contest, read off
    empirically. Apply the resulting value change as a *delta* on top of the
    lineup's own reported figure (not a replacement), so a curve-fit bias at
    a given percentile can't move a zero-exposure lineup's number.

    roi_stddev gets the identical percentile-delta treatment (a second curve
    fit from (raw_percentile, roi_stddev), floored at 0). This matters
    because compute_ceiling_ev regresses roi_stddev on roi and rewards the
    residual as a ceiling bonus — leaving roi_stddev untouched while roi drops
    would make a PPD-exposed lineup look anomalously high-stddev-for-its-roi,
    handing back a spurious ceiling credit for what is actually downside PPD
    risk, not upside variance.
    """
    lineups = pool.lineups
    M = len(lineups)
    if M == 0:
        return
    I = _lineup_indicator_matrix(lineups, sim_results.player_ids)
    scores_raw = (sim_results.results_matrix.astype(np.float32) @ I).T       # (M, n_sims)
    scores_ppd = (sim_results_ppd.results_matrix.astype(np.float32) @ I).T   # (M, n_sims)
    raw_stat = scores_raw.mean(axis=1).astype(np.float64)
    ppd_stat = scores_ppd.mean(axis=1).astype(np.float64)

    mu = float(raw_stat.mean())
    sigma = float(raw_stat.std())
    if sigma < 1e-9:
        return  # degenerate pool (near-identical lineups) — nothing to rank
    raw_percentile = norm.cdf((raw_stat - mu) / sigma)
    ppd_percentile = norm.cdf((ppd_stat - mu) / sigma)
    if np.array_equal(raw_percentile, ppd_percentile):
        return  # no lineup exposed to any at-risk game

    for contest in pool.contests.values():
        curve = _fit_percentile_curve(raw_percentile, contest.roi, min_fit_points)
        if curve is not None:
            knot_x, knot_y = curve
            delta = (
                _interp_extrapolated(ppd_percentile, knot_x, knot_y)
                - _interp_extrapolated(raw_percentile, knot_x, knot_y)
            )
            contest.roi = contest.roi + delta
        if contest.roi_stddev is not None:
            std_curve = _fit_percentile_curve(raw_percentile, contest.roi_stddev, min_fit_points)
            if std_curve is not None:
                knot_x, knot_y = std_curve
                std_delta = (
                    _interp_extrapolated(ppd_percentile, knot_x, knot_y)
                    - _interp_extrapolated(raw_percentile, knot_x, knot_y)
                )
                contest.roi_stddev = np.maximum(contest.roi_stddev + std_delta, 0.0)


def compute_lineup_scores(lineups: list, sim_results) -> np.ndarray:
    """(M, n_sims) float32 simulated lineup scores: each lineup's score in a
    sim is the sum of its 10 players' simulated points (indicator matmul).
    Every pool player is guaranteed present in sim_results (players_df
    includes all pool players), so no -1/missing handling is needed. Split
    out of compute_pool_corr so the p_win EV currency (compute_p_win) and
    the diversity correlation can share one score matrix instead of two."""
    I = _lineup_indicator_matrix(lineups, sim_results.player_ids)
    return (sim_results.results_matrix.astype(np.float32) @ I).T


def compute_pool_corr(
    lineups: list, sim_results, scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(M, M) float32 correlation of simulated lineup scores (points-space).

    `scores` may be passed precomputed (compute_lineup_scores) to avoid
    repeating the indicator matmul when the caller also needs the raw
    per-sim score matrix (the p_win EV currency does).

    A within-pool-rank payout transform (round-12's winning lambda=0
    construction: rank each sim's scores within the pool, map through the
    reference GPP curve scaled to the pool size) was tried here and
    reverted — it collapses the diversity signal for external pools and
    makes the risk sweep produce near-identical portfolios at every risk
    level. Diagnosis: round-12's pools were dominated by tight clusters of
    near-duplicate lineups (shape-preserving mutants of a small set of
    seed parents — see plans/round7 / round-8's "shape mutation"), so
    ranking against the pool itself mostly separated near-duplicates from
    everything else, which the rank transform handles fine. An external
    (SaberSim-style) pool has no such clustering — it's thousands of
    genuinely distinct, comparably-good lineups — and ranking each one
    against thousands of close competitors turns small per-sim noise into
    large rank swings ("crowding"): a synthetic 30-team-stack pool
    (M=3000, realistic overlap/correlation structure) measured
    points-space off-diagonal std=0.149 (44% of pairs |corr|>0.05, a
    risk=1-vs-risk=5 portfolio overlap of 21/150 — healthy
    differentiation) collapsing to std=0.008 (0.4% of pairs |corr|>0.05,
    106/150 overlap — the risk dial going inert) under the rank transform.
    Round-11 measured the points-space/composition-only gap at only
    ~1.5pp mean_pct vs true dollar-space (which needs a real opponent
    field external mode doesn't have) — a much better trade than a
    risk-invariant selector. Every pool player is guaranteed present in
    sim_results (players_df includes all pool players), so the indicator
    matmul needs no -1 handling.
    """
    from src.optimization.gpp_portfolio import DeterminantPortfolioSelector

    M = len(lineups)
    if scores is None:
        scores = compute_lineup_scores(lineups, sim_results)
    pre = DeterminantPortfolioSelector.precompute_pool(scores, float("-inf"))
    assert pre is not None and len(pre[0]) == M
    return pre[2]


def compute_proj_score_floor(
    proj_scores: np.ndarray, floor_percentile: float,
) -> Optional[tuple[float, int]]:
    """(cutoff, n_culled) for a `proj_score_floor_percentile` cull of
    `proj_scores`, or None when disabled (`floor_percentile <= 0`) or there
    are no finite scores to compute a percentile from. Shared by
    `allocate_contests` (applying the cull to `mask`) and the pipeline
    (reporting it once via SSE before the risk sweep — the cull is
    risk-invariant, so the cutoff/count are identical on every risk
    iteration and only need to be computed/reported a single time)."""
    if floor_percentile <= 0:
        return None
    finite = proj_scores[np.isfinite(proj_scores)]
    if finite.size == 0:
        return None
    cutoff = float(np.percentile(finite, floor_percentile))
    n_culled = int(np.sum(~np.isfinite(proj_scores) | (proj_scores < cutoff)))
    return cutoff, n_culled


def compute_pool_proj_scores(lineups: list, players_df: pd.DataFrame) -> np.ndarray:
    """(M,) float64 sum of each lineup's rostered players' projected mean
    (`players_df["mean"]`), for the pool-wide `proj_score_floor_percentile`
    cull in allocate_contests. Every pool player is guaranteed present in
    players_df (build_external_players_df keeps every pool player), so the
    indicator matmul needs no -1 handling."""
    I = _lineup_indicator_matrix(lineups, players_df["player_id"].tolist())
    means = players_df["mean"].to_numpy(dtype=np.float32)
    return (means @ I).astype(np.float64)


def compute_pool_ownership(lineups: list, players_df: pd.DataFrame) -> np.ndarray:
    """(M,) float64 sum of each lineup's rostered players' projected ownership
    (`players_df["ownership"]`, in *percentage points* — see
    build_external_players_df), the ownership half of the `prj_own` EV
    currency. Exact mirror of compute_pool_proj_scores, and it reproduces the
    lineups export's own "Ownership" column, which is likewise the plain sum
    of the 10 players' "My Own" values (verified against a real 7/26 export).
    Computing it here rather than parsing that column keeps `prj_own` working
    for exports that lack it and keeps it on the same footing as
    compute_pool_proj_scores, which reproduces the export's "Proj Score"."""
    I = _lineup_indicator_matrix(lineups, players_df["player_id"].tolist())
    own = players_df["ownership"].to_numpy(dtype=np.float32)
    return (own @ I).astype(np.float64)


def compute_pool_ceiling_proxy(lineups: list, players_df: pd.DataFrame, z: float = 2.33) -> np.ndarray:
    """(M,) float64 `sum(player mean) + z * sqrt(sum(player std_dev^2))` per
    lineup -- a mean+z*sigma ceiling estimate (z=2.33 ~ the 99th percentile
    of a standard normal, matching what a lineup-level "99th percentile"
    column is itself approximating) for lineups that have no such column of
    their own. Built for self-play's generated candidates (see
    src.optimization.self_play): they aren't part of a SaberSim export, so
    they carry no native ceiling signal the way external pool lineups do
    (parse_pool_p99) -- this is a cheap, independence-assuming stand-in, NOT
    a substitute for actual simulation. Treating the 10 rostered players'
    variances as independent is a real simplification (same-team batters
    correlate) -- defensible only as a promotion SCREEN whose output still
    gets validated by real per-sim scoring downstream, not as a final
    ranking signal on its own. `players_df["std_dev"]` is `dk_std` (see
    parse_player_projections), the same source `players_df["mean"]` (`My
    Proj`) already comes from."""
    I = _lineup_indicator_matrix(lineups, players_df["player_id"].tolist())
    means = players_df["mean"].to_numpy(dtype=np.float64)
    variances = players_df["std_dev"].to_numpy(dtype=np.float64) ** 2
    lineup_means = means.astype(np.float32) @ I
    lineup_var = variances.astype(np.float32) @ I
    return lineup_means.astype(np.float64) + z * np.sqrt(np.maximum(lineup_var, 0.0))


def compute_pool_ceiling_scores(pool: "ExternalPool", players_df: pd.DataFrame) -> np.ndarray:
    """(M,) float64 per-lineup ceiling estimate, aligned to `pool.lineups` --
    the basis for the pool-wide `external_pool_proj_score_pct` floor cull in
    place of summed projected mean, so the floor keeps high-ceiling lineups
    a mean-based cull would otherwise drop for being merely median, and
    drops median-ceiling lineups a mean-based cull would otherwise keep.

    Prefers each lineup's own SaberSim "99th" column (`parse_pool_p99`) --
    a real simulated ceiling, not an estimate. Falls back to the
    independence-assuming `compute_pool_ceiling_proxy` for any lineup
    without one: a generated lineup added by
    `augment_topn_pool_with_generated` (no source file, so no "99th" cell
    at all) or a real lineup whose file's "99th" cell was blank/
    unparseable."""
    valid_ids = set(players_df["player_id"].astype(int))
    p99_lookup = parse_pool_p99(pool.source_paths, valid_ids)
    p99 = np.array(
        [p99_lookup.get(frozenset(lu.player_ids), np.nan) for lu in pool.lineups],
        dtype=np.float64,
    )
    missing = ~np.isfinite(p99)
    if missing.any():
        proxy = compute_pool_ceiling_proxy(
            [lu for lu, m in zip(pool.lineups, missing) if m], players_df,
        )
        p99[missing] = proxy
    return p99


_DK_RAKE = 0.16
"""DK's fixed take across contest sizes: prize_pool = entries * entry_fee *
(1 - _DK_RAKE), i.e. the prize pool is only ~84% of collected entry fees
(confirmed against data/payout_structures/dk_classic_gpp.json: $59,452
collected vs $50,000 paid = 15.9% rake). Any estimate of a contest's true
entry count from its parsed prize pool must divide by entry_fee * (1 -
_DK_RAKE), not entry_fee alone, or it undercounts by ~16% (e.g. a real
23,809-entry contest's prize_pool/entry_fee ratio alone reads as 20,000)."""


def implied_field_size(group: "ContestGroup") -> float:
    """Implied entrant count for a contest: prize pool / (entry fee * (1 -
    _DK_RAKE)). DK contest names carry no entrant count (the `[150 Entry
    Max]` tag is a per-user entry cap, not a field size), so the parsed
    prize pool (`dk_entries._parse_prize_pool_cents`, including its
    `Qualifier` divisor) over the rake-adjusted entry fee is the size proxy.
    Note this differs from the *unadjusted* ratio `dk_entries._sort_ratio`
    uses to order entries — that usage only needs relative ordering across
    contests, which the constant rake factor doesn't change, so it's left
    as the raw ratio there.

    Returns 0.0 when either side is missing or non-positive, which
    deliberately zeroes the ownership penalty in `compute_prj_own_ev` — an
    unparseable prize pool degrades that EV to pure projected score rather
    than reshaping a portfolio around a guessed field size."""
    prize = group.prize_pool_cents
    fee = group.entry_fee_cents
    if not prize or not fee or prize <= 0 or fee <= 0:
        return 0.0
    return float(prize) / (float(fee) * (1.0 - _DK_RAKE))


_DEFAULT_OWN_SCALE = 30_000.0

_SMALL_FIELD_THRESHOLD = 5000.0
"""Shared "small field" boundary for proj_top's two field-size-aware
sub-features: the ownership cap (own_cap_start_pct/own_cap_end_pct) and the
ceiling-tier ranking signal (ceiling_tier_boundary) in allocate_contests.
Below this, both are inert -- proj_top is plain mean-proj_score ranking.
Fixed, NOT user-configurable -- only each feature's own dial(s) are exposed
via GppConfig/the UI."""


def compute_prj_own_ev(
    proj_scores: np.ndarray, own_scores: np.ndarray, field_size: float,
    own_scale: float = _DEFAULT_OWN_SCALE,
) -> np.ndarray:
    """The `prj_own` EV currency: projected score minus projected ownership,
    with the ownership penalty scaled by contest size.

        EV = proj_score - own_score * (field_size / own_scale)

    Both inputs are lineup sums: projected score in fantasy points (~70-115
    on a real pool), ownership in percentage points (~10-170). `own_scale`
    is therefore readable as *the field size at which one point of summed
    ownership costs one projected point*.

    Calibrated 2026-07-27 to own_scale=30,000 from two stated indifference
    anchors, which a single linear-in-field-size coefficient satisfies
    exactly:

      * at ~10,000 entries, (proj 95, own 60) should tie (proj 105, own 90)
        — i.e. 10 projected points trade for 30 ownership points, so the
        coefficient there is 1/3 = 10_000/30_000;
      * at 1,000 entries ownership should matter ~10x less, which linear
        scaling gives for free: 1_000/30_000 = 1/30.

    NOTE (2026-07-28): `field_size` is now `implied_field_size(group)`
    rake-adjusted (see _DK_RAKE) -- it reads ~19% higher than it did at
    calibration time for the same contest (dividing by entry_fee*0.84
    instead of entry_fee), so the effective ownership-penalty coefficient
    is now somewhat stronger than the two anchors above intended. Left
    uncorrected for now since ownership itself is externally sourced
    (SaberSim's Adj Own), not our own projection -- revisit own_scale if/
    when prj_own's behavior is recalibrated.

    `field_size=0` (unparseable prize pool) reduces this to plain projected
    score.

    Note on where the projection constraint actually binds: a real pool's
    ownership sums span a far wider range (~125 points) than its projected
    scores (~20), so past roughly 5,000 entries the argmax of this EV is
    essentially the least-owned lineup that cleared the projection floor.
    At large fields `external_pool_proj_score_pct` is the projection lever,
    and this coefficient governs the small/mid-field contests where the two
    terms genuinely trade off."""
    return np.asarray(proj_scores, dtype=np.float64) - (
        np.asarray(own_scores, dtype=np.float64) * (float(field_size) / float(own_scale))
    )


# ---------------------------------------------------------------------------
# p_win EV currency: simulated P(win) against an ownership-sampled field
# ---------------------------------------------------------------------------

_DEFAULT_IMPLIED_ENTRIES = 10_000.0
# Memory guard for the p_win opponent-field sample: field_scores is
# (n_sims, F) float32 plus a sorted copy, so 25k lineups at 25k sims is
# ~2.5 GB each — cap keeps that bounded regardless of contest size.
_PWIN_FIELD_CAP = 25_000


def pwin_implied_entries(groups: list[ContestGroup]) -> dict[str, float]:
    """Per contest_id implied entry count (prize_pool / (entry_fee * (1 -
    _DK_RAKE)) — see implied_field_size for the rake-adjustment reasoning),
    with contests lacking a parsable prize pool borrowing the median of
    those that have one. Used to build the p_win exponent per contest
    (sharpness * size) and to size the shared opponent-field sample
    (pwin_field_size).

    Deliberately different from implied_field_size (used by prj_own), which
    returns 0.0 on a missing prize pool: there, 0.0 fails safe by degrading
    the EV to plain projected score. Here, a p_win exponent of
    sharpness * 0 = 0 would make every lineup's q**0 == 1 for that contest —
    a constant EV that hands the whole ranking to noise — so an unknown
    field size needs a real (borrowed) estimate rather than a safe zero."""
    sizes: dict[str, float] = {}
    known = []
    for g in groups:
        if g.prize_pool_cents and g.entry_fee_cents:
            sizes[g.contest_id] = g.prize_pool_cents / (g.entry_fee_cents * (1.0 - _DK_RAKE))
            known.append(sizes[g.contest_id])
    default = float(np.median(known)) if known else _DEFAULT_IMPLIED_ENTRIES
    for g in groups:
        sizes.setdefault(g.contest_id, default)
    return sizes


def pwin_exponents(
    groups: list[ContestGroup], sharpness: float, flat_reference: float = 0.0,
) -> dict[str, float]:
    """`{contest_id: exponent}` for compute_p_win.

    `flat_reference > 0` substitutes that fixed entry count for every contest's
    own implied entries, i.e. every contest gets the SAME exponent
    (`sharpness * flat_reference`). **Default 0.0 -- per-contest scaling -- is
    the validated setting. Do not enable the flat form without re-reading the
    history below.**

    A flat exponent was briefly shipped (2026-07-30) on evidence that has since
    been RETRACTED. That evidence graded every entry in the portfolio against a
    single archived contest's field and payout curve, so a 352-entry $25 Skipper
    lineup was scored against ~11k-47k opponents and paid from their curve.
    Re-graded per contest against REAL DK payout tables (Skipper 352, Base Hit
    490, Four-Seamer 4,458, Bat Flip 9,803, mini-MAX 17,835 -- 61% of entry
    slots), per-contest scaling WINS:

        flat vs scaled   $/entry -1.75%, better on 1/8 slates, p=0.0042
        floor vs scaled  $/entry -1.64%, better on 1/8 slates, p=0.0153

    Note both flat and floor RAISE P(win) (+2.8%/+3.2%, 7/8 slates) while
    LOSING money -- they chase outright wins in contests whose payout curves do
    not reward winning enough to cover the consistency given up.

    WHY scaling is right, confirmed payout-free: mean ownership of the lineups
    each rule sends to each field size gives a small-minus-large gradient of
    **+23.2** under scaling vs **+6.8** flat. Scaling routes chalk/high-projection
    lineups to small fields and leverage to large ones -- standard DFS strategy
    that the flat form destroys. Small contests get low exponents (a 352-entry
    Skipper gets ~18) and `q**18` rewards consistency; large ones get ~900-2,400
    and reward extreme ceiling. That is the feature, not a bug.

    Also note the payout-modelling trap that produced the retracted result:
    DK payout SHAPE is a property of contest design, not field size. Real
    first-place shares are 20.0% at n=352, 10.0% at n=490, 10.0% at n=4,458,
    33.3% at n=9,803 (a tagged "[$50K to 1st]" format) and 10.0% at n=17,835.
    `scaled_payout_curve` -- which derives a curve from size alone -- cannot
    represent this and pays 84.5% to first at n=416. Never evaluate contest-size
    -dependent behaviour with it.

    `sharpness` keeps its meaning either way: the exponent is
    `sharpness * an entry count`.
    """
    if flat_reference > 0:
        exp = max(1.0, sharpness * float(flat_reference))
        return {g.contest_id: exp for g in groups}
    return {
        cid: max(1.0, sharpness * sz)
        for cid, sz in pwin_implied_entries(groups).items()
    }


def pwin_field_size(groups: list[ContestGroup], floor: int = 5_000,
                    cap: int = _PWIN_FIELD_CAP) -> int:
    """Opponent-field sample size for the p_win percentile estimate: at
    least `floor` (gpp.n_field_lineups), grown to the largest contest's
    implied entry count so q resolves "beats the whole contest" without
    hitting the percentile clip for the biggest field, capped for memory.
    Contest-size differences are the per-contest exponent's job
    (p = q**n_contest); the shared sample only needs enough resolution for
    the largest n — an exactly contest-sized sample would estimate the same
    expectation with a noisy 0/1 indicator per world."""
    sizes = pwin_implied_entries(groups)
    largest = max(sizes.values()) if sizes else float(floor)
    return int(min(max(float(floor), largest), float(cap)))


def _field_percentiles(pool_scores: np.ndarray, field_scores: np.ndarray) -> np.ndarray:
    """(M, S) float32 percentile of each pool lineup's per-world score within
    the simulated opponent field's score distribution for that world.
    `field_scores` is (S, F) as returned by ContestSimulator.score_field.
    Clipped half a field slot away from exact 0/1 so p = q**n stays
    sub-certain under field-sampling noise."""
    S, F = field_scores.shape
    M = pool_scores.shape[0]
    q = np.empty((M, S), dtype=np.float32)
    fs = np.sort(field_scores, axis=1)
    for s in range(S):
        q[:, s] = np.searchsorted(fs[s], pool_scores[:, s], side="left") / F
    np.clip(q, 0.5 / (F + 1), 1.0 - 0.5 / (F + 1), out=q)
    return q


def compute_p_win(
    pool_scores: np.ndarray,
    field_scores: np.ndarray,
    exponents: dict[str, float],
    chunk: int = 1000,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict[str, np.ndarray]:
    """The `p_win` EV currency: `{key: (M,) float64}` of
    `mean_over_worlds(percentile ** exponents[key])` for every requested
    exponent, computed in one pass over the simulated worlds.

    `pool_scores` is (M, S) — S simulated worlds, M pool lineups (see
    compute_lineup_scores). `field_scores` is (S, F), the same S worlds
    scored against an ownership-sampled opponent field
    (ContestSimulator.score_field). Per world, `q` = the candidate's
    percentile against the field that world; averaging `q ** n` over worlds
    with `n = sharpness * implied_entries` gives literal P(win an n-entry
    contest under the field model) at sharpness=1.0, sliding toward
    P(top X%) as sharpness drops. One call serves every contest at once —
    `exponents` is keyed by whatever the caller wants back (contest_id).

    Chunked over worlds (not built as one (M, S) percentile matrix like
    `_field_percentiles`): at S=25k, M=10k that intermediate alone is
    ~1 GB, before a same-sized field_scores copy. Per chunk this holds
    (M, C) pool scores, (C, F) sorted field scores and (M, C) percentiles —
    tens of MB — accumulating mean(q**n) per exponent into one (M,) float64
    each, mirroring ContestScorer's rule that tail metrics stay reduced
    accumulators, never a second (M, n_sims) matrix."""
    M, S = pool_scores.shape
    F = field_scores.shape[1]
    acc = {k: np.zeros(M, dtype=np.float64) for k in exponents}
    n_chunks = (S + chunk - 1) // chunk
    n_done = 0
    for ci, s0 in enumerate(range(0, S, chunk)):
        if stop_check is not None and stop_check():
            break
        s1 = min(s0 + chunk, S)
        fs_chunk = np.sort(field_scores[s0:s1], axis=1)          # (c, F)
        c = s1 - s0
        q = np.empty((M, c), dtype=np.float32)
        for j in range(c):
            q[:, j] = np.searchsorted(fs_chunk[j], pool_scores[:, s0 + j], side="left") / F
        np.clip(q, 0.5 / (F + 1), 1.0 - 0.5 / (F + 1), out=q)
        qd = q.astype(np.float64)
        for key, n_exp in exponents.items():
            acc[key] += np.power(qd, n_exp).sum(axis=1)
        n_done += c
        if progress_cb is not None:
            progress_cb(ci + 1, n_chunks)
    # Divide by worlds actually processed, not S — a stop_check interruption
    # then still yields a correct (if lower-precision) mean rather than a
    # sum silently under-divided by the full world count.
    if n_done > 0:
        for key in acc:
            acc[key] /= n_done
    return acc


def allocate_contests(
    pool: ExternalPool,
    corr_matrix: np.ndarray,
    groups: list[ContestGroup],
    risk: float,
    evw_base: float,
    evw_max: float,
    roi_floor_percentile: float = 40.0,
    proj_scores: Optional[np.ndarray] = None,
    proj_score_floor_percentile: float = 0.0,
    floor_scores: Optional[np.ndarray] = None,
    ev_type: str = "roi",
    own_scores: Optional[np.ndarray] = None,
    own_scale: float = _DEFAULT_OWN_SCALE,
    own_cap_start_pct: float = 100.0,
    own_cap_end_pct: float = 100.0,
    own_cap_field_size_threshold: float = _SMALL_FIELD_THRESHOLD,
    own_cap_max_field_size: Optional[float] = None,
    sim_p95_scores: Optional[np.ndarray] = None,
    sim_p99_scores: Optional[np.ndarray] = None,
    ceiling_tier_boundary: Optional[float] = None,
    p_win_cull: Optional[dict[str, np.ndarray]] = None,
    p_win_select: Optional[dict[str, np.ndarray]] = None,
    p_win_admit_n: int = 0,
    p_win_admit_multiplier: float = 0.0,
    ceiling_weight: float = 0.0,
    cash_anchor_fraction: float = 0.0,
    lane_fraction: float = 0.0,
    lane_evw: float = 0.0,
    rank_normalize: bool = False,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> ExternalAllocation:
    """One risk universe: per-contest greedy selection with shared removal.
    The candidate pool is culled per contest *before* the Det math runs — no
    legacy dollar-EV floor applies here (`ev_floor` is passed as -inf/unused
    since `precomputed` is supplied pre-culled).

    `ev_type` picks the EV currency:

    * `"roi"` (default) — the contest's SaberSim ROI column, with the
      per-contest ROI percentile floor and the ROI-StDev ceiling lean
      described below.
    * `"prj_own"` — `compute_prj_own_ev(proj_scores, own_scores,
      implied_field_size(g), own_scale)`: our own projected score minus
      projected ownership, the penalty scaled by the contest's implied field
      size against the `own_scale` calibration constant.
      Saber's ROI is not consulted at all in this mode, so
      `roi_floor_percentile` and the ceiling lean are both inert (a contest's
      ROI column being blank or unmatched is likewise no longer a reason to
      leave entries unfilled). The pool-wide `proj_score_floor_percentile`
      cull below still applies, and the reported per-lineup EV is the
      prj_own value rather than an ROI. Requires both `proj_scores` and
      `own_scores`.
    * `"p_win"` — `compute_p_win`'s simulated P(win) against an
      ownership-sampled opponent field, `p_win_select[g.contest_id]`. Like
      `prj_own`, Saber's ROI plays no part, so `roi_floor_percentile` and the
      ceiling lean are inert; the pool-wide `proj_score_floor_percentile`
      cull still applies. Two-stage winner's-curse guard, mirroring the
      internal pipeline's fresh-rescore pattern: `p_win_cull[g.contest_id]`
      (estimated on one sim/field draw) trims each contest's candidates to
      the top `p_win_admit_n` *before* `p_win_select[g.contest_id]`
      (estimated on an independent draw) ranks the survivors — a lineup
      that only looks good on the draw used to pick it can't reach the draw
      used to rank it. `p_win_admit_n <= 0` skips the cull (rank the whole
      post-floor pool on p_win_select alone). Requires both `p_win_cull` and
      `p_win_select`, each `{contest_id: (M,) array}`, built from the same
      underlying sims via two calls to compute_p_win.
    * `"proj_top"` — rank directly on `proj_scores` (plain projected mean;
      note the pool-wide floor below culls on `floor_scores`, ceiling by
      default, not this same array, when the caller passes both), no
      ROI/ownership currency at all. Like `prj_own`/`p_win`, Saber's ROI
      plays no part, so `roi_floor_percentile` and the ceiling lean are
      inert; the pool-wide `proj_score_floor_percentile` cull is the only
      cull. Requires `proj_scores`.

      Backtested (tests/backtest_needle.py, 14 archived slates 07/19-08/02):
      the single best-performing currency found for recovering a slate's own
      top-10-real-score lineups from its candidate pool (50% of slates hit
      vs. 14% for a uniform-random pick at the same budget/floor) — every
      diversity-blending or team/overlap-constrained variant tried came back
      at or below this plain ranking. But the portfolios it produces are
      correspondingly concentrated, not diversified: ~34% of a slate's teams
      represented (vs. ~90% for a team-diversity-constrained selection) and
      the single most-rostered player appears in ~76% of the portfolio on
      average — a real correlated-bust risk the needle-hit metric doesn't
      price in. The backtested/validated result is specifically the pure
      ranking (`evw=1.0`, no diversity term at all); the risk sweep's usual
      `evw_base`/`evw_max` (e.g. the shipped 0.10/0.40) means even risk=5
      only *approaches* that, never reproduces it exactly, blending in the
      same correlation diversity term as every other ev_type at every risk
      level — unvalidated for this objective, but a sane default hedge
      against the concentration. Pass `evw_base=evw_max=1.0` to reproduce
      the exact backtested setting.

      `own_cap_start_pct`/`own_cap_end_pct` (both default 100.0, i.e. off)
      add an optional large-field ownership cap on top of the ranking
      above: for any contest with `implied_field_size(g) >=
      own_cap_field_size_threshold` (fixed at 5,000, not user-configurable),
      candidates are further restricted to lineups whose `own_scores` falls
      at or below a percentile of that contest's own ownership distribution
      (the pool-wide-floor survivors). The percentile phases in linearly
      with the contest's own implied field size, from `own_cap_start_pct`
      at the threshold to `own_cap_end_pct` at the largest implied field
      size among that day's own `groups` (self-calibrating anchor, no
      hardcoded number — pass `own_cap_max_field_size` explicitly to reuse
      a precomputed, risk-invariant value across a risk sweep instead of
      recomputing it from `groups` on every call). 100.0 at either end is a
      literal no-op (`np.percentile(dist, 100)` is the distribution max,
      which nothing exceeds), so the default reproduces today's exact
      proj_top behavior byte-for-byte.

      UNVALIDATED: derived from a small-sample (10 archived slates, real
      payout tables) follow-up backtest, not yet promoted to an
      EVIDENCE_LOG.md entry. A FLAT percentile cap at any single value
      (50-95) consistently hurt cash%/top1% vs. uncapped proj_top, worse the
      tighter the cap, whether tested at a 6,000+ or a 5,000+ field-size
      threshold. This GRADUAL phase-in shape traded some of that cash-rate
      consistency (uncapped proj_top still won cash%/top1% in every tested
      variant) for more "big" finishes (payout >= 10x entry — the `ev_tail`
      convention in `tests/bt_core.py`'s `accumulate_currencies`) instead of
      occasional lucky ones — but a nearby start value (85) performed
      erratically worse than its neighbors, a sign the specific percentile
      values aren't yet distinguishable from noise at n=10. It's the
      *shape* (loose start, gentle tightening, large fields only) worth
      shipping as a dialable control, not these particular numbers — hence
      OFF by default.

      `ceiling_tier_boundary` (default `None`, i.e. off) swaps the *ranking
      signal itself* by field size instead of restricting eligibility:
      below `_SMALL_FIELD_THRESHOLD` (5,000, fixed), proj_top stays plain
      mean-`proj_scores` ranking; from there up to `ceiling_tier_boundary`,
      it ranks on `sim_p95_scores` (the 95th percentile of each lineup's
      simulated score distribution); above `ceiling_tier_boundary`, on
      `sim_p99_scores`. Requires both `sim_p95_scores` and `sim_p99_scores`
      (e.g. `np.percentile(lineup_scores, 95/99, axis=1)` over the same
      matrix `compute_lineup_scores` produces).

      VALIDATED more strongly than the ownership cap above (same 10-slate
      real-payout-table backtest, but the pattern held cleanly rather than
      being a single-slate artifact): p95 beat mean, and p99 beat both p95
      and mean, each on its own field-size population, with `drop_max`
      (this contest's largest single payout removed) staying positive for
      every uncapped tier — the signal survives the same robustness check
      that discarded most other findings in this line of testing. The
      medium/large boundary is a real cliff, not a gradual knob: every
      tested value from 10,000 up performed almost identically (differences
      within noise), while 9,000 and below dropped sharply and `drop_max`
      turned negative — driven by a specific recurring contest
      (`Bat Flip`, consistently ~9,900 implied entries) that p99 handles
      poorly and p95 handles well. 15,000 is the shipped default as a
      defensible round number inside the flat, well-supported region — not
      because it beats 10,000-14,000, which it doesn't distinguishably.
      Combining this with the ownership cap above was tested and found to
      hurt (cash%/top1% dropped for every signal it was layered onto,
      including p95/p99) — the two features are independent dials, but
      using both together is not recommended based on what's been tested.

      `p_win_admit_multiplier > 0` scales the cull by each contest's own
      entry count instead of using one flat number across every contest:
      effective_admit_n = max(p_win_admit_n, round(p_win_admit_multiplier *
      len(g.entries))), i.e. `p_win_admit_n` becomes a floor rather than the
      literal cull size. A flat admit_n gives a large contest a much
      *tighter relative* reservoir than a small one (e.g. 250 candidates
      for 72 needed picks vs. 250 for 14) -- confirmed in production data as
      a real cost: the biggest-fill contest on two live slates landed
      hit99=0 while every smaller contest on the same slate caught at least
      one, despite having the most entries (most chances) of any of them.
      Default 0.0 disables scaling -- byte-identical to a flat p_win_admit_n.

    Everything downstream of the EV vector is currency-agnostic: the greedy
    selection, the diversity/hedge terms, the shared-removal `mask` and the
    returned `(lineup, ev)` pairs all work off whichever vector was built.

    `proj_score_floor_percentile` (with `floor_scores`, a `(M,)` per-lineup
    array — pass `compute_pool_ceiling_scores(pool, players_df)`, each
    lineup's own SaberSim "99th" column with a ceiling-proxy fallback, to
    floor on ceiling rather than `compute_pool_proj_scores`'s summed
    projected mean) is a second, pool-wide floor distinct from the
    per-contest ROI floor below: it culls the bottom N% of `floor_scores`
    once, across the *entire* pool, before any contest is processed, by
    seeding the shared-removal `mask` with the exclusion up front — so a
    lineup that fails it is unavailable to every contest, not just the one
    currently being filled. `floor_scores` defaults to `proj_scores` when
    not given, so a caller that only supplies `proj_scores` keeps the old
    mean-based floor unchanged. No-op when `proj_score_floor_percentile <=
    0` or both `floor_scores` and `proj_scores` are None.

    ceiling_weight > 0 (with cash_anchor_fraction, mirroring the internal
    pipeline's ceiling-first `selector_score: tail` pattern) leans the
    *ranking* inside the greedy selection toward each contest's ROI StDev
    (see compute_ceiling_ev) without changing the floor cull, the
    correlation/diversity term, or the reported per-lineup EV — all three
    stay on plain roi. No-ops (falls back to plain roi ranking) when the
    pool's ExternalContest has no roi_stddev (older exports).

    `lane_fraction`/`lane_evw` (both 0.0 = off) reserve the LAST
    ceil(lane_fraction x k) of each contest's entries for a second
    orthogonality lane scored at `lane_evw` — see the LANE SPLIT section of
    DeterminantPortfolioSelector's docstring for why one EVw across the
    whole portfolio conflates two different jobs. Applied per contest, so
    every contest gets both lanes in proportion to its own entry count
    rather than the split falling entirely on whichever contests happen to
    be filled last. Contests with k == 1 take the plain EV argmax and are
    unaffected (there is nothing to be diverse from).

    `rank_normalize` (default False = byte-identical) puts the selector's
    EV and diversity terms on a common ordinal scale -- see RANK
    NORMALISATION in DeterminantPortfolioSelector's docstring.

    A raw ROI cutoff (e.g. >= 0.0) doesn't generalize across contests of
    different sizes/payout structures, so the floor is a percentile of
    that contest's own full ROI column: `roi_floor_percentile=40` culls
    the bottom 40% of `contest.roi` values. The threshold is computed from
    the contest's complete (un-masked) ROI distribution, so which lineups
    get culled for one contest never depends on what another contest culled
    or picked — pools legitimately differ across contests, but only because
    each contest's own ROI distribution differs, not because of cross-contest
    interference. The shared-removal `mask` (a lineup picked for one contest
    is unavailable to the rest) is a separate mechanism and still applies on
    top of the per-contest cull. Blank/unparseable ROI cells always cull
    (they sort below any real percentile). A contest with fewer surviving
    lineups than entries leaves the remainder unfilled rather than
    backfilling with sub-floor lineups.

    An absolute ROI >= 0.0 guard always applies on top of the percentile:
    the effective floor is `max(percentile_threshold, 0.0)`, so a contest
    whose bottom `roi_floor_percentile`% is still non-negative (e.g. a
    strong pool) never admits a lineup projected to lose money, even
    though the percentile alone would have let it through."""
    from src.optimization.gpp_portfolio import DeterminantPortfolioSelector

    if ev_type not in ("roi", "prj_own", "p_win", "proj_top"):
        raise ValueError(
            f"Unknown ev_type {ev_type!r} (expected 'roi', 'prj_own', 'p_win', or 'proj_top')"
        )
    if ev_type == "prj_own" and (proj_scores is None or own_scores is None):
        raise ValueError(
            "ev_type='prj_own' requires both proj_scores and own_scores "
            "(see compute_pool_proj_scores / compute_pool_ownership)"
        )
    if ev_type == "p_win" and (p_win_cull is None or p_win_select is None):
        raise ValueError(
            "ev_type='p_win' requires both p_win_cull and p_win_select "
            "(see compute_p_win)"
        )
    if ev_type == "proj_top" and proj_scores is None:
        raise ValueError(
            "ev_type='proj_top' requires proj_scores (see compute_pool_proj_scores)"
        )
    if (
        ev_type == "proj_top"
        and (own_cap_start_pct < 100.0 or own_cap_end_pct < 100.0)
        and own_scores is None
    ):
        raise ValueError(
            "ev_type='proj_top' with own_cap_start_pct/own_cap_end_pct < 100 "
            "requires own_scores (see compute_pool_ownership)"
        )
    if (
        ev_type == "proj_top" and ceiling_tier_boundary is not None
        and (sim_p95_scores is None or sim_p99_scores is None)
    ):
        raise ValueError(
            "ev_type='proj_top' with ceiling_tier_boundary set requires both "
            "sim_p95_scores and sim_p99_scores (percentiles of the simulated "
            "lineup score matrix, see compute_lineup_scores)"
        )

    M = len(pool.lineups)
    mask = np.ones(M, dtype=bool)
    _floor_basis = floor_scores if floor_scores is not None else proj_scores
    if _floor_basis is not None:
        floor = compute_proj_score_floor(_floor_basis, proj_score_floor_percentile)
        if floor is not None:
            proj_floor, _ = floor
            mask &= np.isfinite(_floor_basis) & (_floor_basis >= proj_floor)

    # Ownership cap (proj_top, large fields only) -- percentile basis and the
    # "largest field seen today" anchor, both computed ONCE here (not per
    # contest; not per risk tier when the caller precomputes and passes
    # own_cap_max_field_size explicitly). own_cap_start_pct == own_cap_end_pct
    # == 100.0 (the default) is a guaranteed no-op: np.percentile(dist, 100)
    # is the distribution max, which every finite own_scores value is <= by
    # construction -- so today's proj_top behavior is unchanged with no
    # special-case "disabled" flag needed.
    _owncap_on = (
        ev_type == "proj_top" and own_scores is not None
        and (own_cap_start_pct < 100.0 or own_cap_end_pct < 100.0)
    )
    _owncap_basis = own_scores[mask & np.isfinite(own_scores)] if _owncap_on else None
    if _owncap_on and (_owncap_basis is None or _owncap_basis.size == 0):
        _owncap_on = False
    _owncap_max_field = own_cap_max_field_size
    if _owncap_on and _owncap_max_field is None:
        _owncap_max_field = max((implied_field_size(g) for g in groups), default=0.0)

    # Ceiling-tier ranking signal (proj_top, off by default): below
    # _SMALL_FIELD_THRESHOLD, proj_top stays plain mean-proj_score ranking;
    # from there up to ceiling_tier_boundary, rank on sim_p95_scores;
    # above ceiling_tier_boundary, sim_p99_scores. ceiling_tier_boundary is
    # None (disabled) by default, reproducing today's proj_top exactly.
    _ceiling_tier_on = (
        ev_type == "proj_top" and ceiling_tier_boundary is not None
        and sim_p95_scores is not None and sim_p99_scores is not None
    )

    idx_of = {id(lu): i for i, lu in enumerate(pool.lineups)}
    portfolio: list = []
    entry_plan: list = []
    unfilled: list = []

    for g in groups:
        if stop_check is not None and stop_check():
            break
        rem_all = np.where(mask)[0]
        if ev_type == "prj_own":
            # No ROI anywhere in this branch: the currency is ours, so a
            # blank/unmatched ROI column can't strand a contest and the ROI
            # floor has nothing to floor. The pool-wide projected-score cull
            # already seeded `mask` above and is the only cull that applies.
            ev_vals = compute_prj_own_ev(
                proj_scores, own_scores, implied_field_size(g), own_scale,
            )
            rem = rem_all[np.isfinite(ev_vals[rem_all])]
        elif ev_type == "p_win":
            ev_vals = p_win_select.get(g.contest_id)
            if ev_vals is None:
                unfilled.extend(g.entries)
                continue
            rem = rem_all[np.isfinite(ev_vals[rem_all])]
            cull_for_contest = p_win_cull.get(g.contest_id)
            effective_admit_n = p_win_admit_n
            if p_win_admit_multiplier > 0:
                effective_admit_n = max(
                    p_win_admit_n, int(round(p_win_admit_multiplier * len(g.entries))),
                )
            if (effective_admit_n and effective_admit_n > 0 and cull_for_contest is not None
                    and len(rem) > effective_admit_n):
                # Stage-A cull on the INDEPENDENT p_win_cull draw — a lineup
                # that only ranks well on the draw used to select it (rem)
                # cannot also be the reason it survives this cull.
                keep = rem[np.argsort(-cull_for_contest[rem])[:effective_admit_n]]
                rem = np.sort(keep)
        elif ev_type == "proj_top":
            # No ROI/ownership currency at all: rank on proj_scores by
            # default, or (optional, off by default) a field-size-tiered
            # ceiling signal instead -- see ceiling_tier_boundary. The
            # pool-wide floor already seeded `mask`, so no further per-
            # contest cull applies here except the two optional large-field
            # features below (ceiling tiering, ownership cap).
            _field_size = (
                implied_field_size(g) if (_ceiling_tier_on or _owncap_on) else None
            )
            ev_vals = proj_scores
            if _ceiling_tier_on and _field_size >= _SMALL_FIELD_THRESHOLD:
                ev_vals = (
                    sim_p99_scores if _field_size >= ceiling_tier_boundary
                    else sim_p95_scores
                )
            rem = rem_all[np.isfinite(ev_vals[rem_all])]
            if _owncap_on and _field_size >= own_cap_field_size_threshold:
                span = _owncap_max_field - own_cap_field_size_threshold
                frac = (
                    (_field_size - own_cap_field_size_threshold) / span
                    if span > 0 else 0.0
                )
                frac = min(max(frac, 0.0), 1.0)
                pct = own_cap_start_pct + frac * (own_cap_end_pct - own_cap_start_pct)
                cutoff = np.percentile(_owncap_basis, pct)
                rem = rem[np.isfinite(own_scores[rem]) & (own_scores[rem] <= cutoff)]
        else:
            contest = pool.contests.get(g.roi_key)
            if contest is None:
                unfilled.extend(g.entries)
                continue
            roi = contest.roi
            finite_roi = roi[np.isfinite(roi)]
            if finite_roi.size == 0:
                unfilled.extend(g.entries)
                continue
            roi_floor = max(float(np.percentile(finite_roi, roi_floor_percentile)), 0.0)
            fill_value = float(finite_roi.min() - 1.0)
            ev_vals = np.nan_to_num(roi, nan=fill_value)
            rem = rem_all[ev_vals[rem_all] >= roi_floor]
        k = min(len(g.entries), len(rem))
        if k < len(g.entries):
            unfilled.extend(g.entries[k:])
        if k == 0:
            continue
        if k == 1:
            picks = [int(rem[int(np.argmax(ev_vals[rem]))])]
            pairs = [(pool.lineups[picks[0]], float(ev_vals[picks[0]]))]
        else:
            ev_override = None
            eff_cash_anchor = 0.0
            if ev_type == "roi":
                stddev = contest.roi_stddev
                ceiling = compute_ceiling_ev(
                    ev_vals[rem], stddev[rem] if stddev is not None else None, ceiling_weight,
                )
                if ceiling is not None:
                    ev_override = np.full(M, np.nan)
                    ev_override[rem] = ceiling
                    eff_cash_anchor = cash_anchor_fraction
            sel = DeterminantPortfolioSelector(
                robust_payout=None,
                candidates=pool.lineups,
                portfolio_size=k,
                risk=risk,
                evw_base=evw_base,
                evw_max=evw_max,
                ev_floor=float("-inf"),
                precomputed=(
                    rem,
                    ev_vals[rem].astype(np.float64),
                    np.ascontiguousarray(corr_matrix[np.ix_(rem, rem)]),
                ),
                ev_override=ev_override,
                cash_anchor_fraction=eff_cash_anchor,
                lane_fraction=lane_fraction,
                lane_evw=lane_evw,
                rank_normalize=rank_normalize,
            )
            pairs = sel.select(stop_check=stop_check, progress_cb=progress_cb)
            picks = [idx_of[id(lu)] for lu, _ in pairs]
        for p, (lu, ev) in zip(picks, pairs):
            mask[p] = False
            portfolio.append((lu, ev))
        entry_plan.extend(g.entries[: len(pairs)])
        if len(pairs) < k:  # stop requested mid-selection
            unfilled.extend(g.entries[len(pairs): k])
            break

    if len(portfolio) != len(entry_plan):
        raise RuntimeError("external allocation invariant broken: portfolio/entry_plan length mismatch")
    return ExternalAllocation(portfolio=portfolio, entry_plan=entry_plan, unfilled=unfilled)


# ---------------------------------------------------------------------------
# Self-play allocation (Phase 1 prototype)
#
# See /home/jduvaleus/.claude/plans/reactive-yawning-cookie.md for the full
# design writeup. Offline/eval use only -- not wired into the live pipeline.
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayAllocation:
    """Same three fields as ExternalAllocation (so anything that already
    knows how to consume allocate_contests's return type works unchanged),
    plus the Phase-1 evaluation deliverables this algorithm's design calls
    for: `source` (parallel to `portfolio` -- "external" or "generated" per
    pick, for the circularity check), `round_log` (the round-loop
    operational log, concatenated across every contest), and
    `refinement_log` (every precision-refinement swap made, concatenated
    across every qualifying contest -- see
    src.optimization.self_play.run_contest_precision_refinement)."""
    portfolio: list               # [(Lineup, roi)] flat, per-contest fill order
    entry_plan: list              # [(contest_id, j)] parallel to portfolio
    unfilled: list                # [(contest_id, j)] pool/opponent exhausted
    source: list                  # str, parallel to portfolio: "external" | "generated"
    round_log: "pd.DataFrame"
    refinement_log: "pd.DataFrame"


def build_self_play_contests(groups: list["ContestGroup"]) -> tuple[list[dict], list[dict]]:
    """Adapt the live pipeline's `ContestGroup` list into the
    `contests: list[dict]` shape `self_play_allocate_contests` expects
    (`{contest_id, n_field, fee, payout_arr, k}`) -- unlike every other
    External Pool `ev_type`, self_play's round loop rebuilds a real payout
    lookup every round and cannot run without one, so every contest here is
    backed by a real DK payout table via `nearest_payout_structure`
    (src/optimization/payout.py). Groups with no entries (`k<=0`) are
    skipped, matching allocate_contests's own convention.

    Returns `(contests, fallback_rows)` -- `fallback_rows` has one dict per
    contest whose payout table was an approximate closest-size match rather
    than an exact name match (`{contest_name, implied_field_size,
    matched_total_entries}`), for the caller to surface as a warning (see
    pipeline.py's `self_play_payout_fallback` progress event)."""
    from src.optimization.payout import nearest_payout_structure, payout_table_to_array

    contests: list[dict] = []
    fallback_rows: list[dict] = []
    for g in groups:
        k = len(g.entries)
        if k <= 0:
            continue
        n_field = implied_field_size(g)
        struct, is_approx = nearest_payout_structure(
            g.contest_name, n_field if n_field > 0 else None,
        )
        if is_approx:
            fallback_rows.append({
                "contest_name": g.contest_name,
                "implied_field_size": n_field,
                "matched_total_entries": int(struct["total_entries"]),
            })
        contests.append({
            "contest_id": g.contest_id,
            "n_field": n_field if n_field > 0 else int(struct["total_entries"]),
            "fee": g.entry_fee_cents / 100.0,
            "payout_arr": payout_table_to_array(struct),
            "k": k,
        })
    return contests, fallback_rows


def remap_self_play_entry_plan(entry_plan: list, groups: list["ContestGroup"]) -> list:
    """Map a self_play `entry_plan`/`unfilled` list (`[(contest_id, j), ...]`)
    back to the shape the rest of the live pipeline's entry-writing tail
    expects (`ExternalAllocation.entry_plan`'s `[(Path, EntryRecord), ...]`),
    via each `ContestGroup`'s own file-order `entries` list. Valid because
    `self_play_allocate_contests`'s per-contest local index `j` always runs
    `0..k-1` where `k == len(g.entries)` (see `build_self_play_contests`,
    which is what supplied that `k` in the first place)."""
    groups_by_id = {g.contest_id: g for g in groups}
    return [groups_by_id[cid].entries[j] for cid, j in entry_plan]


def self_play_allocate_contests(
    contests: list[dict],
    ctx: "SelfPlaySlateContext",
    rng_seed: int = 42,
    shortlist_size: int = 1_000,
    refresh_every: int = 5,
    run_refinement: bool = True,
    refinement_min_field_size: Optional[int] = None,
    refinement_max_swaps: Optional[int] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> SelfPlayAllocation:
    """Self-play sibling to allocate_contests: instead of ranking the whole
    pool once per contest, each contest is filled by iterative best-response
    -- every remaining candidate's real dollar ROI is scored against
    `opponents U own_selections_so_far`, one admission per round (see
    src.optimization.self_play.run_contest_self_play's NO BATCHING note --
    admitting several per round against one static field snapshot would
    reintroduce the correlated-cluster problem self-play exists to avoid),
    refreshing which specific opponent lineups populate the field from
    `ctx`'s once-per-slate base pool. Diversity is a byproduct of genuinely
    competing against prior picks rather than a separate correlation term, so
    unlike allocate_contests this takes no corr_matrix/risk/evw and never
    touches DeterminantPortfolioSelector.

    `contests` is `[{contest_id, n_field, fee, payout_arr, k}, ...]` -- the
    exact shape tests/bt_core.py::build_slate_context's `contests` list
    already produces -- consumed here in CALLER-SUPPLIED order. Pre-sort via
    `tests.bt_core.prod_order` for the real production fill order (entry fee
    desc, prize pool asc, contest_id): which contest claims the shared
    candidate pool first changes who gets what, exactly as it does for
    allocate_contests's own `mask`.

    `ctx` (src.optimization.self_play.SelfPlaySlateContext) is built once for
    the whole slate via build_self_play_context, BEFORE calling this
    function, and is the sole source of the candidate universe: its
    `lineups` (external pool U generated base-pool lineups, source-tagged)
    is what's selected from here -- there's no separate `pool` argument,
    unlike allocate_contests, because ctx already carries everything
    (a generated lineup can be, and is expected to often be, poached into
    the portfolio when it out-ROIs every external candidate, since generated
    lineups vastly outnumber external ones).

    `run_refinement` (default True) runs
    src.optimization.self_play.run_contest_precision_refinement immediately
    after each contest's round loop, for contests with
    `n_field >= refinement_min_field_size` (defaults to that module's
    _REFINEMENT_MIN_FIELD_SIZE) -- small contests are skipped, see that
    module's PRECISION REFINEMENT note for why. No-ops automatically (even
    when True) if `ctx.precise_sim` is None, i.e. `build_self_play_context`
    was called with `precise_n_sims=None`. `refinement_max_swaps` similarly
    defaults to that module's _REFINEMENT_MAX_SWAPS.

    `progress_cb`, if given, is called once per contest with a dict
    `{contest_id, k, n_field, n_rounds, elapsed_s, round_elapsed_s,
    refine_elapsed_s, n_swaps}` right after that contest fills (`elapsed_s` is
    the sum of the two `_elapsed_s` splits; `n_swaps` counts only refinement) --
    the per-contest timing breakdown this function doesn't print on
    its own (callers running this against a real log, e.g.
    scripts/eval_self_play_selector.py, want to see which contests are slow,
    and specifically whether the round loop or the refinement pass is the
    cost driver, without this library function hardcoding a print statement),
    mirroring the progress_cb convention already used by
    ContestScorer.score_candidates and DeterminantPortfolioSelector.select
    elsewhere in this codebase.
    """
    from src.optimization import self_play as sp

    if refinement_min_field_size is None:
        refinement_min_field_size = sp._REFINEMENT_MIN_FIELD_SIZE
    if refinement_max_swaps is None:
        refinement_max_swaps = sp._REFINEMENT_MAX_SWAPS

    candidate_mask = ctx.new_candidate_mask()
    opponent_mask = ctx.new_opponent_mask()
    rng = np.random.default_rng(rng_seed)

    portfolio: list = []
    entry_plan: list = []
    unfilled: list = []
    source: list = []
    round_logs: list = []
    refinement_logs: list = []

    for c in contests:
        k = int(c["k"])
        if k <= 0:
            continue
        t0 = time.time()
        result = sp.run_contest_self_play(
            ctx, contest_id=c["contest_id"], k=k, field_size=c["n_field"],
            payout_arr=c["payout_arr"], entry_fee=c["fee"],
            candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
            rng=rng, shortlist_size=shortlist_size, refresh_every=refresh_every,
        )
        t_round_done = time.time()
        own_idx, own_roi = result.own_idx, result.own_roi
        n_swaps = 0
        if run_refinement and c["n_field"] >= refinement_min_field_size:
            own_idx, own_roi, refine_log = sp.run_contest_precision_refinement(
                ctx, contest_id=c["contest_id"], own_idx=own_idx, own_roi=own_roi,
                final_shortlist_idx=result.final_shortlist_idx,
                field_size=c["n_field"], k=k, payout_arr=c["payout_arr"], entry_fee=c["fee"],
                candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
                rng=rng, max_swaps=refinement_max_swaps,
            )
            if not refine_log.empty:
                refinement_logs.append(refine_log)
                n_swaps = len(refine_log)
            # Refinement's opponents_scores/pool_scores_precise (see
            # self_play._MMAP_THRESHOLD_BYTES's comment) can be multiple GB
            # for a large-field contest -- nudge glibc to actually return
            # that freed memory to the OS before starting the next contest,
            # rather than letting it sit in the arena across the whole
            # per-contest loop.
            sp._release_free_memory()
        t_refine_done = time.time()
        if progress_cb is not None:
            progress_cb({
                "contest_id": c["contest_id"], "k": k, "n_field": c["n_field"],
                "n_rounds": len(result.own_idx), "elapsed_s": t_refine_done - t0,
                "round_elapsed_s": t_round_done - t0,
                "refine_elapsed_s": t_refine_done - t_round_done,
                "n_swaps": n_swaps,
            })
        if not result.round_log.empty:
            round_logs.append(result.round_log)
        entries = [(c["contest_id"], j) for j in range(k)]
        for j, (idx, roi) in enumerate(zip(own_idx, own_roi)):
            portfolio.append((ctx.lineups[idx], roi))
            source.append(str(ctx.source[idx]))
            entry_plan.append(entries[j])
        if len(own_idx) < k:
            unfilled.extend(entries[len(own_idx):])

    if len(portfolio) != len(entry_plan):
        raise RuntimeError(
            "self-play allocation invariant broken: portfolio/entry_plan length mismatch"
        )
    round_log = pd.concat(round_logs, ignore_index=True) if round_logs else pd.DataFrame()
    refinement_log = (
        pd.concat(refinement_logs, ignore_index=True) if refinement_logs else pd.DataFrame()
    )
    return SelfPlayAllocation(
        portfolio=portfolio, entry_plan=entry_plan, unfilled=unfilled,
        source=source, round_log=round_log, refinement_log=refinement_log,
    )


# ---------------------------------------------------------------------------
# Top-N field-coverage allocation
#
# Greedily assigns each contest's entries to whichever remaining candidate
# would have finished top-N most often (a literal rank, not a percentile)
# against a sub-sampled opponent field, across many simulated worlds -- then
# removes the worlds a candidate "claimed" so the next pick has to prove
# itself on worlds nobody has covered yet. Hard-threshold exact set-cover
# (bit-packed popcount greedy, mirroring CoveragePortfolioSelector), unlike
# p_win's smooth q**n probability. No risk sweep: a single portfolio, since
# greedy coverage is diversified by construction. See
# /home/jduvaleus/.claude/plans/given-external-candidate-lineup-optimized-scone.md
# for the full design writeup.
# ---------------------------------------------------------------------------

_TOPN_FIELD_POOL_CAP = 25_000


def build_topn_field_pool(
    players_df: pd.DataFrame,
    ownership_vec: np.ndarray,
    field_pool_size: int,
    rng_seed: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """(field_pool_size, 10) int64 player-id rows -- one
    ContestSimulator.generate_field() call. Meant to be built exactly once
    per slate and reused by every contest's coverage computation (see
    allocate_contests_topn_coverage) -- generation is the expensive step,
    the per-contest field subsets are cheap re-slices of this pool's already-
    simulated scores.

    `stop_check`, if given, is forwarded to generate_field (polled every
    1,000 attempts) -- this can be the single largest uninterruptible chunk
    of a topn_coverage run (100+s at production scale) without it, since
    everything downstream (allocate_contests_topn_coverage's per-pick loop)
    already checks stop_check but can't do anything about a Stop click that
    lands during this call."""
    from src.optimization.contest import ContestSimulator

    return ContestSimulator().generate_field(
        players_df, ownership_vec, n_lineups=field_pool_size, rng_seed=rng_seed,
        progress_cb=progress_cb, stop_check=stop_check,
    )


def augment_topn_pool_with_generated(
    pool: ExternalPool,
    players_df: pd.DataFrame,
    ownership_vec: np.ndarray,
    n_generated: int,
    rng_seed: int,
    stop_check: Optional[Callable[[], bool]] = None,
) -> tuple[ExternalPool, list]:
    """Adds up to `n_generated` additional candidate lineups to `pool`,
    drawn from the same ownership-weighted stacked-lineup generator the
    threshold field pool uses (`build_topn_field_pool` -> ContestSimulator.
    generate_field), so topn_coverage's greedy selection can pick a
    high-performing lineup that's visible in the simulated field but wasn't
    in the real external export -- previously structurally impossible
    (allocate_contests_topn_coverage only ever picks from `pool.lineups`,
    which came exclusively from parse_lineup_pool).

    MUST be called with an `rng_seed` independent of whatever seed the
    threshold field pool (`build_topn_field_pool`) was built with -- the
    two need to be genuinely separate draws, not just separately named
    variables, or a generated candidate could be one of the very field
    lineups a given (draw, world)'s threshold is computed from, comparing
    it against a bar it partly set itself (see the design discussion this
    followed: allocate_contests_topn_coverage's field pool and candidate
    pool were kept as two disjoint arrays from the start specifically to
    avoid needing a self_play-style "remove from field eligibility once
    picked" runtime guard -- two independently seeded draws make that
    guard unnecessary rather than working around not having one).

    Generated lineups are deduplicated against the EXISTING pool using the
    same 9/10-player-overlap rule `parse_lineup_pool` already applies
    (`_find_near_duplicate_removals`), with every real external lineup
    ranked above every generated one so a conflict always keeps the real
    one -- a generated lineup only ever adds NEW shapes, never displaces or
    shadows a real SaberSim pick. Generated-vs-generated conflicts (rare at
    typical pool sizes) break arbitrarily (stable sort order).

    Returns `(augmented_pool, generated_kept)` -- `generated_kept` is the
    list of generated Lineup objects that actually survived dedup (can be
    fewer than `n_generated`, both from near-duplicates and from
    ContestSimulator.generate_field's own rejection sampling not always
    reaching the requested count). `augmented_pool.lineups` is exactly
    `pool.lineups + generated_kept`, in that order -- callers that need to
    tell which of a later allocation's picks were generated-sourced (e.g.
    for progress reporting) can build a boolean mask directly from
    `len(pool.lineups)`/`generated_kept` rather than re-deriving it."""
    if n_generated <= 0:
        return pool, []

    generated_rows = build_topn_field_pool(
        players_df, ownership_vec, n_generated, rng_seed, stop_check=stop_check,
    )
    generated_lineups = [Lineup(player_ids=[int(p) for p in row]) for row in generated_rows]

    combined_lineups = list(pool.lineups) + generated_lineups
    combined_ids = [lu.player_ids for lu in combined_lineups]
    n_external = len(pool.lineups)
    # Every external lineup outranks every generated one, so any 9/10
    # conflict always keeps the real one -- generated lineups only ever
    # fill in NEW shapes.
    priority = np.concatenate([
        np.ones(n_external), np.zeros(len(generated_lineups)),
    ])
    removed = _find_near_duplicate_removals(combined_ids, priority)
    # Every external lineup is guaranteed to survive (they're already
    # mutually deduped from parse_lineup_pool, and outrank every generated
    # one here), so survivors[:n_external] == pool.lineups exactly and
    # survivors[n_external:] is exactly the surviving generated lineups, in
    # their original relative order.
    survivors = [lu for i, lu in enumerate(combined_lineups) if i not in removed]
    generated_kept = survivors[n_external:]

    augmented = ExternalPool(
        lineups=survivors, contests=pool.contests,
        n_dropped_unknown_players=pool.n_dropped_unknown_players,
        n_dropped_duplicates=pool.n_dropped_duplicates,
        n_dropped_near_duplicates=pool.n_dropped_near_duplicates + len(removed),
        source_paths=pool.source_paths,
    )
    return augmented, generated_kept


def _topn_sims_for_field_size(
    field_size_g: float,
    n_sims: int,
    sims_per_contest_fraction: float,
    sims_min: int = 0,
    sims_reference_field_size: float = 0.0,
    sims_power: float = 0.0,
) -> int:
    """Per-contest sim-world budget n_sims_g.

    Default/active path: the field-size-aware formula
    `n_sims_g = clip(round(sims_min * (field_size_g /
    sims_reference_field_size) ** sims_power), sims_min, n_sims)`, used
    whenever `sims_reference_field_size > 0` and `sims_min > 0` (the shipped
    `GppConfig` defaults -- see external_pool_topn_sims_min's calibration
    note in src/api/models.py: 392-1,189-entry fields need ~5,000 sims,
    5,945-17,835-entry fields need ~10,000, per
    scripts/calibrate_topn_sims_per_contest.py against a real archived
    slate). Falls back to a flat fraction of n_sims
    (`sims_per_contest_fraction`) when either of those two is 0 (e.g. a
    caller clears one deliberately)."""
    if sims_reference_field_size > 0 and sims_min > 0:
        n = sims_min * (max(field_size_g, 1.0) / sims_reference_field_size) ** sims_power
        return int(np.clip(round(n), sims_min, n_sims))
    n = int(round(sims_per_contest_fraction * n_sims))
    return int(np.clip(n, 1, n_sims))


def _topn_effective_rank(topn_rank: int, field_size_g: int, percentile_floor: float) -> int:
    """The literal rank a candidate must cross in THIS contest: whichever is
    LOOSER (larger -- easier to clear) of the fixed configured `topn_rank`
    and `percentile_floor` (a fraction, e.g. 0.001 = "top 0.1%") of this
    contest's own field size, rounded up. A flat top-10 bar is a much more
    extreme ask in a 17,000-entry field (top 0.06%) than a 500-entry one
    (top 2%) -- this keeps the bar's real-world difficulty comparable
    across contest sizes instead of literally fixed, while leaving small/
    medium contests (`percentile_floor * field_size_g < topn_rank`)
    untouched at the flat `topn_rank`. `percentile_floor <= 0` disables
    this entirely (pure flat `topn_rank`, the original behavior). Clipped
    to `field_size_g` itself either way (can't rank Nth in a field smaller
    than N)."""
    percentile_rank = int(np.ceil(percentile_floor * field_size_g)) if percentile_floor > 0 else 0
    effective = max(int(topn_rank), percentile_rank)
    return min(effective, field_size_g)


class _SimWorldAllocator:
    """Hands out DISJOINT slices of a shared shuffled permutation of
    `range(n_sims)` to successive `take()` callers, one per contest.

    An earlier version had each contest draw an INDEPENDENT random
    subsample of its own `n_sims_g` worlds. That only reduces *expected*
    overlap between two contests' sim-world sets, it doesn't eliminate it —
    at production scale `n_sims_g` is often a large fraction of `n_sims`
    (the calibrated sizing rule needs ~20-40% of a 25,000-sim run per
    contest), so two independent draws of that size still share a
    substantial fraction of their worlds by chance, which is exactly the
    failure mode per-contest subsampling was introduced to avoid (two
    contests' coverage races converging on the same/near-duplicate
    best-covering lineup because they're chasing largely the same worlds).
    Sequentially consuming disjoint slices of one shared permutation
    guarantees zero overlap between any two contests within a "lap"
    instead of merely making it less likely.

    When cumulative demand exceeds `n_sims` (a real possibility — e.g. many
    large-field contests on one slate), `take()` starts a fresh permutation
    ("lap") rather than raising: mirrors the per-contest greedy loop's own
    coverage-wave-reset (`allocate_contests_topn_coverage`'s `uncovered`
    exhaustion handling) — running out of a disjoint resource forces reuse,
    not failure. Contests within the same lap stay disjoint from each
    other; a contest in a later lap can share worlds with an earlier-lap
    contest, same tradeoff as the within-contest wave reset."""

    def __init__(self, n_sims: int, rng_seed: int) -> None:
        self._n_sims = n_sims
        self._rng = np.random.default_rng(rng_seed)
        self._perm = self._rng.permutation(n_sims)
        self._offset = 0
        self.lap = 0
        self.total_taken = 0

    def take(self, n: int) -> np.ndarray:
        n = min(int(n), self._n_sims)
        if self._offset + n > self._n_sims:
            self.lap += 1
            self._perm = self._rng.permutation(self._n_sims)
            self._offset = 0
        idx = np.sort(self._perm[self._offset:self._offset + n])
        self._offset += n
        self.total_taken += n
        return idx

    @property
    def lap_used_fraction(self) -> float:
        """Fraction of the CURRENT lap's capacity consumed so far -- the
        direct answer to "how close is the next contest to forcing a
        wraparound." 1.0 means the very next `take()` call is guaranteed to
        start a fresh lap regardless of how small its request is."""
        return self._offset / self._n_sims if self._n_sims > 0 else 0.0


def _topn_field_size_for_group(g: "ContestGroup", field_pool_size: int) -> int:
    """field_size_g with the nearest-payout-structure fallback for an
    unparseable prize pool (`implied_field_size` returning 0.0), clipped to
    the field pool's own size. Shared by `allocate_contests_topn_coverage`'s
    per-contest loop and `topn_total_sims_needed`'s pre-simulation sim-demand
    preview (see pipeline.py's topn_coverage auto-sizing step) -- both need
    the identical field-size-with-fallback logic, computed at two different
    points in the pipeline (before vs. during allocation)."""
    from src.optimization.payout import nearest_payout_structure

    implied = implied_field_size(g)
    if implied > 0:
        return int(np.clip(implied, 1, field_pool_size))
    struct, _ = nearest_payout_structure(g.contest_name, None)
    return int(np.clip(int(struct["total_entries"]), 1, field_pool_size))


def topn_total_sims_needed(
    groups: list["ContestGroup"],
    field_pool_size: int,
    sims_per_contest_fraction: float,
    sims_min: int = 0,
    sims_reference_field_size: float = 0.0,
    sims_power: float = 0.0,
) -> int:
    """Sum of every contest's `n_sims_g` (see `_topn_sims_for_field_size`) --
    the total sim-world budget `_SimWorldAllocator` needs to hand out a
    genuinely disjoint set to every contest with zero wraps. Meant to be
    called BEFORE the slate's Monte Carlo simulation runs, so its result can
    become the actual `n_sims` passed to `SimulationEngine.simulate()`
    (removing the "guess a large-enough n_sims" mental math) -- see
    pipeline.py's topn_coverage branch.

    Each contest's own clip-upper-bound inside `_topn_sims_for_field_size`
    is passed as an effectively-unbounded placeholder here (not the final
    `n_sims`, which is exactly what this function is computing) so no
    individual contest's natural formula value gets artificially capped
    before the sum is taken; once the real (now-sized-to-fit) `n_sims` is
    used for the actual allocation later, it's >= every individual
    contest's need by construction, so nothing clips there either."""
    total = 0
    for g in groups:
        if len(g.entries) <= 0:
            continue
        field_size_g = _topn_field_size_for_group(g, field_pool_size)
        n = _topn_sims_for_field_size(
            field_size_g, 2**31 - 1, sims_per_contest_fraction,
            sims_min, sims_reference_field_size, sims_power,
        )
        total += n
    return total


def _score_field_cols_batched(
    sim_sub_matrix: np.ndarray, cols: np.ndarray, batch_size: int = 500,
) -> np.ndarray:
    """`cols`: `(n, 10)` int32 column indices (already resolved from
    player_ids -- see `allocate_contests_topn_coverage`'s precomputed
    `field_lineup_cols`, which skips redoing that id->column lookup on
    every K-draw/contest the way `ContestSimulator.score_field` calling
    it fresh each time would). Batched over `cols`' first axis, mirroring
    `score_field`'s own batching -- an unbatched `sim_sub_matrix[:, cols]`
    materializes a `(n_sims_g, n, 10)` intermediate before the `.sum(axis=2)`
    reduction, which is exactly the blowup this whole restructuring exists
    to avoid (a real one at production scale: ~15GB for a single large
    contest's full field-pool-sized, unbatched score in early testing)."""
    n = cols.shape[0]
    n_sims_g = sim_sub_matrix.shape[0]
    out = np.empty((n_sims_g, n), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[:, start:end] = sim_sub_matrix[:, cols[start:end]].sum(axis=2)
    return out


def topn_payout_rungs(
    contest_name: str, field_size_g: int, effective_rank: int, n_rungs: int,
    tightest_rank: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """`(ranks, weights)` for the payout-weighted rank ladder — the refinement
    that turns the single top-N RANK bar into a PAYOUT-shaped objective.

    The flat bar can't tell "won the contest" from "scraped in at rank N": both
    are one covered slot. This returns `n_rungs` ranks geometrically spaced
    from 1 to `effective_rank` (the bar the single-threshold version already
    uses, so the ladder is a strict REFINEMENT — same outer admission
    boundary, more resolution inside it — not a change of scope; extending the
    rungs down to the cash line would drag the min-cash plateau back into an
    objective built to exclude it).

    `weights[j] = payout(ranks[j]) - payout(ranks[j+1])` (last rung keeps its
    own payout). Because the rungs are NESTED — clearing rank 1 implies
    clearing every looser rank — a candidate placing at realized rank rho
    clears exactly the rungs with `rank >= rho`, so its summed weight
    telescopes to `payout(rank_j)` for the tightest rung it cleared: a
    conservative step approximation of the real payout curve that becomes
    exact as `n_rungs -> effective_rank`. No double counting.

    Weights are normalized to sum to 1. Only their RATIOS matter — gains are
    only ever compared between candidates inside one contest — so the absolute
    dollars (and any mismatch between the matched structure's `total_entries`
    and this contest's real `field_size_g`) scale out entirely. That's why
    `nearest_payout_structure`'s approximation flag is not fatal here.
    """
    from src.optimization.payout import nearest_payout_structure, payout_table_to_array

    n_rungs = max(1, int(n_rungs))
    effective_rank = max(1, int(effective_rank))
    # `tightest_rank` floors how extreme the innermost rung may be. Rank 1 is
    # the rarest event on the ladder AND carries its largest weight, so at a
    # fixed sim budget it is also the least-settled -- a per-candidate claim
    # rate estimated from a handful of worlds, weighted most heavily. Raising
    # this trades payout resolution for estimator stability; see
    # scripts/diagnose_topn_rung_settling.py, which measures the split-half
    # stability of each rung at the production per-contest sim budget.
    lo = int(np.clip(tightest_rank, 1, effective_rank))
    ranks = np.unique(
        np.round(np.geomspace(lo, effective_rank, num=min(n_rungs, effective_rank - lo + 1)))
    ).astype(np.int64)

    struct, approx = nearest_payout_structure(contest_name, field_size_g)
    gross = payout_table_to_array(struct)
    if approx:
        logger.info(
            "topn_coverage ladder: no exact payout table for %r (field ~%d) — "
            "using %r's curve for rung WEIGHTS only (ratios, not dollars).",
            contest_name, field_size_g, struct.get("name", "?"),
        )
    # Rank r -> index r-1, clipped: a rung rank past the table's length keeps
    # the last (smallest) tier rather than indexing out of bounds.
    pay = gross[np.clip(ranks - 1, 0, len(gross) - 1)]
    weights = pay - np.concatenate([pay[1:], [0.0]])
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate table (e.g. every rung inside one flat tier): fall back to
        # equal weights so the ladder degrades to plain multi-rank coverage
        # rather than producing all-zero gains and forcing endless relaxation.
        weights = np.ones(len(ranks), dtype=np.float64)
        total = weights.sum()
    return ranks, (weights / total)


def compute_pool_e_dupes(
    lineups: list,
    players_df: pd.DataFrame,
    intercept: float,
    log_own_coef: float,
    salary_coef: float,
    stack_coef: float,
    salary_cap: float = 50_000.0,
) -> np.ndarray:
    """(M,) E[duplicate copies] per lineup at the dupe model's reference field
    size, ready to hand to `allocate_contests_topn_coverage(e_dupes=...)`.

    Thin adapter: builds the four id->attribute maps
    `gpp_portfolio.expected_dupes` needs out of an external-pool `players_df`
    and delegates. `players_df["ownership"]` is in PERCENTAGE POINTS for this
    pool (parse_player_projections doesn't divide "My Own"/"Adj Own" by 100 --
    see the note in pipeline.py's external branch), while the fitted model
    takes ownership as a FRACTION, so it's divided by 100 here. Getting that
    wrong is silent: percentages would push every `log(own)` positive and
    invert the model's ranking."""
    from src.optimization.gpp_portfolio import expected_dupes

    pids = players_df["player_id"].astype(int).to_numpy()
    own_map = dict(zip(pids, players_df["ownership"].astype(float).to_numpy() / 100.0))
    sal_map = dict(zip(pids, players_df["salary"].astype(float).to_numpy()))
    team_map = dict(zip(pids, players_df["team"]))
    pos_map = dict(zip(pids, players_df["position"]))
    return expected_dupes(
        lineups, own_map, sal_map, team_map, pos_map,
        salary_cap=salary_cap, intercept=intercept, log_own_coef=log_own_coef,
        salary_coef=salary_coef, stack_coef=stack_coef,
    )


def allocate_contests_topn_coverage(
    pool: ExternalPool,
    sim_results,
    groups: list[ContestGroup],
    field_lineups: np.ndarray,
    proj_scores: Optional[np.ndarray] = None,
    proj_score_floor_percentile: float = 0.0,
    floor_scores: Optional[np.ndarray] = None,
    topn_rank: int = 10,
    topn_percentile_floor: float = 0.001,
    field_samples: int = 5,
    sims_per_contest_fraction: float = 0.5,
    sims_min: int = 0,
    sims_reference_field_size: float = 0.0,
    sims_power: float = 0.0,
    relax_step: float = 1.0,
    candidate_batch_size: int = 2_000,
    rng_seed: int = 42,
    field_rng_seed: Optional[int] = None,
    pick_progress_chunk: int = 25,
    e_dupes: Optional[np.ndarray] = None,
    payout_rungs: int = 0,
    payout_tightest_rank: int = 1,
    is_generated: Optional[np.ndarray] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> ExternalAllocation:
    """Top-N field-coverage allocation: for each contest (iterated in the
    caller-supplied `groups` order -- entry fee desc, prize pool asc, same
    convention `allocate_contests`/`self_play_allocate_contests` consume
    without re-sorting), greedily fill its entries with whichever remaining
    candidate crosses a per-simulated-world top-`topn_rank` threshold most
    often among worlds nobody already picked has claimed.

    `sim_results` is a `SimulationResults`-like object (`.results_matrix`
    `(n_sims, n_players)`, `.player_ids`) and `field_lineups` is the RAW
    `(field_pool_size, 10)` player-id array from `build_topn_field_pool` --
    NEITHER is pre-scored against the sim matrix. Both candidate scores and
    field-pool scores are computed ON DEMAND, per contest, restricted to
    that contest's own `n_sims_g`-sized disjoint sim-world slice (and, for
    the field side, only the `field_size_g` lineups actually drawn into a
    given `field_samples` sample) -- never materializing a full `(n_sims,
    field_pool_size)` or `(M, n_sims)` matrix. This replaced an earlier
    version that pre-scored both matrices once up front: harmless at the
    old flat `n_sims=25,000` default, but with n_sims now auto-sized to a
    slate's real total sim-world demand (often 2-3x that, see
    `topn_total_sims_needed`), the pre-scored field matrix alone measured
    ~6.8GB on a real archived slate (auto-sized to ~68,000 sims) --
    pushing peak RSS to ~14.6GB, over the 10GB budget. Per-contest,
    per-K-draw scoring bounds the transient cost by the LARGEST single
    contest's own `(n_sims_g, field_size_g)` need instead of the whole
    slate's, freed again once that contest's picks are done.

    `proj_scores`/`proj_score_floor_percentile`/`floor_scores` mirror
    `allocate_contests`'s own pool-wide projected-score floor exactly
    (`compute_proj_score_floor`, seeded into the shared-removal `mask` once,
    before any contest is processed; `floor_scores` defaults to `proj_scores`
    when not given) -- coverage alone has no notion of "good lineup," only
    "differentiated lineup," so this keeps the mechanism choosing among
    already-viable candidates. The pool's existing 9/10-player-overlap
    near-duplicate cull (`_find_near_duplicate_removals`, applied once in
    `parse_lineup_pool` before any ev_type branch runs) already covers this
    pool regardless of ev_type -- no separate handling needed here.

    Per contest:

    1. `field_size_g` = `implied_field_size(g)` clipped to
       `[1, field_pool_size]` (falls back to the nearest real payout
       table's `total_entries` when unparseable).
    2. `n_sims_g` sim-world indices are taken as a DISJOINT slice of a
       shared shuffled permutation of `range(n_sims)` (`_SimWorldAllocator`
       -- guarantees zero overlap with every other contest's sim-world set
       within the same "lap," unlike an independent random draw, which only
       makes overlap less likely). `sim_results.results_matrix[sim_idx_g]`
       is sliced down to this subsample FIRST, and every subsequent step --
       both candidate scoring and field scoring -- only ever touches that
       small `(n_sims_g, ...)` slice, which is what bounds this contest's
       cost by its own (possibly much smaller) `n_sims_g` and keeps two
       different contests from ever racing for the same worlds (which would
       tend to pick the same lineup's near-neighbors into both).
    3. `field_samples` (K) independent `field_size_g`-lineup index-subsets
       are drawn from `field_lineups` (cheap re-slices of the raw lineup
       array, no new field generation) -- multiple field snapshots per
       contest instead of one static draw, to avoid overfitting to a single
       field's idiosyncrasies. Each draw is scored on demand, against only
       this contest's `n_sims_g` sim-world slice
       (`_score_field_cols_batched`), then discarded once its threshold row
       is extracted.
    4. Per-(field-draw, world) rank-N thresholds are computed via
       `np.partition` (avoids a full sort), where N = `_topn_effective_rank`
       (whichever is LOOSER of the fixed `topn_rank` and
       `topn_percentile_floor` fraction of this contest's own
       `field_size_g`, e.g. `topn_rank=10, topn_percentile_floor=0.001`
       makes a 17,000-entry field effectively top-17 rather than a literal
       top-10, while smaller fields stay at the flat `topn_rank`) --
       keeps the bar's real difficulty comparable across wildly different
       field sizes instead of a fixed rank being a vastly more extreme ask
       in a huge field than a small one.
    5. Crossing bits (`candidate_score >= threshold`) are computed chunked
       over candidates and bit-packed, in the same shape
       `CoveragePortfolioSelector` consumes.
    6. A greedy hard-coverage loop (popcount, `_POPCOUNT_LUT`) picks one
       candidate per step -- the one covering the most still-uncovered
       (draw, world) slots -- then removes those slots from consideration
       for the rest of this contest. If no remaining candidate covers any
       uncovered slot, every threshold is deflated (by the minimum amount
       that lets some candidate cross, equivalent to repeating the literal
       "deflate by `relax_step`" rule that many times) and crossing bits are
       recomputed. If every slot is already covered before the contest is
       full, coverage resets to a fresh "wave" rather than degenerating
       into unresolvable ties.

    Reported EV per pick is the number of (draw, world) slots that
    candidate uniquely claimed at pick time (`f`) -- there's no dollar/ROI
    value native to this currency, but `f` is a meaningful "how much did
    this pick add" figure for display/logging, paralleling every other
    ev_type's `(Lineup, ev)` return shape.

    `payout_rungs`, when > 1, replaces the single rank-N bar with a
    PAYOUT-WEIGHTED LADDER of that many nested rank bars (see
    `topn_payout_rungs`). Each pick's gain becomes
    `sum_j w_j * popcount(bits_j & uncovered_j)` instead of a flat popcount,
    so claiming a world at rank 1 is worth more than scraping in at rank N —
    the single bar scores those identically, which is the "rank bar, not a
    payout" gap. `0`/`1` keeps the original single-threshold behavior exactly.

    Each rung carries its OWN `uncovered` plane, so a world can still be
    claimed once per rung. Known limitation, inherited from the flat version:
    those planes are binary, so once a lineup wins a world outright (clearing
    every rung there) a second lineup in the same world adds nothing, even
    though a second entry really would be paid separately — today's wave-reset
    is still what covers that case. Making a rung's slot depletable by its
    rank WIDTH is the principled fix and is deliberately not attempted here.

    Cost is `R x` the crossing-bit build and `R x` the per-pick popcount;
    threshold extraction is unchanged (one multi-kth `np.partition` over the
    field-score array that was already materialized and is still discarded
    immediately). Memory grows only in the bit-plane array, which is small
    relative to the transient field-score array that already dominates.

    `e_dupes`, if given, is a `(M,)` array of expected duplicate copies per
    candidate at the dupe model's reference field size
    (`gpp_portfolio.expected_dupes` / `DUPE_REF_FIELD_SIZE`), aligned with
    `pool.lineups`. It turns each pick's coverage gain into a
    DUPLICATE-DISCOUNTED gain: `gain x 1/(1 + E[dupes_g])`, where
    `E[dupes_g] = e_dupes * field_size_g / DUPE_REF_FIELD_SIZE` rescales the
    reference-size expectation to THIS contest's field (E[copies] is linear in
    field size; the fitted intercept absorbs the reference size, so skipping
    this overstates duplication by ~20x in a 744-entry contest).

    Rationale: the top-N bar is a RANK bar, not a payout. The field pool is
    ownership-weighted, so the bar already rises in the worlds where a chalk
    play booms -- what it cannot see is that a heavily-duplicated lineup
    SPLITS whatever it wins. Two candidates that claim the same number of
    (draw, world) slots are worth different amounts when one of them is a
    build hundreds of opponents also submitted. This is the cheapest
    available correction: a per-candidate scalar, no per-world cost and no
    extra memory, reusing the already-fitted production model rather than a
    second notion of crowding.

    Only the ARGMAX is weighted. The raw popcount still drives control flow
    (relaxation triggers on nothing crossing, which is a property of the
    thresholds, not of duplication) and is still what's reported as each
    pick's `ev`, so the dumped "slots claimed" figure keeps its meaning and
    stays comparable across runs with and without the discount.

    `is_generated`, if given, is a `(M,)` boolean mask aligned with
    `pool.lineups` (see `augment_topn_pool_with_generated`, which builds
    `pool.lineups` as `real_lineups + generated_lineups` so `is_generated =
    [False]*n_real + [True]*n_generated` lines up directly) -- purely for
    progress reporting (`contest_done`'s `n_generated_picks`), no effect on
    selection itself; a generated lineup competes on identical footing to a
    real one once it's in `pool.lineups`."""
    from src.optimization.gpp_portfolio import DUPE_REF_FIELD_SIZE, _POPCOUNT_LUT
    from src.optimization import self_play as _sp

    _sp._tune_malloc_for_large_arrays()

    M = len(pool.lineups)
    n_sims = sim_results.results_matrix.shape[0]
    K = max(1, int(field_samples))

    # Resolve field_lineups' player_ids to sim-matrix column indices ONCE
    # (cheap -- doesn't scale with n_sims) rather than re-resolving them on
    # every K-draw/contest, which is what repeatedly calling
    # ContestSimulator.score_field would do now that field scoring runs
    # many times instead of once. Drop any lineup with a player missing
    # from sim_results (shouldn't happen in practice -- field_lineups and
    # sim_results are built from the same players_df -- but score_field's
    # original behavior was to drop rather than crash, so this keeps that).
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    _field_cols_raw = np.full((field_lineups.shape[0], 10), -1, dtype=np.int32)
    for _i, _row in enumerate(field_lineups):
        for _j, _pid in enumerate(_row):
            _field_cols_raw[_i, _j] = col_map.get(int(_pid), -1)
    _valid_field = (_field_cols_raw >= 0).all(axis=1)
    field_lineup_cols = _field_cols_raw[_valid_field]  # (field_pool_size, 10) int32
    field_pool_size = field_lineup_cols.shape[0]
    if field_pool_size < field_lineups.shape[0]:
        logger.warning(
            "topn_coverage: dropped %d/%d field lineups with a player missing "
            "from sim_results.", field_lineups.shape[0] - field_pool_size, field_lineups.shape[0],
        )

    # Candidate-lineup indicator matrix, precomputed once for the whole
    # pool (P x M, doesn't scale with n_sims) -- sliced per contest to
    # score only that contest's remaining candidates against only its own
    # sim-world subsample, instead of a precomputed (M, n_sims) matrix for
    # the whole pool.
    I_pool = _lineup_indicator_matrix(pool.lineups, sim_results.player_ids)  # (n_players, M)

    mask = np.ones(M, dtype=bool)
    _floor_basis = floor_scores if floor_scores is not None else proj_scores
    if _floor_basis is not None:
        floor = compute_proj_score_floor(_floor_basis, proj_score_floor_percentile)
        if floor is not None:
            proj_floor, _ = floor
            mask &= np.isfinite(_floor_basis) & (_floor_basis >= proj_floor)

    portfolio: list = []
    entry_plan: list = []
    unfilled: list = []
    sim_allocator = _SimWorldAllocator(n_sims, rng_seed)

    for contest_index, g in enumerate(groups):
        if stop_check is not None and stop_check():
            break
        k = len(g.entries)
        if k <= 0:
            continue
        rem_all = np.where(mask)[0]
        if len(rem_all) == 0:
            unfilled.extend(g.entries)
            continue

        field_size_g = _topn_field_size_for_group(g, field_pool_size)

        n_sims_g = _topn_sims_for_field_size(
            field_size_g, n_sims, sims_per_contest_fraction,
            sims_min, sims_reference_field_size, sims_power,
        )
        # `field_rng_seed` defaults to rng_seed (so a single seed still drives
        # everything, unchanged), but can be set independently to separate the
        # allocator's TWO noise sources: which sim worlds a contest gets
        # (rng_seed, via _SimWorldAllocator) versus which field lineups form
        # its K threshold draws (this rng). They cost very different amounts to
        # buy down -- more sim worlds scale the (n_sims_g x field_size_g)
        # field-score transient linearly, more field samples do not -- so
        # knowing which one drives portfolio variance decides which knob is
        # worth paying for. See scripts/diagnose_topn_variance_decomposition.py.
        _field_seed = rng_seed if field_rng_seed is None else field_rng_seed
        rng = np.random.default_rng(_field_seed + contest_index)  # the K field draws
        sim_idx_g = sim_allocator.take(n_sims_g)

        # Slice down to this contest's sim-world subsample FIRST, then score
        # ON DEMAND from that small slice -- never touching the full n_sims
        # axis again for the rest of this contest. Both of these are
        # transient, freed once this contest's picks are done (see the
        # explicit `del` at the end of the loop body) -- what bounds peak
        # memory by the largest single contest's own need instead of the
        # whole slate's.
        sim_sub_matrix = sim_results.results_matrix[sim_idx_g].astype(np.float32)  # (n_sims_g, n_players)
        cand_sub = (sim_sub_matrix @ I_pool[:, rem_all]).T                          # (len(rem_all), n_sims_g)

        N = _topn_effective_rank(topn_rank, field_size_g, topn_percentile_floor)
        n_slots = K * n_sims_g
        n_bytes_g = -(-n_slots // 8)
        pad = n_bytes_g * 8 - n_slots

        # Rung ranks + payout weights. The single-bar path is the R == 1 case
        # (one rung at rank N, weight 1.0), so everything below is one code
        # path -- no parallel implementation to drift out of sync.
        if payout_rungs and payout_rungs > 1:
            rung_ranks, rung_weights = topn_payout_rungs(
                g.contest_name, field_size_g, N, payout_rungs,
                tightest_rank=payout_tightest_rank,
            )
        else:
            rung_ranks = np.array([N], dtype=np.int64)
            rung_weights = np.array([1.0], dtype=np.float64)
        R = len(rung_ranks)

        def _draw_thresholds() -> np.ndarray:
            """(R, K, n_sims_g) float32 -- one threshold plane per rung.

            All R order statistics come from ONE `np.partition` call per field
            draw (it accepts a sequence of kth), so adding rungs costs no extra
            passes over the big field-score array; that array is still built
            and discarded exactly as before."""
            thr = np.empty((R, K, n_sims_g), dtype=np.float32)
            kths = np.unique(-rung_ranks)  # negative == from the top
            for kk in range(K):
                subset = rng.choice(field_pool_size, size=field_size_g, replace=False)
                field_batch_scores = _score_field_cols_batched(
                    sim_sub_matrix, field_lineup_cols[subset],
                )  # (n_sims_g, field_size_g) -- scored on demand, discarded after this line
                part = np.partition(field_batch_scores, kths, axis=1)
                for r, rank in enumerate(rung_ranks):
                    thr[r, kk] = part[:, -int(rank)]
                del part
            return thr

        def _crossing_bits(thr: np.ndarray) -> np.ndarray:
            """(len(rem_all), R, n_bytes_g) uint8, chunked over candidates.

            Rung-major so each rung's plane stays contiguous for the popcount
            below. The per-batch bool scratch is built one rung at a time --
            materializing (b, R, K, n_sims_g) at once would multiply the single
            largest transient in this function by R for no benefit."""
            n_cand = cand_sub.shape[0]
            bits = np.zeros((n_cand, R, n_bytes_g), dtype=np.uint8)
            for start in range(0, n_cand, candidate_batch_size):
                end = min(start + candidate_batch_size, n_cand)
                batch = cand_sub[start:end]                   # (b, n_sims_g)
                cross = np.empty((end - start, K, n_sims_g), dtype=bool)
                for r in range(R):
                    for kk in range(K):
                        cross[:, kk, :] = batch >= thr[r, kk][None, :]
                    bits[start:end, r, :] = np.packbits(
                        cross.reshape(end - start, n_slots), axis=1,
                    )
            return bits

        def _fresh_uncovered() -> np.ndarray:
            """(R, n_bytes_g) -- one uncovered plane per rung."""
            u = np.full((R, n_bytes_g), 0xFF, dtype=np.uint8)
            if pad:
                u[:, -1] = np.uint8((0xFF << pad) & 0xFF)
            return u

        # Duplicate discount for THIS contest's field size (see the e_dupes
        # paragraph in the docstring). Computed once per contest -- a plain
        # (len(rem_all),) float vector, no per-world or per-pick cost.
        dupe_weight = None
        if e_dupes is not None:
            _ed_g = np.asarray(e_dupes, dtype=np.float64)[rem_all] * (
                field_size_g / DUPE_REF_FIELD_SIZE
            )
            dupe_weight = 1.0 / (1.0 + _ed_g)

        thresholds = _draw_thresholds()
        bits = _crossing_bits(thresholds)
        remaining_local = np.ones(len(rem_all), dtype=bool)
        uncovered = _fresh_uncovered()
        # Unlike `uncovered` (reset to all-1s on every coverage wave, so it
        # only reflects the CURRENT wave), `ever_covered` accumulates via OR
        # across the whole contest, wave resets included -- the persistent
        # record of which (draw, world) slots were claimed by ANY pick,
        # ever, used for the world-level "claimed" summary at contest_done.
        ever_covered = np.zeros((R, n_bytes_g), dtype=np.uint8)

        picks_local: list[int] = []
        picks_ev: list[float] = []
        n_relaxations = 0
        n_wave_resets = 0
        t0 = time.time()
        target = min(k, len(rem_all))
        if progress_cb is not None:
            progress_cb({
                "event": "contest_start", "contest_id": g.contest_id, "k": k,
                "field_size_g": field_size_g, "n_sims_g": n_sims_g,
                "effective_rank": N,
                "contest_index": contest_index, "contests_total": len(groups),
                "sim_lap": sim_allocator.lap,
            })
        while len(picks_local) < target:
            if stop_check is not None and stop_check():
                break
            # (n_cand, R) raw popcounts, one column per rung. `per_rung` drives
            # control flow and reporting; `score` (payout-weighted, then dupe-
            # discounted) drives only the argmax.
            new_bits = np.bitwise_and(bits, uncovered[None, :, :])
            per_rung = _POPCOUNT_LUT[new_bits].sum(axis=2).astype(np.int64)
            # Total across rungs, NOT the outer rung alone: the planes deplete
            # independently, so a candidate can have nothing left to claim at
            # rank N while still being the only one able to claim an unclaimed
            # rank-1 slot. Triggering relaxation on the outer rung alone would
            # throw those picks away.
            gains = per_rung.sum(axis=1)
            score = per_rung @ rung_weights
            gains[~remaining_local] = -1
            score[~remaining_local] = -1.0
            if dupe_weight is not None:
                # Every weight is in (0, 1] and non-remaining rows are already
                # -1, so scaling preserves their "never selected" status.
                score = score * dupe_weight
            best_local = int(np.argmax(score))
            f = int(per_rung[best_local, R - 1])  # outer-rung slots == the
            # single-bar `ev`, so this stays comparable across ladder settings
            if gains[best_local] <= 0:
                if not uncovered.any():
                    # Every slot already claimed by earlier picks in this
                    # contest -- start a fresh coverage wave rather than
                    # degenerating into unresolvable ties.
                    uncovered = _fresh_uncovered()
                    n_wave_resets += 1
                    continue
                # Relax: deflate every threshold by the minimum amount that
                # lets some remaining candidate cross an uncovered slot --
                # equivalent to (and far cheaper than) repeating the literal
                # "deflate by relax_step, retry" rule that many times.
                max_score_per_world = cand_sub[remaining_local].max(axis=0)  # (n_sims_g,)
                # Min gap across EVERY rung's still-uncovered slots, deflating
                # all rungs by that one amount -- keeps the ladder's relative
                # spacing (and so its payout ordering) intact, where relaxing
                # each rung independently would let the rungs cross over.
                uncovered_mask = (
                    np.unpackbits(uncovered, axis=1)[:, :n_slots]
                    .reshape(R, K, n_sims_g).astype(bool)
                )
                gaps = thresholds - max_score_per_world[None, None, :]
                gaps = np.where(uncovered_mask, gaps, np.inf)
                min_gap = float(gaps.min())
                steps = max(1, int(np.ceil(min_gap / relax_step))) if np.isfinite(min_gap) else 1
                thresholds -= steps * relax_step
                bits = _crossing_bits(thresholds)
                n_relaxations += steps
                continue
            picks_local.append(best_local)
            picks_ev.append(float(f))
            remaining_local[best_local] = False
            np.bitwise_and(uncovered, np.bitwise_not(bits[best_local]), out=uncovered)
            np.bitwise_or(ever_covered, bits[best_local], out=ever_covered)
            if progress_cb is not None and (
                len(picks_local) % max(1, pick_progress_chunk) == 0 or len(picks_local) == target
            ):
                progress_cb({
                    "event": "pick", "contest_id": g.contest_id,
                    "pick_num": len(picks_local), "k": k,
                    "uncovered_remaining": int(_POPCOUNT_LUT[uncovered].sum()),
                    "uncovered_total": n_slots * R, "relaxations_so_far": n_relaxations,
                    "elapsed_s": time.time() - t0,
                })

        picks_global = (
            rem_all[np.array(picks_local, dtype=np.int64)] if picks_local
            else np.empty(0, dtype=np.int64)
        )
        for p, ev in zip(picks_global, picks_ev):
            mask[int(p)] = False
            portfolio.append((pool.lineups[int(p)], ev))
        entry_plan.extend(g.entries[: len(picks_global)])
        if len(picks_global) < k:
            unfilled.extend(g.entries[len(picks_global):])
        if progress_cb is not None:
            # worlds_claimed: how many of this contest's OWN n_sims_g
            # sim-worlds were covered by the final portfolio in AT LEAST ONE
            # of the K field draws (ever_covered, OR-accumulated across any
            # coverage-wave resets -- not just the current wave's state, and
            # not double-counted per K draw or per pick). Reported in the
            # SAME units as n_sims_g/sim_total_taken above (raw sim-world
            # count, not (draw, world) slots), so summing worlds_claimed's
            # denominator (n_sims_g) across every contest equals the total
            # auto-sized n_sims -- and worlds_claimed_pct is always in
            # [0, 100], with n_wave_resets called out separately instead of
            # (mis)implied by a >100% figure.
            # Outer rung (rank N) only -- that plane IS the single-bar
            # ever_covered, so worlds_claimed keeps its exact meaning and stays
            # comparable between ladder and single-bar runs. The inner rungs
            # are a strict subset of it by nesting, so they'd add nothing here.
            ever_covered_mask = (
                np.unpackbits(ever_covered[R - 1])[:n_slots].reshape(K, n_sims_g).astype(bool)
            )
            worlds_claimed = int(ever_covered_mask.any(axis=0).sum())
            n_generated_picks = (
                int(is_generated[picks_global].sum()) if is_generated is not None and len(picks_global)
                else 0
            )
            progress_cb({
                "event": "contest_done", "contest_id": g.contest_id, "k": k,
                "n_filled": len(picks_global), "n_relaxations": n_relaxations,
                "n_wave_resets": n_wave_resets, "n_generated_picks": n_generated_picks,
                "elapsed_s": time.time() - t0,
                "sim_lap": sim_allocator.lap,
                "sim_lap_used_pct": round(sim_allocator.lap_used_fraction * 100, 1),
                "sim_total_taken": sim_allocator.total_taken,
                "n_sims_total": n_sims,
                "n_sims_g": n_sims_g,
                "worlds_claimed": worlds_claimed,
                "worlds_claimed_pct": round(100 * worlds_claimed / n_sims_g, 1) if n_sims_g > 0 else 0.0,
            })
        # Explicit del + malloc_trim (see self_play._MMAP_THRESHOLD_BYTES'
        # comment for the same rationale applied here) rather than relying
        # on the next loop iteration's reassignment to free these -- both
        # can be sizable for a large contest, and this is exactly the
        # alloc/free-many-different-sized-arrays pattern that comment
        # documents glibc handling poorly without the tuning below.
        del sim_sub_matrix, cand_sub, bits
        _sp._release_free_memory()
        if len(picks_global) < k and stop_check is not None and stop_check():
            break

    if len(portfolio) != len(entry_plan):
        raise RuntimeError(
            "topn_coverage allocation invariant broken: portfolio/entry_plan length mismatch"
        )
    return ExternalAllocation(portfolio=portfolio, entry_plan=entry_plan, unfilled=unfilled)


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

def archive_external_inputs(
    project_root: Path, slate_path: str, lineups_paths: list, proj_path: Path,
) -> Optional[Path]:
    """Copy the external CSVs (plus DKSalaries, mirroring the server's
    archive convention) into archive/MMDDYYYY derived from the slate's Game
    Info date. Best-effort: returns the archive dir or None.

    DKSalaries.csv is only copied once (the slate itself doesn't change
    intra-day). Every lineups_*.csv for the slate (there may be more than
    one — see discover_external_files) and the MLB_*.csv projections
    companion are always re-copied, overwriting whatever's already archived:
    SaberSim-style exports get refreshed repeatedly as a slate firms up
    (scratches, lineup confirmations), and post-slate analysis
    (analyze_external_pool.py) wants the latest pre-lock snapshot, not
    whatever happened to be captured first — an early snapshot can otherwise
    leave since-scratched players in the archived pool with no way to
    resolve a real FPTS for them.
    """
    try:
        gi = pd.read_csv(slate_path, usecols=["Game Info"])
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(gi["Game Info"].dropna().iloc[0]))
        if not m:
            return None
        mo, dy, yr = m.groups()
        d = project_root / "archive" / f"{mo}{dy}{yr}"
        d.mkdir(parents=True, exist_ok=True)
        copies = [(Path(slate_path), "DKSalaries.csv", False)]
        copies += [(Path(lp), Path(lp).name, True) for lp in lineups_paths]
        copies.append((proj_path, proj_path.name, True))
        for src, dst_name, always_refresh in copies:
            dst = d / dst_name
            if always_refresh or not dst.exists():
                shutil.copy2(str(src), str(dst))
        return d
    except Exception as exc:
        logger.warning("External pool: failed to archive inputs: %s", exc)
        return None
