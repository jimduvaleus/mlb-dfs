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


def _parse_lineup_header(path: Path, rows: list) -> tuple:
    """Identify a lineup file's contest ROI blocks. Returns
    (contest_cols, stddev_cols): norm_name -> (raw prefix, col idx) / col idx."""
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
    if not contest_cols:
        raise ValueError(
            f"External lineup file has no contest ROI blocks "
            f"(no '<name> ROI' column with a '<name> Sim Dupes' sibling): {path}"
        )
    return contest_cols, stddev_cols


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


def parse_lineup_pool(paths, valid_ids: set[int]) -> ExternalPool:
    """Parse one or more lineup exports for the same slate (see
    discover_external_files) with csv.reader on each raw header (duplicate
    'P'/'OF' slot headers and any duplicate contest names must be seen
    verbatim — never pandas' '.1' mangling).

    Lineups are deduplicated by player-id set across *all* files combined —
    a lineup appearing more than once (within one file or across files) is
    kept only once, using the roi/roi_stddev from its first occurrence in
    file order. A second pass then also drops near-duplicates: lineups
    whose 10-player set overlaps another surviving lineup's in exactly 9
    players (a single swapped player). Conflicts are resolved by ROI in
    the largest contest by parsed prize pool (see
    _pick_primary_contest_index) — the lower-ROI lineup of any conflicting
    pair is dropped (see _find_near_duplicate_removals for why this needs
    a priority pass rather than simple equivalence-class grouping).

    Contest ROI blocks are matched across files by normalized name (first
    file to define a given contest wins its raw name / prize pool / single
    entry metadata). A contest present in one file but not another leaves
    NaN roi (and roi_stddev) for the lineups sourced from file(s) missing
    that column — no different from any other blank ROI cell."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    per_file: list[tuple] = []  # (path, data_rows, contest_cols, stddev_cols)
    contest_order: list[str] = []  # norm names, first-seen order across files
    contest_meta: dict[str, tuple] = {}  # norm -> (raw_prefix, prize_pool_cents, single_entry)
    any_stddev: set[str] = set()

    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise ValueError(f"External lineup file is empty: {path}")
        contest_cols, stddev_cols = _parse_lineup_header(path, rows)
        for norm, (prefix, _) in contest_cols.items():
            if norm not in contest_meta:
                contest_order.append(norm)
                contest_meta[norm] = (
                    prefix, _parse_prize_pool_cents(prefix), bool(_SINGLE_ENTRY_RE.search(prefix)),
                )
        any_stddev.update(stddev_cols.keys())
        per_file.append((path, rows[1:], contest_cols, stddev_cols))

    lineups: list[Lineup] = []
    roi_rows: list[list[float]] = []
    stddev_rows: list[list[float]] = []
    seen: set[frozenset[int]] = set()
    n_unknown = 0
    n_dup = 0
    for path, data_rows, contest_cols, stddev_cols in per_file:
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

    # --- Near-duplicate removal (9/10 player overlap) --------------------
    n_near_dup = 0
    if len(lineups) > 1:
        primary_j = _pick_primary_contest_index(contest_order, contest_meta)
        primary_roi = np.array([row[primary_j] for row in roi_rows], dtype=np.float64)
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

    A pitcher row is kept when Status == "Confirmed" (the file lists the
    whole rotation/bullpen, not just the day's starter). For a team with no
    Confirmed pitcher at all, the highest-projected-mean pitcher on that
    team's roster is kept instead as the projected (unconfirmed) starter —
    mirroring how an unconfirmed batter with an Order still shows as the
    amber-bubble projected slot rather than being dropped. A batter row is
    kept only when it carries an Order (its projected batting slot); Status
    == "Confirmed" then distinguishes an officially confirmed slot from a
    merely projected one (slot_confirmed drives the UI's green/amber bubble +
    the team lock icon exactly as it does for the RotoWire/DFF/Market-Odds
    sources, so no separate UI plumbing is needed for SaberSim).
    """
    df = pd.read_csv(path)
    mean_col = "fd_points" if platform == "fanduel" else "dk_points"
    std_col = "fd_std" if platform == "fanduel" else "dk_std"
    is_pitcher = df["Pos"].astype(str) == "P"
    confirmed = df["Status"].astype(str) == "Confirmed"
    order = pd.to_numeric(df.get("Order"), errors="coerce")
    mean_raw = pd.to_numeric(df[mean_col], errors="coerce")
    lineup_slot = pd.Series(np.nan, index=df.index, dtype=float)
    lineup_slot.loc[is_pitcher] = 10
    lineup_slot.loc[~is_pitcher] = order.loc[~is_pitcher]

    keep_pitcher = is_pitcher & confirmed
    pitchers = pd.DataFrame({
        "team": df["Team"], "mean": mean_raw, "confirmed": confirmed,
    })[is_pitcher]
    teams_with_confirmed = set(pitchers.loc[pitchers["confirmed"], "team"])
    fallback_candidates = pitchers[
        ~pitchers["team"].isin(teams_with_confirmed) & pitchers["mean"].notna()
    ]
    if not fallback_candidates.empty:
        fallback_starter_idx = fallback_candidates.groupby("team")["mean"].idxmax()
        keep_pitcher.loc[fallback_starter_idx] = True

    out = pd.DataFrame({
        "player_id": pd.to_numeric(df["DFS ID"], errors="coerce"),
        "name": df["Name"],
        "mean": mean_raw,
        "std_dev": pd.to_numeric(df[std_col], errors="coerce"),
        "lineup_slot": lineup_slot,
        "slot_confirmed": confirmed,
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


def build_quantile_grids(proj_ext: pd.DataFrame, n_points: int = 101) -> dict[int, np.ndarray]:
    """Per-player evenly spaced quantile grids for EmpiricalQuantileMarginal,
    resampled from the file's irregular percentiles. Skips a player (engine
    falls back to Gaussian) on missing/non-monotone knots or a >20% mismatch
    between the grid-implied mean and the file mean."""
    q_levels = np.array([0.25, 0.50, 0.75, 0.85, 0.95, 0.99])
    grid_q = np.linspace(0.0, 1.0, n_points)
    grids: dict[int, np.ndarray] = {}
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
        grids[int(r.player_id)] = grid
    logger.info("External pool: quantile grids built for %d/%d players.", len(grids), len(proj_ext))
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


def implied_field_size(group: "ContestGroup") -> float:
    """Implied entrant count for a contest: prize pool / entry fee. DK contest
    names carry no entrant count (the `[150 Entry Max]` tag is a per-user entry
    cap, not a field size), so the parsed prize pool
    (`dk_entries._parse_prize_pool_cents`, including its `Qualifier` divisor)
    over the entry fee is the size proxy — the same ratio `dk_entries._sort_ratio`
    uses to order entries, and the same prize-pool proxy already used for
    contest matching and fill order here.

    Returns 0.0 when either side is missing or non-positive, which
    deliberately zeroes the ownership penalty in `compute_prj_own_ev` — an
    unparseable prize pool degrades that EV to pure projected score rather
    than reshaping a portfolio around a guessed field size."""
    prize = group.prize_pool_cents
    fee = group.entry_fee_cents
    if not prize or not fee or prize <= 0 or fee <= 0:
        return 0.0
    return float(prize) / float(fee)


_DEFAULT_OWN_SCALE = 30_000.0


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
    """Per contest_id implied entry count (prize_pool / entry_fee — the same
    inference dk_entries' assignment sort uses), with contests lacking a
    parsable prize pool borrowing the median of those that have one. Used to
    build the p_win exponent per contest (sharpness * size) and to size the
    shared opponent-field sample (pwin_field_size).

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
            sizes[g.contest_id] = g.prize_pool_cents / g.entry_fee_cents
            known.append(sizes[g.contest_id])
    default = float(np.median(known)) if known else _DEFAULT_IMPLIED_ENTRIES
    for g in groups:
        sizes.setdefault(g.contest_id, default)
    return sizes


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
    ev_type: str = "roi",
    own_scores: Optional[np.ndarray] = None,
    own_scale: float = _DEFAULT_OWN_SCALE,
    p_win_cull: Optional[dict[str, np.ndarray]] = None,
    p_win_select: Optional[dict[str, np.ndarray]] = None,
    p_win_admit_n: int = 0,
    ceiling_weight: float = 0.0,
    cash_anchor_fraction: float = 0.0,
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

    Everything downstream of the EV vector is currency-agnostic: the greedy
    selection, the diversity/hedge terms, the shared-removal `mask` and the
    returned `(lineup, ev)` pairs all work off whichever vector was built.

    `proj_score_floor_percentile` (with `proj_scores`, a `(M,)` per-lineup
    summed-projected-mean array from `compute_pool_proj_scores`) is a
    second, pool-wide floor distinct from the per-contest ROI floor below:
    it culls the bottom N% of projected score once, across the *entire*
    pool, before any contest is processed, by seeding the shared-removal
    `mask` with the exclusion up front — so a lineup that fails it is
    unavailable to every contest, not just the one currently being filled.
    No-op when `proj_score_floor_percentile <= 0` or `proj_scores` is None.

    ceiling_weight > 0 (with cash_anchor_fraction, mirroring the internal
    pipeline's ceiling-first `selector_score: tail` pattern) leans the
    *ranking* inside the greedy selection toward each contest's ROI StDev
    (see compute_ceiling_ev) without changing the floor cull, the
    correlation/diversity term, or the reported per-lineup EV — all three
    stay on plain roi. No-ops (falls back to plain roi ranking) when the
    pool's ExternalContest has no roi_stddev (older exports).

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

    if ev_type not in ("roi", "prj_own", "p_win"):
        raise ValueError(f"Unknown ev_type {ev_type!r} (expected 'roi', 'prj_own', or 'p_win')")
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

    M = len(pool.lineups)
    mask = np.ones(M, dtype=bool)
    if proj_scores is not None:
        floor = compute_proj_score_floor(proj_scores, proj_score_floor_percentile)
        if floor is not None:
            proj_floor, _ = floor
            mask &= np.isfinite(proj_scores) & (proj_scores >= proj_floor)
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
            if (p_win_admit_n and p_win_admit_n > 0 and cull_for_contest is not None
                    and len(rem) > p_win_admit_n):
                # Stage-A cull on the INDEPENDENT p_win_cull draw — a lineup
                # that only ranks well on the draw used to select it (rem)
                # cannot also be the reason it survives this cull.
                keep = rem[np.argsort(-cull_for_contest[rem])[:p_win_admit_n]]
                rem = np.sort(keep)
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
