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

from src.api.dk_entries import EntryRecord, _parse_prize_pool_cents
from src.optimization.lineup import Lineup

logger = logging.getLogger(__name__)

_N_SLOT_COLS = 10
_SINGLE_ENTRY_RE = re.compile(r"\[\s*single\s+entry\s*\]", re.IGNORECASE)
_LINEUPS_GLOB = "lineups_*.csv"
# 'lineups_dk_mlb_classic_7-17-2026_705pm.csv' -> ('7-17-2026', '705pm')
_LINEUPS_TOKEN_RE = re.compile(r"lineups_.*?_(\d{1,2}-\d{1,2}-\d{4})_(\d{3,4}[ap]m)", re.IGNORECASE)


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
    source_path: Path


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
    """Newest lineups_*.csv in raw_dir + its companion projections CSV.

    Pairing: the slate token from the lineups filename ('7-17-2026', '705pm')
    must appear in the companion as 'YYYY-MM-DD-<time>' ('2026-07-17-705pm');
    falls back to the newest MLB_*_DK_*.csv with paired_by_token=False.
    """
    d = Path(raw_dir)
    out = {"lineups_path": None, "projections_path": None, "paired_by_token": False}
    lineup_files = sorted(d.glob(_LINEUPS_GLOB), key=lambda p: p.stat().st_mtime)
    if not lineup_files:
        return out
    lp = lineup_files[-1]
    out["lineups_path"] = lp
    m = _LINEUPS_TOKEN_RE.search(lp.name)
    if m:
        mo, dy, yr = m.group(1).split("-")
        token = f"{yr}-{int(mo):02d}-{int(dy):02d}-{m.group(2).lower()}"
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
            lp.name, out["projections_path"].name,
        )
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_contest_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def parse_lineup_pool(path: Path, valid_ids: set[int]) -> ExternalPool:
    """Parse the lineup export with csv.reader on the raw header (duplicate
    'P'/'OF' slot headers and any duplicate contest names must be seen
    verbatim — never pandas' '.1' mangling)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"External lineup file is empty: {path}")
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
            logger.warning("External pool: duplicate contest block %r — keeping first.", prefix)
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

    lineups: list[Lineup] = []
    roi_rows: list[list[float]] = []
    stddev_rows: list[list[float]] = []
    seen: set[frozenset[int]] = set()
    n_unknown = 0
    n_dup = 0
    norm_names = list(contest_cols.keys())
    for r in rows[1:]:
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
        for norm in norm_names:
            _, idx = contest_cols[norm]
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

    roi_mat = np.array(roi_rows, dtype=np.float64) if roi_rows else np.zeros((0, len(norm_names)))
    # See ExternalContest.roi_stddev: Saber's raw cell is already pct-pt
    # scaled like `roi * 100`, so /100 puts it on roi's own fraction scale.
    stddev_mat = (
        np.array(stddev_rows, dtype=np.float64) / 100.0
        if stddev_rows else np.zeros((0, len(norm_names)))
    )
    contests = {}
    for j, norm in enumerate(norm_names):
        raw, _ = contest_cols[norm]
        contests[norm] = ExternalContest(
            raw_name=raw,
            norm_name=norm,
            roi=roi_mat[:, j],
            prize_pool_cents=_parse_prize_pool_cents(raw),
            single_entry=bool(_SINGLE_ENTRY_RE.search(raw)),
            roi_stddev=stddev_mat[:, j] if norm in stddev_cols else None,
        )
    logger.info(
        "External pool: %d lineups (%d dropped unknown-player, %d duplicate), %d contests: %s",
        len(lineups), n_unknown, n_dup, len(contests),
        ", ".join(c.raw_name for c in contests.values()),
    )
    return ExternalPool(
        lineups=lineups, contests=contests,
        n_dropped_unknown_players=n_unknown, n_dropped_duplicates=n_dup,
        source_path=path,
    )


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

    cols = ["player_id", "team", "opponent", "slot", "mean", "std_dev",
            "position", "salary", "game", "name", "eligible_positions"]
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


def compute_pool_corr(lineups: list, sim_results) -> np.ndarray:
    """(M, M) float32 correlation of simulated lineup scores (points-space).

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

    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    P = len(col_map)
    M = len(lineups)
    I = np.zeros((P, M), dtype=np.float32)
    for j, lu in enumerate(lineups):
        for pid in lu.player_ids:
            I[col_map[int(pid)], j] = 1.0
    scores = (sim_results.results_matrix.astype(np.float32) @ I).T  # (M, n_sims)
    pre = DeterminantPortfolioSelector.precompute_pool(scores, float("-inf"))
    assert pre is not None and len(pre[0]) == M
    return pre[2]


def allocate_contests(
    pool: ExternalPool,
    corr_matrix: np.ndarray,
    groups: list[ContestGroup],
    risk: float,
    evw_base: float,
    evw_max: float,
    roi_floor_percentile: float = 40.0,
    ceiling_weight: float = 0.0,
    cash_anchor_fraction: float = 0.0,
    stop_check: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> ExternalAllocation:
    """One risk universe: per-contest greedy selection with shared removal.
    EV currency = the contest's ROI column. The candidate pool is culled
    per contest *before* the Det/ROI math runs — no legacy dollar-EV floor
    applies here (`ev_floor` is passed as -inf/unused since `precomputed`
    is supplied pre-culled).

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

    M = len(pool.lineups)
    mask = np.ones(M, dtype=bool)
    idx_of = {id(lu): i for i, lu in enumerate(pool.lineups)}
    portfolio: list = []
    entry_plan: list = []
    unfilled: list = []

    for g in groups:
        if stop_check is not None and stop_check():
            break
        contest = pool.contests[g.roi_key]
        roi = contest.roi
        finite_roi = roi[np.isfinite(roi)]
        if finite_roi.size == 0:
            unfilled.extend(g.entries)
            continue
        roi_floor = max(float(np.percentile(finite_roi, roi_floor_percentile)), 0.0)
        fill_value = float(finite_roi.min() - 1.0)
        roi = np.nan_to_num(roi, nan=fill_value)
        rem_all = np.where(mask)[0]
        rem = rem_all[roi[rem_all] >= roi_floor]
        k = min(len(g.entries), len(rem))
        if k < len(g.entries):
            unfilled.extend(g.entries[k:])
        if k == 0:
            continue
        if k == 1:
            picks = [int(rem[int(np.argmax(roi[rem]))])]
            pairs = [(pool.lineups[picks[0]], float(roi[picks[0]]))]
        else:
            ev_override = None
            eff_cash_anchor = 0.0
            stddev = contest.roi_stddev
            ceiling = compute_ceiling_ev(
                roi[rem], stddev[rem] if stddev is not None else None, ceiling_weight,
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
                    roi[rem].astype(np.float64),
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
    project_root: Path, slate_path: str, lineups_path: Path, proj_path: Path,
) -> Optional[Path]:
    """Copy the two external CSVs (plus DKSalaries, mirroring the server's
    archive convention) into archive/MMDDYYYY derived from the slate's Game
    Info date. Best-effort: returns the archive dir or None.

    DKSalaries.csv is only copied once (the slate itself doesn't change
    intra-day). lineups_*.csv and the MLB_*.csv projections companion are
    always re-copied, overwriting whatever's already archived: SaberSim-style
    exports get refreshed repeatedly as a slate firms up (scratches, lineup
    confirmations), and post-slate analysis (analyze_external_pool.py) wants
    the latest pre-lock snapshot, not whatever happened to be captured first
    — an early snapshot can otherwise leave since-scratched players in the
    archived pool with no way to resolve a real FPTS for them.
    """
    try:
        gi = pd.read_csv(slate_path, usecols=["Game Info"])
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(gi["Game Info"].dropna().iloc[0]))
        if not m:
            return None
        mo, dy, yr = m.groups()
        d = project_root / "archive" / f"{mo}{dy}{yr}"
        d.mkdir(parents=True, exist_ok=True)
        for src, dst_name, always_refresh in [
            (Path(slate_path), "DKSalaries.csv", False),
            (lineups_path, lineups_path.name, True),
            (proj_path, proj_path.name, True),
        ]:
            dst = d / dst_name
            if always_refresh or not dst.exists():
                shutil.copy2(str(src), str(dst))
        return d
    except Exception as exc:
        logger.warning("External pool: failed to archive inputs: %s", exc)
        return None
