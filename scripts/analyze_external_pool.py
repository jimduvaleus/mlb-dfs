"""
Evaluate an EXTERNAL lineup pool (SaberSim-style import, see
src/api/external_pool.py) against actual DraftKings contest results — the
external-pool analogue of analyze_candidate_pool.py.

Differences from the internal candidate-pool analyzer:
  - Loads lineups_*.csv (slot-ordered player-id lineups + per-contest ROI
    blocks, e.g. archive/07192026/lineups_dk_mlb_classic_7-19-2026_135pm.csv)
    instead of candidate_pool_debug.csv. The companion MLB_*_DK_*.csv
    projections file is located the same way the live pipeline does
    (src.api.external_pool.discover_external_files) but isn't required here
    — actual scores come from contest_player_fpts.json, keyed directly by
    the same DK player ids used in the lineup file.
  - There is no single dollar EV in this format — each lineup instead
    carries a distinct ROI/Win Rate/Cash Rate figure per contest tier the
    export was built for (e.g. "MLB $15K mini-MAX [150 Entry Max]"). Pick
    one with --contest (substring match, case-insensitive); use
    --list-contests to see what's available. --sweep sweeps a ROI floor for
    the selected tier instead of the internal script's dollar EV floor. ROI
    is reported in percentage points (71.5 means +71.5%), matching the
    portfolio-panel UI's convention (PortfolioTable.tsx: `mean_ev * 100`).
  - --top N reports the top N lineups by actual score, same as the internal
    script, but reports the selected contest tier's roi/win_rate/cash_rate
    instead of mined_ev/fresh_ev/seed_source/tail_bypass.

Usage
-----
    python scripts/analyze_external_pool.py archive/07192026
    python scripts/analyze_external_pool.py archive/07192026 --list-contests
    python scripts/analyze_external_pool.py archive/07192026 --top 25
    python scripts/analyze_external_pool.py archive/07192026 --top 25 --contest "mini-MAX"

    # Sweep a ROI floor instead of the standard report — same marginal
    # floor/cash-rate/pool-size tradeoff as analyze_candidate_pool.py
    # --sweep, but keyed on one contest tier's ROI column:
    python scripts/analyze_external_pool.py archive/07192026 --sweep --contest "mini-MAX"
    python scripts/analyze_external_pool.py archive/07192026 --sweep --contest "Four-Seamer" \\
        --roi-floor -50 --sweep-end 100 --sweep-step 10

Output
------
  - Per-slate report printed to stdout (default mode): Spearman correlations
    of the selected contest's roi / proj_score / ownership against
    actual_score, plus the same real-field ROI-floor calibration section as
    the internal script (real percentile / approximate cash rate, below vs.
    at-or-above the floor, decile breakdowns).
  - archive/MMDDYYYY/external_pool_eval.csv        (default mode, per-lineup joined table)
  - archive/MMDDYYYY/external_top_candidates.csv   (--top)
  - archive/MMDDYYYY/external_roi_floor_sweep.csv  (--sweep)
  - archive/external_pool_summary.csv              (default mode, appended, multi-slate runs only)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.external_pool import discover_external_files, normalize_contest_name  # noqa: E402

# analyze_candidate_pool.py lives alongside this script — reuse its real-field /
# floor-sweep / decile machinery rather than re-deriving it (it's column-name
# generic apart from the hardcoded "projected_ev" floor column, which the
# external-pool functions below alias the selected contest's ROI column to).
from analyze_candidate_pool import (  # noqa: E402
    _slate_sort_key,
    _append_summary,
    _decile_table,
    add_real_percentile,
    compute_floor_metrics,
    default_cash_threshold,
    load_contest_player_fpts,
    load_real_field_points,
    sweep_floor_table,
)
from scipy.stats import spearmanr  # noqa: E402

_DEFAULT_ROI_FLOOR = 0.0
_N_SLOT_COLS = 10
_ROI_SUFFIX = " ROI"
_SIM_DUPES_SUFFIX = " Sim Dupes"
_WIN_RATE_SUFFIX = " Win Rate"
_CASH_RATE_SUFFIX = " Cash Rate"


def discover_contest_blocks(columns) -> dict:
    """norm_name -> {raw_name, roi, win_rate, cash_rate} for every column
    ending ' ROI' that has a ' Sim Dupes' sibling — the same test
    src.api.external_pool.parse_lineup_pool uses to separate real per-
    contest ROI blocks from the export's generic Slate-size bucket columns
    (e.g. "Large Slate | 10k-50k"), which don't have one.
    """
    col_set = set(columns)
    blocks = {}
    for col in columns:
        if not col.endswith(_ROI_SUFFIX):
            continue
        prefix = col[: -len(_ROI_SUFFIX)]
        if f"{prefix}{_SIM_DUPES_SUFFIX}" not in col_set:
            continue
        norm = normalize_contest_name(prefix)
        blocks[norm] = {
            "raw_name": prefix,
            "roi": col,
            "win_rate": f"{prefix}{_WIN_RATE_SUFFIX}",
            "cash_rate": f"{prefix}{_CASH_RATE_SUFFIX}",
        }
    return blocks


def load_external_lineups(lineups_path: Path) -> tuple[pd.DataFrame, dict, int]:
    """Load the raw export into one row per lineup: lineup_index, player_ids
    (list of 10 DK ids), proj_score, ownership, salary, and
    roi__<contest>/win_rate__<contest>/cash_rate__<contest> for every
    contest tier found. Exact-duplicate lineups (same 10 ids, order-
    independent) are dropped, keeping the first occurrence.
    """
    df = pd.read_csv(lineups_path)
    if len(df.columns) < _N_SLOT_COLS:
        raise ValueError(
            f"{lineups_path} does not look like a lineup export "
            f"(fewer than {_N_SLOT_COLS} columns)"
        )
    id_cols = list(df.columns[:_N_SLOT_COLS])
    player_ids = df[id_cols].astype("int64").values.tolist()

    contest_blocks = discover_contest_blocks(df.columns)
    if not contest_blocks:
        raise ValueError(
            f"{lineups_path}: no contest ROI blocks found (no '<name> ROI' column "
            "with a matching '<name> Sim Dupes' sibling)"
        )

    out = pd.DataFrame({
        "lineup_index": np.arange(len(df)),
        "player_ids": player_ids,
        "proj_score": pd.to_numeric(df.get("Proj Score"), errors="coerce"),
        "ownership": pd.to_numeric(df.get("Ownership"), errors="coerce"),
        "salary": pd.to_numeric(df.get("Salary"), errors="coerce"),
    })
    for norm, block in contest_blocks.items():
        # x100 -> percentage points, matching the portfolio-panel UI's ROI
        # convention (PortfolioTable.tsx does `mean_ev * 100`); the raw file
        # stores ROI as a fraction (1.062 == +106.2%).
        out[f"roi__{norm}"] = pd.to_numeric(df[block["roi"]], errors="coerce") * 100.0
        out[f"win_rate__{norm}"] = pd.to_numeric(df.get(block["win_rate"]), errors="coerce")
        out[f"cash_rate__{norm}"] = pd.to_numeric(df.get(block["cash_rate"]), errors="coerce")

    key = out["player_ids"].apply(lambda ids: frozenset(ids))
    n_before = len(out)
    out = out.loc[~key.duplicated()].reset_index(drop=True)
    n_dup = n_before - len(out)
    return out, contest_blocks, n_dup


def add_actual_score(lineup_df: pd.DataFrame, fpts_map: dict) -> pd.DataFrame:
    """actual_score = sum of each player's real FPTS — NaN if any of the
    lineup's players is missing from fpts_map (partial coverage shouldn't
    silently understate a lineup's score)."""
    df = lineup_df.copy()

    def _score(ids):
        vals = [fpts_map.get(int(pid)) for pid in ids]
        if any(v is None for v in vals):
            return np.nan
        return float(sum(vals))

    df["actual_score"] = df["player_ids"].apply(_score)
    return df


def resolve_contest(contest_blocks: dict, query: str | None) -> str:
    """Return the norm_name of the contest tier matching --contest (substring,
    case-insensitive). Defaults to the first tier in file column order when
    --contest is omitted."""
    if query is None:
        return next(iter(contest_blocks))
    q = query.casefold()
    matches = [norm for norm, b in contest_blocks.items() if q in b["raw_name"].casefold()]
    if not matches:
        available = "\n  ".join(b["raw_name"] for b in contest_blocks.values())
        raise ValueError(f"--contest {query!r} matched no contest tier. Available:\n  {available}")
    if len(matches) > 1:
        names = "\n  ".join(contest_blocks[m]["raw_name"] for m in matches)
        raise ValueError(f"--contest {query!r} matched multiple contest tiers — be more specific:\n  {names}")
    return matches[0]


def print_contest_list(contest_blocks: dict, lineup_df: pd.DataFrame) -> None:
    print(f"{len(contest_blocks)} contest tiers found:")
    for norm, block in contest_blocks.items():
        roi = lineup_df[f"roi__{norm}"]
        print(
            f"  {block['raw_name']!r}  "
            f"roi[min={roi.min():+.1f}% mean={roi.mean():+.1f}% max={roi.max():+.1f}%]"
        )


def top_candidates_table(lineup_df: pd.DataFrame, contest_norm: str, top_n: int) -> pd.DataFrame:
    """Top `top_n` lineups by actual_score, descending. Lineups with
    incomplete actual-score coverage (a player missing from the resolved
    contest fpts map) are excluded — their actual_score is NaN, not low."""
    roi_col, win_col, cash_col = f"roi__{contest_norm}", f"win_rate__{contest_norm}", f"cash_rate__{contest_norm}"
    df = lineup_df.dropna(subset=["actual_score"]).sort_values("actual_score", ascending=False).head(top_n)
    return df[[
        "lineup_index", "actual_score", "salary", "proj_score", "ownership", roi_col, win_col, cash_col,
    ]].rename(columns={
        "proj_score": "mean", roi_col: "roi", win_col: "win_rate", cash_col: "cash_rate",
    }).reset_index(drop=True)


def run_top_candidates(archive_dirs: list[Path], top_n: int, contest_query: str | None) -> None:
    for d in archive_dirs:
        try:
            found = discover_external_files(str(d))
            if not found["lineups_path"]:
                raise FileNotFoundError(f"no lineups_*.csv in {d}")
            lineup_df, contest_blocks, n_dup = load_external_lineups(found["lineups_path"])
            fpts_map = load_contest_player_fpts(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue

        try:
            contest_norm = resolve_contest(contest_blocks, contest_query)
        except ValueError as exc:
            print(f"{d.name}: {exc}")
            continue

        lineup_df = add_actual_score(lineup_df, fpts_map)
        table = top_candidates_table(lineup_df, contest_norm, top_n)

        dup_note = f"  ({n_dup} duplicate lineups dropped)" if n_dup else ""
        print(
            f"\n=== {d.name} === top {len(table)} lineups by actual_score "
            f"[contest={contest_blocks[contest_norm]['raw_name']!r}]{dup_note}"
        )
        # roi is already in percentage points (see load_external_lineups) — format
        # it as "+71.5%" for the printed table, matching the portfolio-panel UI's
        # display convention; the CSV keeps the plain numeric percentage.
        display = table.copy()
        display["roi"] = display["roi"].map(lambda x: f"{x:+.1f}%")
        print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        out_path = d / "external_top_candidates.csv"
        table.to_csv(out_path, index=False)
        print(f"Top candidates written -> {out_path}")


def run_roi_sweep(
    archive_dirs: list[Path], contest_query: str | None, start: float, end: float | None, step: float,
    cash_threshold: float, top_percentile: float,
) -> None:
    for d in archive_dirs:
        try:
            found = discover_external_files(str(d))
            if not found["lineups_path"]:
                raise FileNotFoundError(f"no lineups_*.csv in {d}")
            lineup_df, contest_blocks, _ = load_external_lineups(found["lineups_path"])
            fpts_map = load_contest_player_fpts(d)
            field_points = load_real_field_points(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue

        try:
            contest_norm = resolve_contest(contest_blocks, contest_query)
        except ValueError as exc:
            print(f"{d.name}: {exc}")
            continue
        roi_col = f"roi__{contest_norm}"

        lineup_df = add_actual_score(lineup_df, fpts_map)
        lineup_df = add_real_percentile(lineup_df, field_points, cash_threshold, top_percentile)
        # compute_floor_metrics/sweep_floor_table are written against a
        # hardcoded "projected_ev" column name — alias the selected
        # contest's ROI column to it rather than forking that logic.
        swept = lineup_df.rename(columns={roi_col: "projected_ev"})

        sweep_end = end
        if sweep_end is None:
            max_roi = swept["projected_ev"].max()
            sweep_end = float(np.ceil(max_roi / step) * step)

        table = sweep_floor_table(swept, len(field_points), start, sweep_end, step)
        table = table.rename(columns={"floor": "roi_floor"})

        print(
            f"\n=== {d.name} === [contest={contest_blocks[contest_norm]['raw_name']!r}]  "
            f"n_field={len(field_points)}  cash_threshold={cash_threshold:.2f}  "
            f"top_percentile={top_percentile:.2f}  n_lineups={len(lineup_df)}  "
            f"sweep={start:+.1f}%..{sweep_end:+.1f}% step={step:.1f}%"
        )
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        sweep_path = d / "external_roi_floor_sweep.csv"
        table.to_csv(sweep_path, index=False)
        print(f"Sweep written -> {sweep_path}")


def compute_slate_metrics(lineup_df: pd.DataFrame, contest_norm: str) -> dict:
    roi_col = f"roi__{contest_norm}"
    complete = lineup_df.dropna(subset=["actual_score"])
    n_total = len(lineup_df)
    n_complete = len(complete)
    metrics = {
        "n_lineups": n_total,
        "n_complete": n_complete,
        "coverage": n_complete / n_total if n_total else float("nan"),
        "spearman_roi_actual": float("nan"),
        "spearman_proj_actual": float("nan"),
        "spearman_ownership_actual": float("nan"),
    }
    if n_complete >= 10:
        metrics["spearman_roi_actual"] = spearmanr(complete[roi_col], complete["actual_score"]).correlation
        metrics["spearman_proj_actual"] = spearmanr(complete["proj_score"], complete["actual_score"]).correlation
        metrics["spearman_ownership_actual"] = spearmanr(complete["ownership"], complete["actual_score"]).correlation
    return metrics


def print_condensed_line(name: str, metrics: dict) -> None:
    print(
        f"{name:>12}  n={metrics['n_lineups']:>6}  cov={metrics['coverage'] * 100:5.1f}%  "
        f"r_roi={metrics['spearman_roi_actual']:+.3f}  r_proj={metrics['spearman_proj_actual']:+.3f}  "
        f"r_own={metrics['spearman_ownership_actual']:+.3f}"
    )


def evaluate_archive_dirs(
    archive_dirs: list[Path], contest_query: str | None, roi_floor: float,
    cash_threshold: float, top_percentile: float,
) -> None:
    rows = []
    for d in archive_dirs:
        try:
            found = discover_external_files(str(d))
            if not found["lineups_path"]:
                raise FileNotFoundError(f"no lineups_*.csv in {d}")
            lineup_df, contest_blocks, n_dup = load_external_lineups(found["lineups_path"])
            fpts_map = load_contest_player_fpts(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue

        try:
            contest_norm = resolve_contest(contest_blocks, contest_query)
        except ValueError as exc:
            print(f"{d.name}: {exc}")
            continue
        roi_col = f"roi__{contest_norm}"
        contest_raw = contest_blocks[contest_norm]["raw_name"]

        lineup_df = add_actual_score(lineup_df, fpts_map)
        metrics = compute_slate_metrics(lineup_df, contest_norm)

        try:
            field_points = load_real_field_points(d)
            lineup_df = add_real_percentile(lineup_df, field_points, cash_threshold, top_percentile)
            swept = lineup_df.rename(columns={roi_col: "projected_ev"})
            metrics.update(compute_floor_metrics(swept, roi_floor, len(field_points)))
            metrics["cash_threshold"] = cash_threshold
            metrics["top_percentile"] = top_percentile
            complete = lineup_df.dropna(subset=["would_top_pct"])
            metrics["top_pct_rate"] = complete["would_top_pct"].mean() if len(complete) else float("nan")
            metrics["n_top_pct"] = int(complete["would_top_pct"].sum())

            if len(archive_dirs) == 1:
                print(f"\n=== {d.name} === [contest={contest_raw!r}]")
                print(
                    f"Lineups: {metrics['n_lineups']}  (complete actual-score coverage: "
                    f"{metrics['n_complete']} = {metrics['coverage'] * 100:.1f}%)"
                )
                print(f"Spearman(roi,        actual_score) = {metrics['spearman_roi_actual']:+.3f}")
                print(f"Spearman(proj_score, actual_score) = {metrics['spearman_proj_actual']:+.3f}")
                print(f"Spearman(ownership,  actual_score) = {metrics['spearman_ownership_actual']:+.3f}")

                print("\nROI deciles (1 = lowest ROI .. 10 = highest ROI):")
                roi_dec = _decile_table(lineup_df.rename(columns={roi_col: "roi"}), "roi")
                print(roi_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not roi_dec.empty
                      else "  (not enough complete-coverage lineups for deciles)")

                print("\nOwnership deciles (1 = lowest owned .. 10 = highest owned):")
                own_dec = _decile_table(lineup_df, "ownership")
                print(own_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}") if not own_dec.empty
                      else "  (not enough complete-coverage lineups for deciles)")

                pct_label = top_percentile * 100
                print(
                    f"\nTop {pct_label:.1f}th percentile (real field): "
                    f"{metrics['top_pct_rate'] * 100:.1f}% of lineups ({metrics['n_top_pct']} / {metrics['n_complete']})"
                )
                print(
                    f"\nROI-floor calibration vs. real field (n_field={metrics['n_field']:.0f}, "
                    f"roi_floor={roi_floor:+.1f}%, cash_threshold={cash_threshold:.2f}, "
                    f"top_percentile={top_percentile:.2f}):"
                )
                print(
                    f"  below floor      (n={metrics['n_below_floor']:>6.0f}):  "
                    f"mean real_percentile={metrics['real_percentile_below_floor']:.3f}   "
                    f"cash_rate={metrics['cash_rate_below_floor']:.3f}   "
                    f"top{pct_label:.0f}_rate={metrics['top_pct_rate_below_floor']:.3f}"
                )
                print(
                    f"  at/above floor   (n={metrics['n_at_or_above_floor']:>6.0f}):  "
                    f"mean real_percentile={metrics['real_percentile_at_or_above_floor']:.3f}   "
                    f"cash_rate={metrics['cash_rate_at_or_above_floor']:.3f}   "
                    f"top{pct_label:.0f}_rate={metrics['top_pct_rate_at_or_above_floor']:.3f}"
                )
            else:
                print_condensed_line(d.name, metrics)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{d.name}: ROI-floor calibration skipped ({exc})")
            if len(archive_dirs) > 1:
                print_condensed_line(d.name, metrics)

        eval_path = d / "external_pool_eval.csv"
        lineup_df.to_csv(eval_path, index=False)
        rows.append({"slate": d.name, "contest": contest_raw, **metrics})

    if not rows:
        print("No slates with both a lineups_*.csv export and contest_player_fpts.json found.")
        return

    if len(rows) > 1:
        agg_df = pd.DataFrame(rows)
        print(f"\n=== Aggregate across {len(rows)} slates ===")
        for col in ("spearman_roi_actual", "spearman_proj_actual", "spearman_ownership_actual"):
            vals = agg_df[col].dropna()
            if len(vals):
                print(
                    f"{col}: mean={vals.mean():+.3f}  median={vals.median():+.3f}  "
                    f"std={vals.std():.3f}  n_slates={len(vals)}"
                )
        summary_path = PROJECT_ROOT / "archive" / "external_pool_summary.csv"
        _append_summary(rows, summary_path)
        print(f"\nSummary appended -> {summary_path}")
    else:
        print(f"\nPer-lineup eval written -> {archive_dirs[0] / 'external_pool_eval.csv'}")


def _find_recent_external_slates(n: int) -> list[Path]:
    archive_root = PROJECT_ROOT / "archive"
    candidates = []
    for d in archive_root.iterdir():
        if not d.is_dir():
            continue
        found = discover_external_files(str(d))
        if found["lineups_path"] and (d / "contest_player_fpts.json").exists():
            candidates.append(d)
    return sorted(candidates, key=lambda d: _slate_sort_key(d.name))[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an external (SaberSim-style) lineup pool against actual DK contest results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
        help="Archive directories to evaluate (e.g. archive/07192026). Omit when using --recent.",
    )
    parser.add_argument(
        "--recent", type=int, default=0, metavar="N",
        help="Evaluate the N most recent slates that have both a lineups_*.csv export and "
             "contest_player_fpts.json. Mutually exclusive with positional ARCHIVE_DIR args.",
    )
    parser.add_argument(
        "--contest", type=str, default=None, metavar="SUBSTRING",
        help="Which contest tier's ROI/Win Rate/Cash Rate to use (case-insensitive substring "
             "match against the tier name, e.g. 'mini-MAX' or 'Four-Seamer'). Defaults to the "
             "first tier in the file's column order. Use --list-contests to see all tiers.",
    )
    parser.add_argument(
        "--list-contests", action="store_true",
        help="Print the contest tiers found in the lineup export (with ROI min/mean/max) and exit.",
    )
    parser.add_argument(
        "--top", type=int, default=0, metavar="N",
        help="Instead of the standard report, print (and write "
             "archive/MMDDYYYY/external_top_candidates.csv) the top N lineups by actual "
             "fantasy score, descending, with lineup_index/salary/mean/ownership/roi/win_rate/"
             "cash_rate for the selected --contest tier. Mutually exclusive with --sweep.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Instead of the standard report, sweep a ROI floor (for the selected --contest "
             "tier) from --roi-floor up to --sweep-end in --sweep-step increments, printing the "
             "marginal floor/cash-rate/pool-size tradeoff per slate. Writes "
             "archive/MMDDYYYY/external_roi_floor_sweep.csv.",
    )
    parser.add_argument(
        "--roi-floor", type=float, default=_DEFAULT_ROI_FLOOR, metavar="PCT",
        help=f"ROI floor to calibrate against the real field, or the start of the --sweep range, "
             f"in percentage points — matching the portfolio-panel UI (e.g. 20 for +20%% ROI) "
             f"(default: {_DEFAULT_ROI_FLOOR}, i.e. breakeven).",
    )
    parser.add_argument(
        "--sweep-end", type=float, default=None, metavar="PCT",
        help="Top of the --sweep range, in ROI percentage points (default: rounded up to the "
             "next --sweep-step multiple of the slate's own max ROI for the selected contest, "
             "so the sweep covers the full pool).",
    )
    parser.add_argument(
        "--sweep-step", type=float, default=10.0, metavar="PCT",
        help="Increment between --sweep grid points, in ROI percentage points (default: 10.0).",
    )
    parser.add_argument(
        "--cash-threshold", type=float, default=None, metavar="FRACTION",
        help="Fraction of the real field a lineup must beat to count as an approximate cash "
             "(default: derived from data/payout_structures/dk_classic_gpp.json, falls back to 0.74).",
    )
    parser.add_argument(
        "--top-percentile", type=float, default=0.95, metavar="FRACTION",
        help="Real-field percentile above which a lineup counts toward the 'top percentile' "
             "readout (default: 0.95).",
    )
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR arguments are mutually exclusive.")
    if args.top and args.sweep:
        parser.error("--top and --sweep are mutually exclusive.")

    if args.recent:
        dirs = _find_recent_external_slates(args.recent)
        if not dirs:
            print(f"No slates with both required files found in {PROJECT_ROOT / 'archive'}.")
            sys.exit(1)
        print(f"--recent {args.recent}: selected {[d.name for d in dirs]}")
    else:
        dirs = []
        for raw in args.archive_dirs:
            p = Path(raw)
            if not p.exists():
                print(f"Warning: {p} does not exist — skipping.")
                continue
            if not p.is_dir():
                print(f"Warning: {p} is not a directory — skipping.")
                continue
            dirs.append(p)
        if not dirs:
            print("No valid archive directories found.")
            sys.exit(1)

    if args.list_contests:
        for d in dirs:
            found = discover_external_files(str(d))
            if not found["lineups_path"]:
                print(f"{d.name}: no lineups_*.csv found.")
                continue
            lineup_df, contest_blocks, _ = load_external_lineups(found["lineups_path"])
            print(f"\n=== {d.name} ===")
            print_contest_list(contest_blocks, lineup_df)
        return

    cash_threshold = args.cash_threshold if args.cash_threshold is not None else default_cash_threshold()
    if args.top:
        run_top_candidates(dirs, top_n=args.top, contest_query=args.contest)
    elif args.sweep:
        run_roi_sweep(
            dirs, contest_query=args.contest, start=args.roi_floor, end=args.sweep_end,
            step=args.sweep_step, cash_threshold=cash_threshold, top_percentile=args.top_percentile,
        )
    else:
        evaluate_archive_dirs(
            dirs, contest_query=args.contest, roi_floor=args.roi_floor,
            cash_threshold=cash_threshold, top_percentile=args.top_percentile,
        )


if __name__ == "__main__":
    main()
