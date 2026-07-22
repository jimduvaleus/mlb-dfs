"""
Validate the external-pool "ceiling lean" (`gpp.external_pool_ceiling_weight`,
see src.api.external_pool.compute_ceiling_ev) against REALIZED contest
outcomes, across every archived slate that has both a SaberSim lineup export
and a graded external_pool_eval.csv.

compute_ceiling_ev ranks lineups by roi + weight * resid_z * roi_std, where
resid_z is the z-scored residual of ROI StDev after regressing out ROI (the
part of a lineup's StDev that its ROI doesn't already predict). The theory is
that residual is "extra ceiling" worth reaching for. This script checks
whether that residual actually correlates with what happened in the real
contest, using each archived day's own realized real_percentile (already
computed by scripts/analyze_external_pool.py against the real contest-
standings zip).

Reuses compute_ceiling_ev itself (rather than re-deriving the regression) so
this script can never silently drift from the production ranking math: the
per-lineup "boost" (resid_z * roi_std) is recovered as
compute_ceiling_ev(roi, stddev, weight=1.0) - roi, exploiting linearity in
weight.

Prerequisite per archived day: archive/MMDDYYYY/external_pool_eval.csv must
already exist (built by `python scripts/analyze_external_pool.py
archive/MMDDYYYY`, which needs that day's contest_player_fpts.json + a
contest-standings-*.zip). Days without it are skipped with a note.

Usage
-----
    # every archive day with both required files
    python scripts/validate_ceiling_lean.py

    # specific days only
    python scripts/validate_ceiling_lean.py archive/07192026 archive/07202026

    # test different candidate weights
    python scripts/validate_ceiling_lean.py --weights 0.0 0.25 0.35 0.5

Output
------
Per-day table (one row per contest tier) printed to stdout, a pooled summary
across all analyzed days, and outputs/ceiling_lean_validation.csv (re-run
overwrites rows for any day re-analyzed, so it stays a current snapshot as
new archived days accumulate — treat it as something to re-check every few
slates, not a one-time verdict, especially while day count is small: rows
from the same day share one slate/field and are not independent trials.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.external_pool import (  # noqa: E402
    _MIN_CEILING_FIT_N, compute_ceiling_ev, discover_external_files,
)
from analyze_candidate_pool import _slate_sort_key  # noqa: E402

ARCHIVE_ROOT = PROJECT_ROOT / "archive"
OUT_PATH = PROJECT_ROOT / "outputs" / "ceiling_lean_validation.csv"
DEFAULT_WEIGHTS = [0.0, 0.15, 0.25, 0.35, 0.5]
_ROI_SUFFIX = " ROI"
_STDDEV_SUFFIX = " ROI StDev"


def discover_tier_columns(columns) -> dict[str, tuple[str, str]]:
    """norm_name -> (roi_col, stddev_col) for every '<name> ROI' column that
    has a '<name> ROI StDev' sibling — only these are contest blocks with
    enough data for compute_ceiling_ev (mirrors the Sim-Dupes-sibling test
    src.api.external_pool.parse_lineup_pool uses to separate real per-
    contest columns from the export's generic Slate-size buckets, which
    have neither StDev nor Sim Dupes)."""
    col_set = set(columns)
    tiers = {}
    for col in columns:
        if not col.endswith(_ROI_SUFFIX):
            continue
        prefix = col[: -len(_ROI_SUFFIX)]
        std_col = f"{prefix}{_STDDEV_SUFFIX}"
        if std_col not in col_set:
            continue
        tiers[prefix.casefold()] = (col, std_col)
    return tiers


def find_days() -> list[Path]:
    days = []
    for d in ARCHIVE_ROOT.iterdir():
        if d.is_dir() and (d / "external_pool_eval.csv").exists():
            days.append(d)
    return sorted(days, key=lambda d: _slate_sort_key(d.name))


def analyze_day(day_dir: Path, weights: list[float]) -> list[dict]:
    ev = pd.read_csv(day_dir / "external_pool_eval.csv")
    found = discover_external_files(str(day_dir))
    if not found["lineups_path"]:
        print(f"  ! {day_dir.name}: no lineups_*.csv found — skipping", file=sys.stderr)
        return []
    lu = pd.read_csv(found["lineups_path"])

    # external_pool_eval.csv's own lineup_index always points back to rows
    # in the ORIGINAL raw lineups_*.csv (assigned before analyze_external_
    # pool.py's exact-duplicate-lineup dedup), so this stays correctly
    # aligned even on days where duplicates were dropped and ev is shorter
    # than lu.
    if "lineup_index" not in ev.columns or ev["lineup_index"].max() >= len(lu):
        print(f"  ! {day_dir.name}: lineup_index doesn't align with {found['lineups_path'].name} — skipping",
              file=sys.stderr)
        return []
    lu_aligned = lu.iloc[ev["lineup_index"].to_numpy()].reset_index(drop=True)
    real_pct = ev["real_percentile"].to_numpy()

    rows = []
    for prefix_cf, (roi_col, std_col) in discover_tier_columns(lu.columns).items():
        roi = pd.to_numeric(lu_aligned[roi_col], errors="coerce").to_numpy()
        stddev = pd.to_numeric(lu_aligned[std_col], errors="coerce").to_numpy() / 100.0
        finite = np.isfinite(roi) & np.isfinite(stddev) & np.isfinite(real_pct)
        n = int(finite.sum())
        if n < _MIN_CEILING_FIT_N:
            continue

        # Recover resid_z * roi_std (the weight-free "boost") from the
        # production function itself at weight=1.0 rather than re-deriving
        # the regression here, exploiting compute_ceiling_ev's linearity in
        # weight -- keeps this script honest against whatever the real
        # ranking math does, even if that math changes later.
        ceiling_at_1 = compute_ceiling_ev(roi, stddev, 1.0)
        if ceiling_at_1 is None:
            continue
        boost_full = ceiling_at_1 - roi

        roi_f, real_f, boost = roi[finite], real_pct[finite], boost_full[finite]

        rho_roi, _ = spearmanr(roi_f, real_f)
        rho_boost, p_boost = spearmanr(boost, real_f)
        # Partial correlation: boost vs. real outcome after removing what
        # plain roi alone already predicts about the real outcome -- the
        # genuinely NEW information question, same framing as the boost's
        # own construction (residual of stddev after regressing out roi).
        rp_slope, rp_intercept = np.polyfit(roi_f, real_f, 1)
        real_resid = real_f - (rp_intercept + rp_slope * roi_f)
        rho_partial, p_partial = spearmanr(boost, real_resid)

        row = {
            "day": day_dir.name,
            "tier": roi_col[: -len(_ROI_SUFFIX)],
            "n": n,
            "corr_roi_stddev": float(np.corrcoef(roi_f, stddev[finite])[0, 1]),
            "rho_roi_real": float(rho_roi),
            "rho_boost_real": float(rho_boost),
            "p_boost_real": float(p_boost),
            "rho_boost_real_partial": float(rho_partial),
            "p_boost_real_partial": float(p_partial),
        }
        for w in weights:
            ceiling = roi_f + w * boost
            rho_c, _ = spearmanr(ceiling, real_f)
            row[f"rho_ceiling_w{w:g}_real"] = float(rho_c)
        rows.append(row)
    return rows


def print_day_table(rows: list[dict], weights: list[float]) -> None:
    df = pd.DataFrame(rows)
    cols = ["tier", "n", "corr_roi_stddev", "rho_roi_real", "rho_boost_real_partial", "p_boost_real_partial"]
    cols += [f"rho_ceiling_w{w:g}_real" for w in weights]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_pooled_summary(all_rows: list[dict]) -> None:
    if not all_rows:
        print("\nNo days with usable data.")
        return
    df = pd.DataFrame(all_rows)
    n_days = df["day"].nunique()
    print(f"\n=== Pooled across {n_days} day(s), {len(df)} contest-tier observations ===")
    print(f"Median rho(roi, real):                     {df['rho_roi_real'].median():+.3f}")
    print(f"Median partial rho(residual, real | roi):  {df['rho_boost_real_partial'].median():+.3f}")
    against = (df["rho_boost_real_partial"] < 0).mean()
    print(f"Share of (day, tier) rows where the residual signal points AGAINST real outcome: {against:.0%}")
    if n_days < 5:
        print(
            "\nCaveat: fewer than 5 archived days. Rows from the same day share one slate/field "
            "and are not independent trials -- treat this as a rough, noisy read, not a powered "
            "test. Re-run as more archived days accumulate."
        )


def write_summary(all_rows: list[dict], out_path: Path) -> None:
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    df["run_ts"] = datetime.now().isoformat(timespec="seconds")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_csv(out_path, dtype={"day": str})
        old = old[~old["day"].isin(df["day"].unique())]  # refresh any re-analyzed days
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values("day", key=lambda s: s.map(lambda n: _slate_sort_key(n)))
    df.to_csv(out_path, index=False)
    print(f"\nSummary written -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the external-pool ceiling lean against realized contest outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument(
        "days", nargs="*", metavar="ARCHIVE_DIR",
        help="Specific archive/MMDDYYYY dirs (default: every day with external_pool_eval.csv).",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS, metavar="W",
        help=f"external_pool_ceiling_weight values to test (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    day_dirs = [Path(d) for d in args.days] if args.days else find_days()
    if not day_dirs:
        print(
            "No archive days with external_pool_eval.csv found. Build it first with:\n"
            "  python scripts/analyze_external_pool.py archive/MMDDYYYY"
        )
        return

    all_rows: list[dict] = []
    for day_dir in day_dirs:
        print(f"\n--- {day_dir.name} ---")
        if not (day_dir / "external_pool_eval.csv").exists():
            print(f"  ! no external_pool_eval.csv (run scripts/analyze_external_pool.py {day_dir} first) — skipping")
            continue
        rows = analyze_day(day_dir, args.weights)
        if not rows:
            print("  (no contest tiers with enough paired ROI/StDev data)")
            continue
        print_day_table(rows, args.weights)
        all_rows.extend(rows)

    print_pooled_summary(all_rows)
    write_summary(all_rows, args.out)


if __name__ == "__main__":
    main()
