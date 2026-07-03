"""
Evaluate the candidate pool (pre-portfolio-selection lineups) against actual
DraftKings contest results.

Joins archive/MMDDYYYY/candidate_pool_debug.csv (per-candidate-lineup
projected score/EV/ownership, dumped by the pipeline when dump_candidate_pool
is enabled) against archive/MMDDYYYY/contest_player_fpts.json (the resolved
player_id -> actual FPTS map written by GET /api/contest/analyze) to answer:
does our projected EV / projected score / projected ownership actually
predict real contest performance?

It also calibrates the `ev_floor` ($) config setting against the REAL
contest field rather than our own simulated one: DraftKings' contest-
standings zip actually contains a full per-entry table (Rank, EntryId,
Points, ...) for the entire field, in addition to the player-ownership
sidebar table the rest of this script (and the live "Analyze Contest"
endpoint) already parses. Inserting a candidate's actual_score into that
real field's sorted Points distribution gives a real percentile rank with
no payout-table assumption needed at all — unlike a dollar conversion, which
would require each contest's actual (and not uniformly retrievable) payout
table, since no two DK contests share the same size or prize structure. An
approximate "would this have cashed" binary is also reported, using the
existing dk_classic_gpp.json-derived cash threshold (~74% of the field beaten
to cash, see PipelineRunner._load_cash_threshold) as a genre-level
approximation rather than this specific contest's exact cash line.

Usage
-----
    python scripts/analyze_candidate_pool.py archive/06262026
    python scripts/analyze_candidate_pool.py archive/06262026 archive/06272026
    python scripts/analyze_candidate_pool.py --recent 5
    python scripts/analyze_candidate_pool.py archive/06262026 --ev-floor 0.30

    # Sweep ev_floor instead of the standard report — traces the marginal
    # floor/cash-rate/pool-size tradeoff (e.g. starting at $0.20 in $0.30
    # increments, up to the slate's own max projected_ev by default):
    python scripts/analyze_candidate_pool.py archive/06262026 --sweep
    python scripts/analyze_candidate_pool.py archive/06262026 --sweep --ev-floor 0.20 --sweep-end 6.0 --sweep-step 0.30

Output
------
  - Per-slate report printed to stdout: Spearman correlations of
    projected_ev / projected_score / avg_ownership against actual_score, a
    selected-vs-not-selected breakdown, and an ev_floor calibration section
    (real percentile / approximate cash rate, below vs. at-or-above the
    floor, plus a decile breakdown). Full decile tables are only printed for
    a single slate; multiple slates print a condensed per-slate line plus an
    aggregate table across slates.
  - archive/MMDDYYYY/candidate_pool_eval.csv  (per-lineup joined table).
  - archive/candidate_pool_summary.csv        (one row per slate, appended).
  - With --sweep: archive/MMDDYYYY/candidate_pool_floor_sweep.csv per slate,
    plus the same table printed to stdout, instead of the standard report.
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_EV_FLOOR = 0.20

_REQUIRED_POOL_COLUMNS = {
    "lineup_index", "player_id", "mean", "ownership", "mean_ev", "selected_risks",
}


def _slate_sort_key(name: str) -> tuple:
    """Chronological sort key for archive dir names like '06262026' or
    '06262026e' (MMDDYYYY + optional suffix) — mirrors evaluate_ownership.py."""
    try:
        return (datetime.strptime(name[:8], "%m%d%Y"), name)
    except ValueError:
        return (datetime.min, name)


def load_candidate_pool(archive_dir: Path) -> pd.DataFrame:
    path = archive_dir / "candidate_pool_debug.csv"
    if not path.exists():
        raise FileNotFoundError(f"no candidate_pool_debug.csv in {archive_dir}")
    df = pd.read_csv(path)
    missing = _REQUIRED_POOL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)} — this looks like a "
            "pre-enrichment dump (ownership/mean_ev/selected_risks were added "
            "after the candidate pool dump's write point moved post-sweep); "
            "re-run the pipeline to regenerate it."
        )
    return df


def load_contest_player_fpts(archive_dir: Path) -> dict[int, float]:
    path = archive_dir / "contest_player_fpts.json"
    if not path.exists():
        raise FileNotFoundError(f"no contest_player_fpts.json in {archive_dir}")
    data = json.loads(path.read_text())
    return {int(pid): fpts for pid, fpts in data.get("player_fpts", {}).items()}


def load_real_field_points(archive_dir: Path) -> np.ndarray:
    """Parse the real field's full Points distribution from the
    contest-standings zip's per-entry table (Rank, EntryId, EntryName,
    TimeRemaining, Points, Lineup — the columns before the empty column-7
    gap that separates it from the player-ownership sidebar table GET
    /api/contest/analyze already parses). One row per real entry, so this
    is the true field-size and score distribution for that specific
    contest — no assumption about field size or payout structure needed.
    """
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    if not zips:
        raise FileNotFoundError(f"no contest-standings zip in {archive_dir}")
    with zipfile.ZipFile(zips[0]) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        content = zf.read(csv_name).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise ValueError(f"contest standings CSV in {zips[0].name} is empty")
    points_col = rows[0].index("Points")
    points = []
    for row in rows[1:]:
        if len(row) > points_col and row[points_col].strip():
            try:
                points.append(float(row[points_col]))
            except ValueError:
                continue
    if not points:
        raise ValueError(f"no parseable Points values in {zips[0].name}")
    return np.sort(np.array(points, dtype=np.float64))


def default_cash_threshold() -> float:
    """Fraction of the field a lineup must beat to cash, per the existing
    dk_classic_gpp.json-derived convention (PipelineRunner._load_cash_threshold,
    falls back to 0.74). A genre-level approximation, not this specific
    contest's exact cash line — DK doesn't export per-contest payout tables."""
    try:
        from src.api.pipeline import PipelineRunner
        return PipelineRunner._load_cash_threshold()
    except Exception:
        return 0.74


def add_real_percentile(
    lineup_df: pd.DataFrame, field_points: np.ndarray, cash_threshold: float, top_percentile: float = 0.95,
) -> pd.DataFrame:
    """Add real_percentile (fraction of the real field this candidate's
    actual_score would have beaten-or-tied, via searchsorted on the sorted
    real Points array), would_cash (real_percentile >= cash_threshold), and
    would_top_pct (real_percentile >= top_percentile — the steep, top-heavy
    part of a GPP payout curve where the actual prize money concentrates,
    not just the min-cash line). NaN for candidates with incomplete
    actual-score coverage.
    """
    df = lineup_df.copy()
    has_score = df["actual_score"].notna()
    n_field = len(field_points)
    pct = np.full(len(df), np.nan)
    pct[has_score.values] = (
        np.searchsorted(field_points, df.loc[has_score, "actual_score"].values, side="right") / n_field
    )
    df["real_percentile"] = pct
    # float (NaN/0.0/1.0), not bool — a bool column can't hold NaN for the
    # incomplete-coverage rows, and .mean() over 0.0/1.0 gives the rate
    # exactly the same way it would over True/False.
    df["would_cash"] = np.where(np.isnan(pct), np.nan, (pct >= cash_threshold).astype(float))
    df["would_top_pct"] = np.where(np.isnan(pct), np.nan, (pct >= top_percentile).astype(float))
    return df


def compute_floor_metrics(lineup_df: pd.DataFrame, ev_floor: float, n_field: int) -> dict:
    """Compare real percentile / approximate cash rate for candidates below
    vs. at-or-above ev_floor — the direct test of whether the floor is
    culling the right candidates against what actually happened."""
    complete = lineup_df.dropna(subset=["real_percentile"])
    below = complete[complete["projected_ev"] < ev_floor]
    above = complete[complete["projected_ev"] >= ev_floor]
    metrics = {
        "ev_floor": ev_floor,
        "n_field": n_field,
        "n_below_floor": len(below),
        "n_at_or_above_floor": len(above),
        "cash_rate_below_floor": below["would_cash"].mean() if len(below) else float("nan"),
        "cash_rate_at_or_above_floor": above["would_cash"].mean() if len(above) else float("nan"),
        "top_pct_rate_below_floor": below["would_top_pct"].mean() if len(below) else float("nan"),
        "top_pct_rate_at_or_above_floor": above["would_top_pct"].mean() if len(above) else float("nan"),
        "real_percentile_below_floor": below["real_percentile"].mean() if len(below) else float("nan"),
        "real_percentile_at_or_above_floor": above["real_percentile"].mean() if len(above) else float("nan"),
    }
    return metrics


def sweep_floor_table(lineup_df: pd.DataFrame, n_field: int, start: float, end: float, step: float) -> pd.DataFrame:
    """Re-run compute_floor_metrics across a grid of candidate ev_floor values
    (start..end inclusive, in `step` increments) against the same already-
    joined real-percentile data — the candidate pool dump includes every
    generated candidate regardless of the floor used at run time, so the
    full marginal floor/cash-rate/pool-size tradeoff can be traced out
    retroactively from one archived slate.
    """
    n_steps = int(round((end - start) / step)) + 1
    floors = start + step * np.arange(max(n_steps, 1))
    n_total = len(lineup_df.dropna(subset=["real_percentile"]))
    rows = []
    for fl in floors:
        fl = round(float(fl), 6)
        m = compute_floor_metrics(lineup_df, fl, n_field)
        n_above = m["n_at_or_above_floor"]
        n_below = m["n_below_floor"]
        rows.append({
            "floor": fl,
            "n_below": n_below,
            "n_above": n_above,
            "pct_above": n_above / n_total if n_total else float("nan"),
            "cash_below": m["cash_rate_below_floor"],
            "cash_above": m["cash_rate_at_or_above_floor"],
            "top_pct_below": m["top_pct_rate_below_floor"],
            "n_top_below": (int(round(n_below * m["top_pct_rate_below_floor"])) if not np.isnan(m["top_pct_rate_below_floor"]) else 0),
            "top_pct_above": m["top_pct_rate_at_or_above_floor"],
            "n_top_above": (int(round(n_above * m["top_pct_rate_at_or_above_floor"])) if not np.isnan(m["top_pct_rate_at_or_above_floor"]) else 0),
            "real_pct_below": m["real_percentile_below_floor"],
            "real_pct_above": m["real_percentile_at_or_above_floor"],
        })
    return pd.DataFrame(rows)


def run_floor_sweep(
    archive_dirs: list[Path], start: float, end: float | None, step: float,
    cash_threshold: float, top_percentile: float,
) -> None:
    for d in archive_dirs:
        try:
            pool_df = load_candidate_pool(d)
            fpts_map = load_contest_player_fpts(d)
            field_points = load_real_field_points(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue

        lineup_df = build_lineup_table(pool_df, fpts_map)
        lineup_df = add_real_percentile(lineup_df, field_points, cash_threshold, top_percentile)

        sweep_end = end
        if sweep_end is None:
            max_ev = lineup_df["projected_ev"].max()
            sweep_end = float(np.ceil(max_ev / step) * step)

        table = sweep_floor_table(lineup_df, len(field_points), start, sweep_end, step)

        print(
            f"\n=== {d.name} ===  n_field={len(field_points)}  cash_threshold={cash_threshold:.2f}  "
            f"top_percentile={top_percentile:.2f}  n_candidates={len(lineup_df)}  "
            f"sweep=${start:.2f}..${sweep_end:.2f} step=${step:.2f}"
        )
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        sweep_path = d / "candidate_pool_floor_sweep.csv"
        table.to_csv(sweep_path, index=False)
        print(f"Sweep written -> {sweep_path}")


def build_lineup_table(pool_df: pd.DataFrame, fpts_map: dict[int, float]) -> pd.DataFrame:
    """Collapse the per-player-per-lineup pool rows to one row per candidate
    lineup: projected_score (sum mean), projected_ev (mean_ev, constant per
    lineup), avg_ownership (mean of its players' ownership), actual_score
    (sum of each player's real FPTS — NaN if any of its players is missing
    from fpts_map, so partial-coverage lineups don't silently understate).
    """
    df = pool_df.copy()
    df["actual_fpts"] = df["player_id"].map(fpts_map)

    grouped = df.groupby("lineup_index")
    out = grouped.agg(
        projected_score=("mean", "sum"),
        projected_ev=("mean_ev", "first"),
        avg_ownership=("ownership", "mean"),
        selected_risks=("selected_risks", "first"),
        n_players=("player_id", "count"),
        n_missing=("actual_fpts", lambda s: s.isna().sum()),
        actual_score=("actual_fpts", "sum"),
    ).reset_index()
    out.loc[out["n_missing"] > 0, "actual_score"] = np.nan
    return out


def _decile_table(lineup_df: pd.DataFrame, by: str, value_col: str = "actual_score", n_bins: int = 10) -> pd.DataFrame:
    df = lineup_df.dropna(subset=[by, value_col]).copy()
    if len(df) < n_bins:
        return pd.DataFrame()
    df["decile"] = pd.qcut(df[by], n_bins, labels=False, duplicates="drop") + 1
    return df.groupby("decile").agg(
        n=(by, "count"),
        **{f"{by}_mean": (by, "mean")},
        **{f"{value_col}_mean": (value_col, "mean")},
    ).reset_index()


def compute_slate_metrics(lineup_df: pd.DataFrame) -> dict:
    complete = lineup_df.dropna(subset=["actual_score"])
    n_total = len(lineup_df)
    n_complete = len(complete)
    metrics = {
        "n_candidates": n_total,
        "n_complete": n_complete,
        "coverage": n_complete / n_total if n_total else float("nan"),
        "spearman_ev_actual": float("nan"),
        "spearman_score_actual": float("nan"),
        "spearman_ownership_actual": float("nan"),
        "actual_score_selected_mean": float("nan"),
        "actual_score_unselected_mean": float("nan"),
    }
    if n_complete >= 10:
        metrics["spearman_ev_actual"] = spearmanr(complete["projected_ev"], complete["actual_score"]).correlation
        metrics["spearman_score_actual"] = spearmanr(complete["projected_score"], complete["actual_score"]).correlation
        metrics["spearman_ownership_actual"] = spearmanr(complete["avg_ownership"], complete["actual_score"]).correlation

    selected_mask = complete["selected_risks"].fillna("").astype(str).str.len() > 0
    if selected_mask.any() and (~selected_mask).any():
        metrics["actual_score_selected_mean"] = complete.loc[selected_mask, "actual_score"].mean()
        metrics["actual_score_unselected_mean"] = complete.loc[~selected_mask, "actual_score"].mean()
    return metrics


def print_single_slate_report(name: str, lineup_df: pd.DataFrame, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    print(
        f"Candidates: {metrics['n_candidates']}  (complete actual-score coverage: "
        f"{metrics['n_complete']} = {metrics['coverage'] * 100:.1f}%)"
    )
    print(f"Spearman(projected_ev,    actual_score) = {metrics['spearman_ev_actual']:+.3f}")
    print(f"Spearman(projected_score, actual_score) = {metrics['spearman_score_actual']:+.3f}")
    print(f"Spearman(avg_ownership,   actual_score) = {metrics['spearman_ownership_actual']:+.3f}")
    if not np.isnan(metrics["actual_score_selected_mean"]):
        print(
            f"Mean actual_score — selected into a portfolio: "
            f"{metrics['actual_score_selected_mean']:.2f}   "
            f"not selected: {metrics['actual_score_unselected_mean']:.2f}"
        )

    print("\nProjected-EV deciles (1 = lowest EV .. 10 = highest EV):")
    ev_dec = _decile_table(lineup_df, "projected_ev")
    if not ev_dec.empty:
        print(ev_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  (not enough complete-coverage candidates for deciles)")

    print("\nOwnership deciles (1 = lowest owned .. 10 = highest owned):")
    own_dec = _decile_table(lineup_df, "avg_ownership")
    if not own_dec.empty:
        print(own_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  (not enough complete-coverage candidates for deciles)")

    if metrics.get("n_field") is None:
        print("\nEV-floor calibration: skipped (no contest-standings zip found for the real field).")
        return
    pct_label = metrics["top_percentile"] * 100
    print(
        f"\nTop {pct_label:.1f}th percentile (real field) — where the real money in a GPP "
        f"concentrates, not just the min-cash line:"
    )
    print(f"  {metrics['top_pct_rate'] * 100:.1f}% of candidates ({metrics['n_top_pct']} / {metrics['n_complete']})")

    print(
        f"\nEV-floor calibration vs. real field (n_field={metrics['n_field']:.0f}, "
        f"ev_floor=${metrics['ev_floor']:.2f}, cash_threshold={metrics['cash_threshold']:.2f}, "
        f"top_percentile={metrics['top_percentile']:.2f}):"
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
    print("\nProjected-EV deciles vs. real cash rate (1 = lowest EV .. 10 = highest EV):")
    cash_dec = _decile_table(lineup_df, "projected_ev", value_col="would_cash")
    if not cash_dec.empty:
        print(cash_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  (not enough complete-coverage candidates for deciles)")

    print(f"\nProjected-EV deciles vs. top {pct_label:.0f}th percentile rate (1 = lowest EV .. 10 = highest EV):")
    top_dec = _decile_table(lineup_df, "projected_ev", value_col="would_top_pct")
    if not top_dec.empty:
        print(top_dec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("  (not enough complete-coverage candidates for deciles)")


def print_condensed_line(name: str, metrics: dict) -> None:
    floor_str = ""
    if metrics.get("n_field") is not None:
        floor_str = (
            f"  cash_below={metrics['cash_rate_below_floor']:.3f}  "
            f"cash_above={metrics['cash_rate_at_or_above_floor']:.3f}  "
            f"top{metrics['top_percentile'] * 100:.0f}={metrics['top_pct_rate'] * 100:.1f}%"
        )
    print(
        f"{name:>12}  n={metrics['n_candidates']:>6}  cov={metrics['coverage'] * 100:5.1f}%  "
        f"r_ev={metrics['spearman_ev_actual']:+.3f}  r_score={metrics['spearman_score_actual']:+.3f}  "
        f"r_own={metrics['spearman_ownership_actual']:+.3f}{floor_str}"
    )


def _append_summary(rows: list[dict], summary_path: Path) -> None:
    combined = pd.DataFrame(rows)
    combined["run_ts"] = datetime.now().isoformat(timespec="seconds")
    if summary_path.exists():
        try:
            # dtype=str on "slate" avoids pandas inferring an all-numeric-looking
            # column (e.g. "06262026") as int64 and silently stripping the
            # leading zero on the next write.
            old = pd.read_csv(summary_path, dtype={"slate": str})
            combined = pd.concat([old, combined], ignore_index=True)
        except Exception as exc:
            print(f"Warning: could not read existing {summary_path} ({exc}) — overwriting.")
    combined.to_csv(summary_path, index=False)


def evaluate_archive_dirs(
    archive_dirs: list[Path], ev_floor: float, cash_threshold: float, top_percentile: float,
) -> None:
    rows = []
    for d in archive_dirs:
        try:
            pool_df = load_candidate_pool(d)
            fpts_map = load_contest_player_fpts(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue

        lineup_df = build_lineup_table(pool_df, fpts_map)
        metrics = compute_slate_metrics(lineup_df)

        try:
            field_points = load_real_field_points(d)
            lineup_df = add_real_percentile(lineup_df, field_points, cash_threshold, top_percentile)
            metrics.update(compute_floor_metrics(lineup_df, ev_floor, len(field_points)))
            metrics["cash_threshold"] = cash_threshold
            metrics["top_percentile"] = top_percentile
            complete = lineup_df.dropna(subset=["would_top_pct"])
            metrics["top_pct_rate"] = complete["would_top_pct"].mean() if len(complete) else float("nan")
            metrics["n_top_pct"] = int(complete["would_top_pct"].sum())
        except (FileNotFoundError, ValueError) as exc:
            print(f"{d.name}: EV-floor calibration skipped ({exc})")

        eval_path = d / "candidate_pool_eval.csv"
        lineup_df.to_csv(eval_path, index=False)

        rows.append({"slate": d.name, **metrics})

        if len(archive_dirs) == 1:
            print_single_slate_report(d.name, lineup_df, metrics)
        else:
            print_condensed_line(d.name, metrics)

    if not rows:
        print("No slates with both candidate_pool_debug.csv and contest_player_fpts.json found.")
        return

    if len(rows) > 1:
        agg_df = pd.DataFrame(rows)
        print(f"\n=== Aggregate across {len(rows)} slates ===")
        for col in ("spearman_ev_actual", "spearman_score_actual", "spearman_ownership_actual"):
            vals = agg_df[col].dropna()
            if len(vals):
                print(
                    f"{col}: mean={vals.mean():+.3f}  median={vals.median():+.3f}  "
                    f"std={vals.std():.3f}  n_slates={len(vals)}"
                )
        if "cash_rate_below_floor" in agg_df.columns:
            print(f"\nEV-floor (${ev_floor:.2f}) calibration vs. real field, pooled across slates:")
            for col, label in (
                ("cash_rate_below_floor", "cash_rate below floor"),
                ("cash_rate_at_or_above_floor", "cash_rate at/above floor"),
                ("top_pct_rate_below_floor", "top-pct rate below floor"),
                ("top_pct_rate_at_or_above_floor", "top-pct rate at/above floor"),
                ("real_percentile_below_floor", "real_percentile below floor"),
                ("real_percentile_at_or_above_floor", "real_percentile at/above floor"),
            ):
                vals = agg_df[col].dropna()
                if len(vals):
                    print(f"  {label}: mean={vals.mean():.3f}  median={vals.median():.3f}  n_slates={len(vals)}")
        if "top_pct_rate" in agg_df.columns:
            vals = agg_df["top_pct_rate"].dropna()
            if len(vals):
                tp = agg_df["top_percentile"].dropna().iloc[0] * 100 if "top_percentile" in agg_df.columns else 95
                print(
                    f"\nOverall top {tp:.0f}th percentile rate, pooled across slates: "
                    f"mean={vals.mean() * 100:.1f}%  median={vals.median() * 100:.1f}%  n_slates={len(vals)}"
                )
        summary_path = PROJECT_ROOT / "archive" / "candidate_pool_summary.csv"
        _append_summary(rows, summary_path)
        print(f"\nSummary appended -> {summary_path}")
    else:
        print(f"\nPer-lineup eval written -> {archive_dirs[0] / 'candidate_pool_eval.csv'}")


def _find_recent_pool_slates(n: int) -> list[Path]:
    """N most recent archive subdirectories that have both required files,
    sorted oldest-first (mirrors evaluate_ownership.py's --recent)."""
    archive_root = PROJECT_ROOT / "archive"
    full = sorted(
        (
            d for d in archive_root.iterdir()
            if d.is_dir()
            and (d / "candidate_pool_debug.csv").exists()
            and (d / "contest_player_fpts.json").exists()
        ),
        key=lambda d: _slate_sort_key(d.name),
    )
    return full[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-pool projections against actual DK contest results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
        help="Archive directories to evaluate (e.g. archive/06262026). Omit when using --recent.",
    )
    parser.add_argument(
        "--recent", type=int, default=0, metavar="N",
        help="Evaluate the N most recent slates that have both candidate_pool_debug.csv "
             "and contest_player_fpts.json. Mutually exclusive with positional ARCHIVE_DIR args.",
    )
    parser.add_argument(
        "--ev-floor", type=float, default=_DEFAULT_EV_FLOOR, metavar="$",
        help=f"ev_floor value to calibrate against the real field (default: {_DEFAULT_EV_FLOOR}, "
             "matching the live gpp.ev_floor config default). The candidate pool dump includes "
             "every generated candidate regardless of the floor used at run time, so any "
             "threshold can be tested retroactively.",
    )
    parser.add_argument(
        "--cash-threshold", type=float, default=None, metavar="FRACTION",
        help="Fraction of the real field a lineup must beat to count as an approximate cash "
             "(default: derived from data/payout_structures/dk_classic_gpp.json, falls back to "
             "0.74). A genre-level approximation, not this specific contest's exact cash line.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Instead of the standard report, sweep ev_floor from --ev-floor up to --sweep-end "
             "in --sweep-step increments, printing the marginal floor/cash-rate/pool-size "
             "tradeoff per slate. Writes archive/MMDDYYYY/candidate_pool_floor_sweep.csv.",
    )
    parser.add_argument(
        "--sweep-end", type=float, default=None, metavar="$",
        help="Top of the --sweep range (default: rounded up to the next --sweep-step multiple "
             "of the slate's own max projected_ev, so the sweep always covers the full pool).",
    )
    parser.add_argument(
        "--sweep-step", type=float, default=0.30, metavar="$",
        help="Increment between --sweep grid points (default: $0.30).",
    )
    parser.add_argument(
        "--top-percentile", type=float, default=0.95, metavar="FRACTION",
        help="Real-field percentile above which a lineup counts toward the 'top percentile' "
             "readout (default: 0.95) — where the real money in a top-heavy GPP payout curve "
             "concentrates, distinct from --cash-threshold's min-cash line.",
    )
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR arguments are mutually exclusive.")

    if args.recent:
        dirs = _find_recent_pool_slates(args.recent)
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

    cash_threshold = args.cash_threshold if args.cash_threshold is not None else default_cash_threshold()
    if args.sweep:
        run_floor_sweep(
            dirs, start=args.ev_floor, end=args.sweep_end, step=args.sweep_step,
            cash_threshold=cash_threshold, top_percentile=args.top_percentile,
        )
    else:
        evaluate_archive_dirs(
            dirs, ev_floor=args.ev_floor, cash_threshold=cash_threshold, top_percentile=args.top_percentile,
        )


if __name__ == "__main__":
    main()
