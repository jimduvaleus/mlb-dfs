"""
Evaluate the candidate pool (pre-portfolio-selection lineups) against actual
DraftKings contest results.

Joins archive/MMDDYYYY/candidate_pool_debug.csv (per-candidate-lineup
projected score/EV/ownership, dumped by the pipeline when dump_candidate_pool
is enabled) against archive/MMDDYYYY/contest_player_fpts.json (the resolved
player_id -> actual FPTS map written by GET /api/contest/analyze) to answer:
does our projected EV / projected score / projected ownership actually
predict real contest performance?

Usage
-----
    python scripts/analyze_candidate_pool.py archive/06262026
    python scripts/analyze_candidate_pool.py archive/06262026 archive/06272026
    python scripts/analyze_candidate_pool.py --recent 5

Output
------
  - Per-slate report printed to stdout: Spearman correlations of
    projected_ev / projected_score / avg_ownership against actual_score, plus
    a selected-vs-not-selected breakdown. Full EV/ownership decile tables are
    only printed for a single slate; multiple slates print a condensed
    per-slate line plus an aggregate table across slates.
  - archive/MMDDYYYY/candidate_pool_eval.csv  (per-lineup joined table).
  - archive/candidate_pool_summary.csv        (one row per slate, appended).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def _decile_table(lineup_df: pd.DataFrame, by: str, n_bins: int = 10) -> pd.DataFrame:
    df = lineup_df.dropna(subset=[by, "actual_score"]).copy()
    if len(df) < n_bins:
        return pd.DataFrame()
    df["decile"] = pd.qcut(df[by], n_bins, labels=False, duplicates="drop") + 1
    return df.groupby("decile").agg(
        n=(by, "count"),
        **{f"{by}_mean": (by, "mean")},
        actual_score_mean=("actual_score", "mean"),
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


def print_condensed_line(name: str, metrics: dict) -> None:
    print(
        f"{name:>12}  n={metrics['n_candidates']:>6}  cov={metrics['coverage'] * 100:5.1f}%  "
        f"r_ev={metrics['spearman_ev_actual']:+.3f}  r_score={metrics['spearman_score_actual']:+.3f}  "
        f"r_own={metrics['spearman_ownership_actual']:+.3f}"
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


def evaluate_archive_dirs(archive_dirs: list[Path]) -> None:
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

    evaluate_archive_dirs(dirs)


if __name__ == "__main__":
    main()
