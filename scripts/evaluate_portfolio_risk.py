"""
B3 Option B — evaluate the 5-tier risk sweep (Base/Max EVw) against actual
DraftKings contest results.

Loads archive/MMDDYYYY/portfolio_sweep_{platform}.json (the 5 risk-tier
portfolios, archived by GET /api/contest/analyze alongside the candidate
pool dump) and scores every lineup in every tier against the real field's
actual Points distribution (the same real-percentile / approximate-cash /
top-percentile machinery built for the candidate pool in
analyze_candidate_pool.py, reused here rather than reimplemented).

The question this answers: does a lower EVw (risk 1, diversity-heavy) or a
higher EVw (risk 5, EV-heavy) tier actually perform better in the REAL
contest — both on average, and on a "did at least one of my entries hit a
big outcome" basis, since a multi-entry GPP portfolio's real payoff often
comes from its best lineup, not its average one.

Usage
-----
    python scripts/evaluate_portfolio_risk.py archive/06262026
    python scripts/evaluate_portfolio_risk.py archive/06262026 archive/06272026
    python scripts/evaluate_portfolio_risk.py --recent 5

Output
------
  - Per-slate table printed to stdout: one row per risk tier (1-5) with
    n_lineups, mean actual_score / real_percentile, cash_rate, top_pct_rate,
    best_real_percentile, and any_top_pct (did at least one lineup in that
    tier's portfolio hit the top percentile).
  - An aggregate table across slates, per risk tier — the headline output.
  - archive/MMDDYYYY/portfolio_risk_eval.csv  (per-lineup-per-risk table).
  - archive/portfolio_risk_summary.csv        (one row per slate per risk).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_candidate_pool as acp  # noqa: E402 — reuses real-field loaders, no duplicated logic


def load_portfolio_sweep(archive_dir: Path, platform: str = "draftkings") -> dict[float, list[dict]]:
    path = archive_dir / f"portfolio_sweep_{platform}.json"
    if not path.exists():
        raise FileNotFoundError(f"no portfolio_sweep_{platform}.json in {archive_dir}")
    data = json.loads(path.read_text())
    return {float(tier["risk"]): tier["lineups"] for tier in data.get("sweep", [])}


def evaluate_risk_tier(
    lineups: list[dict], fpts_map: dict[int, float], field_points: np.ndarray,
    cash_threshold: float, top_percentile: float,
) -> dict:
    """Score every lineup in one risk tier's portfolio against the real
    field, then summarize both average and best-of-portfolio outcomes."""
    n_field = len(field_points)
    actual_scores = []
    real_pcts = []
    for lu in lineups:
        pids = [p["player_id"] for p in lu["players"]]
        fpts = [fpts_map.get(pid) for pid in pids]
        if any(f is None for f in fpts):
            actual_scores.append(np.nan)
            real_pcts.append(np.nan)
            continue
        score = sum(fpts)
        pct = np.searchsorted(field_points, score, side="right") / n_field
        actual_scores.append(score)
        real_pcts.append(pct)

    actual_scores = np.array(actual_scores, dtype=np.float64)
    real_pcts = np.array(real_pcts, dtype=np.float64)
    complete = ~np.isnan(real_pcts)
    n_total = len(lineups)
    n_complete = int(complete.sum())

    if n_complete == 0:
        return {
            "n_lineups": n_total, "n_complete": 0,
            "mean_actual_score": float("nan"), "mean_real_percentile": float("nan"),
            "cash_rate": float("nan"), "top_pct_rate": float("nan"),
            "best_real_percentile": float("nan"), "any_top_pct": float("nan"),
        }

    pct_c = real_pcts[complete]
    would_cash = pct_c >= cash_threshold
    would_top = pct_c >= top_percentile
    return {
        "n_lineups": n_total,
        "n_complete": n_complete,
        "mean_actual_score": float(actual_scores[complete].mean()),
        "mean_real_percentile": float(pct_c.mean()),
        "cash_rate": float(would_cash.mean()),
        "top_pct_rate": float(would_top.mean()),
        "best_real_percentile": float(pct_c.max()),
        "any_top_pct": float(would_top.any()),
    }


def evaluate_slate(
    d: Path, platform: str, cash_threshold: float, top_percentile: float,
) -> pd.DataFrame | None:
    try:
        sweep = load_portfolio_sweep(d, platform)
        fpts_map = acp.load_contest_player_fpts(d)
        field_points = acp.load_real_field_points(d)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping {d.name}: {exc}")
        return None

    rows = []
    eval_rows = []
    for risk, lineups in sorted(sweep.items()):
        metrics = evaluate_risk_tier(lineups, fpts_map, field_points, cash_threshold, top_percentile)
        rows.append({"risk": risk, **metrics})
        for lu in lineups:
            pids = [p["player_id"] for p in lu["players"]]
            fpts = [fpts_map.get(pid) for pid in pids]
            actual_score = float("nan") if any(f is None for f in fpts) else sum(fpts)
            pct = (
                float("nan") if np.isnan(actual_score)
                else np.searchsorted(field_points, actual_score, side="right") / len(field_points)
            )
            eval_rows.append({
                "risk": risk, "lineup_index": lu["lineup_index"], "mean_ev": lu.get("mean_ev"),
                "actual_score": actual_score, "real_percentile": pct,
            })

    pd.DataFrame(eval_rows).to_csv(d / "portfolio_risk_eval.csv", index=False)
    table = pd.DataFrame(rows)
    print(f"\n=== {d.name} ===  n_field={len(field_points)}  cash_threshold={cash_threshold:.2f}  top_percentile={top_percentile:.2f}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    table.insert(0, "slate", d.name)
    return table


def _append_summary(rows_df: pd.DataFrame, summary_path: Path) -> None:
    combined = rows_df.copy()
    combined["run_ts"] = datetime.now().isoformat(timespec="seconds")
    if summary_path.exists():
        try:
            old = pd.read_csv(summary_path, dtype={"slate": str})
            combined = pd.concat([old, combined], ignore_index=True)
        except Exception as exc:
            print(f"Warning: could not read existing {summary_path} ({exc}) — overwriting.")
    combined.to_csv(summary_path, index=False)


def evaluate_archive_dirs(
    archive_dirs: list[Path], platform: str, cash_threshold: float, top_percentile: float,
) -> None:
    all_tables = []
    for d in archive_dirs:
        table = evaluate_slate(d, platform, cash_threshold, top_percentile)
        if table is not None:
            all_tables.append(table)

    if not all_tables:
        print("No slates with both portfolio_sweep and contest_player_fpts.json found.")
        return

    combined = pd.concat(all_tables, ignore_index=True)

    if len(all_tables) > 1:
        print(f"\n=== Aggregate across {len(all_tables)} slates, by risk tier ===")
        agg = combined.groupby("risk").agg(
            n_slates=("slate", "count"),
            mean_actual_score=("mean_actual_score", "mean"),
            mean_real_percentile=("mean_real_percentile", "mean"),
            cash_rate=("cash_rate", "mean"),
            top_pct_rate=("top_pct_rate", "mean"),
            best_real_percentile=("best_real_percentile", "mean"),
            any_top_pct_rate=("any_top_pct", "mean"),
        ).reset_index()
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(
            "\n(any_top_pct_rate = fraction of slates where at least one lineup in that "
            "risk tier's portfolio hit the top percentile — the 'hope to hit a winner' metric.)"
        )

    summary_path = PROJECT_ROOT / "archive" / "portfolio_risk_summary.csv"
    _append_summary(combined, summary_path)
    print(f"\nSummary appended -> {summary_path}")


def _find_recent_risk_slates(n: int, platform: str) -> list[Path]:
    archive_root = PROJECT_ROOT / "archive"
    full = sorted(
        (
            d for d in archive_root.iterdir()
            if d.is_dir()
            and (d / f"portfolio_sweep_{platform}.json").exists()
            and (d / "contest_player_fpts.json").exists()
        ),
        key=lambda d: acp._slate_sort_key(d.name),
    )
    return full[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 5-tier risk sweep against actual DK contest results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
        help="Archive directories to evaluate (e.g. archive/06262026). Omit when using --recent.",
    )
    parser.add_argument(
        "--recent", type=int, default=0, metavar="N",
        help="Evaluate the N most recent slates that have both portfolio_sweep_{platform}.json "
             "and contest_player_fpts.json. Mutually exclusive with positional ARCHIVE_DIR args.",
    )
    parser.add_argument("--platform", default="draftkings", help="Platform suffix for portfolio_sweep_{platform}.json (default: draftkings).")
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

    if args.recent:
        dirs = _find_recent_risk_slates(args.recent, args.platform)
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

    cash_threshold = args.cash_threshold if args.cash_threshold is not None else acp.default_cash_threshold()
    evaluate_archive_dirs(dirs, platform=args.platform, cash_threshold=cash_threshold, top_percentile=args.top_percentile)


if __name__ == "__main__":
    main()
