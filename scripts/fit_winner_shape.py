"""Fit the sim-free winner-shape model on real DK contest entries.

For every archived slate with DKSalaries.csv + a contest-standings zip, build
one composition record per real entry (stack shape, salary, real %Drafted
ownership, bring-back — via measure_pool_ceiling's parsers) with label
"finished at/above that contest's real p99". Fit an L2 logistic (numpy IRLS,
src/optimization/winner_shape.py) on per-slate-standardized features.

Walk-forward artifact: for each replay-relevant cutoff date the stored model
is fitted only on slates strictly BEFORE that date, so offline replays of
slate D never score with a model that saw D. "latest" (all slates) is what
the live pipeline uses.

Usage
-----
    python scripts/fit_winner_shape.py                # fit + write artifact
    python scripts/fit_winner_shape.py --report       # also per-cutoff holdout AUC
    python scripts/fit_winner_shape.py --min-slates 8 # min training slates per cutoff

Output: data/processed/winner_shape_model.json
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_candidate_pool as acp  # noqa: E402
import measure_pool_ceiling as mpc  # noqa: E402
from src.optimization.winner_shape import (  # noqa: E402
    DEFAULT_MODEL_PATH, FEATURE_NAMES, fit_logistic_irls, lineup_features, standardize,
)


def _slate_records(archive_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(X_standardized, y) for one slate's real entries, or None."""
    sal_path = archive_dir / "DKSalaries.csv"
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    if not sal_path.exists() or not zips:
        return None
    try:
        field_points = acp.load_real_field_points(archive_dir)
        standings_df, ownership_df = mpc._parse_contest_zip(zips[0])
        salary_map, team_map, opp_map = mpc._load_salary_maps(sal_path)
        _names = pd.read_csv(sal_path)["Name"].str.strip()
        ambiguous = set(_names.value_counts().loc[lambda s: s > 1].index)
        own_by_name = {
            mpc._normalise(str(r.player_name)): float(r.pct_drafted)
            for r in ownership_df.itertuples(index=False)
        }
        records = mpc._real_entry_records(
            standings_df, salary_map, team_map, opp_map, own_by_name, ambiguous,
        )
    except Exception as exc:
        print(f"  {archive_dir.name}: skipped ({exc})")
        return None
    if len(records) < 500:
        return None
    p99 = float(np.quantile(field_points, 0.99))
    df = pd.DataFrame(records)
    y = (df["points"] >= p99).to_numpy(dtype=np.float64)
    X = standardize(lineup_features(df))
    return X, y


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--min-slates", type=int, default=8,
                        help="minimum training slates required per walk-forward cutoff")
    parser.add_argument("--report", action="store_true",
                        help="print per-cutoff holdout AUC (the cutoff slate itself)")
    args = parser.parse_args()

    archive_root = PROJECT_ROOT / "archive"
    slates = []
    for d in sorted(archive_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or d.name.endswith("r"):
            continue
        try:
            date = datetime.strptime(d.name[:8], "%m%d%Y")
        except ValueError:
            continue
        slates.append((date, d))
    slates.sort(key=lambda t: t[0])

    print(f"Building real-entry records for {len(slates)} candidate slates...")
    per_slate: list[tuple[datetime, str, np.ndarray, np.ndarray]] = []
    for date, d in slates:
        out = _slate_records(d)
        if out is None:
            continue
        X, y = out
        per_slate.append((date, d.name, X, y))
        print(f"  {d.name}: {len(y)} entries, {int(y.sum())} top-1%")

    if len(per_slate) < args.min_slates + 1:
        raise SystemExit(f"Only {len(per_slate)} usable slates — need at least {args.min_slates + 1}.")

    models: dict[str, dict] = {}
    report_rows = []
    for i, (date, name, X_hold, y_hold) in enumerate(per_slate):
        if i < args.min_slates:
            continue
        X_tr = np.vstack([X for _, _, X, _ in per_slate[:i]])
        y_tr = np.concatenate([y for _, _, _, y in per_slate[:i]])
        coef, intercept = fit_logistic_irls(X_tr, y_tr, l2=args.l2)
        models[name[:8]] = {
            "coef": coef.tolist(), "intercept": intercept,
            "n_slates": i, "n_entries": int(len(y_tr)),
        }
        if args.report:
            auc = _auc(X_hold @ coef + intercept, y_hold)
            report_rows.append({"cutoff": name, "n_train_slates": i, "holdout_auc": auc})

    # "latest": fitted on everything, for live use.
    X_all = np.vstack([X for _, _, X, _ in per_slate])
    y_all = np.concatenate([y for _, _, _, y in per_slate])
    coef, intercept = fit_logistic_irls(X_all, y_all, l2=args.l2)
    latest_key = per_slate[-1][1][:8]
    # A model fitted on ALL slates must not be keyed as a walk-forward cutoff;
    # store it under its own field.
    artifact = {
        "feature_names": FEATURE_NAMES,
        "fitted_ts": datetime.now().isoformat(timespec="seconds"),
        "l2": args.l2,
        "models": models,
        "latest": max(models) if models else latest_key,
        "full_fit": {
            "coef": coef.tolist(), "intercept": intercept,
            "n_slates": len(per_slate), "n_entries": int(len(y_all)),
        },
    }
    out_path = PROJECT_ROOT / DEFAULT_MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1))
    print(f"\nWalk-forward models: {len(models)} cutoffs -> {out_path}")
    print("Full-fit coefficients (z-scored features):")
    for fn, c in zip(FEATURE_NAMES, coef):
        print(f"  {fn:>10}: {c:+.4f}")
    print(f"  intercept: {intercept:+.4f}")

    if report_rows:
        rep = pd.DataFrame(report_rows)
        print("\nWalk-forward holdout AUC (cutoff slate scored by its own pre-cutoff model):")
        print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"mean AUC = {rep['holdout_auc'].mean():.4f}")


if __name__ == "__main__":
    main()
