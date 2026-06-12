"""
Fit the production ownership calibrator artifact.

Builds the isotonic (PAVA) predicted → actual ownership curves per group
(pitchers / pooled batters) from archived slates with DraftKings contest
standings, and writes data/processed/ownership_calibrator.json.  The pipeline
loads this artifact via load_ownership_calibrator() and applies it after
compute_heuristic_ownership() — see apply_ownership_calibration().

Predictions are recomputed fresh from each archive with the CURRENT model
constants (no dependence on previously written ownership_eval.csv files), and
the artifact records the constants hash it was fitted under.  If the model
constants are later tweaked, load_ownership_calibrator() detects the stale
hash and falls back to uncalibrated ownership until this script is re-run.

Validated 2026-06-11 (W_resid, 21 slates, walk-forward): +0.066 log-RMSE
vs uncalibrated (p=0.0001), rank order preserved, raw RMSE unharmed.

Usage
-----
    # All archived slates with contest standings (recommended)
    python scripts/fit_ownership_calibrator.py

    # N most recent full slates / specific slates
    python scripts/fit_ownership_calibrator.py --recent 15
    python scripts/fit_ownership_calibrator.py archive/05092026 archive/05102026

Re-run whenever new contest standings land in archive/ or after any
ownership.py constants change.
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

from evaluate_ownership import (  # noqa: E402
    _build_player_pool,
    _find_recent_full_slates,
    _fit_isotonic_groups,
    _git_commit_hash,
    _match_ownership,
    _parse_contest_zip,
)
from src.optimization.ownership import (  # noqa: E402
    OWNERSHIP_CALIBRATOR_PATH,
    compute_heuristic_ownership,
    ownership_constants_hash,
)

MIN_SLATES = 5


def _collect_training_points(archive_dirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """
    For each archive slate: build the player pool, compute E_production
    ownership fresh with current constants, and pair it with actual
    %Drafted from the contest standings.

    Returns (all_points_df with columns position/pred/actual, slate_labels).
    """
    frames: list[pd.DataFrame] = []
    labels: list[str] = []
    for d in archive_dirs:
        label = d.name
        zips = sorted(d.glob("contest-standings-*.zip"))
        if not zips:
            print(f"  [{label}] No contest standings zip — skipping.")
            continue
        print(f"  [{label}] Loading ...", end=" ", flush=True)
        try:
            _, ownership_df = _parse_contest_zip(zips[0])
            pool_df = _build_player_pool(d)
            matched = _match_ownership(ownership_df, pool_df)
            if len(matched) < 10:
                print(f"only {len(matched)} matched — skipping.")
                continue
            team_totals = None
            if pool_df["implied_total"].notna().any():
                team_totals = (
                    pool_df[["team", "implied_total"]]
                    .dropna(subset=["implied_total"])
                    .drop_duplicates("team")
                    .set_index("team")["implied_total"]
                    .to_dict()
                )
            ownership = compute_heuristic_ownership(pool_df, team_totals)
            pid_map = dict(zip(pool_df["player_id"], ownership))
            frames.append(pd.DataFrame({
                "position": matched["position"].values,
                "pred": matched["player_id"].map(pid_map).values.astype(float),
                "actual": matched["pct_drafted"].values.astype(float),
            }).dropna())
            labels.append(label)
            print(f"{len(matched)} players matched.")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not frames:
        return pd.DataFrame(columns=["position", "pred", "actual"]), []
    return pd.concat(frames, ignore_index=True), labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the production ownership calibrator artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
                        help="Specific archive directories (default: all full slates).")
    parser.add_argument("--recent", type=int, default=0, metavar="N",
                        help="Use the N most recent full slates.")
    parser.add_argument("--out", type=Path, default=OWNERSHIP_CALIBRATOR_PATH,
                        help=f"Output artifact path (default {OWNERSHIP_CALIBRATOR_PATH}).")
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR are mutually exclusive.")
    if args.recent:
        dirs = _find_recent_full_slates(args.recent)
    elif args.archive_dirs:
        dirs = [Path(d) for d in args.archive_dirs if Path(d).is_dir()]
    else:
        dirs = _find_recent_full_slates(999)
    if not dirs:
        print("No archive directories found.")
        sys.exit(1)

    print(f"Collecting training points from {len(dirs)} slate(s) "
          f"(predictions recomputed with current constants)...")
    points, labels = _collect_training_points(dirs)
    if len(labels) < MIN_SLATES:
        print(f"\nOnly {len(labels)} usable slate(s) — need ≥{MIN_SLATES}. No artifact written.")
        sys.exit(1)

    calibrator = _fit_isotonic_groups(points)
    if not calibrator:
        print("\nToo few training points in every group. No artifact written.")
        sys.exit(1)

    groups_json = {}
    for key in ("P", "bat"):
        if key not in calibrator:
            print(f"  Group {key!r}: too few points — identity (not stored).")
            continue
        x, y = calibrator[key]
        # Rounding to 6 dp can collide closely-spaced knots; re-collapse to
        # keep x strictly increasing and enforce y non-decreasing, matching
        # the loader's validation.
        knots = (
            pd.DataFrame({"x": np.round(x, 6), "y": np.round(y, 6)})
            .groupby("x", as_index=False)["y"].mean()
        )
        xs = knots["x"].values
        ys = np.maximum.accumulate(knots["y"].values)
        groups_json[key] = {"x": [float(v) for v in xs], "y": [round(float(v), 6) for v in ys]}
        n_pts = int(((points["position"] == "P") if key == "P" else (points["position"] != "P")).sum())
        print(f"  Group {key!r}: {n_pts} training points → {len(xs)} knots, "
              f"pred range [{xs.min():.4f}, {xs.max():.4f}] → actual [{ys.min():.4f}, {ys.max():.4f}]")

    artifact = {
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "constants_hash": ownership_constants_hash(),
        "n_slates": len(labels),
        "slates": labels,
        "groups": groups_json,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nCalibrator fitted on {len(labels)} slate(s), constants_hash={artifact['constants_hash']}")
    print(f"Artifact written → {out_path}")


if __name__ == "__main__":
    main()
