"""Leave-one-slate-out calibration of LeveragePortfolioSelector's
coverage_weight against the 10-slate archive -- target_anchor_c held at
1.0 (no field-size gatekeeping) throughout, to isolate the coverage
mechanism itself (regret-minimization on positive-leverage players not yet
in the portfolio) from the separate, messier field-size-gatekeeping
question (see memory project-leverage-anchor-calibration-todo for why that
one isn't ready to calibrate jointly yet).

Mirrors this codebase's established LOSO calibration pattern
(scripts/select_needlunchmoney_pool.py::calibrate_pitcher_coverage_alpha /
load_needlunchmoney_actuals_loso): for each held-out slate, pick the
coverage_weight that performs best (pooled entry-weighted top1% rate --
the metric this exact investigation's own tests/backtest_lab.py repeatedly
documents as having real statistical power at this sample size, unlike
$/entry, which is noise-dominated and already caught one lucky-slate
artifact in this same investigation) on the OTHER 9 slates, then evaluates
ONLY on the held-out slate. Pooling the 10 never-seen-during-calibration
held-out results is the honest estimate of how well "calibrate, then
apply" performs -- avoiding the multiple-comparisons/cherry-picking trap
the earlier in-sample sweep (scripts/sweep_leverage_selector.py) fell into.

Usage
-----
    source venv/bin/activate
    python scripts/calibrate_leverage_coverage_weight.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, grade_joint, load_leverage, load_slate, print_report,
    report, select_greedy, select_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]
GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.01, 0.1))


def pooled_top1(df: pd.DataFrame) -> float:
    if df.empty:
        return float("-inf")
    return 100 * df.top1.mean()


def eval_weight(slates: list, w: float, seed: int = 42, calib: bool = False) -> pd.DataFrame:
    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        picks = select_leverage(sd, coverage_weight=w, target_anchor_c=1.0)
        frames.append(grade_joint(sd, picks, f"cov{w:g}"))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def main() -> None:
    seed, calib = 42, False
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_leverage.npz").exists()]

    held_out_frames = []
    chosen: dict[str, float] = {}
    for held_out in slates:
        train = [s for s in slates if s != held_out]
        best_w, best_score = None, float("-inf")
        for w in GRID:
            score = pooled_top1(eval_weight(train, w, seed, calib))
            if score > best_score:
                best_score, best_w = score, w
        chosen[held_out] = best_w
        print(f"  held-out {held_out}: chosen coverage_weight={best_w:g} "
              f"(train top1%={best_score:.3f} over {len(train)} slates)", flush=True)

        sd = load_slate(held_out, seed, calib)
        picks = select_leverage(sd, coverage_weight=best_w, target_anchor_c=1.0)
        held_out_frames.append(grade_joint(sd, picks, "leverage_selector_loso"))

    df_loso = pd.concat([f for f in held_out_frames if not f.empty], ignore_index=True)

    frames_rank, frames_default = [], []
    for s in slates:
        sd = load_slate(s, seed, calib)
        sel = load_leverage(s)["leverage_ratio_mean"]
        picks = select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000)
        frames_rank.append(grade_joint(sd, picks, "leverage_rank_only"))
        picks = select_leverage(sd)  # untuned default: coverage_weight=0.5, anchor=1.0
        frames_default.append(grade_joint(sd, picks, "leverage_selector_default"))

    combined = pd.concat([df_loso] + frames_rank + frames_default, ignore_index=True)
    out = PROJECT_ROOT / "outputs" / "leverage_coverage_weight_loso.csv"
    out.parent.mkdir(exist_ok=True)
    combined.to_csv(out, index=False)

    res = report(combined, baseline="leverage_rank_only")
    print_report(res, "LOSO-CALIBRATED coverage_weight vs rank-only vs untuned-default selector")

    print("\nchosen coverage_weight per held-out slate:")
    for s, w in chosen.items():
        print(f"  {s}: {w:g}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
