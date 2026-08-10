"""Leave-one-slate-out check on leverage_rank_only's top1% rate, prompted by
a direct observation: in compare_diverse_arms.py's "diversity-gesturing
arms" table, leverage_rank_only has the FEWEST teams/slate (13.0) of all 5
arms in that table -- even though the table's whole point is excluding the
true concentration offenders (proj_score/p_cash, ~7 teams/slate). Its
strong top1% (1.714, best of the 5) could be the same "concentrated on
whichever teams happened to boom on these 10 specific slates" dynamic
flagged for proj_score/p_cash (project-leverage-a1-adjudication-result),
just a milder version -- this directly tests that by dropping each slate in
turn and checking whether the pooled top1% result survives, the same LOSO
methodology already established elsewhere in this codebase for exactly this
concern (e.g. project-leverage-phase-a-results' LOSO_min).

Usage
-----
    source venv/bin/activate
    python scripts/loso_leverage_concentration_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _candidate_currencies, _prod_corr, grade_joint, load_leverage,
    load_slate, select_greedy, select_team_diverse_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]


def main() -> None:
    seed, calib = 42, False
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()
              and (ORACLE_DIR / f"{s}_leverage.npz").exists()]

    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        sel = load_leverage(s)["leverage_ratio_mean"]
        picks = select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000)
        frames.append(grade_joint(sd, picks, "leverage_rank_only"))

        picks = select_team_diverse_leverage(sd)
        frames.append(grade_joint(sd, picks, "team_diverse_leverage"))

        curs = _candidate_currencies(sd)
        corr = _prod_corr(s, seed)
        picks = select_greedy(sd, curs["p_win"], sd.currency("p_win", "A"),
                              floor_pct=30.0, admit_n=2000, evw=0.25, corr=corr)
        frames.append(grade_joint(sd, picks, "prod_faithful"))

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    print("===== per-slate top1% by arm =====")
    per_slate = combined.groupby(["arm", "slate"])["top1"].mean().unstack("slate") * 100
    print(per_slate.round(3).to_string())

    print("\n===== LOSO: pooled top1% dropping each slate in turn =====")
    rows = []
    for arm in combined["arm"].unique():
        g = combined[combined.arm == arm]
        pooled = 100 * g["top1"].mean()
        loso_vals = []
        for s in slates:
            rest = g[g["slate"] != s]
            if rest.empty:
                continue
            loso_vals.append(100 * rest["top1"].mean())
        rows.append({
            "arm": arm, "pooled_top1%": pooled,
            "LOSO_min": min(loso_vals), "LOSO_max": max(loso_vals),
            "spread": max(loso_vals) - min(loso_vals),
        })
    df = pd.DataFrame(rows).sort_values("pooled_top1%", ascending=False)
    print(df.round(3).to_string(index=False))
    print("\nLOSO_min close to pooled (small spread) = robust, not carried by any single slate.")
    print("Large spread / LOSO_min << pooled = the pooled number leans heavily on one slate.")


if __name__ == "__main__":
    main()
