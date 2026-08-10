"""Evaluate select_team_diverse_leverage (tests/backtest_lab.py) across the
10-slate archive: BOTH its team-concentration structure (does it actually
achieve real-pro-like spread, unlike proj_score/p_cash's ~7-team pileup)
AND its profitability (grade_joint dollars/rates vs the other arms already
tested), in one pass -- structure without profitability, or profitability
without structure, are both half the answer.

Usage
-----
    source venv/bin/activate
    python scripts/eval_team_diverse_leverage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _candidate_currencies, _primary_teams, grade_joint,
    load_leverage, load_slate, print_report, report, select_greedy,
    select_team_diverse_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]


def team_concentration(picks_by_slate: dict) -> dict:
    pooled: dict[str, int] = {}
    per_slate_n = []
    for s, picks in picks_by_slate.items():
        with np.load(ORACLE_DIR / f"{s}_real.npz", allow_pickle=False) as z:
            pids_arr = z["player_ids"]
        primary = _primary_teams(s, pids_arr)
        slate_teams: dict[str, int] = {}
        for idx_list in picks.values():
            for i in idx_list:
                t = primary[i]
                slate_teams[t] = slate_teams.get(t, 0) + 1
                pooled[t] = pooled.get(t, 0) + 1
        per_slate_n.append(len(slate_teams))
    total = sum(pooled.values())
    return {
        "pooled_distinct_teams": len(pooled),
        "mean_teams_per_slate": float(np.mean(per_slate_n)),
        "top_team_share_pct": 100 * max(pooled.values()) / total if total else float("nan"),
    }


def main() -> None:
    seed, calib = 42, False
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_leverage.npz").exists()]

    picks_by_slate = {}
    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        picks = select_team_diverse_leverage(sd)
        picks_by_slate[s] = picks
        frames.append(grade_joint(sd, picks, "team_diverse_leverage"))

        sel = load_leverage(s)["leverage_ratio_mean"]
        picks_rank = select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000)
        frames.append(grade_joint(sd, picks_rank, "leverage_rank_only"))

        proj = _candidate_currencies(sd)["proj_score"]
        picks_proj = select_greedy(sd, proj, proj, floor_pct=30.0, admit_n=2000)
        frames.append(grade_joint(sd, picks_proj, "proj_score"))

    conc = team_concentration(picks_by_slate)
    print("\n===== team_diverse_leverage STRUCTURE =====")
    print(pd.Series(conc).to_string())
    print("reference: real pros ~19-21 teams/slate, ~7% top-team share; "
          "proj_score/p_cash measured at 6.8-7.0 teams/slate, ~30.6% top-team share")

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = PROJECT_ROOT / "outputs" / "team_diverse_leverage_eval.csv"
    out.parent.mkdir(exist_ok=True)
    combined.to_csv(out, index=False)
    res = report(combined, baseline="leverage_rank_only")
    print_report(res, "team_diverse_leverage vs leverage_rank_only vs proj_score")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
