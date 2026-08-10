"""Evaluate select_team_diverse_leverage's pitcher_quality_weight nudge
(needlunchmoney's validated pitcher-selection edge, applied LIGHTLY on top
of the already-real-pro-matching team-diverse structure -- see memory
project-leverage-session-handoff's "Real-pro team-concentration check")
across a range of light weights, checking BOTH profitability and that
team structure is preserved (this only changes within-bucket ranking, not
team allocation, so structure should be unaffected).

Usage
-----
    source venv/bin/activate
    python scripts/eval_pitcher_quality_nudge.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _primary_teams, grade_joint, load_slate, print_report,
    report, select_team_diverse_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def structure(picks_by_slate: dict) -> dict:
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
        "teams_per_slate": float(np.mean(per_slate_n)),
        "top_team_share_pct": 100 * max(pooled.values()) / total if total else float("nan"),
    }


def main() -> None:
    seed, calib = 42, False
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_leverage.npz").exists()]

    frames = []
    struct_rows = []
    for w in WEIGHTS:
        name = f"pq_w{w:g}"
        picks_by_slate = {}
        for s in slates:
            sd = load_slate(s, seed, calib)
            picks = select_team_diverse_leverage(sd, pitcher_quality_weight=w)
            picks_by_slate[s] = picks
            frames.append(grade_joint(sd, picks, name))
        st = structure(picks_by_slate)
        struct_rows.append({"arm": name, **st})

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = PROJECT_ROOT / "outputs" / "pitcher_quality_nudge_eval.csv"
    out.parent.mkdir(exist_ok=True)
    combined.to_csv(out, index=False)

    res = report(combined, baseline="pq_w0")
    struct_df = pd.DataFrame(struct_rows).set_index("arm")
    res = res.set_index("arm").join(struct_df).reset_index()
    print_report(res, "PITCHER-QUALITY NUDGE sweep (baseline = weight 0.0, i.e. original team_diverse_leverage)")
    print("\n-- structure (should stay ~constant across weights -- only within-bucket ranking changes) --")
    print(res[["arm", "teams_per_slate", "top_team_share_pct"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
