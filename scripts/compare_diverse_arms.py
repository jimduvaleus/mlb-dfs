"""Rate-ladder + team-concentration comparison across arms, restricted to
ones that at least gesture at diversity (excludes proj_score/p_cash-style
pure EV rank, which "wins" the realized-world comparison mainly by piling
~30% of every entry onto one team -- see memory
project-leverage-a1-adjudication-result's concentration finding).

Usage
-----
    source venv/bin/activate
    python scripts/compare_diverse_arms.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _candidate_currencies, _prod_corr, _primary_teams,
    _selfcheck_prod_faithful, grade_joint, load_leverage, load_slate,
    select_greedy, select_team_diverse_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]


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
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()
              and (ORACLE_DIR / f"{s}_leverage.npz").exists()]

    print("self-check: prod_faithful reproduces production's allocate_contests")
    _selfcheck_prod_faithful()

    arm_picks: dict[str, dict[str, dict]] = {
        "prod_faithful": {}, "coverage_light": {},
        "leverage_rank_only": {}, "team_diverse_leverage": {},
        "random@floor30": {},
    }
    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        curs = _candidate_currencies(sd)
        corr = _prod_corr(s, seed)

        picks = select_greedy(sd, curs["p_win"], sd.currency("p_win", "A"),
                              floor_pct=30.0, admit_n=2000, evw=0.25, corr=corr)
        arm_picks["prod_faithful"][s] = picks
        frames.append(grade_joint(sd, picks, "prod_faithful"))

        from tests.backtest_lab import _composition_overlap_fn
        comp_fn = _composition_overlap_fn(s)
        sel = curs["proj_score"]
        picks = select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000,
                              evw=0.25, corr_fn=comp_fn)
        arm_picks["coverage_light"][s] = picks
        frames.append(grade_joint(sd, picks, "coverage_light"))

        sel = load_leverage(s)["leverage_ratio_mean"]
        picks = select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000)
        arm_picks["leverage_rank_only"][s] = picks
        frames.append(grade_joint(sd, picks, "leverage_rank_only"))

        picks = select_team_diverse_leverage(sd)
        arm_picks["team_diverse_leverage"][s] = picks
        frames.append(grade_joint(sd, picks, "team_diverse_leverage"))

        import zlib
        rng = np.random.default_rng(zlib.crc32(f"{s}|{seed}|random".encode()) & 0xFFFFFFFF)
        picks = select_greedy(sd, curs["p_win"], sd.currency("p_win", "A"),
                              floor_pct=30.0, admit_n=0, rng=rng)
        arm_picks["random@floor30"][s] = picks
        frames.append(grade_joint(sd, picks, "random@floor30"))

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    rows = []
    for arm, picks_by_slate in arm_picks.items():
        g = combined[combined.arm == arm]
        st = structure(picks_by_slate)
        rows.append({
            "arm": arm, "cash%": 100 * g.cash.mean(), "top1%": 100 * g.top1.mean(),
            "top01%": 100 * g.top01.mean(), "$/entry": (g.won.sum() - g.fee.sum()) / len(g),
            **st,
        })
    df = pd.DataFrame(rows).sort_values("top1%", ascending=False)
    print("\n===== DIVERSITY-GESTURING ARMS: rate ladder + structure =====")
    print(df.round(3).to_string(index=False))
    print("\n(excluded: proj_score, p_cash, p_cash@assign -- pure EV rank, no "
          "diversity mechanism, ~7 teams/slate / ~30% top-team concentration)")


if __name__ == "__main__":
    main()
