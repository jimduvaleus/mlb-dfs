"""Build real (no-duplicate-lineup) portfolios for named backtest_lab arms
across the 10-slate archive and measure their team-concentration structure
-- primary-stack team spread, top-team exposure share -- against the
already-documented real-pro reference (needlunchmoney/ShaidyAdvice:
~19-21 distinct primary teams, near-uniform allocation, see memory
project-rival-portfolio-shaidyadvice). Also dumps human-readable lineups
for a chosen slate/arm so a currency's actual picks can be inspected
directly, not just its aggregate stats.

Usage
-----
    source venv/bin/activate
    python scripts/inspect_arm_portfolios.py                       # concentration summary, all arms, all slates
    python scripts/inspect_arm_portfolios.py --dump proj_score 07222026   # print actual lineups
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import ORACLE_DIR, load_leverage, load_slate, select_greedy  # noqa: E402

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]
ARMS = ("proj_score", "p_cash", "leverage_rank_only")


def team_and_position_maps(slate: str) -> tuple[dict, dict]:
    sal = pd.read_csv(PROJECT_ROOT / "archive" / slate / "DKSalaries.csv")
    team_map = dict(zip(sal["ID"].astype(int), sal["TeamAbbrev"].astype(str)))
    # GOTCHA (memory project-rival-portfolio-shaidyadvice): DKSalaries
    # Position is SP/RP, never plain "P" -- must substring-match.
    is_pitcher = sal["Position"].astype(str).str.contains("P")
    pos_map = dict(zip(sal["ID"].astype(int), is_pitcher))
    return team_map, pos_map


def primary_team(player_ids: np.ndarray, team_map: dict, pos_map: dict) -> str:
    hitters = [pid for pid in player_ids if not pos_map.get(int(pid), False)]
    teams = pd.Series([team_map.get(int(pid), "?") for pid in hitters])
    return teams.value_counts().idxmax()


def arm_currency(sd, arm: str):
    if arm == "leverage_rank_only":
        sel = load_leverage(sd.slate)["leverage_ratio_mean"]
        return sel, sel
    if arm == "proj_score":
        from tests.backtest_lab import _candidate_currencies
        sel = _candidate_currencies(sd)["proj_score"]
        return sel, sel
    if arm == "p_cash":
        return sd.currency("p_cash", "B"), sd.currency("p_cash", "A")
    raise ValueError(arm)


def concentration_summary(seed: int = 42, calib: bool = False) -> None:
    rows = []
    for arm in ARMS:
        pooled_teams: dict[str, int] = {}
        per_slate_n_teams = []
        for s in SLATES:
            if not (ORACLE_DIR / f"{s}_real.npz").exists():
                continue
            sd = load_slate(s, seed, calib)
            sel, cull = arm_currency(sd, arm)
            picks = select_greedy(sd, sel, cull, floor_pct=30.0, admit_n=2000)
            team_map, pos_map = team_and_position_maps(s)
            with np.load(ORACLE_DIR / f"{s}_real.npz", allow_pickle=False) as z:
                pids_arr = z["player_ids"]
            slate_teams: dict[str, int] = {}
            for idx_list in picks.values():
                for i in idx_list:
                    t = primary_team(pids_arr[i], team_map, pos_map)
                    slate_teams[t] = slate_teams.get(t, 0) + 1
                    pooled_teams[t] = pooled_teams.get(t, 0) + 1
            per_slate_n_teams.append(len(slate_teams))
        total = sum(pooled_teams.values())
        top_share = 100 * max(pooled_teams.values()) / total if total else float("nan")
        rows.append({
            "arm": arm, "pooled_distinct_teams": len(pooled_teams),
            "mean_teams_per_slate": np.mean(per_slate_n_teams),
            "top_team_share_pct": top_share, "total_entries": total,
        })
    df = pd.DataFrame(rows)
    print("\n===== TEAM-CONCENTRATION STRUCTURE (pooled over 10 slates) =====")
    print(df.to_string(index=False))
    print("\nreference (memory project-rival-portfolio-shaidyadvice, per-slate, "
          "150-entry real portfolios): needlunchmoney/ShaidyAdvice ~19-21 distinct "
          "primary teams per slate, near-uniform (~7 entries/team, min 1 max 15-22)")


def dump_lineups(arm: str, slate: str, seed: int = 42, calib: bool = False, limit: int = 20) -> None:
    sd = load_slate(slate, seed, calib)
    sel, cull = arm_currency(sd, arm)
    picks = select_greedy(sd, sel, cull, floor_pct=30.0, admit_n=2000)
    team_map, pos_map = team_and_position_maps(slate)
    sal = pd.read_csv(PROJECT_ROOT / "archive" / slate / "DKSalaries.csv")
    name_map = dict(zip(sal["ID"].astype(int), sal["Name"].astype(str)))
    with np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False) as z:
        pids_arr = z["player_ids"]

    n_printed = 0
    for cid, idx_list in picks.items():
        print(f"\n--- contest {cid} ({len(idx_list)} entries) ---")
        for i in idx_list:
            if n_printed >= limit:
                print("  ... (truncated)")
                return
            pt = primary_team(pids_arr[i], team_map, pos_map)
            names = [f"{name_map.get(int(p), '?')}({team_map.get(int(p), '?')})" for p in pids_arr[i]]
            print(f"  [{pt}] " + ", ".join(names))
            n_printed += 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", nargs=2, metavar=("ARM", "SLATE"), default=None)
    args = p.parse_args()
    if args.dump:
        dump_lineups(args.dump[0], args.dump[1])
    else:
        concentration_summary()


if __name__ == "__main__":
    main()
