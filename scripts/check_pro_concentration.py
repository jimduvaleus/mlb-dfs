"""Measure real named pros' actual team-concentration structure (primary-
stack team spread, top-team share) on the 10-slate archive, using their
REAL submitted lineups from the standings zips -- not inferred from an
older/different metric (correlation-based Dn saturation, a different
slate sample). Same methodology as scripts/inspect_arm_portfolios.py's
measurement of our own arms, so the two are directly comparable.

Usage
-----
    source venv/bin/activate
    python scripts/check_pro_concentration.py needlunchmoney youdacao hishboo
"""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tests.bt_core import load_real_contests  # noqa: E402
from scripts.analyze_rival_portfolio import parse_standings_rows  # noqa: E402

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]


def team_and_position_maps(slate: str) -> tuple[dict, dict]:
    sal = pd.read_csv(PROJECT_ROOT / "archive" / slate / "DKSalaries.csv")
    sal["Name"] = sal["Name"].astype(str).str.strip()
    team_map = dict(zip(sal["Name"], sal["TeamAbbrev"].astype(str)))
    is_pitcher = sal["Position"].astype(str).str.contains("P")
    pos_map = dict(zip(sal["Name"], is_pitcher))
    return team_map, pos_map


def primary_team(names: tuple, team_map: dict, pos_map: dict) -> str:
    hitters = [n for n in names if not pos_map.get(n, False)]
    teams = pd.Series([team_map.get(n, "?") for n in hitters])
    return teams.value_counts().idxmax() if len(teams) else "?"


def measure(handle: str) -> dict:
    pooled: dict[str, int] = {}
    per_slate_n = []
    total_entries = 0
    for s in SLATES:
        d = PROJECT_ROOT / "archive" / s
        if not d.exists():
            continue
        team_map, pos_map = team_and_position_maps(s)
        slate_teams: dict[str, int] = {}
        for c in load_real_contests(d):
            z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
            with zipfile.ZipFile(z) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                rows = list(csv.reader(io.StringIO(
                    zf.read(name).decode("utf-8-sig", errors="replace"))))
            e, _ = parse_standings_rows(rows)
            mine = e[e.handle == handle]
            for names in mine["names"]:
                t = primary_team(names, team_map, pos_map)
                slate_teams[t] = slate_teams.get(t, 0) + 1
                pooled[t] = pooled.get(t, 0) + 1
                total_entries += 1
        if slate_teams:
            per_slate_n.append(len(slate_teams))
    total = sum(pooled.values())
    return {
        "handle": handle, "entries": total_entries,
        "distinct_teams_pooled": len(pooled),
        "mean_teams_per_slate": float(np.mean(per_slate_n)) if per_slate_n else float("nan"),
        "top_team_share_pct": 100 * max(pooled.values()) / total if total else float("nan"),
        "n_slates_active": len(per_slate_n),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("handles", nargs="+")
    args = p.parse_args()
    rows = [measure(h) for h in args.handles]
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
