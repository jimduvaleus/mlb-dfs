"""
One-time recovery script: regenerate ownership_projections.csv for a slate
archive using the same logic the pipeline runs at contest time.

Usage:
    python scripts/recover_ownership_projections.py archive/05222026
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.ownership import compute_heuristic_ownership


def _parse_eligible_positions(raw: str) -> list[str]:
    _CANONICAL = {
        "SP": "P", "RP": "P",
        "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
        "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
        "UTIL": "UTIL",
    }
    parts = [p.strip() for p in str(raw).split("/")]
    return [_CANONICAL.get(p, p) for p in parts if p in _CANONICAL]


def build_player_pool(archive_dir: Path) -> pd.DataFrame:
    sal_df = pd.read_csv(archive_dir / "DKSalaries.csv")
    sal_df.rename(columns={
        "ID": "player_id", "Name": "name", "Salary": "salary",
        "TeamAbbrev": "team", "Game Info": "game", "Position": "raw_position",
    }, inplace=True)
    sal_df["player_id"] = sal_df["player_id"].astype(int)
    sal_df["eligible_positions"] = sal_df["raw_position"].apply(_parse_eligible_positions)
    sal_df["position"] = sal_df["eligible_positions"].str[0]

    def _opponent(row):
        m = re.match(r"(\w+)@(\w+)", str(row["game"]))
        if m:
            away, home = m.group(1), m.group(2)
            return home if row["team"] == away else away
        return ""

    sal_df["opponent"] = sal_df.apply(_opponent, axis=1)
    sal_df = sal_df[
        ["player_id", "name", "salary", "position", "eligible_positions", "team", "opponent", "game"]
    ].drop_duplicates("player_id")

    # Prefer the server's merged projections file (data/processed/projections.csv) — it
    # matches what the server actually fed to compute_heuristic_ownership at pipeline time,
    # including players that had no DFF entry but appeared only in market_odds_projections.
    # Fall back to rebuilding from the archive if the live file is absent.
    live_proj_path = PROJECT_ROOT / "data" / "processed" / "projections.csv"
    if live_proj_path.exists():
        proj_df = pd.read_csv(live_proj_path)
        proj_df["player_id"] = proj_df["player_id"].astype(int)
        proj_cols = ["player_id", "mean", "std_dev"]
        if "lineup_slot" in proj_df.columns:
            proj_cols.append("lineup_slot")
        proj_df = proj_df[proj_cols]
        print(f"Using live projections: {live_proj_path}")
    else:
        dff_df = pd.read_csv(archive_dir / "dff_projections.csv")
        dff_df["player_id"] = dff_df["player_id"].astype(int)
        dff_cols = ["player_id", "mean", "std_dev"]
        if "lineup_slot" in dff_df.columns:
            dff_cols.append("lineup_slot")

        mo_path = archive_dir / "market_odds_projections.csv"
        if mo_path.exists():
            mo_df = pd.read_csv(mo_path)
            mo_df["player_id"] = mo_df["player_id"].astype(int)
            mo_lookup = mo_df.set_index("player_id")[["mean", "std_dev"]]
            has_mo = dff_df["player_id"].isin(mo_lookup.index)
            dff_df.loc[has_mo, "mean"] = dff_df.loc[has_mo, "player_id"].map(mo_lookup["mean"])
            dff_df.loc[has_mo, "std_dev"] = dff_df.loc[has_mo, "player_id"].map(mo_lookup["std_dev"])
        proj_df = dff_df[dff_cols]
        print("Live projections not found — rebuilding from archive DFF/MO CSVs")

    pool = sal_df.merge(proj_df, on="player_id", how="left")
    pool = pool.dropna(subset=["mean", "std_dev"])

    tt_path = archive_dir / "team_totals.csv"
    if tt_path.exists():
        tt_df = pd.read_csv(tt_path)
        if {"team", "implied_total"}.issubset(tt_df.columns):
            tt_map = tt_df.set_index("team")["implied_total"].to_dict()
            pool["implied_total"] = pool["team"].map(tt_map)

    mo_fair_path = archive_dir / "market_odds_fair_odds.json"
    if mo_fair_path.exists():
        with open(mo_fair_path) as f:
            fair = json.load(f)
        hr_market = fair.get("Player Home Runs", {})
        if hr_market:
            hr_map = {int(pid): v.get("over_0.5") for pid, v in hr_market.items() if "over_0.5" in v}
            pool["hr_prob"] = pool["player_id"].map(hr_map)

    return pool


def find_slate_exclusions(archive_dir: Path, team_reductions: dict) -> dict:
    """Find the slate_exclusions.json entry whose team_ownership_reductions matches."""
    excl_path = PROJECT_ROOT / "data" / "slate_exclusions.json"
    if not excl_path.exists():
        return {}
    with open(excl_path) as f:
        all_excl = json.load(f)
    for entry in all_excl.values():
        if entry.get("team_ownership_reductions") == team_reductions:
            return entry
    return {}


def main(archive_path: str) -> None:
    archive_dir = Path(archive_path)
    out_path = archive_dir / "ownership_projections.csv"

    settings_path = archive_dir / "ownership_settings.json"
    team_reductions = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        team_reductions = settings.get("team_ownership_reductions", {})
        if team_reductions:
            print(f"Applying team ownership reductions: {team_reductions}")

    excl = find_slate_exclusions(archive_dir, team_reductions)
    excluded_ids = set(excl.get("excluded_player_ids", []))
    excluded_games = set(excl.get("excluded_games", []))
    excluded_teams = set(excl.get("excluded_teams", []))
    if excluded_ids or excluded_games or excluded_teams:
        print(f"Applying slate exclusions: {len(excluded_ids)} players, games={excluded_games or None}, teams={excluded_teams or None}")

    pool = build_player_pool(archive_dir)
    if excluded_ids:
        pool = pool[~pool["player_id"].isin(excluded_ids)]
    if excluded_games:
        pool = pool[~pool["game"].str.startswith(tuple(excluded_games))]
    if excluded_teams:
        pool = pool[~pool["team"].isin(excluded_teams)]
    pool = pool.reset_index(drop=True)
    print(f"Player pool: {len(pool)} players")

    team_totals = None
    if "implied_total" in pool.columns and pool["implied_total"].notna().any():
        team_totals = (
            pool[["team", "implied_total"]].dropna(subset=["implied_total"])
            .drop_duplicates("team").set_index("team")["implied_total"].to_dict()
        )

    ownership_arr = compute_heuristic_ownership(
        pool,
        team_totals=team_totals,
        team_ownership_reductions=team_reductions or None,
    )

    rows = []
    for i, row in pool.reset_index(drop=True).iterrows():
        rows.append({
            "player_id": row["player_id"],
            "name": row["name"],
            "team": row["team"],
            "game": row["game"],
            "position": row["position"],
            "ownership_pct": round(float(ownership_arr[i]) * 100, 4),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Written → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <archive_dir>")
        sys.exit(1)
    main(sys.argv[1])
