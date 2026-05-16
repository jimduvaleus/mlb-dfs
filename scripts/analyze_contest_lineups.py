"""
Analyze real DraftKings contest entries from archived slates to extract
calibration statistics for field lineup generation: stacking patterns,
salary distributions, and bring-back rates.

Usage
-----
    python scripts/analyze_contest_lineups.py               # all archive/ slates
    python scripts/analyze_contest_lineups.py archive/05082026  # one slate

Output
------
    data/processed/contest_stats.json   calibration params for contest.py
    Prints per-slate and aggregate tables to stdout.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ownership import _parse_contest_zip, _LINEUP_POSITIONS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SALARY_CAP = 50_000
_BATTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF"}


# ---------------------------------------------------------------------------
# Lineup string parser
# ---------------------------------------------------------------------------

def _parse_lineup_string(lineup_str: str) -> list[tuple[str, str]]:
    """
    Parse "1B Josh Naylor 2B Xavier Edwards ..." into [(pos, name), ...].
    Returns [] if the string is malformed, contains LOCKED entries, or
    doesn't yield exactly 10 players.
    """
    if not lineup_str or "LOCKED" in lineup_str:
        return []
    tokens = lineup_str.split()
    result: list[tuple[str, str]] = []
    current_pos: str | None = None
    name_tokens: list[str] = []

    for tok in tokens:
        if tok in _LINEUP_POSITIONS:
            if current_pos is not None:
                if not name_tokens:
                    return []
                result.append((current_pos, " ".join(name_tokens)))
            current_pos = tok
            name_tokens = []
        else:
            name_tokens.append(tok)

    if current_pos is not None and name_tokens:
        result.append((current_pos, " ".join(name_tokens)))

    return result if len(result) == 10 else []


# ---------------------------------------------------------------------------
# Salary map loader
# ---------------------------------------------------------------------------

def _load_salary_maps(
    sal_path: Path,
) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    """
    Read DKSalaries.csv → (name→salary, name→team, name→opponent).
    Opponent is derived from the "Team@Opponent Date Time" game string.
    """
    import pandas as pd

    df = pd.read_csv(sal_path)
    df.rename(
        columns={"ID": "player_id", "Name": "name", "Salary": "salary",
                 "TeamAbbrev": "team", "Game Info": "game"},
        inplace=True,
    )

    salary_map: dict[str, int] = {}
    team_map: dict[str, str] = {}
    opp_map: dict[str, str] = {}

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        salary_map[name] = int(row["salary"])
        team = str(row["team"]).strip()
        team_map[name] = team
        m = re.match(r"(\w+)@(\w+)", str(row["game"]))
        if m:
            away, home = m.group(1), m.group(2)
            opp_map[name] = home if team == away else away
        else:
            opp_map[name] = ""

    return salary_map, team_map, opp_map


# ---------------------------------------------------------------------------
# Per-slate analysis
# ---------------------------------------------------------------------------

def analyze_slate(slate_dir: Path) -> dict | None:
    """Parse one archive slate directory and return a stats dict."""
    sal_path = slate_dir / "DKSalaries.csv"
    zip_files = list(slate_dir.glob("contest-standings-*.zip"))

    if not sal_path.exists() or not zip_files:
        print(f"  Skipping {slate_dir.name}: missing DKSalaries.csv or contest ZIP")
        return None

    salary_map, team_map, opp_map = _load_salary_maps(sal_path)

    try:
        standings_df, _ = _parse_contest_zip(zip_files[0])
    except Exception as exc:
        print(f"  Skipping {slate_dir.name}: failed to parse ZIP — {exc}")
        return None

    n_entries = len(standings_df)
    n_parsed = 0
    n_salary_miss = 0

    primary_stacks: list[int] = []
    salaries: list[int] = []
    bringbacks: list[bool] = []

    for _, row in standings_df.iterrows():
        players = _parse_lineup_string(str(row.get("lineup_str", "")))
        if not players:
            continue

        miss = sum(1 for _, name in players if name not in salary_map)
        if miss > 2:
            n_salary_miss += 1
            continue

        n_parsed += 1

        team_batter_counts: dict[str, int] = {}
        pitcher_opps: list[str] = []

        for pos, name in players:
            team = team_map.get(name, "")
            if pos in _BATTER_POSITIONS:
                team_batter_counts[team] = team_batter_counts.get(team, 0) + 1
            else:
                pitcher_opps.append(opp_map.get(name, ""))

        primary_stack = max(team_batter_counts.values()) if team_batter_counts else 0
        primary_stacks.append(primary_stack)

        total_sal = sum(salary_map.get(name, 0) for _, name in players)
        if total_sal > 0:
            salaries.append(total_sal)

        batter_teams = set(team_batter_counts.keys())
        bringbacks.append(any(opp and opp in batter_teams for opp in pitcher_opps))

    if not primary_stacks:
        print(f"  {slate_dir.name}: no parseable lineups")
        return None

    print(f"  {slate_dir.name}: {n_entries:,} entries, {n_parsed:,} parsed, {n_salary_miss} salary misses")

    stacks_arr = np.array(primary_stacks)
    sals_arr = np.array(salaries)
    bb_arr = np.array(bringbacks)

    stack_dist = {str(k): float((stacks_arr == k).mean()) for k in range(1, 7)}

    stacked_mask = stacks_arr >= 4
    stack_probability = float(stacked_mask.mean())
    stacked_stacks = stacks_arr[stacked_mask]
    stack_size_4_prob = float((stacked_stacks == 4).mean()) if len(stacked_stacks) > 0 else 0.5

    at_cap = float((sals_arr >= SALARY_CAP).mean())
    under_cap = sals_arr[sals_arr < SALARY_CAP]
    underspend = SALARY_CAP - under_cap if len(under_cap) > 0 else np.array([0])

    return {
        "slate": slate_dir.name,
        "n_entries": n_entries,
        "n_parsed": n_parsed,
        "n_salary_miss": n_salary_miss,
        "stack_dist": stack_dist,
        "stack_probability": stack_probability,
        "stack_size_4_prob": stack_size_4_prob,
        "bringback_rate": float(bb_arr.mean()),
        "salary_mean": float(sals_arr.mean()),
        "salary_median": float(np.median(sals_arr)),
        "salary_stdev": float(sals_arr.std()),
        "salary_p5": float(np.percentile(sals_arr, 5)),
        "salary_p10": float(np.percentile(sals_arr, 10)),
        "salary_p25": float(np.percentile(sals_arr, 25)),
        "salary_p75": float(np.percentile(sals_arr, 75)),
        "salary_p90": float(np.percentile(sals_arr, 90)),
        "salary_at_cap_frac": at_cap,
        "underspend_mean": float(underspend.mean()),
        "underspend_median": float(np.median(underspend)),
        "underspend_p25": float(np.percentile(underspend, 25)),
        "underspend_p75": float(np.percentile(underspend, 75)),
        "underspend_p90": float(np.percentile(underspend, 90)),
        # Raw arrays preserved for aggregate pooling; stripped before JSON output.
        "_primary_stacks": primary_stacks,
        "_salaries": salaries,
        "_bringbacks": [bool(b) for b in bringbacks],
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_stats(slate_results: list[dict]) -> dict:
    """Pool raw data across all slates and compute aggregate distributions."""
    all_stacks: list[int] = []
    all_sals: list[int] = []
    all_bb: list[bool] = []

    for r in slate_results:
        all_stacks.extend(r["_primary_stacks"])
        all_sals.extend(r["_salaries"])
        all_bb.extend(r["_bringbacks"])

    stacks_arr = np.array(all_stacks)
    sals_arr = np.array(all_sals)
    bb_arr = np.array(all_bb)

    stack_dist = {str(k): float((stacks_arr == k).mean()) for k in range(1, 7)}

    stacked_mask = stacks_arr >= 4
    stack_probability = float(stacked_mask.mean())
    stacked_stacks = stacks_arr[stacked_mask]
    stack_size_4_prob = float((stacked_stacks == 4).mean()) if len(stacked_stacks) > 0 else 0.5

    at_cap = float((sals_arr >= SALARY_CAP).mean())
    under_cap = sals_arr[sals_arr < SALARY_CAP]
    underspend = SALARY_CAP - under_cap if len(under_cap) > 0 else np.array([0])

    return {
        "slates_analyzed": len(slate_results),
        "total_entries": sum(r["n_entries"] for r in slate_results),
        "total_parsed": sum(r["n_parsed"] for r in slate_results),
        "stack_dist": stack_dist,
        "stack_probability": stack_probability,
        "stack_size_4_prob": stack_size_4_prob,
        "bringback_rate": float(bb_arr.mean()),
        "salary_mean": float(sals_arr.mean()),
        "salary_median": float(np.median(sals_arr)),
        "salary_stdev": float(sals_arr.std()),
        "salary_p5": float(np.percentile(sals_arr, 5)),
        "salary_p10": float(np.percentile(sals_arr, 10)),
        "salary_p25": float(np.percentile(sals_arr, 25)),
        "salary_p75": float(np.percentile(sals_arr, 75)),
        "salary_p90": float(np.percentile(sals_arr, 90)),
        "salary_at_cap_frac": at_cap,
        "underspend_mean": float(underspend.mean()),
        "underspend_median": float(np.median(underspend)),
        "underspend_p25": float(np.percentile(underspend, 25)),
        "underspend_p75": float(np.percentile(underspend, 75)),
        "underspend_p90": float(np.percentile(underspend, 90)),
    }


def derive_calibration_params(agg: dict) -> dict:
    """Translate aggregate stats into calibration constants for contest.py."""
    from datetime import date

    p10 = agg["salary_p10"]
    salary_floor = int((p10 // 500) * 500)

    return {
        "stack_probability": round(agg["stack_probability"], 4),
        "stack_size_4_prob": round(agg["stack_size_4_prob"], 4),
        "salary_floor_field": salary_floor,
        "bringback_rate": round(agg["bringback_rate"], 4),
        "calibrated_from": str(date.today()),
        "n_entries": agg["total_parsed"],
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> None:
    header = (
        f"{'Slate':<12} {'Entries':>8} {'Parsed':>8} {'Stack≥4':>8} "
        f"{'4-stk%':>7} {'5-stk%':>7} {'SalMean':>8} {'SalP10':>7} "
        f"{'@Cap%':>6} {'BringBk':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        p4 = r["stack_size_4_prob"]
        print(
            f"{r['slate']:<12} {r['n_entries']:>8,} {r['n_parsed']:>8,} "
            f"{r['stack_probability']:>8.1%} {p4:>7.1%} {1 - p4:>7.1%} "
            f"{r['salary_mean']:>8,.0f} {r['salary_p10']:>7,.0f} "
            f"{r['salary_at_cap_frac']:>6.1%} {r['bringback_rate']:>8.1%}"
        )


def _print_aggregate(agg: dict, cal: dict) -> None:
    print(f"\n=== AGGREGATE ({agg['slates_analyzed']} slates, {agg['total_parsed']:,} lineups) ===")

    print("\nStack distribution (primary-stack size):")
    for k in sorted(agg["stack_dist"].keys(), key=int):
        bar = "█" * int(agg["stack_dist"][k] * 40)
        print(f"  {k}: {agg['stack_dist'][k]:>6.1%}  {bar}")

    print(f"\nStack probability (≥4 batters from one team): {agg['stack_probability']:.1%}")
    print(f"  Of stacked lineups — 4-stack: {agg['stack_size_4_prob']:.1%}  "
          f"5-stack: {1 - agg['stack_size_4_prob']:.1%}")

    print(f"\nSalary usage:")
    print(f"  Mean: {agg['salary_mean']:,.0f}  Median: {agg['salary_median']:,.0f}  "
          f"StDev: {agg['salary_stdev']:,.0f}")
    print(f"  p5={agg['salary_p5']:,.0f}  p10={agg['salary_p10']:,.0f}  "
          f"p25={agg['salary_p25']:,.0f}  p75={agg['salary_p75']:,.0f}  "
          f"p90={agg['salary_p90']:,.0f}")
    print(f"  At cap ({SALARY_CAP:,}): {agg['salary_at_cap_frac']:.1%}")
    print(f"  Underspend (when under cap): "
          f"mean={agg['underspend_mean']:,.0f}  "
          f"median={agg['underspend_median']:,.0f}  "
          f"p90={agg['underspend_p90']:,.0f}")

    print(f"\nBring-back rate (informational): {agg['bringback_rate']:.1%}")

    print(f"\n=== CALIBRATION PARAMS ===")
    for k, v in cal.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DK contest entry data to calibrate field lineup generation."
    )
    parser.add_argument(
        "slate_dirs", nargs="*",
        help="Slate directories to analyze (default: all archive/MMDDYYYY/ subdirs)",
    )
    args = parser.parse_args()

    if args.slate_dirs:
        slate_dirs = [Path(d) for d in args.slate_dirs]
    else:
        archive_root = PROJECT_ROOT / "archive"
        slate_dirs = sorted(
            p for p in archive_root.iterdir()
            if p.is_dir() and re.match(r"\d{8}$", p.name)
        )

    if not slate_dirs:
        print("No slate directories found.")
        sys.exit(1)

    print(f"Analyzing {len(slate_dirs)} slate(s)...\n")

    results: list[dict] = []
    for sd in slate_dirs:
        print(f"[{sd.name}]")
        r = analyze_slate(sd)
        if r:
            results.append(r)

    if not results:
        print("No parseable data found.")
        sys.exit(1)

    print()
    _print_table(results)

    agg = aggregate_stats(results)
    cal = derive_calibration_params(agg)
    _print_aggregate(agg, cal)

    per_slate = {
        r["slate"]: {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    }
    output = {"calibration_params": cal, "aggregate": agg, "per_slate": per_slate}

    out_path = PROJECT_ROOT / "data" / "processed" / "contest_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
