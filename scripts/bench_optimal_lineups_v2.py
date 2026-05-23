#!/usr/bin/env python3
"""
Timing benchmark v2: top-N optimal lineups via iterative ILP (OR-Tools CBC).

Uses Google OR-Tools pywraplp (CBC solver) with:
  - One binary variable per (player, eligible_position) pair (xp).
  - Binary z[t] variables for stack-team designation.
  - Aggregate pitcher-batter conflict (one row per pitcher vs 189 per pair in v1).
  - Parallel execution across min_uniques settings (one worker per mu).

For each (min_uniques, target_n):
  1. Build a fresh CBC model (per worker process).
  2. Solve for lineup #1.
  3. Append no-good cut: sum(xp[j] for j in lineup) <= 10 - min_uniques.
  4. Repeat until target_n lineups found or infeasible.

Requires: pip install ortools (Google OR-Tools, includes CBC).

Slates: 05172026, 05192026

Usage:
    source venv/bin/activate
    python scripts/bench_optimal_lineups_v2.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.dk_slate import DraftKingsSlateIngestor

ARCHIVE_DIR = ROOT / "archive"
SLATES = ["05172026", "05192026"]

MIN_STACK = 4
POS_REQUIREMENTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}

TARGET_NS = (50, 100)
MIN_UNIQUES_RANGE = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Data loading  (identical to v1)
# ---------------------------------------------------------------------------

def load_slate(slate: str) -> pd.DataFrame:
    d = ARCHIVE_DIR / slate
    salary_df = DraftKingsSlateIngestor(str(d / "DKSalaries.csv")).get_slate_dataframe()
    proj_df = pd.read_csv(
        d / "dff_projections.csv",
        usecols=["player_id", "mean", "std_dev", "lineup_slot", "slot_confirmed"],
    )
    df = salary_df.merge(proj_df, on="player_id", how="inner")
    df["slot_confirmed"] = df["slot_confirmed"].apply(
        lambda x: str(x).strip().lower() == "true"
    )
    df = df[(df["position"] == "P") | df["slot_confirmed"]].copy()
    df.sort_values("mean", ascending=False, inplace=True, ignore_index=True)
    return df


def _build_meta(df: pd.DataFrame) -> dict:
    return {
        int(r.player_id): {
            "position": r.position,
            "eligible_positions": list(r.eligible_positions),
            "salary": float(r.salary),
            "team": r.team,
            "opponent": r.opponent,
            "game": r.game,
        }
        for r in df.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# OR-Tools CBC model builder
# ---------------------------------------------------------------------------

def _build_model(df: pd.DataFrame):
    """Build a fresh CBC model. Returns (solver, xp_vars, z_vars, xp_list, player_to_js, mean_map)."""
    from ortools.linear_solver import pywraplp

    meta = _build_meta(df)
    player_ids = df["player_id"].astype(int).tolist()
    mean_map: dict[int, float] = {int(r.player_id): float(r.mean) for r in df.itertuples(index=False)}

    # --- (player, pos) variable index structures ---
    xp_list: list[tuple[int, str]] = []
    player_to_js: dict[int, list[int]] = {pid: [] for pid in player_ids}
    pos_to_js: dict[str, list[int]] = {}

    for pid in player_ids:
        for pos in meta[pid]["eligible_positions"]:
            j = len(xp_list)
            xp_list.append((pid, pos))
            player_to_js[pid].append(j)
            pos_to_js.setdefault(pos, []).append(j)

    n_xp = len(xp_list)

    pitcher_pids = [pid for pid in player_ids if meta[pid]["position"] == "P"]
    batter_pids  = [pid for pid in player_ids if meta[pid]["position"] != "P"]
    batter_teams = sorted({meta[pid]["team"] for pid in batter_pids})
    T = len(batter_teams)

    team_batter_js: dict[str, list[int]] = {tm: [] for tm in batter_teams}
    game_js: dict[str, list[int]] = {}
    for j, (pid, _pos) in enumerate(xp_list):
        if meta[pid]["position"] != "P":
            team_batter_js[meta[pid]["team"]].append(j)
        g = meta[pid]["game"]
        if g:
            game_js.setdefault(g, []).append(j)

    pitcher_team_js: dict[str, list[int]] = {}
    for pp in pitcher_pids:
        for j in player_to_js[pp]:
            pitcher_team_js.setdefault(meta[pp]["team"], []).append(j)

    opp_of = {pid: meta[pid]["opponent"] for pid in pitcher_pids}

    # --- Create solver ---
    solver = pywraplp.Solver.CreateSolver("CBC")
    solver.SuppressOutput()

    xp = [solver.BoolVar(f"xp{j}") for j in range(n_xp)]
    z  = [solver.BoolVar(f"z{t}") for t in range(T)]

    # Objective: maximise sum(mean * xp)
    obj = solver.Objective()
    obj.SetMaximization()
    for j, (pid, _) in enumerate(xp_list):
        obj.SetCoefficient(xp[j], mean_map[pid])

    # C1: each multi-position player selected at most once
    for pid, js in player_to_js.items():
        if len(js) > 1:
            c = solver.Constraint(0, 1)
            for j in js:
                c.SetCoefficient(xp[j], 1)

    # C2: exact position slot counts
    for pos, count in POS_REQUIREMENTS.items():
        c = solver.Constraint(count, count)
        for j in pos_to_js.get(pos, []):
            c.SetCoefficient(xp[j], 1)

    # C3: salary cap <= 50,000
    c = solver.Constraint(0, 50_000)
    for j, (pid, _) in enumerate(xp_list):
        c.SetCoefficient(xp[j], meta[pid]["salary"])

    # C4: <= 5 batters per team
    for tm in batter_teams:
        c = solver.Constraint(0, 5)
        for j in team_batter_js[tm]:
            c.SetCoefficient(xp[j], 1)

    # C5: pitcher–batter conflict (aggregate big-M per pitcher, 21 rows vs 189 in v1)
    #     sum(opp_batter_xp) + n_opp * sum(pitcher_xp) <= n_opp
    for pp in pitcher_pids:
        opp = opp_of[pp]
        opp_batter_js = [j for bp in batter_pids if meta[bp]["team"] == opp
                         for j in player_to_js[bp]]
        n_opp = len([bp for bp in batter_pids if meta[bp]["team"] == opp])
        if opp_batter_js and player_to_js[pp]:
            c = solver.Constraint(-solver.infinity(), float(n_opp))
            for j in opp_batter_js:
                c.SetCoefficient(xp[j], 1)
            for j in player_to_js[pp]:
                c.SetCoefficient(xp[j], float(n_opp))

    # C6: <= 1 pitcher per team (forces pitchers from different teams)
    for tm, js in pitcher_team_js.items():
        if len(js) > 1:
            c = solver.Constraint(0, 1)
            for j in js:
                c.SetCoefficient(xp[j], 1)

    # C7: sum(z[t]) >= 1 (at least one designated stack team)
    c = solver.Constraint(1, T)
    for t in range(T):
        c.SetCoefficient(z[t], 1)

    # C8: sum(batter_xp[team_t]) - MIN_STACK * z[t] >= 0  (stack linkage)
    for t, tm in enumerate(batter_teams):
        c = solver.Constraint(0, solver.infinity())
        for j in team_batter_js[tm]:
            c.SetCoefficient(xp[j], 1)
        c.SetCoefficient(z[t], -float(MIN_STACK))

    # C9: <= 9 players per game (ensures >= 2 games)
    for _g, js in game_js.items():
        c = solver.Constraint(0, 9)
        for j in js:
            c.SetCoefficient(xp[j], 1)

    return solver, xp, z, xp_list, player_to_js, mean_map


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess for each mu setting)
# ---------------------------------------------------------------------------

def _solve_mu_worker(args: tuple) -> tuple:
    """Solve n_target lineups for one (slate, min_uniques) pair.

    Returns (min_uniques, lineups, checkpoints, n_vars, n_base_rows).
    Runs in a subprocess — imports OR-Tools fresh.
    """
    from ortools.linear_solver import pywraplp

    slate, min_uniques, n_target = args

    df = load_slate(slate)
    solver, xp, z, xp_list, player_to_js, mean_map = _build_model(df)
    n_xp = len(xp_list)

    lineups: list[tuple[float, frozenset]] = []
    checkpoints: list[float] = []
    t0 = time.perf_counter()

    for _ in range(n_target):
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            break

        pids = frozenset(xp_list[j][0] for j in range(n_xp) if xp[j].solution_value() > 0.5)
        score = sum(mean_map[p] for p in pids)
        lineups.append((score, pids))
        checkpoints.append(time.perf_counter() - t0)

        # No-good cut: sum(xp[j] for j in lineup_vars) <= 10 - min_uniques
        cut_js = sorted({j for pid in pids for j in player_to_js[pid]})
        c = solver.Constraint(0, float(10 - min_uniques))
        for j in cut_js:
            c.SetCoefficient(xp[j], 1)

    return min_uniques, lineups, checkpoints


# ---------------------------------------------------------------------------
# Benchmark table helpers  (identical to v1)
# ---------------------------------------------------------------------------

def _cell(lineups: list, checkpoints: list, n: int) -> str:
    cnt = min(len(lineups), n)
    if cnt == 0:
        return "infeasible"
    scores = [s for s, _ in lineups[:cnt]]
    t      = checkpoints[cnt - 1]
    avg_ms = t / cnt * 1000
    return f"{cnt}/{n}  {scores[0]:.2f}–{scores[-1]:.2f}  [{t:.1f}s, {avg_ms:.0f}ms/lu]"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_slate(slate: str) -> None:
    bar = "=" * 74
    print(f"\n{bar}")
    print(f"Slate: {slate}")
    print(bar)

    t0 = time.perf_counter()
    df = load_slate(slate)
    t_load = time.perf_counter() - t0
    n_p = int((df["position"] == "P").sum())
    n_b = int((df["position"] != "P").sum())
    print(f"  Players: {len(df)}  ({n_p} P, {n_b} confirmed batters)  [load {t_load:.3f}s]")

    # Count model size without building a full OR-Tools solver
    max_n = max(TARGET_NS)
    col_w = 46

    hdr = f"  {'min_u':^5}  │"
    sep = f"  {'─────':^5}  ┼"
    for n in TARGET_NS:
        hdr += f"  {'top-' + str(n):<{col_w}}│"
        sep += "─" * (col_w + 2) + "┼"

    # Run all min_uniques in parallel (one subprocess per mu)
    t_bench_start = time.perf_counter()
    args = [(slate, mu, max_n) for mu in MIN_UNIQUES_RANGE]

    results: dict[int, tuple] = {}

    with ProcessPoolExecutor(max_workers=len(MIN_UNIQUES_RANGE)) as ex:
        futures = {ex.submit(_solve_mu_worker, a): a[1] for a in args}
        for fut in as_completed(futures):
            mu, lineups, checkpoints = fut.result()
            results[mu] = (lineups, checkpoints)

    t_bench = time.perf_counter() - t_bench_start
    print(f"  Parallel solve ({len(MIN_UNIQUES_RANGE)} workers): {t_bench:.1f}s wall time\n")

    print(hdr)
    print(sep)
    for mu in MIN_UNIQUES_RANGE:
        lineups, checkpoints = results[mu]
        row = f"    {mu:2d}     │"
        for n in TARGET_NS:
            row += f"  {_cell(lineups, checkpoints, n):<{col_w}}│"
        print(row)

    print()


def main() -> None:
    grand_t0 = time.perf_counter()
    for slate in SLATES:
        bench_slate(slate)
    total = time.perf_counter() - grand_t0
    print(f"Total wall time (v2 / OR-Tools CBC, parallel): {total:.2f}s")


if __name__ == "__main__":
    main()
