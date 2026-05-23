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
from src.optimization.optimal_lineups import generate_optimal_lineups

ARCHIVE_DIR = ROOT / "archive"
SLATES = ["05172026", "05192026"]

TARGET_NS = (50, 100)
MIN_UNIQUES_RANGE = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess for each mu setting)
# ---------------------------------------------------------------------------

def _solve_mu_worker(args: tuple) -> tuple:
    """Solve n_target lineups for one (slate, min_uniques) pair.

    Returns (min_uniques, lineups, checkpoints).
    Runs in a subprocess.
    """
    slate, min_uniques, n_target = args

    df = load_slate(slate)
    mean_map = {int(r.player_id): float(r.mean) for r in df.itertuples(index=False)}

    lineups: list[tuple[float, frozenset]] = []
    checkpoints: list[float] = []
    t0 = time.perf_counter()

    def _on_lineup(n_found: int) -> None:
        lu = result_lineups[n_found - 1]
        score = sum(mean_map[p] for p in lu.player_ids)
        lineups.append((score, frozenset(lu.player_ids)))
        checkpoints.append(time.perf_counter() - t0)

    result_lineups = generate_optimal_lineups(
        df, n=n_target, min_uniques=min_uniques, progress_cb=_on_lineup
    )

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
