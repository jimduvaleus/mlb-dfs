#!/usr/bin/env python3
"""Is sigma_dG new information, and does the external pool already span it?

Two questions, one cheap pass over a slate. Both are GATES -- each can stop
work rather than justify it.

Q1. IS sigma_dG NEW? Haugh & Singal's `sigma_{delta,G}` -- per-player
    covariance with the field's payout cutoff -- is field-aware in a way none
    of our currencies are. But this repo has repeatedly found new currencies
    that turned out to be re-parameterisations of old ones (p_win vs tail
    metrics, rho 0.88-0.95; memory project-external-pool-currency-comparison).
    If per-lineup w'sigma_dG correlates ~1 with projected ownership, it is
    ownership arithmetic with extra steps and the whole term should be dropped.
    FAILURE LOOKS LIKE: |rho| > 0.9 against ownership or proj_score.

Q2. DOES THE POOL ALREADY SPAN THE LOW-sigma REGION? dR can only select what
    is in the pool, and the SaberSim pool is a shared commodity -- a rival
    played our pool's #3 lineup with all 10 players matching
    (project-pipeline-is-a-random-draw). But this repo also measured ILP
    supplements at 0.57-0.68x the pool's p_win with ZERO reaching the combined
    top 1% across 7 slates (diagnose_ilp_supplement_pwin). So generation has to
    earn its place: if the pool already reaches where the sigma frontier goes,
    skip it and let that prior negative stand.
    FAILURE LOOKS LIKE: decision == "pool-already-spans".

Q3 (free, since the block is computed anyway). ASSUMPTION 5.2: is
    Cov(delta_p, G^(r_d)) effectively constant across payout tiers d? The paper
    assumes it and proves it in the O -> infinity limit (Prop 5.1). If it holds
    here, the (n_players, T) block collapses to one vector per slate.

EPISTEMIC CEILING: this measures STRUCTURE (is the signal distinct, is the
region reachable), never profitability. Nothing here can say sigma_dG makes
money. Per PROSPECTIVE_PROTOCOL that verdict comes from a live A/B.

Usage:
    python scripts/eval_field_covariance.py 07222026 07242026
    python scripts/eval_field_covariance.py --all

Env: FCV_NSIMS (default 4000), FCV_SEED (42), FCV_FIELD (10000),
     FCV_LAMBDAS (comma list), FCV_PER_LAMBDA (30), FCV_FORCE=1
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.mrp.field_covariance import (  # noqa: E402
    assumption_52_report,
    field_order_statistics,
    lineup_sigma_scores,
    payout_weighted_sigma,
    player_field_covariance,
    tier_boundary_ranks,
)
from src.optimization.mrp.sigma_frontier import pool_sigma_coverage, sigma_frontier  # noqa: E402
from src.optimization.payout import payout_table_to_array, structure_for_contest  # noqa: E402
from tests import bt_core  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / "outputs" / "field_covariance" / "results.csv"
SIM_CACHE = PROJECT_ROOT / "outputs" / "replay" / "sim_cache"

N_SIMS = int(os.environ.get("FCV_NSIMS", "4000"))
SEED = int(os.environ.get("FCV_SEED", "42"))
FIELD_SIZE = int(os.environ.get("FCV_FIELD", "10000"))
PER_LAMBDA = int(os.environ.get("FCV_PER_LAMBDA", "30"))
LAMBDAS = tuple(float(x) for x in os.environ.get(
    "FCV_LAMBDAS", "0.0,0.1,0.2,0.3,0.5,0.7,1.0").split(","))
FORCE = os.environ.get("FCV_FORCE") == "1"


def _append_and_reload(csv_path: Path, slate: str, rows: list[dict]) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"slate": str})
        old = old[old["slate"] != slate]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"slate": str})


def _done_slates(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path, dtype={"slate": str})["slate"].unique())


def run_slate(d: Path, slate: str) -> list[dict]:
    real = bt_core.load_real_contests(d)
    if not real:
        raise SystemExit("no named standings zips")

    ctx = bt_core.build_slate_context(
        d, seed=SEED, calibrated=False, real=real, n_sims=N_SIMS,
        sharpness=0.05, sim_cache_dir=SIM_CACHE, want_corr=False, want_pwin=False,
    )
    pool, players_df, sim_results = ctx["pool"], ctx["players_df"], ctx["sim_results"]
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    sim_matrix = sim_results.results_matrix.astype(np.float32)

    # Largest contest by field size drives the tier grid -- it is where the
    # top-heavy weight (and therefore this whole objective) actually lives.
    biggest = max(real, key=lambda c: len(c["sorted_scores"]))
    payout = biggest["payout_arr"]
    ranks = tier_boundary_ranks(payout)

    sim = ContestSimulator()
    own = players_df["ownership"].to_numpy(dtype=np.float64)
    field_lineups = sim.generate_field(players_df, own, n_lineups=FIELD_SIZE, rng_seed=SEED)
    field_scores = sim.score_field(field_lineups, sim_matrix, col_map)
    field_sorted = np.sort(field_scores, axis=1)
    del field_scores

    thr = field_order_statistics(field_sorted, ranks)
    sigma_block = player_field_covariance(sim_matrix, thr)          # (P, T)
    a52 = assumption_52_report(sigma_block)
    sigma_vec = payout_weighted_sigma(sigma_block, payout, ranks)   # (P,)
    del field_sorted

    # --- Q1: is it new? ------------------------------------------------------
    cols = np.array([[col_map[int(p)] for p in lu.player_ids] for lu in pool.lineups])
    pool_sigma = lineup_sigma_scores(sigma_vec, cols)
    pool_own = ep.compute_pool_ownership(pool.lineups, players_df)
    pool_proj = ep.compute_pool_proj_scores(pool.lineups, players_df)
    rho_own = float(spearmanr(pool_sigma, pool_own).statistic)
    rho_proj = float(spearmanr(pool_sigma, pool_proj).statistic)

    # --- Q2: does the pool span the frontier? --------------------------------
    pdf = players_df.copy()
    if "eligible_positions" not in pdf.columns:
        pdf["eligible_positions"] = [[p] for p in pdf["position"]]
    sig_by_row = np.array([sigma_vec[col_map[int(p)]] for p in pdf["player_id"]])
    t_gen = time.time()
    frontier = sigma_frontier(pdf, sig_by_row, lambdas=LAMBDAS,
                              n_per_lambda=PER_LAMBDA, min_uniques=3, min_stack=4)
    gen_s = time.time() - t_gen

    if frontier:
        fcols = np.array([[col_map[int(p)] for p in lu.player_ids] for lu in frontier])
        frontier_sigma = lineup_sigma_scores(sigma_vec, fcols)
    else:
        frontier_sigma = np.array([])
    cov = pool_sigma_coverage(pool_sigma, frontier_sigma)

    row = {
        "slate": slate, "seed": SEED, "n_sims": N_SIMS, "field_size": FIELD_SIZE,
        "n_pool": len(pool.lineups), "n_frontier": len(frontier),
        "frontier_gen_s": round(gen_s, 1),
        "contest": biggest["contest"], "n_field_real": len(biggest["sorted_scores"]),
        "rho_sigma_vs_own": round(rho_own, 4),
        "rho_sigma_vs_proj": round(rho_proj, 4),
        "is_new_signal": bool(abs(rho_own) < 0.9 and abs(rho_proj) < 0.9),
        "decision": cov["decision"],
    }
    row.update({k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in cov.items() if k != "decision"})
    row.update({f"a52_{k}": (round(v, 5) if isinstance(v, float) else v)
                for k, v in a52.items()})
    return [row]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:
        slates = sorted(p.name for p in (PROJECT_ROOT / "archive").iterdir()
                        if p.is_dir() and (p / "DKSalaries.csv").exists())
    else:
        slates = args or list(bt_core.BACKTEST_SLATES)

    done = set() if FORCE else _done_slates(RESULTS_CSV)
    todo = [s for s in slates if s not in done]
    print(f"slates: {len(slates)} requested, {len(done)} done, {len(todo)} to run")
    print(f"n_sims={N_SIMS} seed={SEED} field={FIELD_SIZE} lambdas={LAMBDAS}\n")

    for slate in todo:
        t0 = time.time()
        try:
            rows = run_slate(PROJECT_ROOT / "archive" / slate, slate)
        except SystemExit as exc:
            print(f"{slate}: SKIPPED -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{slate}: FAILED -- {type(exc).__name__}: {exc}")
            continue
        _append_and_reload(RESULTS_CSV, slate, rows)
        r = rows[0]
        print(f"{slate}: rho_own={r['rho_sigma_vs_own']:+.3f} "
              f"rho_proj={r['rho_sigma_vs_proj']:+.3f} "
              f"new={r['is_new_signal']} decision={r['decision']} "
              f"({time.time() - t0:.0f}s)")

    if RESULTS_CSV.exists():
        report(pd.read_csv(RESULTS_CSV, dtype={"slate": str}))
    return 0


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("sigma_dG STRUCTURE -- gates, not profitability")
    print("=" * 78)
    print(f"slates: {len(df)}")
    print(f"\nQ1 IS IT NEW?  rho vs ownership  mean {df['rho_sigma_vs_own'].mean():+.3f} "
          f"(|max| {df['rho_sigma_vs_own'].abs().max():.3f})")
    print(f"               rho vs proj_score mean {df['rho_sigma_vs_proj'].mean():+.3f} "
          f"(|max| {df['rho_sigma_vs_proj'].abs().max():.3f})")
    n_new = int(df["is_new_signal"].sum())
    print(f"               distinct on {n_new}/{len(df)} slates"
          + ("" if n_new == len(df) else "   <-- NOT distinct everywhere"))

    print(f"\nQ2 POOL COVERAGE: " + ", ".join(
        f"{k}={v}" for k, v in df["decision"].value_counts().items()))
    if "frontier_pct_in_pool" in df:
        print(f"   frontier's most extreme lineup sits at pool percentile "
              f"{df['frontier_pct_in_pool'].mean():.3f}% (mean)")
    if (df["decision"] == "pool-already-spans").all():
        print("   -> SKIP generation. The pool reaches everywhere the frontier does,")
        print("      and the repo's prior negative on augmentation stands.")

    if "a52_median_rel_spread" in df:
        print(f"\nQ3 ASSUMPTION 5.2: median relative spread across tiers "
              f"{df['a52_median_rel_spread'].mean():.4f}, "
              f"min column corr {df['a52_min_col_corr_vs_mean'].min():.4f}")
        print("   (small spread + corr near 1 => the (P,T) block collapses to one vector)")


if __name__ == "__main__":
    raise SystemExit(main())
