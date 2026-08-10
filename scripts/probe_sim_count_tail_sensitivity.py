"""Throwaway probe: is round_n_sims=2,000 enough to resolve rare-tail-event
ROI differences, or is it noise at the depth hit99.9 needs?

Motivation (see conversation / plan): self_play's hit99.9 came back near-zero
in the full backtest while hit99 roughly doubled production. Hypothesis: a
top-heavy DK payout curve's true top band (top ~10-30 of a 10k-30k field) is
a rare event -- if a candidate's true probability of landing there is on the
order of 1-in-2,000 or rarer, a 2,000-sim ROI estimate has only ~1 expected
"hit" to work with, which is noise, not signal. hit99 (~1-in-100) is fine at
n=2,000 (~20 expected hits); hit99.9 (~1-in-1,000 to 1-in-tens-of-thousands
depending on contest size) may not be.

Method: reuse the ALREADY-CACHED n_sims=25,000 Monte Carlo for 07222026 (no
new simulation needed) and the SAME base opponent pool + SAME fixed opponent
draw across sim-depth levels (only pool composition, not its scoring, is
n_sims-independent) -- so the ONLY thing that changes between comparisons is
how many of the 25,000 cached sim columns get used to compute ROI. This
isolates sim-count effects from RNG-draw effects.

Usage
-----
    source venv/bin/activate
    python scripts/probe_sim_count_tail_sensitivity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization import self_play  # noqa: E402
from src.optimization.gpp_portfolio import _build_payout_lookup, _payout_cumsum  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402
from tests.bt_core import build_slate_context, load_real_contests  # noqa: E402

SLATE = "07222026"
SEED = 42
SIM_LEVELS = [2_000, 5_000, 10_000, 25_000]  # nested subsets of the same 25k cache
BASE_POOL_SIZE = 30_000


def main() -> None:
    d = PROJECT_ROOT / "archive" / SLATE
    real = load_real_contests(d)
    sim_cache_dir = PROJECT_ROOT / "outputs" / "self_play_eval" / "sim_cache"

    t0 = time.time()
    ctx = build_slate_context(
        d, SEED, False, real, n_sims=25_000, sharpness=0.05,
        sim_cache_dir=sim_cache_dir, want_corr=False, want_pwin=False,
    )
    print(f"context build (cached): {time.time() - t0:.0f}s")
    full_sim = ctx["sim_results"]
    own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()

    # One fixed random permutation of sim indices -> each SIM_LEVELS[i] is a
    # strict superset of SIM_LEVELS[i-1]'s columns (mimics "simulate more",
    # not "simulate different").
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(full_sim.results_matrix.shape[0])

    # Base opponent pool: generated ONCE, reused at every sim-depth level --
    # pool composition (which players/lineups) doesn't depend on n_sims, only
    # scoring them does.
    t0 = time.time()
    base_pool = self_play.build_base_opponent_pool(
        ctx["players_df"], own_vec, BASE_POOL_SIZE, rng_seed=SEED,
    )
    print(f"base pool ({BASE_POOL_SIZE:,} lineups): {time.time() - t0:.0f}s")
    from src.optimization.lineup import Lineup
    generated_lineups = [Lineup(player_ids=[int(p) for p in row]) for row in base_pool]
    n_external = len(ctx["pool"].lineups)
    all_lineups = list(ctx["pool"].lineups) + generated_lineups
    source = np.array(["external"] * n_external + ["generated"] * len(generated_lineups))

    biggest = max([c for c in ctx["contests"] if c["k"] > 0], key=lambda c: c["n_field"])
    k = int([c for c in ctx["contests"] if c["contest_id"] == biggest["contest_id"]][0]["k"])
    n_field = biggest["n_field"]
    n_opponents = n_field - k
    print(f"\nbiggest contest: {biggest['contest']} n_field={n_field:,} k={k} "
          f"n_opponents={n_opponents:,}")

    # Fixed opponent draw (same specific opponent LINEUPS across every sim-depth
    # level -- only the depth of simulation used to SCORE them changes).
    opp_rng = np.random.default_rng(SEED + 1)
    opp_idx = opp_rng.choice(len(generated_lineups), size=n_opponents, replace=False) + n_external

    results_by_level: dict[int, pd.DataFrame] = {}
    for n_sims in SIM_LEVELS:
        idx = np.sort(perm[:n_sims])
        sub_sim = SimulationResults(full_sim.player_ids, full_sim.results_matrix[idx])
        t0 = time.time()
        scores = ep.compute_lineup_scores(all_lineups, sub_sim).astype(np.float32)
        opponents_scores = scores[opp_idx].T  # (n_sims, n_opponents)
        field_sorted = np.ascontiguousarray(np.sort(opponents_scores, axis=1))
        lookup = _build_payout_lookup(biggest["payout_arr"], N=field_sorted.shape[1], entry_fee=biggest["fee"])
        cumsum = _payout_cumsum(lookup)
        dilute = np.zeros_like(cumsum)

        # Score every EXTERNAL candidate (the pool self-play actually favored,
        # 72% of real picks) -- smaller universe than the full 34k, keeps this
        # probe fast, and it's the population the tail question is really about.
        cand_idx = np.arange(n_external)
        roi = self_play._score_against_field(scores[cand_idx], field_sorted, cumsum, dilute)
        dt = time.time() - t0
        print(f"  n_sims={n_sims:>6,}: scored {len(cand_idx):,} candidates in {dt:.1f}s")
        results_by_level[n_sims] = pd.DataFrame({"cand_idx": cand_idx, "roi": roi}).set_index("cand_idx")

    print("\n===== ranking stability across sim depth (Spearman rho, consecutive levels) =====")
    prev = None
    for n_sims in SIM_LEVELS:
        cur = results_by_level[n_sims]["roi"]
        if prev is not None:
            rho = prev.corr(cur, method="spearman")
            top50_prev = set(prev.sort_values(ascending=False).head(50).index)
            top50_cur = set(cur.sort_values(ascending=False).head(50).index)
            overlap = len(top50_prev & top50_cur)
            print(f"  vs previous level: rho={rho:.3f}  top-50 overlap={overlap}/50")
        prev = cur

    print("\n===== top-10 candidates at n_sims=2,000 vs their ROI at n_sims=25,000 =====")
    low = results_by_level[2_000]["roi"]
    high = results_by_level[25_000]["roi"]
    top10_low = low.sort_values(ascending=False).head(10)
    for idx, roi_2k in top10_low.items():
        rank_at_25k = int((high > high.loc[idx]).sum()) + 1
        print(f"  cand {idx:>6d}  roi@2k={roi_2k:8.3f}  roi@25k={high.loc[idx]:8.3f}  "
              f"rank@25k={rank_at_25k:>6d} (of {len(high):,})")

    print("\n===== top-10 candidates at n_sims=25,000 vs their ROI at n_sims=2,000 =====")
    top10_high = high.sort_values(ascending=False).head(10)
    for idx, roi_25k in top10_high.items():
        rank_at_2k = int((low > low.loc[idx]).sum()) + 1
        print(f"  cand {idx:>6d}  roi@25k={roi_25k:8.3f}  roi@2k={low.loc[idx]:8.3f}  "
              f"rank@2k={rank_at_2k:>6d} (of {len(low):,})")


if __name__ == "__main__":
    main()
