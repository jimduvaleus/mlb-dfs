"""Throwaway follow-up profiling script (self-play selector, retooling after
the first full-size eval run on 07222026 was killed after 18+ minutes with
process RSS ~11.5GB and system swap 90% full).

scripts/profile_self_play_costs.py only measured generate_field() (cheap,
one-time per slate) and _build_payout_lookup() (cheap, O(N) Python loop) --
it never measured the actual per-round cost at real n_sims=25,000 scale:
_merge_and_sort_field's np.concatenate+np.sort over a (n_sims, N) array
(N up to ~29k for the slate's biggest contest) and _score_against_field over
the FULL remaining candidate universe. This script measures exactly that,
once, directly -- reusing the sim cache the killed run already wrote to disk
so it doesn't re-pay the 134s context-build cost.

Usage
-----
    source venv/bin/activate
    python scripts/profile_self_play_round_cost.py [slate] [base_pool_size]
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization import self_play  # noqa: E402
from src.optimization.gpp_portfolio import _build_payout_lookup, _payout_cumsum  # noqa: E402
from tests.bt_core import LIVE_CFG, build_slate_context, load_real_contests  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux


def main() -> None:
    slate = sys.argv[1] if len(sys.argv) > 1 else "07222026"
    base_pool_size = int(sys.argv[2]) if len(sys.argv) > 2 else self_play._SELF_PLAY_POOL_CAP
    seed = 42
    n_sims = int(LIVE_CFG["simulation"]["n_sims"])

    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    sim_cache_dir = PROJECT_ROOT / "outputs" / "self_play_eval" / "sim_cache"

    print(f"{slate}: n_sims={n_sims:,}, base_pool_size={base_pool_size:,}")
    print(f"RSS at start: {rss_mb():.0f} MB")

    t0 = time.perf_counter()
    ctx = build_slate_context(d, seed, False, real, n_sims=n_sims, sharpness=0.05,
                              sim_cache_dir=sim_cache_dir, want_corr=False, want_pwin=False)
    print(f"context build (cached): {time.perf_counter() - t0:.0f}s, RSS: {rss_mb():.0f} MB")

    own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()
    t0 = time.perf_counter()
    sp_ctx = self_play.build_self_play_context(
        ctx["sim_results"], ctx["players_df"], own_vec, ctx["pool"],
        base_pool_size=base_pool_size, base_pool_seed=seed,
    )
    print(f"self-play context build: {time.perf_counter() - t0:.0f}s, RSS: {rss_mb():.0f} MB")
    print(f"  candidate universe: {len(sp_ctx.lineups):,} lineups "
          f"({sp_ctx.n_external:,} external + {len(sp_ctx.lineups) - sp_ctx.n_external:,} generated)")
    print(f"  ctx.scores: shape={sp_ctx.scores.shape}, {sp_ctx.scores.nbytes / 1e9:.2f} GB")

    biggest = max(real, key=lambda c: c["n_field"])
    n_field = biggest["n_field"]
    k = 100  # representative entry count for a cost profile, doesn't need to be exact
    n_opponents = min(max(n_field - k, 0), len(sp_ctx.lineups) - sp_ctx.n_external)
    print(f"\nbiggest contest: {biggest['contest']} n_field={n_field:,}, "
          f"profiling with n_opponents={n_opponents:,}")

    rng = np.random.default_rng(seed)
    opp_idx = np.flatnonzero(sp_ctx.new_opponent_mask())
    chosen = rng.choice(opp_idx, size=n_opponents, replace=False)

    t0 = time.perf_counter()
    opponents_scores = sp_ctx.scores[chosen].T
    own_scores = np.zeros((sp_ctx.n_sims, 0), dtype=np.float32)
    field_sorted = self_play._merge_and_sort_field(opponents_scores, own_scores)
    t1 = time.perf_counter()
    print(f"_merge_and_sort_field: {t1 - t0:.2f}s, "
          f"field_sorted shape={field_sorted.shape}, {field_sorted.nbytes / 1e9:.2f} GB")
    print(f"  RSS: {rss_mb():.0f} MB")

    lookup = _build_payout_lookup(biggest["payout_arr"], N=field_sorted.shape[1], entry_fee=biggest["fee"])
    cumsum = _payout_cumsum(lookup)
    dilute = np.zeros_like(cumsum)
    t2 = time.perf_counter()

    remaining_idx = np.arange(len(sp_ctx.lineups))
    roi = self_play._score_against_field(sp_ctx.scores[remaining_idx], field_sorted, cumsum, dilute)
    t3 = time.perf_counter()
    print(f"_score_against_field (full {len(remaining_idx):,}-lineup candidate universe, "
          f"first call incl. numba JIT compile): {t3 - t2:.2f}s")
    print(f"  RSS: {rss_mb():.0f} MB")

    # second call, JIT already compiled -- the number that actually matters for round 2+.
    t4 = time.perf_counter()
    roi = self_play._score_against_field(sp_ctx.scores[remaining_idx], field_sorted, cumsum, dilute)
    t5 = time.perf_counter()
    print(f"_score_against_field (repeat call, JIT warm): {t5 - t4:.2f}s")

    per_round = (t1 - t0) + (t5 - t4)
    print(f"\nper-round estimate (merge_sort + score, JIT-warm): {per_round:.2f}s")
    print(f"  x ~15 rounds/contest x {len(real)} contests on this slate "
          f"~ {per_round * 15 * len(real):.0f}s just for this cost, "
          f"before payout-lookup rebuild / bookkeeping / smaller contests")


if __name__ == "__main__":
    main()
