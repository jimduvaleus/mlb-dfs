"""Throwaway profiling script (self-play selector, Phase 1 step 0).

Measures the two costs the self-play round loop pays repeatedly:

  * ContestSimulator.generate_field() -- lineups/sec, to size the once-per-
    slate base opponent pool affordably (analogous to _PWIN_FIELD_CAP's
    25k-lineup memory bound in src/api/external_pool.py).
  * gpp_portfolio._build_payout_lookup() -- its _band_average helper is a
    pure-Python loop over N+1 bins, and the self-play design rebuilds this
    every round (N = len(opponents) + len(own_selections_so_far) grows each
    round), so its per-call cost x rounds-per-contest is a real budget item
    separate from the numba-parallel scoring kernel.

Usage
-----
    source venv/bin/activate
    python scripts/profile_self_play_costs.py [slate ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # flush per-line when redirected to a log file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.gpp_portfolio import _build_payout_lookup  # noqa: E402
from src.optimization.payout import load_payout_structure, payout_table_to_array  # noqa: E402
from tests.bt_core import build_slate_context, load_real_contests  # noqa: E402

DEFAULT_SLATES = ["07222026", "07292026"]


def profile_generate_field(players_df, own_vec, seed: int) -> None:
    cs = ContestSimulator()
    print("  generate_field:")
    for n in (5_000, 10_000, 20_000, 30_000):
        t0 = time.perf_counter()
        field = cs.generate_field(players_df, own_vec, n_lineups=n, rng_seed=seed)
        dt = time.perf_counter() - t0
        rate = len(field) / dt if dt > 0 else float("nan")
        print(f"    n={n:>6,}  got={len(field):>6,}  {dt:6.2f}s  {rate:8.0f} lineups/sec")


def profile_payout_lookup(payout_arr: np.ndarray) -> None:
    print("  _build_payout_lookup (rebuilt every self-play round):")
    for n in (100, 1_000, 5_000, 15_000, 29_411):
        t0 = time.perf_counter()
        _build_payout_lookup(payout_arr, N=n, entry_fee=4.0)
        dt = time.perf_counter() - t0
        # ~15 rounds/contest is the target from self_play_batch_size's default.
        print(f"    N={n:>6,}  {dt * 1000:7.2f}ms/call  -> ~{dt * 15 * 1000:8.1f}ms for 15 rounds")


def main() -> None:
    slates = [s for s in sys.argv[1:] if s.isdigit()] or DEFAULT_SLATES
    seed = 42
    sim_cache_dir = PROJECT_ROOT / "outputs" / "replay" / "sim_cache"
    sim_cache_dir.mkdir(parents=True, exist_ok=True)

    # Real DK's largest registered payout table (Rally Cap, 29,411 entries)
    # -- worst case for the per-round lookup rebuild.
    big_struct = load_payout_structure("dk_rally_cap_29411")
    profile_payout_lookup(payout_table_to_array(big_struct))

    for slate in slates:
        d = PROJECT_ROOT / "archive" / slate
        if not d.exists():
            print(f"\n{slate}: no archive dir, skipping")
            continue
        print(f"\n{slate}:")
        real = load_real_contests(d)
        t0 = time.perf_counter()
        ctx = build_slate_context(
            d, seed, False, real,
            n_sims=2000, sharpness=0.05, sim_cache_dir=sim_cache_dir,
            want_corr=False, want_pwin=False,
        )
        print(f"  context build (players_df/sims only): {time.perf_counter() - t0:.0f}s")
        own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()
        profile_generate_field(ctx["players_df"], own_vec, seed)


if __name__ == "__main__":
    main()
