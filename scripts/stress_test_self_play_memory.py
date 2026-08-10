"""Throwaway stress test for the malloc-tuning fix (self_play._tune_malloc_
for_large_arrays / _release_free_memory, 2026-08-08): simulates the exact
alloc/free PATTERN that made self_play_allocate_contests's RSS climb --
repeated large-but-differently-sized precise-tier score arrays, one per
contest, including sizes BIGGER than this session's real test slate's
largest contest (mini-max, ~23,781 implied entries) per the user's point
that bigger contests do occur. Uses synthetic Lineup/SimulationResults data
(no real slate needed) so this runs in seconds, not minutes.

Usage
-----
    source venv/bin/activate
    python scripts/stress_test_self_play_memory.py [--no-tune]
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.external_pool import compute_lineup_scores  # noqa: E402
from src.optimization import self_play  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

N_PLAYERS = 400
PRECISE_N_SIMS = 20_000
# Contest field sizes to simulate scoring opponents for, in the SAME mixed,
# non-monotonic order real contests get processed in (prod_order sorts by
# fee/prize pool, not by size) -- includes two sizes bigger than this
# session's real mini-max (23,781), per the user's "we play even bigger
# contests sometimes" point.
FIELD_SIZES = [476, 11574, 396, 992, 11905, 714, 2381, 5952, 794, 1190, 7143, 23781, 40000, 60000]


def rss_mb() -> float:
    # CURRENT resident memory, NOT resource.getrusage().ru_maxrss -- that
    # stat is a monotonic HIGH-WATER MARK that never decreases, so it can't
    # show whether a free actually returned memory (a first-draft version
    # of this script used ru_maxrss and consequently showed "no improvement"
    # even when the fix was working, because ru_maxrss just kept reporting
    # the same already-reached peak after every free regardless).
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-tune", action="store_true", help="skip the mmap-threshold fix, for comparison")
    args = p.parse_args()

    if not args.no_tune:
        self_play._tune_malloc_for_large_arrays()
        print("malloc tuning: ON")
    else:
        print("malloc tuning: OFF (baseline)")

    rng = np.random.default_rng(0)
    sim = SimulationResults(
        player_ids=list(range(N_PLAYERS)),
        results_matrix=rng.normal(8.0, 4.0, size=(PRECISE_N_SIMS, N_PLAYERS)).astype(np.float64),
    )
    max_field = max(FIELD_SIZES)
    all_lineups = [
        Lineup(player_ids=list(rng.choice(N_PLAYERS, size=10, replace=False)))
        for _ in range(max_field)
    ]

    peak = rss_mb()
    print(f"baseline RSS: {peak:.0f} MB")
    for field_size in FIELD_SIZES:
        lineups = all_lineups[:field_size]
        scores = compute_lineup_scores(lineups, sim).astype(np.float32)
        gb = scores.nbytes / 1e9
        peak = max(peak, rss_mb())
        del scores
        gc.collect()
        self_play._release_free_memory()
        after = rss_mb()
        peak = max(peak, after)
        print(f"  field_size={field_size:>6,}  array={gb:.2f}GB  RSS after free: {after:.0f} MB")

    print(f"PEAK RSS: {peak:.0f} MB, FINAL (current) RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
