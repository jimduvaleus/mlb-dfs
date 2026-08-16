"""Production-scale memory/timing smoke test for the topn_coverage external-
pool allocator (src/api/external_pool.py::allocate_contests_topn_coverage),
per CLAUDE.md's "any script that runs for more than a few seconds ... must
checkpoint" rule and the user's explicit ask for this feature: build a
realistic (n_sims, field_pool, candidate_pool) at production scale and
confirm peak RSS stays comfortably under the ~10GB budget the user set.

CORRECTNESS NOTE (2026-08-09): the first version of this script scored a
synthetic field pool with pure i.i.d. Gaussian noise instead of the real
production field generator. This version uses a real archived slate
throughout: a real players_df/simulation via tests/bt_core.py's
build_slate_context, the REAL field generator
(src.api.external_pool.build_topn_field_pool -> ContestSimulator.
generate_field, the same ownership-weighted stacked-lineup sampler
production uses), and the slate's own real external candidate pool and real
per-contest field sizes -- exactly what production's pipeline.py topn_
coverage branch builds and allocates over.

AUTO-SIZED n_sims (2026-08-09): pipeline.py's topn_coverage branch now sizes
n_sims to the SUM of every contest's own n_sims_g (ep.topn_total_sims_needed),
not a flat default, so _SimWorldAllocator can hand every contest a fully
disjoint sim-world set with zero wraps (see the "truly independent" request
that motivated _SimWorldAllocator in the first place). This script mirrors
that: a cheap throwaway build_slate_context call at a tiny n_sims first
(just to read real per-contest field sizes off the slate's own zips, no real
simulation needed for that), computes the real total demand from those field
sizes, then re-builds at that (larger, uncached -- this is the actual point
of the stress test) size. Reports each contest's sim_lap/sim_lap_used_pct at
the end to confirm the auto-sizing actually avoided wraps, not just that it
ran.

ON-DEMAND SCORING (2026-08-09): auto-sizing alone measured peak RSS at
~14.6GB on this slate (a pre-scored (n_sims, field_pool_size) matrix alone
was ~6.8GB at ~68,000 auto-sized sims) -- over the 10GB budget.
allocate_contests_topn_coverage no longer accepts pre-scored pool_scores/
field_pool_scores matrices; it now takes sim_results + raw field_lineups
directly and scores both the candidate pool and each field-sample draw ON
DEMAND, per contest, restricted to that contest's own small sim-world/
field-size slice, freed again once that contest's picks are done (see the
function's docstring). This script mirrors that -- no more score_field
pre-scoring step at all.

REALISTIC CANDIDATE POOL SIZE (2026-08-09): this archived slate's real
external pool is only 4,049 lineups after near-duplicate removal -- smaller
than a typical live run (~8,000, per a real production progress-panel line:
"7,919 lineups imported ... 2,218 near-duplicates removed"). Padded up to
TARGET_POOL_SIZE with additional lineups from the same real stacked-lineup
generator (a second, independent build_topn_field_pool draw), not just the
real 4,049, so the candidate-scoring cost is representative too.

Checkpoint / resume
--------------------
Each contest's timing/RSS numbers are appended to
outputs/topn_coverage_stress/results.csv immediately after that contest's
allocate_contests_topn_coverage call returns -- mirrors tests/backtest_lab.py
/ scripts/eval_self_play_selector.py's `_append_and_reload` pattern. A
contest already present in results.csv is skipped on the next invocation, so
an interrupted run (Ctrl-C, kill, OOM) resumes instead of restarting; the
final summary always reflects every contest on disk, not just the ones this
invocation processed. Set TOPN_STRESS_FORCE=1 to redo everything.

Usage
-----
    source venv/bin/activate
    python scripts/stress_test_topn_coverage_memory.py

Env vars
--------
    TOPN_STRESS_SLATE  archive dir name (default 07222026)
    TOPN_STRESS_FORCE  "1" re-runs contests already in results.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402
from tests.bt_core import build_slate_context, load_real_contests  # noqa: E402

SLATE = os.environ.get("TOPN_STRESS_SLATE", "07222026")
FORCE = os.environ.get("TOPN_STRESS_FORCE") == "1"
SEED = 42
FIELD_POOL_SIZE = 25_000  # external_pool_topn_field_pool_size default
MIN_N_SIMS = 25_000  # configured simulation.n_sims floor -- auto-sizing only grows from here
TARGET_POOL_SIZE = 8_000  # realistic live-run candidate pool size, see module docstring

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_coverage_stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def rss_mb() -> float:
    """CURRENT resident memory (not the monotonic-high-water-mark
    ru_maxrss) -- see stress_test_self_play_memory.py's rss_mb for why the
    distinction matters when checking whether memory is actually released
    between contests."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _append_and_reload(csv_path: Path, contest_id: str, row: dict) -> pd.DataFrame:
    """Replace `contest_id`'s row in csv_path with `row` and return the full
    accumulated table read back from disk -- mirrors scripts/
    eval_self_play_selector.py's helper of the same name/purpose."""
    df = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def _done_contests() -> set:
    if FORCE or not RESULTS_CSV.exists():
        return set()
    return set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"].unique())


class _StressGroup:
    """Minimal stand-in for ep.ContestGroup: allocate_contests_topn_coverage
    only reads .contest_id, .entries, and (via implied_field_size) .prize_
    pool_cents/.entry_fee_cents. We already know each real contest's true
    field size from ctx["contests"] (load_real_contests's own zip-derived
    n_field), so entry_fee/prize_pool are set to reproduce that exact size
    through implied_field_size rather than re-deriving it approximately."""
    def __init__(self, contest_id: str, n_field: int, k: int):
        self.contest_id = contest_id
        self.contest_name = contest_id
        self.entry_fee_cents = 100
        self.prize_pool_cents = int(round(n_field * 100 * (1 - ep._DK_RAKE)))
        self.single_entry_tag = False
        self.roi_key = ""
        self.roi_fallback = True
        self.entries = [(Path("x/Entries.csv"), _make_entry(contest_id, f"e{i}")) for i in range(k)]


def _make_entry(contest_id: str, entry_id: str):
    from src.api.dk_entries import EntryRecord
    return EntryRecord(
        entry_id=entry_id, contest_name=contest_id, contest_id=contest_id,
        entry_fee_cents=100, entry_fee_raw="$1", prize_pool_cents=None,
    )


def main() -> None:
    print(f"slate={SLATE} field_pool_size={FIELD_POOL_SIZE:,}")
    print(f"baseline RSS: {rss_mb():.0f} MB")

    d = PROJECT_ROOT / "archive" / SLATE
    real = load_real_contests(d)
    sim_cache_dir = PROJECT_ROOT / "outputs" / "self_play_eval" / "sim_cache"

    # Cheap throwaway preview: real per-contest field sizes come straight off
    # the slate's own archived zips (load_real_contests/build_slate_context's
    # display-size matching), not from anything that needs a real
    # simulation -- a tiny n_sims here just avoids a second code path.
    t0 = time.time()
    preview_ctx = build_slate_context(
        d, SEED, False, real, n_sims=100, sharpness=0.05,
        sim_cache_dir=sim_cache_dir, want_corr=False, want_pwin=False,
    )
    print(f"preview (real field sizes, throwaway n_sims=100): {time.time() - t0:.1f}s")
    from src.api.models import GppConfig
    _defaults = GppConfig()
    total_demand = 0
    for c in preview_ctx["contests"]:
        if c["k"] <= 0:
            continue
        n = ep._topn_sims_for_field_size(
            c["n_field"], 2**31 - 1, _defaults.external_pool_topn_sims_per_contest_fraction,
            _defaults.external_pool_topn_sims_min, _defaults.external_pool_topn_sims_reference_field_size,
            _defaults.external_pool_topn_sims_power,
        )
        total_demand += n
    n_sims = max(MIN_N_SIMS, total_demand)
    print(f"real total sim demand across {len(preview_ctx['contests'])} contests: "
          f"{total_demand:,} -> n_sims={n_sims:,} (configured floor {MIN_N_SIMS:,})")
    del preview_ctx

    t0 = time.time()
    ctx = build_slate_context(
        d, SEED, False, real, n_sims=n_sims, sharpness=0.05,
        sim_cache_dir=sim_cache_dir, want_corr=False, want_pwin=False,
    )
    print(f"context build (real n_sims, likely uncached): {time.time() - t0:.1f}s, RSS {rss_mb():.0f} MB")
    players_df, pool, sim_results = ctx["players_df"], ctx["pool"], ctx["sim_results"]
    own_vec = players_df["ownership"].astype(float).to_numpy()

    t0 = time.time()
    print(f"building real field pool ({FIELD_POOL_SIZE:,} lineups via the actual "
          f"stacked-lineup generator, NOT pre-scored)...")
    field_lineups = ep.build_topn_field_pool(players_df, own_vec, FIELD_POOL_SIZE, SEED)
    print(f"  {field_lineups.shape[0]:,} lineups generated, "
          f"{time.time() - t0:.1f}s, RSS {rss_mb():.0f} MB")

    if len(pool.lineups) < TARGET_POOL_SIZE:
        n_pad = TARGET_POOL_SIZE - len(pool.lineups)
        t0 = time.time()
        print(f"padding real {len(pool.lineups):,}-lineup candidate pool with "
              f"{n_pad:,} more (same generator, independent seed) to reach a "
              f"realistic ~{TARGET_POOL_SIZE:,}...")
        pad_rows = ep.build_topn_field_pool(players_df, own_vec, n_pad, SEED + 1)
        pad_lineups = [Lineup(player_ids=[int(p) for p in row]) for row in pad_rows]
        pool = ep.ExternalPool(
            lineups=pool.lineups + pad_lineups,
            contests=pool.contests, n_dropped_unknown_players=pool.n_dropped_unknown_players,
            n_dropped_duplicates=pool.n_dropped_duplicates,
            n_dropped_near_duplicates=pool.n_dropped_near_duplicates,
            source_paths=pool.source_paths,
        )
        print(f"  candidate pool now {len(pool.lineups):,} lineups, "
              f"{time.time() - t0:.1f}s, RSS {rss_mb():.0f} MB")

    contests = sorted(ctx["contests"], key=lambda c: -c["n_field"])
    print(f"contests={len(contests)}")
    groups = [_StressGroup(c["contest_id"], int(c["n_field"]), int(c["k"])) for c in contests]

    # allocate_contests_topn_coverage must be called ONCE across every
    # contest together -- the whole point being stress-tested is
    # _SimWorldAllocator's cross-contest disjoint state, which resets on
    # every call. That means this step isn't per-contest-resumable the way
    # the setup steps above are (their cost is already absorbed by
    # build_slate_context's sim cache + determinism, so a redo is cheap):
    # if any contest is missing from results.csv, the WHOLE allocation call
    # reruns and re-checkpoints every contest via progress_cb below, rather
    # than resuming mid-call.
    done = _done_contests()
    if len(done) >= len(contests):
        print(f"all {len(contests)} contests already in {RESULTS_CSV.name}, skipping allocation")
    else:
        peak_state = {"peak": rss_mb()}

        def _progress(info: dict) -> None:
            if info.get("event") != "contest_done":
                return
            cur_rss = rss_mb()
            peak_state["peak"] = max(peak_state["peak"], cur_rss)
            print(
                f"[{info['contest_id']}] filled={info['n_filled']}/{info['k']} "
                f"{info['elapsed_s']:.1f}s RSS {cur_rss:.0f} MB (peak {peak_state['peak']:.0f} MB) "
                f"-- sim lap {info['sim_lap']} ({info['sim_lap_used_pct']:.0f}% used, "
                f"{info['sim_total_taken']:,} total assigned)"
            )
            _append_and_reload(RESULTS_CSV, info["contest_id"], {
                "contest_id": info["contest_id"], "n_filled": info["n_filled"],
                "k": info["k"], "elapsed_s": info["elapsed_s"], "rss_mb": cur_rss,
                "sim_lap": info["sim_lap"], "sim_lap_used_pct": info["sim_lap_used_pct"],
                "sim_total_taken": info["sim_total_taken"],
            })

        t0 = time.time()
        ep.allocate_contests_topn_coverage(
            pool, sim_results, groups, field_lineups,
            topn_rank=_defaults.external_pool_topn_rank,
            topn_percentile_floor=_defaults.external_pool_topn_percentile_floor,
            field_samples=_defaults.external_pool_topn_field_samples,
            sims_per_contest_fraction=_defaults.external_pool_topn_sims_per_contest_fraction,
            sims_min=_defaults.external_pool_topn_sims_min,
            sims_reference_field_size=_defaults.external_pool_topn_sims_reference_field_size,
            sims_power=_defaults.external_pool_topn_sims_power,
            rng_seed=SEED, progress_cb=_progress,
        )
        print(f"allocation for all {len(contests)} contests: {time.time() - t0:.1f}s total")

    final = pd.read_csv(RESULTS_CSV, dtype={"contest_id": str}) if RESULTS_CSV.exists() else pd.DataFrame()
    disk_peak = final["rss_mb"].max() if not final.empty else float("nan")
    max_lap = int(final["sim_lap"].max()) if not final.empty else -1
    print(f"\n{len(final)}/{len(contests)} contests done on disk. Peak RSS across all: {disk_peak:.0f} MB")
    print(f"Max sim_lap reached: {max_lap} (0 means the auto-sizing avoided every wrap, as intended)")
    if disk_peak > 10_000:
        print("WARNING: peak RSS exceeded the 10GB budget.")


if __name__ == "__main__":
    main()
