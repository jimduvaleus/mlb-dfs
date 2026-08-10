"""Throwaway single-slate memory profile of self_play's live-pipeline
construction path (build_self_play_context + self_play_allocate_contests),
motivated by this session's OOM kill of the offline multi-slate backtest
loop (kernel-confirmed via dmesg, RSS reached 15GB) -- see the self_play
production-wiring plan's "Memory safety pre-flight" section. A live server
run only builds ONE slate per invocation (unlike that batch script's
cross-slate loop, where slate N+1's arrays are built before slate N's are
necessarily released), so this isolates that single-slate case's peak RSS
rather than assuming the batch-script number transfers directly.

Usage
-----
    source venv/bin/activate
    python scripts/profile_self_play_memory.py [slate]
"""
import resource
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization import self_play  # noqa: E402
from tests.bt_core import LIVE_CFG, build_slate_context, load_real_contests, prod_order  # noqa: E402


def rss_mb() -> float:
    # ru_maxrss is peak RSS so far for the whole process, in KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    slate = sys.argv[1] if len(sys.argv) > 1 else "07222026"
    print(f"baseline RSS: {rss_mb():.0f} MB")

    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    t0 = time.time()
    ctx = build_slate_context(
        d, 42, False, real, n_sims=int(LIVE_CFG["simulation"]["n_sims"]),
        sharpness=0.05,
        sim_cache_dir=PROJECT_ROOT / "outputs" / "self_play_eval" / "sim_cache",
    )
    print(f"after build_slate_context: {rss_mb():.0f} MB ({time.time() - t0:.0f}s)")

    own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()
    t0 = time.time()
    sp_ctx = self_play.build_self_play_context(
        ctx["sim_results"], ctx["players_df"], own_vec, ctx["pool"],
    )
    print(f"after build_self_play_context: {rss_mb():.0f} MB ({time.time() - t0:.0f}s)")

    contests = [c for c in ctx["contests"] if c["k"] > 0]
    prize_pool = {c["contest_id"]: float(c["payout_arr"].sum()) for c in contests}
    order = prod_order(
        [c["contest_id"] for c in contests], [c["fee"] for c in contests], prize_pool,
    )
    ordered = [contests[i] for i in order]

    t0 = time.time()
    alloc = ep.self_play_allocate_contests(ordered, sp_ctx, rng_seed=42)
    print(
        f"after self_play_allocate_contests: {rss_mb():.0f} MB ({time.time() - t0:.0f}s) "
        f"({len(alloc.portfolio)} entries, {len(alloc.unfilled)} unfilled)"
    )

    print(f"PEAK RSS for this single-slate run: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
