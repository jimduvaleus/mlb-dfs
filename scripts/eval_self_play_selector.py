"""Evaluate the self-play portfolio-construction algorithm (Phase 1 prototype,
src/optimization/self_play.py + src/api/external_pool.py::self_play_allocate_contests)
against production's real-money `allocate_contests(ev_type="p_win")` baseline,
using the same real-payout-table / real-field grading every other selector
experiment in this codebase has been judged by (tests/bt_core.py).

See /home/jduvaleus/.claude/plans/reactive-yawning-cookie.md for the full
design writeup.

Usage
-----
    source venv/bin/activate
    python scripts/eval_self_play_selector.py [slate ...]

Pause/resume: each slate's rows are appended to outputs/self_play_eval/results.csv
(and round_log.csv / refinement_log.csv) as soon as that slate finishes -- mirrors
tests/backtest.py's _append_and_reload. A slate already present in results.csv is
skipped on the next invocation (set SELF_PLAY_FORCE=1 to redo it anyway), so a long
multi-slate run can be interrupted (Ctrl-C, a kill, a crash) and resumed later with
the same command line; the printed summary always reflects every slate on disk, not
just the ones processed in the current invocation.

Long-running and typically redirected to a log file -- stdout is forced to
line-buffering (see sys.stdout.reconfigure below) so `tail -f` on that log shows
progress live instead of nothing until the process exits (file redirection makes
Python fall back to full block buffering by default).

Env vars (mirroring tests/backtest.py's conventions)
-----------------------------------------------------
    BT_NSIMS              n_sims for the base Monte Carlo sim (default: config.yaml
                          simulation.n_sims) -- NOT the self-play round loop's sim
                          count, see SELF_PLAY_ROUND_NSIMS below.
    SELF_PLAY_POOL_SIZE   base opponent pool size per slate (default: 30_000,
                          see scripts/profile_self_play_costs.py)
    SELF_PLAY_ROUND_NSIMS sims subsampled for the round loop itself (default: 2_000,
                          see self_play._ROUND_N_SIMS_DEFAULT -- production grading
                          uses real historical outcomes regardless, so this only
                          controls selection-time ranking noise, not grading fidelity)
    SELF_PLAY_PRECISE_NSIMS  sims used for the post-round-loop precision-refinement
                          pass on contests with n_field >= _REFINEMENT_MIN_FIELD_SIZE
                          (default: 20_000, see self_play._PRECISE_N_SIMS_DEFAULT and
                          the module's PRECISION REFINEMENT note). Set to 0 to disable
                          refinement entirely (build_self_play_context gets
                          precise_n_sims=None).
    SELF_PLAY_REFRESH_EVERY  redraw the opponent subsample every N rounds instead of
                          every round (default: 5, see run_contest_self_play)
    SELF_PLAY_SEED        rng seed (default: 42)
    SELF_PLAY_FORCE       "1" re-runs slates already present in results.csv

DK stops serving contest standings ~10 days out, so the archive only grows
forward -- BACKTEST_SLATES is a source-committed snapshot updated at
milestones and can be stale; a slate directory existing under archive/ is
the source of truth for what's currently gradeable (see the slate-list note
in the approved plan).
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
from src.optimization import self_play  # noqa: E402
from tests.bt_core import (  # noqa: E402
    LIVE_CFG, _FakeGroup, build_slate_context, grade_portfolio,
    load_real_contests, prod_order, verify_slate,
)

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
POOL_SIZE = int(os.environ.get("SELF_PLAY_POOL_SIZE", self_play._SELF_PLAY_POOL_CAP))
ROUND_N_SIMS = int(os.environ.get("SELF_PLAY_ROUND_NSIMS", self_play._ROUND_N_SIMS_DEFAULT))
_precise_env = os.environ.get("SELF_PLAY_PRECISE_NSIMS")
PRECISE_N_SIMS = (
    None if _precise_env == "0" else int(_precise_env) if _precise_env is not None
    else self_play._PRECISE_N_SIMS_DEFAULT
)
REFRESH_EVERY = int(os.environ.get("SELF_PLAY_REFRESH_EVERY", "5"))
SEED = int(os.environ.get("SELF_PLAY_SEED", "42"))
FORCE = os.environ.get("SELF_PLAY_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "self_play_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
ROUND_LOG_CSV = OUT_DIR / "round_log.csv"
REFINEMENT_LOG_CSV = OUT_DIR / "refinement_log.csv"

# Flat-2000 production baseline (see project memory: made permanent 2026-07-31).
PWIN_ADMIT_FLOOR, PWIN_ADMIT_MULT, PWIN_EVW = 2000, 0.0, 0.25


def _append_and_reload(csv_path: Path, slate: str, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Replace `slate`'s rows in csv_path with new_rows and return the full
    accumulated table read back from disk -- so an interrupted run still
    leaves every completed slate's results on disk and gradeable, and a
    later invocation's summary reflects the whole archive processed so far,
    not just this process's slates. Mirrors tests/backtest.py's helper of
    the same name/purpose, extended to DROP any existing rows for `slate`
    first (not just append) -- SELF_PLAY_FORCE reprocessing an already-done
    slate must overwrite its old rows, not duplicate them alongside the new
    ones; new_rows can legitimately be empty (e.g. a slate with no logged
    rounds), so `slate` is taken explicitly rather than inferred from it."""
    df = new_rows.copy()
    if not df.empty:
        df["slate"] = df["slate"].astype(str)
    if csv_path.exists():
        # dtype pin is load-bearing: slate names are all-digit (07222026),
        # so an untyped read_csv infers int64 and silently drops the
        # leading zero -- breaking later string comparisons against slates.
        old = pd.read_csv(csv_path, dtype={"slate": str})
        old = old[old["slate"] != slate]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def _done_slates() -> set:
    if FORCE or not RESULTS_CSV.exists():
        return set()
    return set(pd.read_csv(RESULTS_CSV, dtype={"slate": str})["slate"].unique())


def _stage_metrics(gross: np.ndarray, rank: np.ndarray, fee: float, n_field: int) -> dict:
    finite = np.isfinite(gross)
    g, r = gross[finite], rank[finite]
    n = len(g)
    if n == 0:
        return dict(n=0, hit99=np.nan, hit999=np.nan, cash_rate=np.nan, mean_net=np.nan)
    net = g - fee
    return dict(
        n=n,
        hit99=float(np.mean(r <= max(1, round(n_field * 0.01)))),
        hit999=float(np.mean(r <= max(1, round(n_field * 0.001)))),
        cash_rate=float(np.mean(g > 0)),
        mean_net=float(np.mean(net)),
    )


def run_slate(slate: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per_entry rows, round_log rows, refinement_log rows) for this slate's two arms."""
    slate_t0 = time.time()
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    raw = pd.read_csv(d / "DKSalaries.csv")
    nm = raw[["ID", "Name"]].astype({"ID": str})
    fpts = verify_slate(d, real, nm)
    print(f"{slate}: verified against realized FPTS")

    t0 = time.time()
    ctx = build_slate_context(
        d, SEED, False, real, n_sims=N_SIMS, sharpness=SHARPNESS,
        sim_cache_dir=SIM_CACHE_DIR,
    )
    print(f"  context built in {time.time() - t0:.0f}s")

    own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()
    t0 = time.time()
    sp_ctx = self_play.build_self_play_context(
        ctx["sim_results"], ctx["players_df"], own_vec, ctx["pool"],
        base_pool_size=POOL_SIZE, base_pool_seed=SEED,
        round_n_sims=ROUND_N_SIMS, round_sims_seed=SEED,
        precise_n_sims=PRECISE_N_SIMS,
    )
    print(f"  self-play base pool ({POOL_SIZE:,} lineups) built in {time.time() - t0:.0f}s "
          f"(precise_n_sims={sp_ctx.precise_n_sims}, promoted={len(sp_ctx.promoted_idx)})")

    contests = [c for c in ctx["contests"] if c["k"] > 0]
    prize_pool = {c["contest_id"]: float(c["payout_arr"].sum()) for c in contests}
    order = prod_order([c["contest_id"] for c in contests], [c["fee"] for c in contests], prize_pool)
    ordered = [contests[i] for i in order]

    def _sp_progress(info: dict) -> None:
        print(f"    [{info['contest_id']}] k={info['k']} n_field={info['n_field']:,} "
              f"rounds={info['n_rounds']} swaps={info.get('n_swaps', 0)} "
              f"{info['elapsed_s']:.1f}s (round={info.get('round_elapsed_s', float('nan')):.1f}s "
              f"refine={info.get('refine_elapsed_s', float('nan')):.1f}s)")

    t0 = time.time()
    alloc_sp = ep.self_play_allocate_contests(
        ordered, sp_ctx, rng_seed=SEED, refresh_every=REFRESH_EVERY, progress_cb=_sp_progress,
    )
    print(f"  self-play allocation done in {time.time() - t0:.0f}s "
          f"({len(alloc_sp.portfolio)} entries, {len(alloc_sp.unfilled)} unfilled)")

    t0 = time.time()
    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ordered]
    alloc_pw = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0,
        evw_base=PWIN_EVW, evw_max=PWIN_EVW, ev_type="p_win",
        p_win_cull=ctx["p_win_cull"], p_win_select=ctx["p_win_select"],
        p_win_admit_n=PWIN_ADMIT_FLOOR, p_win_admit_multiplier=PWIN_ADMIT_MULT,
    )
    print(f"  p_win baseline allocation done in {time.time() - t0:.0f}s "
          f"({len(alloc_pw.portfolio)} entries, {len(alloc_pw.unfilled)} unfilled)")

    t0 = time.time()
    # actual FPTS lookup covers BOTH external pool lineups and generated
    # base-pool lineups (self-play can poach the latter into its portfolio;
    # the p_win baseline only ever picks from ctx["pool"].lineups, a subset
    # of sp_ctx.lineups with identical object identity -- see build_self_play_context).
    actual_of: dict[int, float] = {}
    for lu in sp_ctx.lineups:
        actual_of[id(lu)] = sum(fpts.get(int(p), float("nan")) for p in lu.player_ids)

    # proj_fpts (sum of players_df's own projected means, ep.compute_pool_proj_scores)
    # is a snapshot-time value, computed identically for external and generated
    # lineups from the SAME projections file -- unlike actual FPTS/gross $/rank,
    # it isn't contaminated by which lineups happened to roster a since-scratched
    # player, so it's a confound-free complementary check on whether the self-play
    # selection mechanism prefers higher-EV candidates, independent of real-outcome
    # luck. See SelfPlaySlateContext's PRE-LOCK STALENESS note for why actual-
    # outcome comparisons (this row's gross/rank) don't have that property.
    proj_scores = ep.compute_pool_proj_scores(sp_ctx.lineups, ctx["players_df"])
    proj_of: dict[int, float] = {id(lu): float(p) for lu, p in zip(sp_ctx.lineups, proj_scores)}

    def _entries_for(entry_plan, portfolio, source):
        by_contest: dict[str, list[int]] = {}
        for j, (cid, _e) in enumerate(entry_plan):
            by_contest.setdefault(cid, []).append(j)
        rows = []
        for c in ordered:
            idxs = by_contest.get(c["contest_id"], [])
            if not idxs:
                continue
            scores = np.array([actual_of[id(portfolio[j][0])] for j in idxs])
            gross, rank = grade_portfolio(scores, c["sorted_scores"], c["payout_arr"])
            for j_local, j in enumerate(idxs):
                rows.append(dict(
                    slate=slate, contest_id=c["contest_id"], fee=c["fee"], n_field=c["n_field"],
                    gross=gross[j_local], rank=rank[j_local],
                    proj_fpts=proj_of[id(portfolio[j][0])],
                    source=(source[j] if source else "external"),
                ))
        return rows

    rows = []
    rows += [{**r, "arm": "self_play"} for r in
              _entries_for(alloc_sp.entry_plan, alloc_sp.portfolio, alloc_sp.source)]
    rows += [{**r, "arm": "p_win"} for r in
              _entries_for(alloc_pw.entry_plan, alloc_pw.portfolio, None)]
    print(f"  grading done in {time.time() - t0:.0f}s ({len(rows)} graded entries)")

    round_log = alloc_sp.round_log.copy()
    if not round_log.empty:
        round_log["slate"] = slate

    # Enrich the swap log with what actually happened to both lineups in
    # reality (actual_fpts) and at the frozen-snapshot level (proj_fpts) --
    # "does this swap look like something we believe" needs both: proj_fpts
    # confirms the SELECTION step reasoned correctly at swap time, actual_fpts
    # shows how it turned out (subject to the same pre-lock-staleness/scratch
    # luck every other real-outcome number in this file carries).
    refinement_log = alloc_sp.refinement_log.copy()
    if not refinement_log.empty:
        refinement_log["slate"] = slate
        for prefix in ("swapped_out", "swapped_in"):
            lus = [sp_ctx.lineups[int(i)] for i in refinement_log[f"{prefix}_idx"]]
            refinement_log[f"{prefix}_actual_fpts"] = [actual_of[id(lu)] for lu in lus]
            refinement_log[f"{prefix}_proj_fpts"] = [proj_of[id(lu)] for lu in lus]

    print(f"  {slate}: total slate time {time.time() - slate_t0:.0f}s")
    return pd.DataFrame(rows), round_log, refinement_log


def print_summary(combined: pd.DataFrame, round_logs: pd.DataFrame, refinement_logs: pd.DataFrame) -> None:
    if combined.empty:
        print("\n[summary] no results")
        return

    print("\n===== pooled per-entry metrics (self_play vs p_win) =====")
    for arm, g in combined.groupby("arm"):
        fees = g["fee"].sum()
        won = g["gross"].sum()
        print(f"  {arm:>10s}  n={len(g):5d}  cash_rate={100*np.mean(g['gross']>0):5.1f}%  "
              f"net_$={won - fees:9.2f}  roi={100*(won-fees)/fees:6.1f}%")

    print("\n===== pooled per-entry metrics, split by source (self_play only) =====")
    print("  CAVEAT: both external AND generated are built from this slate's frozen pre-lock")
    print("  snapshot (~1s apart -- neither has an edge over the other), so a gap between them")
    print("  here isn't pool-quality signal. Both are handicapped the same way vs. the REAL")
    print("  field, whose entrants could make post-lock swaps neither pool reflects. See")
    print("  SelfPlaySlateContext's PRE-LOCK STALENESS note.")
    sp = combined[combined["arm"] == "self_play"]
    for source, g in sp.groupby("source"):
        fees, won = g["fee"].sum(), g["gross"].sum()
        print(f"  {source:>10s}  n={len(g):5d}  cash_rate={100*np.mean(g['gross']>0):5.1f}%  "
              f"net_$={won - fees:9.2f}  roi={100*(won-fees)/fees:6.1f}%")

    if "proj_fpts" in sp.columns:
        print("\n===== proj_fpts by source (self_play only) -- confound-free: same snapshot, =====")
        print("===== not contaminated by post-hoc scratch luck (see PRE-LOCK STALENESS)  =====")
        for source, g in sp.groupby("source"):
            print(f"  {source:>10s}  n={len(g):5d}  mean_proj_fpts={g['proj_fpts'].mean():6.2f}  "
                  f"median={g['proj_fpts'].median():6.2f}")

    print("\n===== hit99 / hit99.9 (self_play vs p_win) =====")
    for arm, g in combined.groupby("arm"):
        by_contest = []
        for cid, gc in g.groupby("contest_id"):
            n_field = int(gc["n_field"].iloc[0])
            by_contest.append(_stage_metrics(gc["gross"].to_numpy(), gc["rank"].to_numpy(),
                                              fee=gc["fee"].iloc[0], n_field=n_field))
        bc = pd.DataFrame(by_contest)
        print(f"  {arm:>10s}  mean_hit99={100*bc['hit99'].mean():5.2f}%  "
              f"mean_hit999={100*bc['hit999'].mean():5.2f}%")

    if not round_logs.empty:
        print(f"\n===== round-loop operational log (n={len(round_logs)} logged rounds) =====")
        print(f"  full rescores (shortlist exhausted mid-contest): "
              f"{int(round_logs['full_rescore'].sum())} / {len(round_logs)} rounds "
              f"({100*round_logs['full_rescore'].mean():.2f}%)")
        print("  n_scored percentiles (candidates re-scored per round, shortlist-restricted "
              "rounds should sit near shortlist_size):")
        print(round_logs["n_scored"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())

    if refinement_logs.empty:
        print("\n===== precision refinement: no swaps made =====")
    else:
        rl = refinement_logs
        print(f"\n===== precision refinement (n={len(rl)} swaps across "
              f"{rl['contest_id'].nunique()} contests) =====")
        print("  swap direction (source -> source):")
        print(rl.groupby(["swapped_out_source", "swapped_in_source"]).size()
              .rename("n").to_string())
        # "Do we believe this" check: did the swapped-in lineup actually score
        # higher in reality, not just by the precise ROI estimate that picked
        # it? (subject to the same pre-lock-staleness caveat as any other
        # real-outcome number here -- see PRE-LOCK STALENESS).
        actual_delta = rl["swapped_in_actual_fpts"] - rl["swapped_out_actual_fpts"]
        proj_delta = rl["swapped_in_proj_fpts"] - rl["swapped_out_proj_fpts"]
        print(f"  swapped-in beat swapped-out on ACTUAL fpts: "
              f"{100*(actual_delta > 0).mean():.1f}% of swaps (mean delta {actual_delta.mean():+.2f})")
        print(f"  swapped-in beat swapped-out on PROJECTED fpts (confound-free): "
              f"{100*(proj_delta > 0).mean():.1f}% of swaps (mean delta {proj_delta.mean():+.2f})")
        print(f"\nwrote {REFINEMENT_LOG_CSV}")


def main() -> None:
    slates = [s for s in sys.argv[1:] if s.isdigit()]
    if not slates:
        raise SystemExit(
            "usage: python scripts/eval_self_play_selector.py <slate MMDDYYYY> [<slate> ...]\n"
            "(pass every currently-archived slate explicitly -- see the slate-list "
            "note in the approved plan; BACKTEST_SLATES can be stale)"
        )

    done = _done_slates()
    combined = pd.read_csv(RESULTS_CSV, dtype={"slate": str}) if RESULTS_CSV.exists() else pd.DataFrame()
    round_logs = pd.read_csv(ROUND_LOG_CSV, dtype={"slate": str}) if ROUND_LOG_CSV.exists() else pd.DataFrame()
    refinement_logs = (
        pd.read_csv(REFINEMENT_LOG_CSV, dtype={"slate": str}) if REFINEMENT_LOG_CSV.exists() else pd.DataFrame()
    )

    for slate in slates:
        if slate in done:
            print(f"{slate}: already in {RESULTS_CSV.name}, skipping (SELF_PLAY_FORCE=1 to redo)")
            continue
        d = PROJECT_ROOT / "archive" / slate
        if not d.exists():
            print(f"{slate}: no archive dir, skipping")
            continue
        df, log, refine_log = run_slate(slate)
        combined = _append_and_reload(RESULTS_CSV, slate, df)
        round_logs = _append_and_reload(ROUND_LOG_CSV, slate, log)
        refinement_logs = _append_and_reload(REFINEMENT_LOG_CSV, slate, refine_log)
        print(f"  wrote {RESULTS_CSV.name} ({len(combined)} rows accumulated)")

    print_summary(combined, round_logs, refinement_logs)


if __name__ == "__main__":
    main()
