"""Sweep self_play's round-loop sim count (round_n_sims) to see the actual
cost/benefit curve, motivated by 2026-08-08's mini-max result: the precision-
refinement pass spent 159-164s on a 72-round contest to make exactly 1 swap,
raising the question of whether that time budget would be better spent making
every ROUND-LOOP pick less noisy in the first place (round_n_sims currently
defaults to 2,000 -- see self_play._ROUND_N_SIMS_DEFAULT and the module's
SHORTLIST RESTRICTION note on why round_n_sims was cut that low originally:
tractability, not accuracy).

This does NOT touch outputs/self_play_eval/results.csv / round_log.csv /
refinement_log.csv (scripts/eval_self_play_selector.py's checkpointed
production-comparison state) -- those are keyed by slate only, and re-running
the same slate at a different round_n_sims here would silently clobber that
comparison. Writes to its own sweep_round_nsims.csv / sweep_round_nsims_contests.csv
instead, keyed by (slate, round_n_sims[, contest_id]).

Usage
-----
    source venv/bin/activate
    python scripts/sweep_round_nsims.py [slate ...]

Env vars
--------
    SWEEP_ROUND_NSIMS   comma-separated round_n_sims values to try
                        (default: "2000,5000,10000")
    BT_NSIMS, SELF_PLAY_POOL_SIZE, SELF_PLAY_PRECISE_NSIMS, SELF_PLAY_REFRESH_EVERY,
    SELF_PLAY_SEED      same meaning/defaults as scripts/eval_self_play_selector.py
    SWEEP_FORCE         "1" re-runs (slate, round_n_sims) combos already on disk
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
    LIVE_CFG, build_slate_context, grade_portfolio, load_real_contests,
    prod_order, verify_slate,
)

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
POOL_SIZE = int(os.environ.get("SELF_PLAY_POOL_SIZE", self_play._SELF_PLAY_POOL_CAP))
PRECISE_N_SIMS = int(os.environ.get("SELF_PLAY_PRECISE_NSIMS", self_play._PRECISE_N_SIMS_DEFAULT))
REFRESH_EVERY = int(os.environ.get("SELF_PLAY_REFRESH_EVERY", "5"))
SEED = int(os.environ.get("SELF_PLAY_SEED", "42"))
FORCE = os.environ.get("SWEEP_FORCE") == "1"
ROUND_NSIMS_VALUES = [int(x) for x in os.environ.get("SWEEP_ROUND_NSIMS", "2000,5000,10000").split(",")]

OUT_DIR = PROJECT_ROOT / "outputs" / "self_play_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_CSV = OUT_DIR / "sweep_round_nsims.csv"
SWEEP_CONTESTS_CSV = OUT_DIR / "sweep_round_nsims_contests.csv"


def _append_and_reload(csv_path: Path, key_cols: list, key_vals: tuple, new_rows: pd.DataFrame) -> pd.DataFrame:
    df = new_rows.copy()
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"slate": str})
        mask = np.ones(len(old), dtype=bool)
        for col, val in zip(key_cols, key_vals):
            mask &= (old[col] != val)
        old = old[mask]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def _done_combos() -> set:
    if FORCE or not SWEEP_CSV.exists():
        return set()
    df = pd.read_csv(SWEEP_CSV, dtype={"slate": str})
    return set(zip(df["slate"], df["round_n_sims"]))


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


def run_slate(slate: str) -> None:
    done = _done_combos()
    to_run = [n for n in ROUND_NSIMS_VALUES if (slate, n) not in done]
    if not to_run:
        print(f"{slate}: all round_n_sims values already done, skipping")
        return

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

    contests = [c for c in ctx["contests"] if c["k"] > 0]
    prize_pool = {c["contest_id"]: float(c["payout_arr"].sum()) for c in contests}
    order = prod_order([c["contest_id"] for c in contests], [c["fee"] for c in contests], prize_pool)
    ordered = [contests[i] for i in order]

    for round_n_sims in to_run:
        t_variant0 = time.time()
        print(f"  --- round_n_sims={round_n_sims} ---")

        t0 = time.time()
        sp_ctx = self_play.build_self_play_context(
            ctx["sim_results"], ctx["players_df"], own_vec, ctx["pool"],
            base_pool_size=POOL_SIZE, base_pool_seed=SEED,
            round_n_sims=round_n_sims, round_sims_seed=SEED,
            precise_n_sims=PRECISE_N_SIMS,
        )
        print(f"    self-play base pool ({POOL_SIZE:,} lineups) built in {time.time() - t0:.0f}s "
              f"(precise_n_sims={sp_ctx.precise_n_sims}, promoted={len(sp_ctx.promoted_idx)})")

        contest_rows = []

        def _sp_progress(info: dict) -> None:
            print(f"      [{info['contest_id']}] k={info['k']} n_field={info['n_field']:,} "
                  f"rounds={info['n_rounds']} swaps={info.get('n_swaps', 0)} "
                  f"{info['elapsed_s']:.1f}s (round={info.get('round_elapsed_s', float('nan')):.1f}s "
                  f"refine={info.get('refine_elapsed_s', float('nan')):.1f}s)")
            contest_rows.append(dict(
                slate=slate, round_n_sims=round_n_sims, contest_id=info["contest_id"],
                k=info["k"], n_field=info["n_field"], n_rounds=info["n_rounds"],
                n_swaps=info.get("n_swaps", 0),
                round_elapsed_s=info.get("round_elapsed_s", float("nan")),
                refine_elapsed_s=info.get("refine_elapsed_s", float("nan")),
            ))

        t0 = time.time()
        alloc_sp = ep.self_play_allocate_contests(
            ordered, sp_ctx, rng_seed=SEED, refresh_every=REFRESH_EVERY, progress_cb=_sp_progress,
        )
        alloc_elapsed = time.time() - t0
        print(f"    self-play allocation done in {alloc_elapsed:.0f}s "
              f"({len(alloc_sp.portfolio)} entries, {len(alloc_sp.unfilled)} unfilled)")

        actual_of: dict[int, float] = {}
        for lu in sp_ctx.lineups:
            actual_of[id(lu)] = sum(fpts.get(int(p), float("nan")) for p in lu.player_ids)

        by_contest: dict[str, list[int]] = {}
        for j, (cid, _e) in enumerate(alloc_sp.entry_plan):
            by_contest.setdefault(cid, []).append(j)

        all_gross, all_rank, all_fee, all_nfield = [], [], [], []
        for c in ordered:
            idxs = by_contest.get(c["contest_id"], [])
            if not idxs:
                continue
            scores = np.array([actual_of[id(alloc_sp.portfolio[j][0])] for j in idxs])
            gross, rank = grade_portfolio(scores, c["sorted_scores"], c["payout_arr"])
            all_gross.append(gross)
            all_rank.append(rank)
            all_fee.extend([c["fee"]] * len(idxs))
            all_nfield.extend([c["n_field"]] * len(idxs))

        gross = np.concatenate(all_gross)
        rank = np.concatenate(all_rank)
        fee = np.array(all_fee)
        nfield = np.array(all_nfield)
        finite = np.isfinite(gross)
        net = gross[finite] - fee[finite]
        hit99 = float(np.mean(rank[finite] <= np.maximum(1, np.round(nfield[finite] * 0.01))))
        hit999 = float(np.mean(rank[finite] <= np.maximum(1, np.round(nfield[finite] * 0.001))))

        cdf = pd.DataFrame(contest_rows)
        total_round_s = cdf["round_elapsed_s"].sum()
        total_refine_s = cdf["refine_elapsed_s"].sum()
        total_swaps = int(cdf["n_swaps"].sum())

        summary_row = pd.DataFrame([dict(
            slate=slate, round_n_sims=round_n_sims,
            n_entries=int(finite.sum()), hit99=hit99, hit999=hit999,
            mean_net=float(np.mean(net)), cash_rate=float(np.mean(gross[finite] > 0)),
            total_round_s=float(total_round_s), total_refine_s=float(total_refine_s),
            total_swaps=total_swaps, alloc_elapsed_s=float(alloc_elapsed),
            variant_elapsed_s=float(time.time() - t_variant0),
        )])
        _append_and_reload(SWEEP_CSV, ["slate", "round_n_sims"], (slate, round_n_sims), summary_row)
        _append_and_reload(
            SWEEP_CONTESTS_CSV, ["slate", "round_n_sims"], (slate, round_n_sims), cdf,
        )
        print(f"    round_n_sims={round_n_sims}: hit99={hit99:.4f} hit999={hit999:.4f} "
              f"mean_net={np.mean(net):.2f} total_round_s={total_round_s:.0f} "
              f"total_refine_s={total_refine_s:.0f} swaps={total_swaps} "
              f"variant_time={time.time() - t_variant0:.0f}s")


def main() -> None:
    slates = sys.argv[1:] or ["07222026"]
    for slate in slates:
        run_slate(slate)
    if SWEEP_CSV.exists():
        print("\n===== sweep summary =====")
        print(pd.read_csv(SWEEP_CSV).to_string(index=False))


if __name__ == "__main__":
    main()
