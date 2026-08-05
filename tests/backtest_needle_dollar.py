"""$-denominated backtest of proj_top and stack_max (see backtest_needle.py
for their origin -- the needle-in-haystack framing: recovering a slate's own
top-10-real-score pool lineups) against real contest standings and payout
tables.

Companion to backtest_needle.py, NOT part of the protocol-governed arm set in
bt_core.ARMS / backtest_lab.build_arms(): per PROSPECTIVE_PROTOCOL.md's
mining control ("no new arm, currency, or hypothesis is run against the
archive without a dated EVIDENCE_LOG.md entry first"), proj_top/stack_max
haven't been pre-registered there, so this stays a separate file rather than
silently joining the standing arm set a gated adjudication run would pull in.

proj_top and stack_max are both field-agnostic by construction: same ranking
regardless of a contest's size, entry fee, or payout shape, unlike p_win
(field-size-scaled exponent) or roi (Saber's own field-aware simulated ROI).

Uses production's real per-contest fill order (bt_core.prod_order: entry fee
desc, prize pool asc, contest_id) rather than whichever order the oracle's
contest_id array happens to store contests in -- a shared candidate-removal
mask across contests means fill order can change who gets what.

    source venv/bin/activate
    python tests/backtest_needle_dollar.py                 # all Tier-1 slates, arms + evw sweep
    python tests/backtest_needle_dollar.py 07222026         # subset
    BT_EVW_GRID=1,0.8,0.5,0 python tests/backtest_needle_dollar.py
"""
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import prod_order  # noqa: E402
from tests.backtest_needle import TIER1_SLATES, load_pool_context, primary_stack_teams  # noqa: E402
from tests.backtest_lab import (  # noqa: E402
    A1_BASELINE, A1_NULL, _candidate_currencies, _prod_corr, grade_joint,
    load_field, load_slate, print_report, proj_floor_mask, report, select_greedy,
)

SEED = 42
CALIB = False
EVW_GRID = [float(x) for x in os.environ.get(
    "BT_EVW_GRID", "1.0,0.8,0.6,0.4,0.25,0.10,0.05,0.0").split(",")]
OUT_DIR = PROJECT_ROOT / "tests" / "backtest_output"


def select_stack_greedy(sd, sel, cull, stack_team, *, floor_pct=30.0, admit_n=2000,
                        order=None) -> dict:
    """Per-contest team-constrained greedy, mirroring select_greedy's exact
    cull/floor/shared-removal shape -- only the within-contest ranking
    differs: force distinct primary-team coverage first (by `sel`), falling
    back to plain `sel` ranking once every team represented in that
    contest's admit window has been used once. Per-contest sibling of
    backtest_needle.select_stack_diverse (which operates on one pooled
    synthetic budget instead of real per-contest sizes)."""
    mask = proj_floor_mask(sd, floor_pct)
    picks: dict = {}
    for ci in (order if order is not None else range(len(sd.cids))):
        kk = int(sd.k[ci])
        if kk <= 0:
            continue
        rem = np.where(mask & np.isfinite(sel[ci]))[0]
        if admit_n > 0 and len(rem) > admit_n:
            rem = np.sort(rem[np.argsort(-cull[ci][rem])[:admit_n]])
        kk = min(kk, len(rem))
        if kk == 0:
            continue
        order_local = rem[np.argsort(-sel[ci][rem])]
        teams = stack_team[order_local]
        chosen: list[int] = []
        used: set = set()
        leftover: list[int] = []
        for idx, team in zip(order_local, teams):
            if team not in used:
                chosen.append(int(idx))
                used.add(team)
            else:
                leftover.append(int(idx))
            if len(chosen) == kk:
                break
        if len(chosen) < kk:
            chosen.extend(leftover[:kk - len(chosen)])
        picks[sd.cids[ci]] = chosen
        mask[chosen] = False
    return picks


def fill_order_for(sd) -> list:
    field = load_field(sd.slate)
    prize_pool = {cid: float(field[cid][1].sum()) for cid in sd.cids}
    return prod_order(sd.cids, sd.fee, prize_pool)


def main() -> None:
    slates = [s for s in sys.argv[1:] if s.isdigit()] or TIER1_SLATES

    arm_frames = []
    evw_frames = []
    for slate in slates:
        sd = load_slate(slate, SEED, CALIB)
        curs = _candidate_currencies(sd)
        corr = _prod_corr(slate, SEED)
        fill_order = fill_order_for(sd)

        ctx = load_pool_context(slate, seed=SEED, calib=CALIB)
        stack_team = primary_stack_teams(ctx)
        proj = curs["proj_score"]

        null_rng = np.random.default_rng(
            zlib.crc32(f"{sd.slate}|{sd.seed}|{A1_NULL}".encode()) & 0xFFFFFFFF)
        picks_by_arm = {
            A1_BASELINE: select_greedy(sd, curs["p_win"], sd.currency("p_win", "A"),
                                       floor_pct=30.0, admit_n=2000, evw=0.25,
                                       corr=corr, order=fill_order),
            A1_NULL: select_greedy(sd, curs["p_win"], sd.currency("p_win", "A"),
                                   floor_pct=30.0, admit_n=0,
                                   rng=null_rng, order=fill_order),
            "proj_top": select_greedy(sd, proj, proj, floor_pct=30.0, admit_n=2000,
                                      order=fill_order),
            "stack_max": select_stack_greedy(sd, proj, proj, stack_team,
                                             floor_pct=30.0, admit_n=2000,
                                             order=fill_order),
        }
        for arm, picks in picks_by_arm.items():
            arm_frames.append(grade_joint(sd, picks, arm))

        for evw in EVW_GRID:
            picks = select_greedy(sd, proj, proj, floor_pct=30.0, admit_n=2000,
                                  evw=evw, corr=corr, order=fill_order)
            evw_frames.append(grade_joint(sd, picks, f"proj_top@evw{evw:g}"))

        print(f"  {slate} done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arm_df = pd.concat(arm_frames, ignore_index=True)
    arm_out = OUT_DIR / "needle_dollar_arms.csv"
    arm_df.to_csv(arm_out, index=False)
    res = report(arm_df, baseline=A1_BASELINE)
    print_report(res, f"PROJ_TOP / STACK_MAX $ BACKTEST (seed {SEED}, "
                      f"{arm_df.slate.nunique()} Tier-1 slates)")
    print(f"\nwrote {arm_out}")

    evw_df = pd.concat(evw_frames, ignore_index=True)
    evw_out = OUT_DIR / "needle_dollar_evw_sweep.csv"
    evw_df.to_csv(evw_out, index=False)
    baseline_evw = f"proj_top@evw{EVW_GRID[0]:g}" if EVW_GRID else None
    if baseline_evw and baseline_evw in set(evw_df.arm):
        res2 = report(evw_df, baseline=baseline_evw)
        print_report(res2, f"PROJ_TOP EVW SWEEP, $ TERMS (seed {SEED}, "
                           f"{evw_df.slate.nunique()} Tier-1 slates)")
    print(f"wrote {evw_out}")


if __name__ == "__main__":
    main()
