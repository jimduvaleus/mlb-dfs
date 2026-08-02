"""
Real-money backtest: does the External Pool p_win funnel benefit from
diversifying harder than production does?

--------------------------------------------------------------------------
THE QUESTION
--------------------------------------------------------------------------
Projections are only directionally accurate; day-to-day variance in a GPP
dwarfs any marginal edge a currency can carry. The theory under test: cull
the pool to plausibly-profitable candidates (keep this — it's not up for
debate), then LEAN HARDER on diversification than production's EVw=0.25 does,
so a portfolio doesn't cannibalize its own chances of catching a lottery
ticket by covering similar outcomes many times over. Compared against pure
randomness (no ranking signal at all) as the baseline that actually matters —
if a fancy currency can't beat a coin flip, the currency isn't the story.

--------------------------------------------------------------------------
METHOD
--------------------------------------------------------------------------
Rebuilds the exact External Pool / p_win funnel from archived slate inputs
(archive/MMDDYYYY/: DKSalaries.csv, lineups_*.csv, MLB_*.csv, and one or more
contest-standings-*.zip per real contest entered) using the REAL production
functions (src/api/external_pool.py, src/optimization/gpp_portfolio.py,
src/optimization/contest.py) rather than a hand-rolled reimplementation —
compute_lineup_scores, compute_pool_corr, compute_p_win, and especially
allocate_contests/DeterminantPortfolioSelector are called directly, so an
arm's selection logic is never more than "production with different EVw/
admit-window arguments." Two arms ("random"/"cull_rnd") aren't expressible
through allocate_contests (it always ranks by an EV vector) — those use a
small local helper that reuses the SAME cull step verbatim and only swaps
the final ranking for a uniform draw.

Every candidate's real payout comes from the exact DK payout table for the
contest it was entered in (data/payout_structures/, matched by (field size,
entry fee) via structure_for_contest — see git log on that module for how
thoroughly this was validated) and the REAL field of opponents who actually
played that contest (parsed from the standings zip, our candidate inserted
as one more competitor with ties split evenly). This replaced an earlier
approach that scored every entry against one borrowed field and a payout
curve scaled from a single reference size — both were shown to be
materially wrong (see git log on src/api/payout.py and
external_pool.py::pwin_exponents's docstring).

Per-contest p_win exponent uses the REAL field size from the standings zip
(ground truth), not production's implied-entries-from-parsed-prize-pool
estimate — strictly better information than production has at run time,
appropriate for a backtest.

Two-stage winner's-curse guard (mirrors src/api/pipeline.py's external-pool
branch exactly): n_sims is split into disjoint A/B halves, an independent
opponent field is generated for each half, p_win is computed once per half.
The cull ranks on the A draw, selection ranks survivors on the B draw — a
lineup that only looks good on the draw used to pick it can't also be why
it survives to be ranked.

PPD (postponed game) risk-adjustment is NOT replicated — that machinery
haircuts EV for a live run's uncertain future; here we already know exactly
what happened, so realized FPTS already reflects any real-world PPD.

--------------------------------------------------------------------------
CAVEAT (read before trusting a dollar figure out of this)
--------------------------------------------------------------------------
DK payouts are extremely top-heavy (CV of a single entry's payout commonly
30-50+; see the arm summary table this script prints). Getting a
statistically meaningful read on mean $/entry needs a LOT of slates — do
not conclude anything from a handful. The rate ladder (top-1%/top-0.1%/
top-10 hit counts) is the more efficient discriminator: a Bernoulli event
at p~1-2% has far lower relative variance per entry than the payout itself.
Report both, lead with the ladder, and bootstrap the dollar CIs so their
width is visible rather than implying false precision.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    source venv/bin/activate
    python tests/backtest.py 07282026 07292026 07302026

Env vars:
    BT_SEEDS=42,137,4242   comma-separated RNG seeds (default: 42)
    BT_NSIMS=2000          override simulation.n_sims for a fast smoke test
                            (default: config.yaml's simulation.n_sims)

Output: tests/backtest_output/results.csv (appended) + summary tables on
stdout (net by slate, ROI/win-rate table, rate ladder, bootstrap CIs).

This lives in tests/ rather than a throwaway scratch directory because it
had to be rebuilt once already after a scratchpad reset — it's cheap
infrastructure worth keeping. It is NOT picked up by `pytest tests/`
(doesn't match the test_*.py discovery pattern) and is not part of the
`python -m pytest tests/` suite CLAUDE.md documents; run it directly.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from tests.bt_core import (  # noqa: E402
    LIVE_CFG, build_slate_context, grade_pick,
    load_real_contests, verify_slate, weighted_proj_scores, _FakeGroup,
)

SLATES = [s for s in sys.argv[1:] if s.isdigit()]
if not SLATES:
    raise SystemExit("usage: python tests/backtest.py <slate MMDDYYYY> [<slate> ...]")
SEEDS = [int(s) for s in os.environ.get("BT_SEEDS", "42").split(",")]

OUT_DIR = PROJECT_ROOT / "tests" / "backtest_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
FLOOR_PCT = float(LIVE_CFG["gpp"].get("external_pool_proj_score_pct", 30.0))

# (uses_calibration, admit_floor, admit_mult, EVw)
# EVw=None means "ignore p_win ranking entirely, draw uniformly" -- the cull
# (admit_floor/admit_mult) still applies unless it's also zero.
ARMS: dict[str, tuple] = {
    "old":         (False, 250,  12.0, 0.25),  # 2026-07-28 to 07-30 era: uncalibrated grids, 250-floor/12x window
    "flat2000_uc": (False, 2000, 0.0,  0.25),  # 2026-07-27 17:16 to 07-28 11:22 era: uncalibrated grids, LITERAL flat-2000 window (no multiplier existed yet) -- the config actually being asked about here
    "flat2000_c":  (True,  2000, 0.0,  0.25),  # same flat-2000 window, but on TODAY's calibrated sims -- isolates "was it the window" from "was it that era's uncalibrated grids"
    "new":         (True,  100,  1.5,  0.25),  # 2026-07-30 to 07-31 era: calibrated, tight max(100, 1.5x) admit window.
                                                # NOT current production -- flat2000_uc was promoted to the permanent
                                                # default on 2026-07-31 (commit 81db769), so THAT arm is the live config.
    "cull_lo":     (True,  100,  1.5,  0.10),  # cull as production, lean hard on diversity (EVw 0.10)
    "cull_d0":     (True,  100,  1.5,  0.00),  # cull as production, pure diversity (no EV term)
    "cull_rnd":    (True,  100,  1.5,  None),  # cull as production (NARROW window), then draw uniformly from survivors
    "flat2000_rnd":(False, 2000, 0.0,  None),  # cull as flat2000_uc (WIDE window, uncalibrated), then draw uniformly --
                                                # isolates "does the wide cull's floor help random coverage" from cull_rnd's
                                                # narrow-window version, which was the single worst arm in the 8-slate run
    "wide":        (True,    0,  0.0,  0.25),  # no cull at all, production EVw -- isolates the cull's effect
    "wide_lo":     (True,    0,  0.0,  0.10),  # no cull, heavy diversity
    "random":      (True,    0,  0.0,  None),  # no cull, no ranking signal -- the baseline that matters most
}

# Frozen BEFORE any BT_ARMS filtering, because the random arms seed their RNG
# off an arm's position in this list (see run_arm). Taking the position from
# the filtered dict instead would give "random" a different draw depending on
# what else was requested that run -- while still writing it under the same
# label, silently pooling two different draws in results.csv.
_ARMS_RNG_ORDER = list(ARMS)

# BT_ARMS=name1,name2 restricts a run to a subset of ARMS (e.g. testing one
# new arm against the existing 8-slate archive without re-running, and
# duplicating, every other arm's already-collected rows in results.csv).
_arm_subset = os.environ.get("BT_ARMS")
if _arm_subset:
    _wanted = [a.strip() for a in _arm_subset.split(",")]
    _missing = [a for a in _wanted if a not in ARMS]
    if _missing:
        raise SystemExit(f"BT_ARMS names not in ARMS: {_missing}")
    ARMS = {a: ARMS[a] for a in _wanted}

# BT_FLOOR_PCTS=0,10,20,30,40,50 sweeps the proj-score-floor percentile for
# every active arm, at each (slate, seed) context -- cheap to add since the
# floor is a pure selection-time parameter (see run_arm's floor_pct):
# no new sims/corr/p_win needed, just extra allocate_contests/_random_pick
# calls against contexts already being built. Output arm names become
# "<arm>@floor<pct>" so results.csv can distinguish sweep points from the
# single-floor (module FLOOR_PCT) runs. A single value (or unset) leaves
# arm names unchanged, matching every prior run's schema.
_floor_sweep = os.environ.get("BT_FLOOR_PCTS")
FLOOR_SWEEP: list = [float(x) for x in _floor_sweep.split(",")] if _floor_sweep else [None]

# BT_PITCHER_WEIGHTS=1.0,1.5,2.0,2.5,3.0 sweeps the pitcher weight in the
# proj-score-floor basis (see weighted_proj_scores) -- same "cheap, no new
# sims" property as FLOOR_SWEEP. Output arm names get a "@pwN" suffix.
# Motivated by pitcher proj correlating with real FPTS 2.5x as strongly as
# hitter proj (0.415 vs 0.167, all 8 archived slates) -- the floor currently
# trusts both equally.
_pw_sweep = os.environ.get("BT_PITCHER_WEIGHTS")
PW_SWEEP: list = [float(x) for x in _pw_sweep.split(",")] if _pw_sweep else [1.0]

# BT_DIVERSITY_CORR=full,hitter_only sweeps which (M,M) correlation matrix
# feeds DeterminantPortfolioSelector's diversity/hedge terms (see
# hitter_only_corr). Computed once per context, only when "hitter_only" is
# requested, so an ordinary run pays nothing extra. Output arm names get a
# "@corr<variant>" suffix; "full" (the default/no-op) is unchanged.
_VALID_CORR_VARIANTS = {"full", "hitter_only"}
_corr_sweep = os.environ.get("BT_DIVERSITY_CORR")
DIVERSITY_CORR_SWEEP: list = (
    [x.strip() for x in _corr_sweep.split(",")] if _corr_sweep else ["full"]
)
_bad_corr = [c for c in DIVERSITY_CORR_SWEEP if c not in _VALID_CORR_VARIANTS]
if _bad_corr:
    raise SystemExit(f"BT_DIVERSITY_CORR values not in {_VALID_CORR_VARIANTS}: {_bad_corr}")

# ---------------------------------------------------------------------------
# Pipeline replication (real production functions only)
# ---------------------------------------------------------------------------

def _random_pick(ctx: dict, admit_floor: int, admit_mult: float, rng: np.random.Generator,
                  floor_pct: float = None, pitcher_weight: float = 1.0):
    """cull_rnd / random arms: not expressible through allocate_contests
    (it always ranks by an EV vector). Reuses allocate_contests's exact
    proj-floor + p_win cull logic (see the docstring on allocate_contests
    in src/api/external_pool.py for the formula this mirrors), then draws
    uniformly from the survivors instead of ranking them."""
    proj_scores = weighted_proj_scores(ctx, pitcher_weight)
    floor = ep.compute_proj_score_floor(proj_scores, FLOOR_PCT if floor_pct is None else floor_pct)
    mask = np.isfinite(proj_scores)
    if floor is not None:
        mask &= proj_scores >= floor[0]
    picks: dict[str, list[int]] = {}
    for c in ctx["contests"]:
        k = c["k"]
        if k <= 0:
            continue
        rem = np.where(mask)[0]
        if admit_floor > 0 or admit_mult > 0:
            cull_v = ctx["p_win_cull"].get(c["contest_id"])
            eff_n = max(admit_floor, int(round(admit_mult * k))) if admit_mult > 0 else admit_floor
            if eff_n > 0 and cull_v is not None and len(rem) > eff_n:
                rem = np.sort(rem[np.argsort(-cull_v[rem])[:eff_n]])
        take = rng.choice(len(rem), size=min(k, len(rem)), replace=False)
        idx = [int(rem[i]) for i in take]
        for p in idx:
            mask[p] = False
        picks[c["contest_id"]] = idx
    return picks


def run_arm(ctx: dict, arm: str, seed: int, floor_pct: float = None,
            pitcher_weight: float = 1.0, corr_variant: str = "full",
            ) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Returns (picks, unfilled_by_contest) -- callers must check the
    second dict before trusting per-entry $ metrics (see _FakeGroup).
    floor_pct overrides the module-level FLOOR_PCT (config default) for a
    single call -- used by the proj-score-floor calibration sweep, which
    needs the SAME arm run at several floor percentiles within one context
    build (floor is a pure selection-time parameter, so this needs no new
    sims/corr/p_win -- see build_slate_context). pitcher_weight similarly
    overrides the basis the floor is computed from -- see
    weighted_proj_scores; 1.0 is the unweighted baseline. corr_variant picks
    which (M,M) correlation matrix drives the diversity/hedge terms -- see
    hitter_only_corr; "full" (ctx["corr"]) is the unweighted baseline.
    _random_pick never reads ctx["corr"] at all, so corr_variant is simply
    unused for evw=None arms (random/cull_rnd/flat2000_rnd)."""
    _, admit_floor, admit_mult, evw = ARMS[arm]
    eff_floor_pct = FLOOR_PCT if floor_pct is None else floor_pct
    if evw is None:
        # offset by the arm's position (not hash(arm) -- string hashing is
        # randomized per process by default, which would make "random"/
        # "cull_rnd" non-reproducible across runs of the same seed).
        # Position comes from the UNFILTERED order so BT_ARMS can't change
        # the draw behind a label that stays the same.
        rng = np.random.default_rng(seed * 1000 + _ARMS_RNG_ORDER.index(arm))
        picks = _random_pick(ctx, admit_floor, admit_mult, rng, floor_pct=eff_floor_pct,
                              pitcher_weight=pitcher_weight)
        unfilled = {c["contest_id"]: c["k"] - len(picks.get(c["contest_id"], []))
                    for c in ctx["contests"]}
        return picks, unfilled

    corr = ctx["corr"] if corr_variant == "full" else ctx["hitter_corr"]
    if corr is None:
        raise ValueError(
            f"corr_variant={corr_variant!r} needs ctx['hitter_corr'] -- "
            "was BT_DIVERSITY_CORR set to include 'hitter_only'?"
        )
    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ctx["contests"] if c["k"] > 0]
    alloc = ep.allocate_contests(
        ctx["pool"], corr, groups, risk=3.0,
        evw_base=evw, evw_max=evw,
        proj_scores=weighted_proj_scores(ctx, pitcher_weight), proj_score_floor_percentile=eff_floor_pct,
        ev_type="p_win", p_win_cull=ctx["p_win_cull"], p_win_select=ctx["p_win_select"],
        p_win_admit_n=admit_floor, p_win_admit_multiplier=admit_mult,
    )
    idx_of = {id(lu): i for i, lu in enumerate(ctx["pool"].lineups)}
    unfilled_by_contest: dict[str, int] = {}
    for cid, _j in alloc.unfilled:
        unfilled_by_contest[cid] = unfilled_by_contest.get(cid, 0) + 1
    picks: dict[str, list[int]] = {}
    i = 0
    for g in groups:
        filled_n = len(g.entries) - unfilled_by_contest.get(g.contest_id, 0)
        picks[g.contest_id] = [idx_of[id(lu)] for lu, _ in alloc.portfolio[i:i + filled_n]]
        i += filled_n
    unfilled = {c["contest_id"]: unfilled_by_contest.get(c["contest_id"], 0)
                for c in ctx["contests"]}
    return picks, unfilled


def _append_and_reload(new_rows: list[dict]) -> pd.DataFrame:
    """Append this slate's rows to the results CSV immediately and return
    the full accumulated table read back from disk -- so a crash, kill, or
    just impatience mid-run still leaves every completed slate's results on
    disk and gradeable, instead of everything living only in this
    process's memory until the very last slate finishes."""
    df = pd.DataFrame(new_rows)
    if not df.empty:
        df["slate"] = df["slate"].astype(str)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        # dtype pin is load-bearing: every slate name is all-digit
        # (07262026), so an untyped read_csv infers int64 and silently
        # drops the leading zero -- breaking every later string comparison
        # against SLATES/argv.
        old = pd.read_csv(RESULTS_CSV, dtype={"slate": str})
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df


def print_summary(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"\n[{label}] no results")
        return
    df = df.copy()
    df["fees"] = df["n"] * df["fee"]
    seen = set(df["arm"].unique())
    # floor-sweep output arms are "<arm>@floor<pct>", not literal ARMS keys --
    # order them after their base arm's ARMS position, sorted by floor value.
    present_arms = [a for a in ARMS if a in seen]
    extra = sorted(a for a in seen if a not in ARMS)
    present_arms += extra
    print(f"\n===== {label} =====\n")
    p = df.groupby(["slate", "arm"], as_index=False)[["fees", "won"]].sum()
    p["net"] = p["won"] - p["fees"]
    piv = p.pivot_table(index="slate", columns="arm", values="net")[present_arms]
    print("  net by slate:")
    print(piv.round(2).to_string())

    c = df.groupby("arm").agg(entries=("n", "sum"), fees=("fees", "sum"), won=("won", "sum"))
    c["net"] = c["won"] - c["fees"]
    c["ROI"] = 100 * c["net"] / c["fees"]
    print("\n" + c.loc[present_arms].round(2).to_string())

    r = df.groupby("arm").agg(top1=("top1", "sum"), top01=("top01", "sum"),
                              best_rank=("best_rank", "min"))
    print("\n" + r.loc[present_arms].to_string())


def main() -> None:
    all_fill_events = []
    full_df = pd.DataFrame()
    for slate in SLATES:
        rows = []
        fill_events = []
        d = PROJECT_ROOT / "archive" / slate
        real = load_real_contests(d)
        # DKSalaries.csv's raw Name/ID columns -- verify_slate needs the
        # exact display name to join against the zip's Player/FPTS table,
        # and duplicate-name detection has to run on this, not slate_df.
        raw = pd.read_csv(d / "DKSalaries.csv")
        nm = raw[["ID", "Name"]].astype({"ID": str})
        fpts = verify_slate(d, real, nm)
        print(f"{slate}: all zips verified against this slate's realized FPTS", flush=True)

        for seed in SEEDS:
            ctxs = {}
            for calib in (False, True):
                t0 = time.time()
                ctxs[calib] = build_slate_context(
                    d, seed, calib, real,
                    n_sims=N_SIMS, sharpness=SHARPNESS, sim_cache_dir=SIM_CACHE_DIR,
                    want_hitter_corr="hitter_only" in DIVERSITY_CORR_SWEEP,
                )
                print(f"    seed {seed} calib={calib} context built in {time.time() - t0:.0f}s", flush=True)

            for arm in ARMS:
                calib_flag = ARMS[arm][0]
                evw = ARMS[arm][3]
                ctx = ctxs[calib_flag]
                # random/cull_rnd/flat2000_rnd (evw is None) never read
                # ctx["corr"] at all -- don't burn a sweep point on them.
                corr_variants = DIVERSITY_CORR_SWEEP if evw is not None else DIVERSITY_CORR_SWEEP[:1]
                for floor_pct in FLOOR_SWEEP:
                    for pw in PW_SWEEP:
                        for corr_variant in corr_variants:
                            out_arm = arm
                            if floor_pct is not None:
                                out_arm += f"@floor{floor_pct:g}"
                            # Suffix whenever a sweep was actually REQUESTED
                            # (env var set), not just when the value differs
                            # from the no-op default -- an explicit sweep that
                            # happens to include the default value (e.g.
                            # BT_PITCHER_WEIGHTS=1.0,1.5,...) must not collide
                            # with pre-existing unsuffixed baseline rows from
                            # an earlier, unrelated run. (Found the hard way:
                            # a pw sweep silently duplicated/corrupted
                            # "random"/"flat2000_uc"'s bare-label rows.)
                            if _pw_sweep is not None:
                                out_arm += f"@pw{pw:g}"
                            if _corr_sweep is not None:
                                out_arm += f"@corr{corr_variant}"
                            picks, unfilled = run_arm(ctx, arm, seed, floor_pct=floor_pct,
                                                       pitcher_weight=pw, corr_variant=corr_variant)
                            for cid, n_unfilled in unfilled.items():
                                if n_unfilled > 0:
                                    fill_events.append({
                                        "slate": slate, "seed": seed, "arm": out_arm,
                                        "contest_id": cid, "unfilled": n_unfilled,
                                    })
                            actual = {i: sum(fpts.get(int(p), float("nan")) for p in lu.player_ids)
                                      for i, lu in enumerate(ctx["pool"].lineups)}
                            for c in ctx["contests"]:
                                idxs = picks.get(c["contest_id"], [])
                                n_ambiguous = 0
                                for i in idxs:
                                    a = actual[i]
                                    if not np.isfinite(a):
                                        n_ambiguous += 1  # rostered an ambiguous-name player -- see verify_slate
                                        continue
                                    gross, rank = grade_pick(a, c["sorted_scores"], c["payout_arr"])
                                    rows.append({
                                        "slate": slate, "seed": seed, "arm": out_arm,
                                        "contest": c["contest"], "n": 1, "fee": c["fee"],
                                        "won": gross, "best_rank": rank, "n_field": c["n_field"],
                                        "top1": int(rank <= max(1, c["n_field"] // 100)),
                                        "top01": int(rank <= max(1, c["n_field"] // 1000)),
                                    })
                                if n_ambiguous:
                                    fill_events.append({
                                        "slate": slate, "seed": seed, "arm": out_arm,
                                        "contest_id": c["contest_id"] + " [ambiguous-name drop]",
                                        "unfilled": n_ambiguous,
                                    })

        print(f"\n===== FILL CHECK [{slate}] =====")
        if fill_events:
            fe = pd.DataFrame(fill_events)
            print("  UNFILLED ENTRIES FOUND -- per-entry $ metrics for the affected "
                  "(seed, arm) below are on a SMALLER denominator than intended. An "
                  "arm that silently drops hard-to-fill entries must not read as "
                  "'better' per-entry just because it graded fewer of them.")
            print(fe.groupby(["arm"])["unfilled"].agg(["sum", "count"]).to_string())
        else:
            print("  clean -- every arm filled every contest at its intended size.")
        all_fill_events.extend(fill_events)

        # Flushed to disk (and printed) as soon as THIS slate finishes --
        # a crash, kill, or a check-in partway through a multi-hour run
        # still has every completed slate's results available, instead of
        # everything living only in memory until the very last slate.
        full_df = _append_and_reload(rows)
        print(f"results -> {RESULTS_CSV}  ({len(full_df)} rows total)")
        print_summary(full_df[full_df["slate"].astype(str) == slate], f"THIS SLATE ({slate})")

    print_summary(full_df, f"POOLED ACROSS {full_df['slate'].nunique()} SLATE(S)")


if __name__ == "__main__":
    main()
