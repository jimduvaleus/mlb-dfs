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
import csv as csv_mod
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.gpp_portfolio import (  # noqa: E402
    DeterminantPortfolioSelector, _HEDGE_WEIGHT_FRACTION,
)
from src.optimization.payout import payout_table_to_array, structure_for_contest  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

SLATES = [s for s in sys.argv[1:] if s.isdigit()]
if not SLATES:
    raise SystemExit("usage: python tests/backtest.py <slate MMDDYYYY> [<slate> ...]")
SEEDS = [int(s) for s in os.environ.get("BT_SEEDS", "42").split(",")]

OUT_DIR = PROJECT_ROOT / "tests" / "backtest_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "config.yaml") as f:
    LIVE_CFG = yaml.safe_load(f)
N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
FLOOR_PCT = float(LIVE_CFG["gpp"].get("external_pool_proj_score_pct", 30.0))
_PWIN_FIELD_CAP = 25_000

# (uses_calibration, admit_floor, admit_mult, EVw)
# EVw=None means "ignore p_win ranking entirely, draw uniformly" -- the cull
# (admit_floor/admit_mult) still applies unless it's also zero.
ARMS: dict[str, tuple] = {
    "old":         (False, 250,  12.0, 0.25),  # 2026-07-28 to 07-30 era: uncalibrated grids, 250-floor/12x window
    "flat2000_uc": (False, 2000, 0.0,  0.25),  # 2026-07-27 17:16 to 07-28 11:22 era: uncalibrated grids, LITERAL flat-2000 window (no multiplier existed yet) -- the config actually being asked about here
    "flat2000_c":  (True,  2000, 0.0,  0.25),  # same flat-2000 window, but on TODAY's calibrated sims -- isolates "was it the window" from "was it that era's uncalibrated grids"
    "new":         (True,  100,  1.5,  0.25),  # current production: calibrated, tight admit window
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

# standings zip stem (lowercased, no extension) -> display key matching
# portfolio_sweep_draftkings.json's contest_name AND the payout registry's
# lowercased contest key (structure_for_contest lowercases internally).
ZIP_TO_CONTEST = {
    "4k-base-hit": "Base Hit", "6k-base-hit": "Base Hit",
    "10k-base-hit": "Base Hit", "base-hit": "Base Hit",
    "bat-flip": "Bat Flip", "chin-music": "Chin Music",
    "five-tool-player": "Five-Tool Player", "four-seamer": "Four-Seamer",
    "hot-corner": "Hot Corner", "mini-max": "mini-MAX",
    "pickoff": "Pickoff", "rally-cap": "Rally Cap",
    "relay-throw": "Relay Throw", "skipper": "Skipper",
    "solo-shot": "Solo Shot", "moonshot": "Moonshot",
    "knuckleball": "Knuckleball",
    # 07/25 archive: misnamed file, 2,972 entries matches four-seamer-payout.txt
    # exactly and no four-seamer.zip exists otherwise -- see git log f20550a.
    "sour-seamer": "Four-Seamer",
}


# ---------------------------------------------------------------------------
# Real-field / real-money grading
# ---------------------------------------------------------------------------

def load_real_contests(d: Path) -> list[dict]:
    """[{contest, contest_id, n_field, fee, sorted_scores, payout_arr}] --
    one entry per real standings zip in this slate's archive dir.

    `contest-standings-*.zip` is always excluded: on every slate checked
    it's a byte-identical (or, once, a stale same-night pre-stat-correction)
    duplicate of one of the named zips, kept around only because the
    Analyze Contest UI feature expects a file with that name to exist.
    """
    out = []
    for z in sorted(d.glob("*.zip")):
        if z.name.startswith("contest-standings"):
            continue
        stem = z.stem.lower()
        contest = ZIP_TO_CONTEST.get(stem)
        if contest is None:
            raise SystemExit(
                f"unmapped zip {z.name} in {d.name} -- add it to ZIP_TO_CONTEST. "
                "Silently skipping a contest would drop real entries from the backtest."
            )
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv_mod.reader(
                zf.read(name).decode("utf-8-sig", errors="replace").splitlines()
            ))
        body = rows[1:]
        scores = sorted(float(r[4]) for r in body if r and r[0].strip().isdigit())
        n_field = len(scores)
        struct = structure_for_contest(contest, n_entries=n_field)
        table_n = struct["total_entries"] if struct is not None else None
        # DK's payout tiers are fixed to the contest's DESIGNED/advertised
        # size at creation, not to how many people actually showed up that
        # day -- a guaranteed contest that under-fills still pays from the
        # full-size table (confirmed 2026-07-31: two 07/22 captures had the
        # table's OWN stated header size, not that day's real fill, and were
        # byte-identical to already-registered tables at the larger size).
        # So table_n >= n_field is expected and fine; only a real MISMATCH
        # (table smaller than the real field, or a big relative gap --
        # meaning nearest-match grabbed the wrong contest's table) is an error.
        gap = None if table_n is None else table_n - n_field
        if struct is None or gap < 0 or gap > max(50, round(0.05 * table_n)):
            raise SystemExit(
                f"no matching payout table for {contest!r} at real fill n={n_field:,} "
                f"(zip {z.name} in {d.name}, nearest table n={table_n}) -- "
                "capture and register the right one."
            )
        if gap > 0:
            print(f"    note: {contest} real fill {n_field:,} < designed table size "
                  f"{table_n:,} (gap {gap}, {100*gap/table_n:.1f}%) -- using the full-size "
                  "table, DK's payout tiers don't shrink for an under-filled guarantee",
                  flush=True)
        out.append({
            "contest": contest,
            "contest_id": f"{d.name}:{stem}",
            "n_field": n_field,
            "fee": struct["entry_fee"],
            "sorted_scores": np.array(scores, dtype=np.float64),
            "payout_arr": payout_table_to_array(struct),
        })
    return out


def verify_slate(d: Path, real: list[dict], nm: dict) -> dict:
    """{player_id: actual_fpts} rebuilt fresh from every real zip's embedded
    per-player FPTS side table (never trusts a separately-archived
    contest_player_fpts.json -- that file was found stale once already,
    see git log). Every zip must agree with every other on every
    unambiguous player or this raises: a mismatch across INDEPENDENTLY
    downloaded real files means one of them is wrong (a DK retroactive
    stat correction between download times, most often) and grading
    against it would silently score real money wrong.

    DKSalaries names that map to more than one player_id that slate (two
    real MLB players who happen to share a name) can't be resolved from
    the zip's Player/FPTS side table alone (no ID column) -- both ids are
    left out of the returned map, and any of our lineups rostering either
    one drops out of grading (NaN actual_score) rather than risk crediting
    the wrong player's score.
    """
    dup_names = set(nm["Name"][nm["Name"].duplicated(keep=False)])
    id_by_name = {r.Name: str(r.ID) for r in nm.itertuples() if r.Name not in dup_names}
    merged: dict[str, float] = {}
    for c in real:
        z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv_mod.reader(
                zf.read(name).decode("utf-8-sig", errors="replace").splitlines()
            ))
        body = rows[1:]
        emb = {
            r[7].strip(): float(r[10]) for r in body
            if len(r) > 10 and r[7].strip() and r[10] not in ("", "FPTS")
        }
        for pname, fp in emb.items():
            pid = id_by_name.get(pname)
            if pid is None:
                continue
            if pid in merged and abs(merged[pid] - fp) >= 0.011:
                raise SystemExit(
                    f"{d.name}: real zips disagree on {pname} ({pid}) -- "
                    f"{merged[pid]} vs {fp} from {c['contest_id']}. "
                    "Independently downloaded real files should never differ; "
                    "investigate before trusting this slate."
                )
            merged[pid] = fp
    return {int(k): v for k, v in merged.items()}


# ---------------------------------------------------------------------------
# Pipeline replication (real production functions only)
# ---------------------------------------------------------------------------

def build_slate_context(d: Path, seed: int, calibrated: bool, real: list[dict]):
    """Everything needed to run every arm for one (slate, seed, calibration)
    combination: pool, players_df, sim_results, corr, proj_scores, p_win
    cull/select dicts (two-stage), and the per-contest real allocation
    (sizes/fees keyed to `real`, splitting a display name shared by more
    than one real contest that slate -- e.g. "Base Hit" covering both a
    $4K and $10K variant -- proportionally by real field size)."""
    raw_dir = str(d)
    found = ep.discover_external_files(raw_dir)
    if not found["lineups_paths"] or not found["projections_path"]:
        raise SystemExit(f"{d.name}: no lineups_*.csv / projections CSV pair found.")

    slate_df = DraftKingsSlateIngestor(str(d / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    if not pool.lineups:
        raise SystemExit(f"{d.name}: every lineup dropped (unknown player ids).")
    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )

    copula = EmpiricalCopula(str(PROJECT_ROOT / LIVE_CFG["paths"]["copula"]))
    if calibrated:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=True, scratch_prob=0.02,
            mean_calib_batter=0.88, mean_calib_pitcher=1.0,
        )
    else:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=False, scratch_prob=0.0,
            mean_calib_batter=1.0, mean_calib_pitcher=1.0,
        )
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)

    cache_path = SIM_CACHE_DIR / f"{d.name}_{N_SIMS}_{seed}_calib{calibrated}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
    else:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(N_SIMS)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache_path,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )

    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
    corr = ep.compute_pool_corr(pool.lineups, sim_results, scores=lineup_scores)
    proj_scores = ep.compute_pool_proj_scores(pool.lineups, players_df)

    # --- real per-contest sizes: split a shared display name proportionally
    # by real field size (e.g. "Base Hit" covering both a $4K and $10K
    # contest the same slate) ---
    sw = __import__("json").loads((d / "portfolio_sweep_draftkings.json").read_text())
    r1 = next(x for x in sw["sweep"] if x["risk"] == 1.0)
    display_sizes: dict[str, int] = {}
    for lu in r1["lineups"]:
        display_sizes[lu["contest_name"]] = display_sizes.get(lu["contest_name"], 0) + 1
    by_display: dict[str, list[dict]] = {}
    for c in real:
        by_display.setdefault(c["contest"], []).append(c)
    contests: list[dict] = []
    for display, n_total in display_sizes.items():
        variants = by_display.get(display)
        if not variants:
            continue  # no real zip for this display name -- can't grade it
        if len(variants) == 1:
            contests.append({**variants[0], "k": n_total})
            continue
        total_field = sum(v["n_field"] for v in variants)
        alloc = [int(round(n_total * v["n_field"] / total_field)) for v in variants]
        alloc[-1] += n_total - sum(alloc)  # fix rounding drift on the last one
        for v, k in zip(variants, alloc):
            if k > 0:
                contests.append({**v, "k": k})

    # --- p_win: two-stage winner's-curse guard, mirrors pipeline.py exactly ---
    own_vec = players_df["ownership"].astype(float).to_numpy()
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    n_half = N_SIMS // 2
    sims_A, sims_B = sim_results.results_matrix[:n_half], sim_results.results_matrix[n_half:2 * n_half]
    scores_A, scores_B = lineup_scores[:, :n_half], lineup_scores[:, n_half:2 * n_half]

    field_n = int(min(max(5_000, max(c["n_field"] for c in contests)), _PWIN_FIELD_CAP))
    cs = ContestSimulator()
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed)
    field_B = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed + 1)
    field_scores_A = cs.score_field(field_A, sims_A, col_map)
    field_scores_B = cs.score_field(field_B, sims_B, col_map)

    # Ground-truth field size (from the real zip), not an implied estimate --
    # strictly better information than production has at run time.
    exponents = {c["contest_id"]: max(1.0, SHARPNESS * c["n_field"]) for c in contests}
    p_win_cull = ep.compute_p_win(scores_A, field_scores_A, exponents)
    p_win_select = ep.compute_p_win(scores_B, field_scores_B, exponents)

    return dict(
        pool=pool, corr=corr, proj_scores=proj_scores, players_df=players_df,
        contests=contests, p_win_cull=p_win_cull, p_win_select=p_win_select,
    )


class _FakeGroup:
    """Minimal stand-in for ep.ContestGroup: allocate_contests's p_win
    branch only reads .contest_id and len(.entries). We know real per-
    contest entry counts from portfolio_sweep_draftkings.json directly
    (the pipeline's actual submitted counts -- overflow entries some days
    come from a second account and shouldn't be replicated here), so a
    dummy .entries list of the right length is sufficient and exact.

    Entries are tagged (contest_id, j) rather than bare ints: when a
    contest's pool is exhausted before it fills, allocate_contests reports
    the shortfall via a flat `unfilled` list with no group boundary
    markers -- globally-unique entries let run_arm trace an unfilled
    placeholder back to the contest it belongs to (see the partial-fill
    trap this guards against: an arm that silently drops entries must not
    look "better" per-entry just because it graded fewer of them)."""
    def __init__(self, contest_id: str, k: int):
        self.contest_id = contest_id
        self.entries = [(contest_id, j) for j in range(k)]


def weighted_proj_scores(ctx: dict, pitcher_weight: float) -> np.ndarray:
    """Pool proj-score sum with each pitcher's projected mean multiplied by
    `pitcher_weight` before summing (hitters unweighted) -- pitcher_weight=1.0
    reproduces ctx["proj_scores"] exactly. Motivated by a direct measurement
    against real outcomes across all 8 archived slates: pitcher projected
    mean correlates with real FPTS more than 2x as strongly as hitter
    projected mean does (r=0.415 vs 0.167; see memory
    project-cull-calibration-followup). The floor currently sums both
    equally, trusting a noisy hitter signal exactly as much as a much more
    reliable pitcher one -- this tests whether weighting the sum toward the
    more trustworthy half changes what the floor admits/culls."""
    if pitcher_weight == 1.0:
        return ctx["proj_scores"]
    players_df = ctx["players_df"]
    w = np.where(players_df["position"].to_numpy() == "P", pitcher_weight, 1.0)
    weighted_means = players_df["mean"].to_numpy(dtype=np.float64) * w
    tmp = players_df.copy()
    tmp["mean"] = weighted_means
    return ep.compute_pool_proj_scores(ctx["pool"].lineups, tmp)


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
            pitcher_weight: float = 1.0,
            ) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Returns (picks, unfilled_by_contest) -- callers must check the
    second dict before trusting per-entry $ metrics (see _FakeGroup).
    floor_pct overrides the module-level FLOOR_PCT (config default) for a
    single call -- used by the proj-score-floor calibration sweep, which
    needs the SAME arm run at several floor percentiles within one context
    build (floor is a pure selection-time parameter, so this needs no new
    sims/corr/p_win -- see build_slate_context). pitcher_weight similarly
    overrides the basis the floor is computed from -- see
    weighted_proj_scores; 1.0 is the unweighted baseline."""
    _, admit_floor, admit_mult, evw = ARMS[arm]
    eff_floor_pct = FLOOR_PCT if floor_pct is None else floor_pct
    if evw is None:
        # offset by the arm's position (not hash(arm) -- string hashing is
        # randomized per process by default, which would make "random"/
        # "cull_rnd" non-reproducible across runs of the same seed)
        rng = np.random.default_rng(seed * 1000 + list(ARMS).index(arm))
        picks = _random_pick(ctx, admit_floor, admit_mult, rng, floor_pct=eff_floor_pct,
                              pitcher_weight=pitcher_weight)
        unfilled = {c["contest_id"]: c["k"] - len(picks.get(c["contest_id"], []))
                    for c in ctx["contests"]}
        return picks, unfilled

    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ctx["contests"] if c["k"] > 0]
    alloc = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0,
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


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_pick(actual_score: float, sorted_real: np.ndarray, payout_arr: np.ndarray):
    """(gross_$, rank) inserting our lineup as one more competitor in the
    real field, ties split evenly across the tie band (including us)."""
    n_field = len(sorted_real)
    right = int(np.searchsorted(sorted_real, actual_score, side="right"))
    left = int(np.searchsorted(sorted_real, actual_score, side="left"))
    n_above, n_tied = n_field - right, right - left
    rank = n_above + 1
    lo, hi = n_above, min(n_above + n_tied + 1, len(payout_arr))
    band = payout_arr[lo:hi] if lo < len(payout_arr) else np.array([])
    return (float(band.mean()) if len(band) else 0.0), rank


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
                ctxs[calib] = build_slate_context(d, seed, calib, real)
                print(f"    seed {seed} calib={calib} context built in {time.time() - t0:.0f}s", flush=True)

            for arm in ARMS:
                calib_flag = ARMS[arm][0]
                ctx = ctxs[calib_flag]
                for floor_pct in FLOOR_SWEEP:
                    for pw in PW_SWEEP:
                        out_arm = arm
                        if floor_pct is not None:
                            out_arm += f"@floor{floor_pct:g}"
                        if pw != 1.0:
                            out_arm += f"@pw{pw:g}"
                        picks, unfilled = run_arm(ctx, arm, seed, floor_pct=floor_pct, pitcher_weight=pw)
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
