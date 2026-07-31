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
    "old":      (False, 250, 12.0, 0.25),   # pre-2026-07-30: uncalibrated grids, wide admit window
    "new":      (True,  100, 1.5,  0.25),   # current production: calibrated, tight admit window
    "cull_lo":  (True,  100, 1.5,  0.10),   # cull as production, lean hard on diversity (EVw 0.10)
    "cull_d0":  (True,  100, 1.5,  0.00),   # cull as production, pure diversity (no EV term)
    "cull_rnd": (True,  100, 1.5,  None),   # cull as production, then draw uniformly from survivors
    "wide":     (True,    0, 0.0,  0.25),   # no cull at all, production EVw -- isolates the cull's effect
    "wide_lo":  (True,    0, 0.0,  0.10),   # no cull, heavy diversity
    "random":   (True,    0, 0.0,  None),   # no cull, no ranking signal -- the baseline that matters most
}

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
        fee_guess = None  # resolved against the payout table below
        struct = structure_for_contest(contest, n_entries=n_field)
        if struct is None or struct["total_entries"] != n_field:
            raise SystemExit(
                f"no exact payout table for {contest!r} at n={n_field:,} "
                f"(zip {z.name} in {d.name}) -- capture and register it first."
            )
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
        pool=pool, corr=corr, proj_scores=proj_scores,
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


def _random_pick(ctx: dict, admit_floor: int, admit_mult: float, rng: np.random.Generator):
    """cull_rnd / random arms: not expressible through allocate_contests
    (it always ranks by an EV vector). Reuses allocate_contests's exact
    proj-floor + p_win cull logic (see the docstring on allocate_contests
    in src/api/external_pool.py for the formula this mirrors), then draws
    uniformly from the survivors instead of ranking them."""
    proj_scores = ctx["proj_scores"]
    floor = ep.compute_proj_score_floor(proj_scores, FLOOR_PCT)
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


def run_arm(ctx: dict, arm: str, seed: int) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Returns (picks, unfilled_by_contest) -- callers must check the
    second dict before trusting per-entry $ metrics (see _FakeGroup)."""
    _, admit_floor, admit_mult, evw = ARMS[arm]
    if evw is None:
        # offset by the arm's position (not hash(arm) -- string hashing is
        # randomized per process by default, which would make "random"/
        # "cull_rnd" non-reproducible across runs of the same seed)
        rng = np.random.default_rng(seed * 1000 + list(ARMS).index(arm))
        picks = _random_pick(ctx, admit_floor, admit_mult, rng)
        unfilled = {c["contest_id"]: c["k"] - len(picks.get(c["contest_id"], []))
                    for c in ctx["contests"]}
        return picks, unfilled

    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ctx["contests"] if c["k"] > 0]
    alloc = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0,
        evw_base=evw, evw_max=evw,
        proj_scores=ctx["proj_scores"], proj_score_floor_percentile=FLOOR_PCT,
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


def main() -> None:
    rows = []
    fill_events = []
    for slate in SLATES:
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
                picks, unfilled = run_arm(ctx, arm, seed)
                for cid, n_unfilled in unfilled.items():
                    if n_unfilled > 0:
                        fill_events.append({
                            "slate": slate, "seed": seed, "arm": arm,
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
                            "slate": slate, "seed": seed, "arm": arm,
                            "contest": c["contest"], "n": 1, "fee": c["fee"],
                            "won": gross, "best_rank": rank, "n_field": c["n_field"],
                            "top1": int(rank <= max(1, c["n_field"] // 100)),
                            "top01": int(rank <= max(1, c["n_field"] // 1000)),
                        })
                    if n_ambiguous:
                        fill_events.append({
                            "slate": slate, "seed": seed, "arm": arm,
                            "contest_id": c["contest_id"] + " [ambiguous-name drop]",
                            "unfilled": n_ambiguous,
                        })

    print("\n===== FILL CHECK =====")
    if fill_events:
        fe = pd.DataFrame(fill_events)
        print("  UNFILLED ENTRIES FOUND -- per-entry $ metrics for the affected "
              "(slate, seed, arm) below are on a SMALLER denominator than intended. "
              "An arm that silently drops hard-to-fill entries must not read as "
              "'better' per-entry just because it graded fewer of them.")
        print(fe.groupby(["arm"])["unfilled"].agg(["sum", "count"]).to_string())
    else:
        print("  clean -- every arm filled every contest at its intended size.")

    df = pd.DataFrame(rows)
    if df.empty:
        print("no results")
        return
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        df = pd.concat([pd.read_csv(RESULTS_CSV), df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nresults -> {RESULTS_CSV}")

    df["fees"] = df["n"] * df["fee"]
    print("\n===== REAL-MONEY BACKTEST =====\n")
    p = df.groupby(["slate", "arm"], as_index=False)[["fees", "won"]].sum()
    p["net"] = p["won"] - p["fees"]
    piv = p.pivot_table(index="slate", columns="arm", values="net")[list(ARMS)]
    print("  net by slate:")
    print(piv.round(2).to_string())

    c = df.groupby("arm").agg(entries=("n", "sum"), fees=("fees", "sum"), won=("won", "sum"))
    c["net"] = c["won"] - c["fees"]
    c["ROI"] = 100 * c["net"] / c["fees"]
    print("\n" + c.loc[list(ARMS)].round(2).to_string())

    r = df.groupby("arm").agg(top1=("top1", "sum"), top01=("top01", "sum"),
                              best_rank=("best_rank", "min"))
    print("\n" + r.loc[list(ARMS)].to_string())


if __name__ == "__main__":
    main()
