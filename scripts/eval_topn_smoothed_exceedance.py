"""Does smoothed exceedance actually settle the top-N currency?

`allocate_contests_topn_coverage`'s hard `1[score >= threshold]` indicator is a
rare-event estimator: at a tight rung a candidate crosses in a handful of
worlds and never in the rest, so its whole claim rate rests on ~1.9 events
(split-half rho_full ~0.30 at rank 1 -- scripts/diagnose_topn_rung_settling.py).
Every top-heavy refinement built on top of that inherits the noise, which is
why the payout ladder was rejected: 96% of its weight landed on rungs with
rho_full < 0.55.

`smooth_tau_scale > 0` replaces the indicator with P(threshold <= score) under
the rank-N order statistic's own sampling distribution (see
`external_pool.smoothing_tau`, which this script CALLS rather than
reimplements, so the measurement is of production's formula). This script is
the direct A/B of the claim that motivates the change.

Method -- deliberately the same shape as diagnose_topn_rung_settling.py so the
numbers are comparable to the pre-change baseline:

  * one production-scale run per contest, real slate inputs, cached sim
  * each contest's own disjoint sim-world slice, consumed in lockstep with
    production so every contest sees the worlds it really would
  * each rung's per-candidate currency computed BOTH ways from the SAME
    thresholds and the SAME worlds -- only the indicator-vs-probability step
    differs, so nothing else can explain a gap
  * worlds split into two disjoint halves; each candidate's per-rung currency
    is computed on each half independently and the halves compared by Spearman
    rho over the pool. Reported per rung as `rho_half`, and Spearman-Brown
    stepped up to `rho_full` (the reliability of the FULL-budget estimate,
    which is what the greedy actually consumes).
  * `top50_overlap` alongside it, because the greedy only ever eats the very
    top of the ranking, not the whole of it.

RELIABILITY IS NOT VALIDITY, and this is the trap this script must not walk
into. rho only asks "would I rank the same way on independent worlds"; a
currency that is perfectly reliable and WRONG scores rho = 1.0. As tau grows
the logistic flattens, z -> 0, and the ranking collapses toward a monotone
function of the candidate's MEAN score -- which is very reliable and is
precisely the field-blind ceiling that measured rho = -0.18 against realized
score and buys chalk. So higher rho at a larger tau_scale is NOT automatically
better. Two columns separate denoising from drifting:

  rho_vs_hard   full-budget smoothed currency vs the hard currency. High =>
                same statistic, less noise (what we want).
  rho_vs_mean   full-budget currency vs the candidate's plain mean score over
                this contest's worlds, reported for the hard arm too as the
                baseline. Climbing toward 1.0 as tau rises => the currency has
                stopped being field-relative and is drifting into chalk.

tau_scale = 1.0 is the value with a derivation behind it (the order
statistic's actual sampling sd); the sweep exists to show the shape of the
trade, not to pick the arm with the biggest rho.

Unlike the settling script, both arms accumulate over (draw, world) SLOTS
rather than OR-ing the K field draws together -- that is the allocator's real
slot semantics, and OR-ing would hide exactly the threshold-draw noise tau is
meant to absorb.

Checkpoint / resume per CLAUDE.md: rows appended per contest to
outputs/topn_smoothed_exceedance/results.csv; contests already on disk are
skipped, so an interrupted run resumes on the same command line.

Usage
-----
    source venv/bin/activate
    python scripts/eval_topn_smoothed_exceedance.py 2>&1 | tee /tmp/smooth.log

Env vars
--------
    TOPN_REQ_RAW          slate input dir (default data/raw)
    TOPN_SMOOTH_RUNGS     rungs to probe (default 5, matching the baseline)
    TOPN_SMOOTH_SCALES    comma-separated tau scales (default "0.5,1.0,2.0")
    TOPN_SMOOTH_FORCE     "1" re-runs contests already in results.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import parse_entry_file  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
N_RUNGS = int(os.environ.get("TOPN_SMOOTH_RUNGS", "5"))
SCALES = [float(x) for x in os.environ.get("TOPN_SMOOTH_SCALES", "0.5,1.0,2.0").split(",")]
FORCE = os.environ.get("TOPN_SMOOTH_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_smoothed_exceedance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"  # reuse sim/field caches


def _append_and_reload(csv_path: Path, contest_id: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"contest_id": str})


def _spearman_brown(rho_half: float) -> float:
    """Half-length reliability -> full-length reliability. The split-half rho
    measures an estimate built on n_sims_g/2 worlds; the greedy consumes the
    full budget, so this is the figure that answers "is the currency the
    allocator sees settled"."""
    if not np.isfinite(rho_half):
        return float("nan")
    denom = 1.0 + rho_half
    return float(2.0 * rho_half / denom) if denom > 1e-9 else float("nan")


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _reliability(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if a.std() <= 0 or b.std() <= 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(a, b).statistic)
    top_a = set(np.argsort(-a)[:50].tolist())
    top_b = set(np.argsort(-b)[:50].tolist())
    return rho, _spearman_brown(rho), len(top_a & top_b) / 50.0


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp, paths = cfg["gpp"], cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)

    found = ep.discover_external_files(RAW_DIR)
    slate_df = DraftKingsSlateIngestor(str(PROJECT_ROOT / paths["dk_slate"])).get_slate_dataframe()
    pool = ep.parse_lineup_pool(
        found["lineups_paths"], set(slate_df["player_id"].astype(int)), require_roi_blocks=False,
    )
    proj_ext = ep.parse_player_projections(found["projections_path"])
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, {int(p) for lu in pool.lineups for p in lu.player_ids},
        PipelineRunner._derive_opponent,
    )
    all_file_entries = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            all_file_entries.append((p, recs))
    groups = ep.group_and_match_contests(all_file_entries, pool)

    fp_size = int(gpp.get("external_pool_topn_field_pool_size", 25_000))
    n_sims_cfg = int(cfg["simulation"].get("n_sims", 25_000))
    frac = float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5))
    smin = int(gpp.get("external_pool_topn_sims_min", 0))
    sref = float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0))
    spow = float(gpp.get("external_pool_topn_sims_power", 0.0))
    n_sims = max(n_sims_cfg, ep.topn_total_sims_needed(groups, fp_size, frac, smin, sref, spow))

    cache = SHARED / "sim_cache" / f"{found['projections_path'].stem}_{n_sims}_{seed}.npz"
    if cache.exists():
        with np.load(cache) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
    else:
        grids = ep.build_quantile_grids(
            proj_ext,
            zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
            scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
            mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
            mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)),
        )
        engine = SimulationEngine(
            EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
            batter_pca_model=None, score_grid=None, quantile_grids=grids,
        )
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(n_sims)
        np.random.set_state(rng_state)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
    print(f"sim {sim_results.results_matrix.shape}, contests={len(groups)}")

    own_vec = players_df["ownership"].astype(float).to_numpy()
    fp_cache = SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{seed}.npy"
    field_lineups = np.load(fp_cache) if fp_cache.exists() else ep.build_topn_field_pool(
        players_df, own_vec, fp_size, seed,
    )

    floor_scores = ep.compute_pool_ceiling_scores(pool, players_df)
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)),
    )
    elig = np.ones(len(pool.lineups), dtype=bool)
    if floor is not None:
        elig &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    elig_lineups = [lu for lu, e in zip(pool.lineups, elig) if e]
    print(f"eligible candidates: {len(elig_lineups)}  scales={SCALES}")

    I_pool = ep._lineup_indicator_matrix(elig_lineups, sim_results.player_ids)
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    fcols = np.array([[col_map[int(p)] for p in r] for r in field_lineups], dtype=np.int32)

    K = int(gpp.get("external_pool_topn_field_samples", 3))
    rank = int(gpp.get("external_pool_topn_rank", 10))
    pct_floor = float(gpp.get("external_pool_topn_percentile_floor", 0.001))
    allocator = ep._SimWorldAllocator(n_sims, seed)

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"])

    for ci, g in enumerate(groups):
        if not g.entries:
            continue
        field_size_g = ep._topn_field_size_for_group(g, fcols.shape[0])
        n_sims_g = ep._topn_sims_for_field_size(field_size_g, n_sims, frac, smin, sref, spow)
        # Consume in lockstep with production even for skipped contests, so a
        # resumed run gives every contest the same world slice as a fresh one.
        sim_idx_g = allocator.take(n_sims_g)
        if g.contest_id in done:
            print(f"[skip] {g.contest_name}")
            continue
        t0 = time.time()
        N = ep._topn_effective_rank(rank, field_size_g, pct_floor)
        rung_ranks, rung_weights = ep.topn_payout_rungs(g.contest_name, field_size_g, N, N_RUNGS)
        sub = sim_results.results_matrix[sim_idx_g].astype(np.float32)
        cand = (sub @ I_pool).T                       # (n_cand, n_sims_g)
        n_cand = cand.shape[0]
        rng = np.random.default_rng(seed + ci)

        perm = np.random.default_rng(seed + 1000 + ci).permutation(n_sims_g)
        h1, h2 = perm[: n_sims_g // 2], perm[n_sims_g // 2:]

        R = len(rung_ranks)
        arms = ["hard"] + [f"smooth{s:g}" for s in SCALES]
        # Accumulate half-sums per arm instead of materializing the full
        # (R, n_arms, n_cand, n_sims_g) currency -- that array would be ~10GB
        # here, which is the exact blowup CLAUDE.md's matrix-op rule warns
        # about. Only one (n_cand, n_sims_g) transient is ever live.
        acc = {(a, r, half): np.zeros(n_cand, dtype=np.float64)
               for a in arms for r in range(R) for half in (0, 1)}
        events = np.zeros(R, dtype=np.float64)
        tau_med = {r: [] for r in range(R)}

        # Every rung's threshold AND its bracket ranks come out of one
        # partition per field draw (same trick production uses).
        wanted = set()
        for rk in rung_ranks:
            lo, hi = ep._rung_bracket_ranks(int(rk), field_size_g)
            wanted |= {int(rk), lo, hi}
        kths = np.unique(-np.array(sorted(wanted), dtype=np.int64))

        for kk in range(K):
            subset = rng.choice(fcols.shape[0], size=field_size_g, replace=False)
            fs = ep._score_field_cols_batched(sub, fcols[subset])
            part = np.partition(fs, kths, axis=1)
            del fs
            for r, rk in enumerate(rung_ranks):
                thr = part[:, -int(rk)]                        # (n_sims_g,)
                hard = (cand >= thr[None, :])
                events[r] += float(hard.sum(axis=1).mean())
                acc[("hard", r, 0)] += hard[:, h1].sum(axis=1)
                acc[("hard", r, 1)] += hard[:, h2].sum(axis=1)
                del hard
                for s in SCALES:
                    tau = ep.smoothing_tau(part, int(rk), field_size_g, s)
                    tau_med[r].append(float(np.median(tau)))
                    z = (cand - thr[None, :]) / np.maximum(tau, 1e-6)[None, :]
                    soft = expit(ep._LOGISTIC_NORMAL_SCALE * z)
                    del z
                    acc[(f"smooth{s:g}", r, 0)] += soft[:, h1].sum(axis=1)
                    acc[(f"smooth{s:g}", r, 1)] += soft[:, h2].sum(axis=1)
                    del soft
            del part

        # Full-budget currency per arm/rung (both halves together) -- what the
        # greedy really consumes, and the basis for the drift columns.
        full = {(a, r): acc[(a, r, 0)] + acc[(a, r, 1)] for a in arms for r in range(R)}
        mean_score = cand.mean(axis=1)

        rows = []
        for r, rk in enumerate(rung_ranks):
            for a in arms:
                rho, rho_full, ov = _reliability(acc[(a, r, 0)], acc[(a, r, 1)])
                rows.append({
                    "contest_id": g.contest_id, "contest": g.contest_name,
                    "k": len(g.entries), "field_size_g": field_size_g,
                    "n_sims_g": n_sims_g, "N": N, "rung_rank": int(rk),
                    "weight": round(float(rung_weights[r]), 4), "arm": a,
                    "events_per_cand": round(float(events[r]), 2),
                    "tau_median": round(float(np.median(tau_med[r])), 4) if a != "hard" else 0.0,
                    "rho_half": round(rho, 4), "rho_full": round(rho_full, 4),
                    "top50_overlap": round(ov, 3),
                    # Denoising vs drifting -- see the module docstring.
                    "rho_vs_hard": round(_safe_spearman(full[(a, r)], full[("hard", r)]), 4),
                    "rho_vs_mean": round(_safe_spearman(full[(a, r)], mean_score), 4),
                })
        df = _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        base = {r["rung_rank"]: r for r in rows if r["arm"] == "hard"}
        summary = " | ".join(
            f"r{rk}: hard {base[rk]['rho_full']:.2f}"
            + "".join(
                f" / {a[6:]}x {[x for x in rows if x['arm'] == a and x['rung_rank'] == rk][0]['rho_full']:.2f}"
                for a in arms if a != "hard"
            )
            for rk in sorted(base)
        )
        print(f"[{ci+1}/{len(groups)}] {g.contest_name} F={field_size_g} "
              f"sims={n_sims_g} ({time.time()-t0:.0f}s)\n    {summary}")
        del sub, cand, acc

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        print("\n=== weighted mean rho_full by arm (payout-weight weighted) ===")
        for a, sub_df in df.groupby("arm"):
            w = sub_df["weight"] * sub_df["k"]
            print(f"  {a:>12}: {np.average(sub_df['rho_full'], weights=w):.4f}  "
                  f"(unweighted {sub_df['rho_full'].mean():.4f}, "
                  f"top50 {sub_df['top50_overlap'].mean():.3f}, "
                  f"vs_hard {sub_df['rho_vs_hard'].mean():.3f}, "
                  f"vs_mean {sub_df['rho_vs_mean'].mean():.3f})")
        print("\n  vs_mean climbing with tau = the currency drifting from "
              "field-relative toward field-blind chalk; see the module docstring.")
        print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
