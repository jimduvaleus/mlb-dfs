"""Does raising the sim budget actually buy portfolio stability, and at what
memory cost?

scripts/diagnose_topn_variance_decomposition.py attributed ~95% of
topn_coverage's cross-seed portfolio variance to SIM-WORLD resampling (vs ~36%
to field draws), so `external_pool_topn_sims_min` is the knob that matters.
This runs that same 2x2 at 1x / 2x / 3x the calibrated `sims_min`, with K
already cut 5 -> 3 (see models.py's note on that change).

FALSIFIABLE PREDICTION: if the cross-seed variation is ordinary Monte Carlo
error, the ace-exposure gap should shrink as 1/sqrt(n) --

    1x: 6.7pp (measured at K=5)   2x: ~4.7pp   3x: ~3.9pp

and per-player exposure rho should rise correspondingly. If the gap does NOT
shrink at that rate, the variation is structural -- genuine near-ties in an
objective with many equally-good optima -- and more sims will never resolve
it, which is a reason to STOP buying rather than buy more.

Note the 1x arm here is re-measured at K=3, so it is also the clean read on
what cutting K did on its own (the earlier 6.7pp was at K=5).

Each (multiplier, arm) records wall clock and sampled peak RSS; the user's
ceiling is ~11GB. n_sims is auto-sized per multiplier via
`topn_total_sims_needed`, so the simulation itself is re-run and cached once
per multiplier.

Checkpoint / resume per CLAUDE.md: rows land in
outputs/topn_sim_budget_sweep/results.csv as each arm finishes; arms already
on disk are skipped. TOPN_SWEEP_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/sweep_topn_sim_budget.py

Env vars
--------
    TOPN_SWEEP_MULTS   comma-separated sim_min multipliers (default 1,2,3)
    TOPN_SWEEP_FORCE   "1" re-runs arms already in results.csv
"""
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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
MULTS = [int(x) for x in os.environ.get("TOPN_SWEEP_MULTS", "1,2,3").split(",")]
FORCE = os.environ.get("TOPN_SWEEP_FORCE") == "1"
TRACK_PLAYER = "Tarik Skubal"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_sim_budget_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
PAIRS_CSV = OUT_DIR / "pairs.csv"
SIM_CACHE = OUT_DIR / "sim_cache"
SIM_CACHE.mkdir(parents=True, exist_ok=True)
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"

ARMS = [("A0_w42_f42", 42, 42), ("A1_w137_f42", 137, 42),
        ("A2_w42_f137", 42, 137), ("A3_w137_f137", 137, 137)]


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


class PeakRSS:
    def __init__(self, interval: float = 0.05):
        self.interval, self.peak = interval, 0.0
        self._stop, self._t = threading.Event(), None

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_gb())
                self._stop.wait(self.interval)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss_gb())
        return False


def exposures(lineups, pids):
    idx = {int(p): i for i, p in enumerate(pids)}
    out = np.zeros(len(pids))
    for lu in lineups:
        for p in lu.player_ids:
            j = idx.get(int(p))
            if j is not None:
                out[j] += 1
    return out / max(1, len(lineups))


def stack_exposure(lineups, team_of, pos_of):
    counts = {}
    for lu in lineups:
        teams = {}
        for p in lu.player_ids:
            if pos_of.get(int(p), "") != "P":
                t = team_of.get(int(p))
                if t:
                    teams[t] = teams.get(t, 0) + 1
        for t, c in teams.items():
            if c >= 3:
                counts[t] = counts.get(t, 0) + 1
    return {t: c / max(1, len(lineups)) for t, c in counts.items()}


def nn_mean(A, B):
    Bs = [set(lu.player_ids) for lu in B]
    return float(np.mean([max(len(set(a.player_ids) & b) for b in Bs) for a in A]))


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp, paths = cfg["gpp"], cfg["paths"]
    base_seed = int(gpp.get("rng_seed") or 42)
    K = int(gpp.get("external_pool_topn_field_samples", 3))
    base_sims_min = int(gpp.get("external_pool_topn_sims_min", 0))
    print(f"K={K}, base sims_min={base_sims_min}, multipliers={MULTS}")

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
    field_lineups = np.load(
        SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{base_seed}.npy")

    pids = players_df["player_id"].astype(int).to_numpy()
    team_of = dict(zip(pids, players_df["team"]))
    pos_of = dict(zip(pids, players_df["position"]))
    pid_track = int(players_df.loc[players_df["name"] == TRACK_PLAYER, "player_id"].iloc[0])

    grids = None
    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV)["key"].astype(str))

    alloc_pool = None
    for mult in MULTS:
        sims_min = base_sims_min * mult
        n_sims = max(int(cfg["simulation"].get("n_sims", 25_000)), ep.topn_total_sims_needed(
            groups, fp_size,
            float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
            sims_min,
            float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
            float(gpp.get("external_pool_topn_sims_power", 0.0)),
        ))
        print(f"\n########## {mult}x  (sims_min={sims_min}, n_sims={n_sims}) ##########")
        cache = SIM_CACHE / f"{found['projections_path'].stem}_{n_sims}_{base_seed}.npz"
        if cache.exists():
            with np.load(cache) as z:
                sim_results = SimulationResults(
                    [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64))
            print(f"sim: cache hit {sim_results.results_matrix.shape}")
        else:
            if grids is None:
                grids = ep.build_quantile_grids(
                    proj_ext,
                    zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
                    scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
                    mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
                    mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)),
                )
            engine = SimulationEngine(
                EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
                batter_pca_model=None, score_grid=None, quantile_grids=grids)
            t0 = time.time()
            rng_state = np.random.get_state()
            np.random.seed(base_seed)
            sim_results = engine.simulate(n_sims)
            np.random.set_state(rng_state)
            np.savez_compressed(
                cache, player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
                results_matrix=sim_results.results_matrix.astype(np.float32))
            print(f"sim: {n_sims} worlds in {time.time() - t0:.0f}s")

        if alloc_pool is None:
            from src.optimization.leverage import compute_generation_ownership_vec
            gen_own = compute_generation_ownership_vec(
                pool.lineups, sim_results, players_df,
                field_size=float(ep.pwin_field_size(
                    groups, floor=int(gpp.get("n_field_lineups", 5_000)))),
                blend_weight=float(gpp.get("external_pool_topn_generated_leverage_weight", 0.0)),
                sharpness=float(gpp.get("external_pool_pwin_sharpness", 0.05)))
            alloc_pool, _ = ep.augment_topn_pool_with_generated(
                pool, players_df, gen_own,
                int(gpp.get("external_pool_topn_generated_pool_size", 0)), base_seed + 1)
            floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
            print(f"alloc pool {len(alloc_pool.lineups)}")
        idx_of = {id(lu): i for i, lu in enumerate(alloc_pool.lineups)}

        common = dict(
            proj_scores=None,
            proj_score_floor_percentile=float(gpp.get("external_pool_proj_score_pct", 30.0)),
            floor_scores=floor_scores,
            topn_rank=int(gpp.get("external_pool_topn_rank", 10)),
            topn_percentile_floor=float(gpp.get("external_pool_topn_percentile_floor", 0.001)),
            field_samples=K,
            sims_per_contest_fraction=float(
                gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
            sims_min=sims_min,
            sims_reference_field_size=float(
                gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
            sims_power=float(gpp.get("external_pool_topn_sims_power", 0.0)),
        )

        for arm, w_seed, f_seed in ARMS:
            key = f"{mult}x_{arm}"
            pick_f = OUT_DIR / f"pick_idx_{key}.npy"
            if key in done and pick_f.exists():
                print(f"  [skip] {key}")
                continue
            cdone = []
            t0 = time.time()
            with PeakRSS() as peak:
                alloc = ep.allocate_contests_topn_coverage(
                    alloc_pool, sim_results, groups, field_lineups,
                    rng_seed=w_seed, field_rng_seed=f_seed, **common,
                    progress_cb=lambda i: cdone.append(i)
                    if i.get("event") == "contest_done" else None)
            pick_idx = np.array([idx_of[id(lu)] for lu, _ in alloc.portfolio])
            np.save(pick_f, pick_idx)
            wc = sum(r["worlds_claimed"] for r in cdone)
            ns = sum(r["n_sims_g"] for r in cdone)
            row = {
                "key": key, "mult": mult, "sims_min": sims_min, "n_sims": n_sims, "K": K,
                "arm": arm, "worlds_seed": w_seed, "fields_seed": f_seed,
                "elapsed_s": round(time.time() - t0, 1),
                "peak_rss_gb": round(peak.peak, 2),
                "ru_maxrss_gb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
                "track_exposure": round(float(np.mean(
                    [pid_track in lu.player_ids for lu, _ in alloc.portfolio])), 4),
                "worlds_claimed_pct": round(100 * wc / ns, 3),
                "mean_ceiling": round(float(floor_scores[pick_idx].mean()), 2),
                "n_sims_g_max": max(r["n_sims_g"] for r in cdone),
            }
            print(f"  {key}: {row['elapsed_s']:.0f}s  peak {row['peak_rss_gb']:.2f} GB  "
                  f"{TRACK_PLAYER} {row['track_exposure'] * 100:.1f}%  "
                  f"worlds {row['worlds_claimed_pct']}%")
            df = pd.DataFrame([row])
            if RESULTS_CSV.exists():
                old = pd.read_csv(RESULTS_CSV)
                df = pd.concat([old[old["key"].astype(str) != key], df], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
        del sim_results

    # --- pairwise divergence per multiplier ---
    res = pd.read_csv(RESULTS_CSV)
    rows = []
    for mult in sorted(res["mult"].unique()):
        def load(arm):
            return [alloc_pool.lineups[i]
                    for i in np.load(OUT_DIR / f"pick_idx_{mult}x_{arm}.npy")]
        try:
            A0 = load("A0_w42_f42")
        except FileNotFoundError:
            continue
        e0 = exposures(A0, pids)
        s0 = stack_exposure(A0, team_of, pos_of)
        x0 = res[(res["mult"] == mult) & (res["arm"] == "A0_w42_f42")].iloc[0]["track_exposure"]
        for arm, label in (("A1_w137_f42", "sim worlds only"),
                           ("A2_w42_f137", "field draws only"),
                           ("A3_w137_f137", "both")):
            try:
                B = load(arm)
            except FileNotFoundError:
                continue
            eb = exposures(B, pids)
            used = (e0 + eb) > 0
            sb = stack_exposure(B, team_of, pos_of)
            teams = sorted(set(s0) | set(sb))
            xb = res[(res["mult"] == mult) & (res["arm"] == arm)].iloc[0]["track_exposure"]
            rows.append({
                "mult": mult, "changed": label,
                "identity_overlap": round(
                    len(set(map(id, A0)) & set(map(id, B))) / len(A0), 3),
                "nn_mean": round(nn_mean(A0, B), 2),
                "expo_rho": round(float(spearmanr(e0[used], eb[used]).statistic), 3),
                "stack_rho": round(float(spearmanr(
                    [s0.get(t, 0) for t in teams], [sb.get(t, 0) for t in teams]).statistic), 3),
                "ace_gap_pp": round(abs(xb - x0) * 100, 1),
            })
    pdf = pd.DataFrame(rows)
    pdf.to_csv(PAIRS_CSV, index=False)

    print("\n=== cost per multiplier (max over arms) ===")
    print(res.groupby("mult").agg(
        n_sims=("n_sims", "max"), n_sims_g_max=("n_sims_g_max", "max"),
        elapsed_s=("elapsed_s", "max"), peak_rss_gb=("peak_rss_gb", "max"),
    ).to_string())
    print("\n=== divergence vs A0, per multiplier ===")
    print(pdf.to_string(index=False))

    both = pdf[pdf["changed"] == "both"].sort_values("mult")
    if len(both) > 1:
        base = both.iloc[0]
        print(f"\n=== 1/sqrt(n) check (ace-exposure gap, 'both' arm) ===")
        for _, r in both.iterrows():
            pred = base["ace_gap_pp"] / np.sqrt(r["mult"] / base["mult"])
            print(f"  {int(r['mult'])}x: measured {r['ace_gap_pp']:.1f}pp   "
                  f"1/sqrt(n) predicts {pred:.1f}pp")
    print(f"\nfull tables: {RESULTS_CSV} , {PAIRS_CSV}")


if __name__ == "__main__":
    main()
