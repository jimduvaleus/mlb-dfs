"""A/B the duplicate-discounted coverage gain (`allocate_contests_topn_
coverage(e_dupes=...)`) at PRODUCTION scale, with wall-clock and peak-RSS.

Background (2026-08-11): the 08/10 portfolio came back 84.9% Tarik Skubal vs
47.2% field ownership. Two diagnostics established that the coverage
objective only WEAKLY prefers that concentration -- capping exposure at 50%
costs ~3.3% of covered slots in multi-entry contests (scripts/
diagnose_topn_exposure_cap.py) -- while the objective is structurally unable
to price one thing the concentration costs: the top-N bar is a RANK bar, so a
heavily-duplicated lineup that crosses it splits whatever it wins, and
nothing in the greedy sees that. `e_dupes` is the cheapest correction: a
per-candidate scalar 1/(1 + E[dupes_g]), reusing the already-fitted
production dupe model (`gpp.dupe_*` / scripts/fit_dupe_model.py).

FALSIFIABLE PREDICTIONS being tested here (stated before the run):

  P1  Skubal exposure falls -- he is the chalkiest player on the slate.
  P2  the portfolio's mean E[dupes] falls MATERIALLY. This is the mechanism
      check: if it doesn't move, the discount is inert and P1/P3 are noise.
  P3  worlds_claimed falls only into the CHEAP band (~1-3%), not the 9-27%
      band the cap sweep showed is expensive.

Any of: exposure flat, E[dupes] flat, or coverage falling into the expensive
band => the discount is not doing what it is supposed to do.

Production-faithful, unlike the two earlier diagnostics (which shared one
sim-world set across contests for comparability): this calls the REAL
`allocate_contests_topn_coverage` with `_SimWorldAllocator`'s disjoint
per-contest slices and n_sims auto-sized by `topn_total_sims_needed` exactly
as pipeline.py's topn_coverage branch does, so the timing and memory numbers
are the ones a live run would see.

Peak RSS is sampled by a background thread (50ms) rather than read once at
the end -- the allocator's whole design is transient per-contest arrays that
are freed again (see its docstring), so a post-hoc reading would miss the
peak entirely. ru_maxrss is reported alongside as an independent check.

Checkpoint / resume per CLAUDE.md: each arm's summary row is appended to
outputs/topn_dupe_discount/results.csv as it finishes; an arm already on disk
is skipped. TOPN_DUPE_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/eval_topn_dupe_discount.py

Env vars
--------
    TOPN_REQ_RAW     slate input dir (default data/raw)
    TOPN_DUPE_FORCE  "1" re-runs arms already in results.csv
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
FORCE = os.environ.get("TOPN_DUPE_FORCE") == "1"
TRACK_PLAYER = "Tarik Skubal"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_dupe_discount"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
CONTEST_CSV = OUT_DIR / "per_contest.csv"
SIM_CACHE = OUT_DIR / "sim_cache"
SIM_CACHE.mkdir(parents=True, exist_ok=True)


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


class PeakRSS:
    """Background sampler: the allocator frees its big per-contest arrays as
    it goes, so only a running sample catches the true high-water mark."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = None

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


def _append_and_reload(csv_path: Path, key_col: str, key: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old[key_col].astype(str) != str(key)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


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

    # --- n_sims auto-sizing, exactly as pipeline.py's topn branch ---
    n_sims_cfg = int(cfg["simulation"].get("n_sims", 25_000))
    demand = ep.topn_total_sims_needed(
        groups, fp_size,
        float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
        int(gpp.get("external_pool_topn_sims_min", 0)),
        float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
        float(gpp.get("external_pool_topn_sims_power", 0.0)),
    )
    n_sims = max(n_sims_cfg, demand)
    print(f"contests={len(groups)} entries={sum(len(g.entries) for g in groups)} "
          f"pool={len(pool.lineups)}  n_sims {n_sims_cfg} -> {n_sims} (auto-sized)")

    cache = SIM_CACHE / f"{found['projections_path'].stem}_{n_sims}_{seed}.npz"
    if cache.exists():
        with np.load(cache) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
        print(f"sim: cache hit ({sim_results.results_matrix.shape})")
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
        t0 = time.time()
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(n_sims)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache, player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
        print(f"sim: {n_sims} worlds in {time.time() - t0:.0f}s")

    own_vec = players_df["ownership"].astype(float).to_numpy()
    fp_cache = OUT_DIR / f"field_pool_{found['projections_path'].stem}_{fp_size}_{seed}.npy"
    if fp_cache.exists():
        field_lineups = np.load(fp_cache)
    else:
        t0 = time.time()
        field_lineups = ep.build_topn_field_pool(players_df, own_vec, fp_size, seed)
        np.save(fp_cache, field_lineups)
        print(f"field pool: {fp_size} in {time.time() - t0:.0f}s")

    # --- generated-pool augmentation + floor scores, as pipeline does ---
    n_gen = int(gpp.get("external_pool_topn_generated_pool_size", 0))
    alloc_pool = pool
    if n_gen > 0:
        gen_w = float(gpp.get("external_pool_topn_generated_leverage_weight", 0.0))
        gen_own = own_vec
        if gen_w > 0:
            from src.optimization.leverage import compute_generation_ownership_vec
            gen_own = compute_generation_ownership_vec(
                pool.lineups, sim_results, players_df,
                field_size=float(ep.pwin_field_size(
                    groups, floor=int(gpp.get("n_field_lineups", 5_000)))),
                blend_weight=gen_w,
                sharpness=float(gpp.get("external_pool_pwin_sharpness", 0.05)),
            )
        alloc_pool, _ = ep.augment_topn_pool_with_generated(
            pool, players_df, gen_own, n_gen, seed + 1,
        )
    floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
    print(f"alloc pool: {len(alloc_pool.lineups)}")

    # --- E[dupes] at the model's reference field size (once, whole pool) ---
    t0 = time.time()
    e_dupes = ep.compute_pool_e_dupes(
        alloc_pool.lineups, players_df,
        intercept=float(gpp.get("dupe_intercept", 3.698)),
        log_own_coef=float(gpp.get("dupe_log_own_coef", 0.212)),
        salary_coef=float(gpp.get("dupe_salary_coef", 0.089)),
        stack_coef=float(gpp.get("dupe_stack_coef", 0.024)),
    )
    print(f"E[dupes] (ref 14,863 entries): p10={np.percentile(e_dupes, 10):.2f} "
          f"p50={np.percentile(e_dupes, 50):.2f} p90={np.percentile(e_dupes, 90):.2f} "
          f"max={e_dupes.max():.1f}  [{time.time() - t0:.1f}s for {len(e_dupes)} lineups]")

    pid_track = int(players_df.loc[players_df["name"] == TRACK_PLAYER, "player_id"].iloc[0])
    common = dict(
        proj_scores=None,
        proj_score_floor_percentile=float(gpp.get("external_pool_proj_score_pct", 30.0)),
        floor_scores=floor_scores,
        topn_rank=int(gpp.get("external_pool_topn_rank", 10)),
        topn_percentile_floor=float(gpp.get("external_pool_topn_percentile_floor", 0.001)),
        field_samples=int(gpp.get("external_pool_topn_field_samples", 5)),
        sims_per_contest_fraction=float(
            gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
        sims_min=int(gpp.get("external_pool_topn_sims_min", 0)),
        sims_reference_field_size=float(
            gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
        sims_power=float(gpp.get("external_pool_topn_sims_power", 0.0)),
        rng_seed=seed,
    )

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV)["arm"].astype(str))

    idx_of = {id(lu): i for i, lu in enumerate(alloc_pool.lineups)}
    for arm, ed in [("baseline", None), ("dupe_discount", e_dupes)]:
        if arm in done:
            print(f"[skip] arm {arm}")
            continue
        contest_rows = []

        def cb(info, _rows=contest_rows):
            if info.get("event") == "contest_done":
                _rows.append(info)

        print(f"\n--- arm: {arm} ---")
        t0 = time.time()
        with PeakRSS() as peak:
            alloc = ep.allocate_contests_topn_coverage(
                alloc_pool, sim_results, groups, field_lineups,
                e_dupes=ed, progress_cb=cb, **common,
            )
        elapsed = time.time() - t0
        ru_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

        picks = [lu for lu, _ in alloc.portfolio]
        pick_idx = np.array([idx_of[id(lu)] for lu in picks])
        # Pool indices of this arm's picks, in fill order -- lets the summary
        # report how many of the 119 picks actually CHANGED between arms,
        # which is what separates "a few swaps" from "a broad reshuffle"
        # behind the same aggregate E[dupes] move.
        np.save(OUT_DIR / f"pick_idx_{arm}.npy", pick_idx)
        exposure = float(np.mean([pid_track in lu.player_ids for lu in picks]))
        worlds_claimed = sum(r["worlds_claimed"] for r in contest_rows)
        n_sims_g_total = sum(r["n_sims_g"] for r in contest_rows)
        row = {
            "arm": arm, "n_picks": len(picks),
            "elapsed_s": round(elapsed, 1),
            "peak_rss_gb": round(peak.peak, 2),
            "ru_maxrss_gb": round(ru_gb, 2),
            "track_exposure": round(exposure, 4),
            "portfolio_mean_e_dupes": round(float(e_dupes[pick_idx].mean()), 3),
            "portfolio_median_e_dupes": round(float(np.median(e_dupes[pick_idx])), 3),
            "worlds_claimed": worlds_claimed,
            "n_sims_g_total": n_sims_g_total,
            "worlds_claimed_pct": round(100 * worlds_claimed / n_sims_g_total, 3),
            "mean_slots_per_pick": round(float(np.mean([ev for _, ev in alloc.portfolio])), 1),
            "n_relaxations": sum(r["n_relaxations"] for r in contest_rows),
            "n_wave_resets": sum(r["n_wave_resets"] for r in contest_rows),
            "n_unfilled": len(alloc.unfilled),
        }
        print(f"  {elapsed:.0f}s  peak RSS {peak.peak:.2f} GB (ru_maxrss {ru_gb:.2f})  "
              f"{TRACK_PLAYER} {exposure * 100:.1f}%  "
              f"E[dupes] mean {row['portfolio_mean_e_dupes']}  "
              f"worlds {row['worlds_claimed_pct']}%")
        _append_and_reload(RESULTS_CSV, "arm", arm, [row])
        _append_and_reload(CONTEST_CSV, "arm", arm, [
            {"arm": arm, "contest_id": r["contest_id"], "k": r["k"],
             "n_filled": r["n_filled"], "n_sims_g": r["n_sims_g"],
             "worlds_claimed": r["worlds_claimed"],
             "worlds_claimed_pct": r["worlds_claimed_pct"],
             "n_relaxations": r["n_relaxations"], "n_wave_resets": r["n_wave_resets"],
             "elapsed_s": round(r["elapsed_s"], 1)}
            for r in contest_rows
        ])

    t = pd.read_csv(RESULTS_CSV).set_index("arm")
    print("\n=== A/B summary ===")
    print(t.to_string())
    if {"baseline", "dupe_discount"} <= set(t.index):
        b, d = t.loc["baseline"], t.loc["dupe_discount"]
        print(f"\nP1 {TRACK_PLAYER} exposure : {b.track_exposure * 100:.1f}% -> "
              f"{d.track_exposure * 100:.1f}%  ({(d.track_exposure - b.track_exposure) * 100:+.1f} pp)")
        print(f"P2 portfolio mean E[dupes]: {b.portfolio_mean_e_dupes:.2f} -> "
              f"{d.portfolio_mean_e_dupes:.2f}  "
              f"({100 * (d.portfolio_mean_e_dupes / b.portfolio_mean_e_dupes - 1):+.1f}%)")
        print(f"P3 worlds claimed         : {b.worlds_claimed_pct:.2f}% -> "
              f"{d.worlds_claimed_pct:.2f}%  "
              f"({100 * (d.worlds_claimed / b.worlds_claimed - 1):+.2f}% relative)")
        print(f"   wall clock             : {b.elapsed_s:.0f}s -> {d.elapsed_s:.0f}s")
        print(f"   peak RSS               : {b.peak_rss_gb:.2f} -> {d.peak_rss_gb:.2f} GB")
        pb = OUT_DIR / "pick_idx_baseline.npy"
        pd_ = OUT_DIR / "pick_idx_dupe_discount.npy"
        if pb.exists() and pd_.exists():
            a, c = set(np.load(pb).tolist()), set(np.load(pd_).tolist())
            print(f"   picks changed          : {len(a - c)} of {len(a)} "
                  f"({100 * len(a - c) / len(a):.0f}% of the portfolio replaced)")
    print(f"\nper-contest: {CONTEST_CSV}")


if __name__ == "__main__":
    main()
