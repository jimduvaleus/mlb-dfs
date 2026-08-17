"""Can the p_win stage afford 25,000 sim worlds PER STAGE (n_sims 50,000)?

[[project-pwin-reliability]] measured p_win's split-half rho at 0.880 and
attributed 100% of the divergence to the sim-world axis and ~2% to field
sampling. The A/B winner's-curse guard halves n_sims per stage, so it spends
2x of the one resource that owns the error; Spearman-Brown puts 25,000
worlds/stage at rho 0.935 vs the shipped 0.880. The fix is `simulation.n_sims`
rather than a new estimator -- IF it fits. This measures whether it does.

Two constraints, both measured here rather than estimated:

  memory  the largest arrays scale linearly in n_sims. Critically,
          pipeline.py's p_win branch builds `_field_scores_A` AND
          `_field_scores_B` -- each (n_sims/2 x F) float32 -- and holds BOTH
          alive while it computes p_win from each in turn. At F=25,000 that
          pair is 2.5GB today and 5.0GB at n_sims=50,000. This script also
          measures an INTERLEAVED ordering (score A -> p_win A -> free A ->
          score B -> p_win B) which is behaviour-identical, since
          _p_win_cull only ever reads field_scores_A and _p_win_select only
          field_scores_B.

  time    simulate() + 2x generate_field + 2x score_field + 2x compute_p_win.
          n_sims is a GLOBAL knob, so the simulation and every other
          n_sims-scaling consumer (compute_lineup_scores, compute_pool_corr)
          are timed too, not just the p_win stage.

Arms are run in separate SUBPROCESSES so one arm's peak RSS cannot be
attributed to another and glibc cannot carry a freed arena across arms.

Checkpoint / resume per CLAUDE.md: one row per arm appended to
outputs/pwin_scalecheck/results.csv; arms already on disk are skipped.

Usage
-----
    source venv/bin/activate
    python scripts/scalecheck_pwin_n_sims.py 2>&1 | tee /tmp/pwinsc.log

Env vars
--------
    TOPN_REQ_RAW       slate input dir (default data/raw)
    PWIN_SC_ARMS       comma-separated (default "25000,50000,50000_interleaved,50000_interleaved_lean")
    PWIN_SC_FORCE      "1" re-runs arms already in results.csv
    PWIN_SC_CHILD      internal -- marks the subprocess arm
"""
import json
import os
import subprocess
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

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
ARMS = os.environ.get(
    "PWIN_SC_ARMS",
    "25000,50000,50000_interleaved,50000_interleaved_lean,50000_corrsub",
).split(",")
FORCE = os.environ.get("PWIN_SC_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "pwin_scalecheck"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


class PeakRSS:
    """Background sampler -- the stage frees its big arrays as it goes, so only
    a running sample catches the true high-water mark."""

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


def run_arm(arm: str) -> dict:
    """One arm, in-process (this is the child). Reproduces pipeline.py's p_win
    branch faithfully, including which arrays are alive at the same time."""
    import yaml as _yaml
    from src.api import external_pool as ep
    from src.api.dk_entries import parse_entry_file
    from src.api.pipeline import PipelineRunner
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    from src.models.copula import EmpiricalCopula
    from src.optimization.contest import ContestSimulator
    from src.simulation.engine import SimulationEngine

    interleaved = "interleaved" in arm
    # "_lean" adds the two cost fixes the probe isolated (see the module
    # docstring's memory note): score_field against a float32 view of the sim
    # matrix, a smaller batch, and an explicit malloc_trim after freeing each
    # field-score array.
    lean = "lean" in arm
    corrsub = "corrsub" in arm
    n_sims = int(arm.split("_")[0])

    cfg = _yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp, paths = cfg["gpp"], cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)
    timings = {}
    peak_after = {}
    overall = PeakRSS()
    overall.__enter__()

    t = time.time()
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
    fe = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            fe.append((p, recs))
    groups = ep.group_and_match_contests(fe, pool)
    timings["load"] = time.time() - t

    # --- simulate (the global n_sims knob) -------------------------------
    t = time.time()
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
    st = np.random.get_state()
    np.random.seed(seed)
    sim_results = engine.simulate(n_sims)
    np.random.set_state(st)
    timings["simulate"] = time.time() - t
    peak_after["simulate"] = rss_gb()

    # --- n_sims-scaling consumers shared with every ev_type --------------
    t = time.time()
    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)   # (M, S) f32
    timings["lineup_scores"] = time.time() - t
    # "_corrsub" feeds the DIVERSITY term's correlation a fixed 25,000-world
    # subsample while p_win keeps every world. Justified because the two terms
    # have wildly different budget needs: the diversity ordering measures
    # split-half rho 0.976-0.999 (it is a bulk statistic over all worlds and is
    # already fully settled), whereas p_win concentrates its weight on the few
    # worlds where a candidate tops the field. precompute_pool is the single
    # largest allocation in the whole path -- ~5.6x the (M x n_sims) float32
    # score matrix -- so this is where raising n_sims actually costs.
    corr_scores = lineup_scores[:, :25_000] if corrsub else lineup_scores
    t = time.time()
    corr = ep.compute_pool_corr(pool.lineups, sim_results, scores=corr_scores)  # (M, M) f32
    timings["pool_corr"] = time.time() - t
    peak_after["pool_corr"] = rss_gb()

    # --- p_win stage, mirroring pipeline.py ------------------------------
    sharpness = float(gpp.get("external_pool_pwin_sharpness", 0.05))
    flat_ref = float(gpp.get("external_pool_pwin_flat_reference", 0.0))
    fs_cfg = int(gpp.get("external_pool_pwin_field_size", 0))
    field_n = fs_cfg if fs_cfg > 0 else ep.pwin_field_size(
        groups, floor=int(gpp.get("n_field_lineups", 5_000)))
    exponents = ep.pwin_exponents(groups, sharpness, flat_ref)
    n_half = n_sims // 2
    own_vec = players_df["ownership"].astype(float).to_numpy()
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    sims_A = sim_results.results_matrix[:n_half]
    sims_B = sim_results.results_matrix[n_half:2 * n_half]
    scores_A = lineup_scores[:, :n_half]
    scores_B = lineup_scores[:, n_half:2 * n_half]

    cs = ContestSimulator()
    t = time.time()
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed)
    field_B = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed + 1)
    timings["generate_field_x2"] = time.time() - t

    from src.optimization.self_play import _release_free_memory

    # score_field does `sim_matrix[:, batch_cols]`, materializing a
    # (n_sims, batch, 10) intermediate per batch. At n_sims=25,000/batch=500
    # off a float64 results_matrix that is a 1.0 GB transient, allocated and
    # freed ~50 times -- the peak driver, and the alloc/free churn that leaves
    # glibc holding the arena afterwards.
    sf_kw = {} if not lean else {"batch_size": 125}
    if lean:
        sims_A = sims_A.astype(np.float32)
        sims_B = sims_B.astype(np.float32)

    def _free():
        if lean:
            _release_free_memory()

    t = time.time()
    if interleaved:
        # Behaviour-identical reorder: _p_win_cull only reads field_scores_A and
        # _p_win_select only field_scores_B, so B need not exist while A is used.
        fsa = cs.score_field(field_A, sims_A, col_map, **sf_kw)
        p_win_cull = ep.compute_p_win(scores_A, fsa, exponents)
        del fsa
        _free()
        fsb = cs.score_field(field_B, sims_B, col_map, **sf_kw)
        p_win_select = ep.compute_p_win(scores_B, fsb, exponents)
        del fsb
        _free()
    else:
        # Exactly today's ordering: BOTH field-score arrays alive at once.
        fsa = cs.score_field(field_A, sims_A, col_map, **sf_kw)
        fsb = cs.score_field(field_B, sims_B, col_map, **sf_kw)
        p_win_cull = ep.compute_p_win(scores_A, fsa, exponents)
        p_win_select = ep.compute_p_win(scores_B, fsb, exponents)
        del fsa, fsb
    timings["pwin_stage"] = time.time() - t
    peak_after["pwin_stage"] = rss_gb()
    overall.__exit__()

    return {
        "arm": arm, "n_sims": n_sims, "worlds_per_stage": n_half,
        "interleaved": interleaved, "lean": lean, "corrsub": corrsub,
        "field_n": field_n,
        "pool_M": len(pool.lineups), "n_contests": len(groups),
        "peak_rss_gb": round(overall.peak, 2),
        "total_s": round(sum(timings.values()), 1),
        **{f"t_{k}": round(v, 1) for k, v in timings.items()},
        **{f"rss_after_{k}_gb": round(v, 2) for k, v in peak_after.items()},
        "sanity_contests_scored": len(p_win_select),
    }


def main() -> None:
    if os.environ.get("PWIN_SC_CHILD"):
        row = run_arm(os.environ["PWIN_SC_CHILD"])
        print("__RESULT__" + json.dumps(row))
        return

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV)["arm"].astype(str))

    for arm in ARMS:
        if arm in done:
            print(f"[skip] {arm}")
            continue
        print(f"\n--- arm: {arm} (subprocess) ---")
        env = {**os.environ, "PWIN_SC_CHILD": arm}
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
            raise SystemExit(f"arm {arm} failed ({proc.returncode})")
        line = next(l for l in proc.stdout.splitlines() if l.startswith("__RESULT__"))
        row = json.loads(line[len("__RESULT__"):])
        df = pd.DataFrame([row])
        if RESULTS_CSV.exists():
            old = pd.read_csv(RESULTS_CSV)
            old = old[old["arm"].astype(str) != arm]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"  wall {time.time()-t0:.0f}s  peak RSS {row['peak_rss_gb']} GB  "
              f"simulate {row['t_simulate']}s  pwin_stage {row['t_pwin_stage']}s")

    t = pd.read_csv(RESULTS_CSV).set_index("arm")
    cols = [c for c in ["n_sims", "worlds_per_stage", "peak_rss_gb", "total_s",
                        "t_simulate", "t_lineup_scores", "t_pool_corr",
                        "t_generate_field_x2", "t_pwin_stage"] if c in t.columns]
    print("\n=== p_win n_sims scale check ===")
    print(t[cols].to_string())


if __name__ == "__main__":
    main()
