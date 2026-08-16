"""Has each payout-ladder RUNG settled at the per-contest sim budget?

`external_pool_topn_sims_min/reference_field_size/power` were calibrated
(scripts/calibrate_topn_sims_per_contest.py) for the single top-N RANK bar to
settle. The payout ladder (`allocate_contests_topn_coverage(payout_rungs=...)`)
adds much tighter rungs, and rank 1 is a far rarer event than rank N -- ~18x
rarer in a 17,857-entry contest -- while carrying the ladder's LARGEST weight.
A rung whose per-candidate claim rate is estimated from a handful of worlds,
weighted most heavily, is exactly the shape of a winner's curse: the greedy
would chase sampling noise.

Raising n_sims is not free: the transient field-score array is
(n_sims_g x field_size_g) float32, ~768MB for the largest contest at the
current budget, so it scales linearly with any sim increase. This script
measures whether the increase is NEEDED before anyone pays for it.

Two measurements per rung, at the production per-contest sim budget:

  events   mean (draw, world) slots a candidate claims. The raw sample size
           behind every per-candidate estimate on that rung.

  split-half stability
           the contest's sim worlds are split into two disjoint halves, each
           candidate's per-rung claim RATE is computed on each half
           independently, and the two are compared by Spearman rho over the
           candidate pool. This is the direct question the greedy cares about:
           would I rank candidates the same way on an independent set of
           worlds? rho ~1 means settled; rho near 0 means that rung is noise
           and the ladder is chasing it. Reported alongside `top50_overlap`
           (Jaccard of each half's top-50 candidates), because the greedy
           only ever consumes the very top of the ranking, not the whole of it.

Both halves use the SAME thresholds (drawn on the full world set) so the split
isolates sim-world sampling noise in the CANDIDATE estimates, which is what
n_sims controls -- not field-draw noise, which is what `field_samples` (K)
controls and which no amount of extra sims would fix.

Checkpoint / resume per CLAUDE.md: rows appended per contest to
outputs/topn_rung_settling/results.csv; contests already on disk are skipped.
TOPN_SETTLE_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/diagnose_topn_rung_settling.py

Env vars
--------
    TOPN_REQ_RAW        slate input dir (default data/raw)
    TOPN_SETTLE_RUNGS   rungs to probe (default 5)
    TOPN_SETTLE_FORCE   "1" re-runs contests already in results.csv
"""
import os
import sys
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
N_RUNGS = int(os.environ.get("TOPN_SETTLE_RUNGS", "5"))
FORCE = os.environ.get("TOPN_SETTLE_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_rung_settling"
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
    n_sims = max(n_sims_cfg, ep.topn_total_sims_needed(
        groups, fp_size,
        float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
        int(gpp.get("external_pool_topn_sims_min", 0)),
        float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
        float(gpp.get("external_pool_topn_sims_power", 0.0)),
    ))

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
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)),
    )
    elig = np.ones(len(alloc_pool.lineups), dtype=bool)
    if floor is not None:
        elig &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    elig_lineups = [lu for lu, e in zip(alloc_pool.lineups, elig) if e]
    print(f"eligible candidates: {len(elig_lineups)}")

    I_pool = ep._lineup_indicator_matrix(elig_lineups, sim_results.player_ids)
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    fcols = np.array([[col_map[int(p)] for p in r] for r in field_lineups], dtype=np.int32)

    K = int(gpp.get("external_pool_topn_field_samples", 5))
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
        n_sims_g = ep._topn_sims_for_field_size(
            field_size_g, n_sims,
            float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
            int(gpp.get("external_pool_topn_sims_min", 0)),
            float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
            float(gpp.get("external_pool_topn_sims_power", 0.0)),
        )
        # Consume the allocator in lockstep with production so each contest
        # sees the same world slice it would in a real run.
        sim_idx_g = allocator.take(n_sims_g)
        if g.contest_id in done:
            print(f"[skip] {g.contest_name}")
            continue
        t0 = time.time()
        N = ep._topn_effective_rank(rank, field_size_g, pct_floor)
        rung_ranks, rung_weights = ep.topn_payout_rungs(
            g.contest_name, field_size_g, N, N_RUNGS,
        )
        sub = sim_results.results_matrix[sim_idx_g].astype(np.float32)
        cand = (sub @ I_pool).T                       # (n_cand, n_sims_g)
        rng = np.random.default_rng(seed + ci)

        R = len(rung_ranks)
        claims = np.zeros((R, cand.shape[0], n_sims_g), dtype=bool)
        kths = np.unique(-rung_ranks)
        for kk in range(K):
            subset = rng.choice(fcols.shape[0], size=field_size_g, replace=False)
            fs = ep._score_field_cols_batched(sub, fcols[subset])
            part = np.partition(fs, kths, axis=1)
            for r, rk in enumerate(rung_ranks):
                claims[r] |= cand >= part[:, -int(rk)][None, :]
            del fs, part

        # Disjoint halves of THIS contest's worlds; same thresholds for both,
        # so only candidate-side sim noise is being measured.
        perm = np.random.default_rng(seed + 1000 + ci).permutation(n_sims_g)
        h1, h2 = perm[: n_sims_g // 2], perm[n_sims_g // 2:]
        rows = []
        for r, rk in enumerate(rung_ranks):
            a = claims[r][:, h1].mean(axis=1)
            b = claims[r][:, h2].mean(axis=1)
            rho = float(spearmanr(a, b).statistic) if a.std() > 0 and b.std() > 0 else float("nan")
            top_a = set(np.argsort(-a)[:50].tolist())
            top_b = set(np.argsort(-b)[:50].tolist())
            rows.append({
                "contest_id": g.contest_id, "contest": g.contest_name,
                "k": len(g.entries), "field_size_g": field_size_g,
                "n_sims_g": n_sims_g, "N": N, "rung_rank": int(rk),
                "weight": round(float(rung_weights[r]), 4),
                "events_per_cand": round(float(claims[r].sum(axis=1).mean()), 1),
                "claim_rate": round(float(claims[r].mean()), 5),
                "split_half_rho": round(rho, 4),
                "top50_overlap": round(len(top_a & top_b) / 50.0, 3),
            })
        _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        print(f"[{ci + 1}/{len(groups)}] {g.contest_name[:34]:<34s} "
              f"field={field_size_g:6d} n_sims_g={n_sims_g:6d} N={N:3d} ({time.time() - t0:.0f}s)")
        for row in rows:
            print(f"     rank {row['rung_rank']:>3d}  w={row['weight']:.3f}  "
                  f"events={row['events_per_cand']:>7.1f}  rho={row['split_half_rho']:.3f}  "
                  f"top50={row['top50_overlap']:.2f}")
        del sub, cand, claims
        from src.optimization import self_play as _sp
        _sp._release_free_memory()

    t = pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})
    print("\n=== entry-weighted per-rung settling (rank as a fraction of N) ===")
    t["rank_frac"] = (t["rung_rank"] / t["N"]).round(2)
    agg = t.groupby("rank_frac").apply(lambda d: pd.Series({
        "mean_weight": np.average(d["weight"], weights=d["k"]),
        "events_per_cand": np.average(d["events_per_cand"], weights=d["k"]),
        "split_half_rho": np.average(d["split_half_rho"], weights=d["k"]),
        "top50_overlap": np.average(d["top50_overlap"], weights=d["k"]),
    }), include_groups=False)
    print(agg.round(3).to_string())
    print(f"\nfull table: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
