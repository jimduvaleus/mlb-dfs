"""How much of a pitcher's topn_coverage exposure is mechanically REQUIRED?

Motivating question (2026-08-11): the 08/10 portfolio came back 84.9% Tarik
Skubal under `external_pool_ev_type: topn_coverage`, against 47.2% field
ownership. Is that what the coverage objective genuinely demands, or is the
greedy over-buying?

`allocate_contests_topn_coverage` maximizes coverage of (field draw, sim
world) slots where a candidate crosses the field's rank-N bar. So the
mechanically-justified exposure floor for pitcher X is answerable directly,
without running the greedy at all:

    for each (field draw, world) slot, does the BEST candidate in the
    eligible pool that does NOT roster X still cross that slot's bar?

Taking the max over every non-X candidate is the right question because it's
an upper bound on what any non-X *portfolio* could claim: if the best non-X
lineup misses the bar in a world, no non-X lineup claims that world, no
matter how many you select. So

    required_X = P(slot is claimable at all  AND  not claimable without X)

is a LOWER bound on the exposure to X that pure coverage requires. Exposure
materially above `required_X / claimable` is the greedy's pick-ordering
(marginal-gain sequencing, per-contest `uncovered` resets), not the objective.

Mirrors the production topn path exactly for everything that sets the bar:
same pool parse + near-duplicate cull, same `build_external_players_df`,
same quantile grids/config knobs, same ownership-weighted field pool
(`build_topn_field_pool`), same leverage-weighted generated-pool
augmentation, same ceiling-based 30% floor cull, same per-contest
`field_size_g` / `_topn_effective_rank` / `field_samples` / per-contest RNG
seeding.

ONE DELIBERATE DEVIATION: every contest is evaluated over the SAME shared
sim-world set rather than `_SimWorldAllocator`'s disjoint per-contest slices.
Disjointness exists to stop two contests' greedy races converging on the same
lineup -- there is no greedy here, and using one common world set makes the
per-contest numbers directly comparable (the only thing that differs between
contests is then the bar itself: field_size_g and N).

Checkpoint / resume (CLAUDE.md's long-running-script rule): each contest's
rows are appended to outputs/topn_pitcher_requirement/results.csv as soon as
that contest finishes; a contest already on disk is skipped on re-invocation.
Set TOPN_REQ_FORCE=1 to redo everything.

Usage
-----
    source venv/bin/activate
    python scripts/diagnose_topn_pitcher_requirement.py

Env vars
--------
    TOPN_REQ_RAW     slate input dir (default data/raw)
    TOPN_REQ_NSIMS   shared sim-world count (default 12000)
    TOPN_REQ_NPITCH  how many pitchers to report (default 12)
    TOPN_REQ_FORCE   "1" re-runs contests already in results.csv
"""
import os
import sys
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
N_SIMS = int(os.environ.get("TOPN_REQ_NSIMS", "12000"))
N_PITCHERS = int(os.environ.get("TOPN_REQ_NPITCH", "12"))
FORCE = os.environ.get("TOPN_REQ_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_pitcher_requirement"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SIM_CACHE = OUT_DIR / "sim_cache"
SIM_CACHE.mkdir(parents=True, exist_ok=True)


def _append_and_reload(csv_path: Path, contest_id: str, rows: list[dict]) -> pd.DataFrame:
    """Replace `contest_id`'s rows in csv_path with `rows`, return the whole
    accumulated table off disk (tests/backtest.py::_append_and_reload pattern)."""
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"contest_id": str})


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp = cfg["gpp"]
    paths = cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)

    # --- slate inputs, exactly as pipeline.py's external branch builds them ---
    found = ep.discover_external_files(RAW_DIR)
    slate_path = PROJECT_ROOT / paths["dk_slate"]
    slate_df = DraftKingsSlateIngestor(str(slate_path)).get_slate_dataframe()
    valid_ids = set(slate_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )
    print(f"pool={len(pool.lineups)} lineups (after near-dupe cull), "
          f"players={len(players_df)}, files={[p.name for p in found['lineups_paths']]}")

    # --- contest groups from the real *Entries.csv, production ordering ---
    all_file_entries = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            all_file_entries.append((p, recs))
    groups = ep.group_and_match_contests(all_file_entries, pool)
    print(f"contests={len(groups)}, entries={sum(len(g.entries) for g in groups)}")

    # --- simulation (cached) ---
    grids = ep.build_quantile_grids(
        proj_ext,
        zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
        scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
        mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
        mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)),
    )
    sig = found["projections_path"].stem
    cache = SIM_CACHE / f"{sig}_{N_SIMS}_{seed}.npz"
    if cache.exists():
        with np.load(cache) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
        print(f"sim: loaded cache {cache.name}")
    else:
        t0 = time.time()
        engine = SimulationEngine(
            EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
            batter_pca_model=None, score_grid=None, quantile_grids=grids,
        )
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(N_SIMS)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
        print(f"sim: {N_SIMS} worlds in {time.time() - t0:.0f}s -> {cache.name}")

    own_vec = players_df["ownership"].astype(float).to_numpy()
    field_pool_size_cfg = int(gpp.get("external_pool_topn_field_pool_size", 25_000))

    # --- field pool + generated-pool augmentation, production settings ---
    t0 = time.time()
    field_lineups = ep.build_topn_field_pool(players_df, own_vec, field_pool_size_cfg, seed)
    print(f"field pool: {field_lineups.shape[0]} lineups in {time.time() - t0:.0f}s")

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
        alloc_pool, gen_kept = ep.augment_topn_pool_with_generated(
            pool, players_df, gen_own, n_gen, seed + 1,
        )
        print(f"pool augmented: +{len(gen_kept)} generated -> {len(alloc_pool.lineups)}")

    # --- ceiling-based floor cull, exactly as production seeds `mask` ---
    floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)),
    )
    eligible = np.ones(len(alloc_pool.lineups), dtype=bool)
    if floor is not None:
        eligible &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    print(f"eligible after {gpp.get('external_pool_proj_score_pct')}% ceiling floor: "
          f"{int(eligible.sum())}/{len(alloc_pool.lineups)}")

    # --- pitcher roster membership over the ELIGIBLE candidate pool ---
    name_by_id = dict(zip(players_df["player_id"].astype(int), players_df["name"])) \
        if "name" in players_df.columns else {}
    pos_by_id = dict(zip(players_df["player_id"].astype(int), players_df["position"]))
    elig_idx = np.where(eligible)[0]
    elig_lineups = [alloc_pool.lineups[i] for i in elig_idx]
    n_cand = len(elig_lineups)

    pitcher_ids = [int(p) for p, pos in pos_by_id.items() if str(pos).startswith("P")]
    has_p = {}
    for pid in pitcher_ids:
        v = np.fromiter((pid in lu.player_ids for lu in elig_lineups), dtype=bool, count=n_cand)
        if v.any():
            has_p[pid] = v
    top_pitchers = sorted(has_p, key=lambda p: -has_p[p].sum())[:N_PITCHERS]
    print("\npitcher share of ELIGIBLE candidate pool:")
    for pid in top_pitchers:
        print(f"  {name_by_id.get(pid, pid):<20s} {has_p[pid].mean() * 100:5.1f}%")

    # --- candidate scores over the shared world set (built once) ---
    I_pool = ep._lineup_indicator_matrix(elig_lineups, sim_results.player_ids)
    sim_mat = sim_results.results_matrix.astype(np.float32)
    cand = (sim_mat @ I_pool).T  # (n_cand, N_SIMS)
    print(f"\ncandidate score matrix: {cand.shape} ({cand.nbytes / 1e9:.2f} GB)")

    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    fcols = np.array([[col_map[int(p)] for p in row] for row in field_lineups], dtype=np.int32)

    K = int(gpp.get("external_pool_topn_field_samples", 5))
    rank = int(gpp.get("external_pool_topn_rank", 10))
    pct_floor = float(gpp.get("external_pool_topn_percentile_floor", 0.001))

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"])

    table = None
    for ci, g in enumerate(groups):
        if not g.entries:
            continue
        if g.contest_id in done:
            print(f"[skip] {g.contest_name} (already in results.csv)")
            continue
        t0 = time.time()
        field_size_g = ep._topn_field_size_for_group(g, fcols.shape[0])
        N = ep._topn_effective_rank(rank, field_size_g, pct_floor)
        rng = np.random.default_rng(seed + ci)

        thr = np.empty((K, N_SIMS), dtype=np.float32)
        for kk in range(K):
            subset = rng.choice(fcols.shape[0], size=field_size_g, replace=False)
            fs = ep._score_field_cols_batched(sim_mat, fcols[subset])
            thr[kk] = np.partition(fs, -N, axis=1)[:, -N]
            del fs

        max_all = cand.max(axis=0)                      # (N_SIMS,)
        cross_all = max_all[None, :] >= thr             # (K, N_SIMS)
        n_slots = cross_all.size
        claimable = float(cross_all.mean())

        rows = []
        for pid in top_pitchers:
            keep = ~has_p[pid]  # has_p is already indexed over elig_lineups
            if not keep.any():
                continue
            max_wo = cand[keep].max(axis=0)
            cross_wo = max_wo[None, :] >= thr
            required = float((cross_all & ~cross_wo).sum()) / n_slots
            rows.append({
                "contest_id": g.contest_id, "contest": g.contest_name,
                "k": len(g.entries), "field_size_g": field_size_g, "N": N,
                "claimable_frac": round(claimable, 4),
                "pitcher": name_by_id.get(pid, str(pid)),
                "pool_share": round(float(has_p[pid].mean()), 4),
                # share of ALL slots that only this pitcher's lineups can claim
                "required_abs": round(required, 4),
                # ... as a share of the slots claimable at all == the exposure
                # a pure-coverage portfolio must give this pitcher
                "required_of_claimable": round(required / claimable, 4) if claimable else 0.0,
                "claimable_without": round(float(cross_wo.mean()), 4),
            })
            del max_wo, cross_wo
        table = _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        print(f"[{ci + 1}/{len(groups)}] {g.contest_name:<14s} k={len(g.entries):3d} "
              f"field={field_size_g:6d} N={N:3d} claimable={claimable * 100:5.1f}%  "
              f"({time.time() - t0:.0f}s)")

    if table is None:
        table = pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})

    # --- entry-weighted summary (contest bias, not contest-count bias --
    # see the decile-lift weighting fix; a 60-entry contest should not count
    # the same as a 2-entry one) ---
    print("\n=== entry-weighted required exposure across the slate ===")
    agg = (
        table.assign(w=table["k"])
        .groupby("pitcher")
        .apply(lambda d: pd.Series({
            "pool_share": np.average(d["pool_share"], weights=d["w"]),
            "required_of_claimable": np.average(d["required_of_claimable"], weights=d["w"]),
            "claimable_without": np.average(d["claimable_without"], weights=d["w"]),
        }), include_groups=False)
        .sort_values("required_of_claimable", ascending=False)
    )
    print((agg * 100).round(1).to_string())
    print(f"\nfull per-contest table: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
