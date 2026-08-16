"""What does capping a pitcher's exposure actually COST the topn_coverage
objective, at the real per-contest entry budget?

Companion to diagnose_topn_pitcher_requirement.py, which asked the
budget-FREE question ("what share of slots can no non-X lineup claim?") and
answered 14.3% for Skubal on 08/10 against an 84.9% realized exposure. That
statistic takes the max over every non-X candidate, so it's a permissive
upper bound on non-X claimability -- with only k lineups you cannot hold the
best non-X lineup for every world, so it's a LOWER bound on required
exposure and possibly a weak one.

This script closes that gap by running the greedy itself under a hard
exposure cap and measuring the objective directly: how many (field draw,
world) slots does the portfolio cover after the contest's real k picks, at
cap in {no cap, 70%, 50%, 47% (= field ownership)}? If coverage is
materially flat across caps, the concentration is not being bought by the
objective and a cap is free. If it falls off, the exposure is earned.

The greedy loop here mirrors `allocate_contests_topn_coverage`'s inner loop
exactly -- same bit-packed crossing matrix, same `_POPCOUNT_LUT` popcount
gain, same minimum-deflation relaxation rule, same coverage-wave reset --
with ONE addition: once `floor(cap * k)` picks rostering the capped pitcher
have been made, remaining candidates rostering them are dropped from
contention for the rest of that contest. Production is NOT modified; this is
a measurement.

Same deliberate deviation as the companion script: one shared sim-world set
for every contest instead of `_SimWorldAllocator`'s disjoint slices, so the
per-contest numbers are directly comparable (see that script's docstring).
Each contest is run standalone against the full eligible pool rather than
inheriting earlier contests' removals, for the same reason.

Checkpoint / resume per CLAUDE.md: rows land in
outputs/topn_pitcher_requirement/cap_results.csv as each contest finishes;
contests already on disk are skipped. TOPN_CAP_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/diagnose_topn_exposure_cap.py

Env vars
--------
    TOPN_REQ_RAW    slate input dir (default data/raw)
    TOPN_REQ_NSIMS  shared sim-world count (default 12000)
    TOPN_CAP_PLAYER capped player's name (default "Tarik Skubal")
    TOPN_CAP_FORCE  "1" re-runs contests already in cap_results.csv
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
from src.optimization.gpp_portfolio import _POPCOUNT_LUT  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
N_SIMS = int(os.environ.get("TOPN_REQ_NSIMS", "12000"))
CAP_PLAYER = os.environ.get("TOPN_CAP_PLAYER", "Tarik Skubal")
FORCE = os.environ.get("TOPN_CAP_FORCE") == "1"
CAPS = [1.01, 0.70, 0.50, 0.47]  # 1.01 == no cap (production behavior)

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_pitcher_requirement"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "cap_results.csv"
SIM_CACHE = OUT_DIR / "sim_cache"
SIM_CACHE.mkdir(parents=True, exist_ok=True)


def _append_and_reload(csv_path: Path, contest_id: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"contest_id": str})


def greedy(cand: np.ndarray, thr: np.ndarray, k: int, capped: np.ndarray,
           cap: float, relax_step: float = 1.0) -> dict:
    """One contest's greedy fill under an exposure cap on `capped`.

    Mirrors allocate_contests_topn_coverage's inner loop; returns the
    objective (union of covered slots) plus the diagnostics that explain it."""
    K, n_sims_g = thr.shape
    n_slots = K * n_sims_g
    n_bytes = -(-n_slots // 8)
    pad = n_bytes * 8 - n_slots
    n_cand = cand.shape[0]

    def crossing_bits(t: np.ndarray) -> np.ndarray:
        bits = np.zeros((n_cand, n_bytes), dtype=np.uint8)
        for s in range(0, n_cand, 2000):
            e = min(s + 2000, n_cand)
            cross = np.empty((e - s, K, n_sims_g), dtype=bool)
            for kk in range(K):
                cross[:, kk, :] = cand[s:e] >= t[kk][None, :]
            bits[s:e] = np.packbits(cross.reshape(e - s, n_slots), axis=1)
        return bits

    def fresh() -> np.ndarray:
        u = np.full(n_bytes, 0xFF, dtype=np.uint8)
        if pad:
            u[-1] = np.uint8((0xFF << pad) & 0xFF)
        return u

    thresholds = thr.copy()
    bits = crossing_bits(thresholds)
    remaining = np.ones(n_cand, dtype=bool)
    uncovered = fresh()
    ever = np.zeros(n_bytes, dtype=np.uint8)

    cap_budget = int(np.floor(cap * k))
    n_capped_picked = 0
    picks, n_relax, n_waves = [], 0, 0
    while len(picks) < k:
        if n_capped_picked >= cap_budget:
            remaining &= ~capped  # cap binds: drop the capped player entirely
            if not remaining.any():
                break
        gains = _POPCOUNT_LUT[np.bitwise_and(bits, uncovered[None, :])].sum(axis=1).astype(np.int64)
        gains[~remaining] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            if not uncovered.any():
                uncovered = fresh()
                n_waves += 1
                continue
            max_per_world = cand[remaining].max(axis=0)
            unc = np.unpackbits(uncovered)[:n_slots].reshape(K, n_sims_g).astype(bool)
            gaps = np.where(unc, thresholds - max_per_world[None, :], np.inf)
            g_min = float(gaps.min())
            steps = max(1, int(np.ceil(g_min / relax_step))) if np.isfinite(g_min) else 1
            thresholds -= steps * relax_step
            bits = crossing_bits(thresholds)
            n_relax += steps
            continue
        picks.append(best)
        remaining[best] = False
        if capped[best]:
            n_capped_picked += 1
        np.bitwise_and(uncovered, np.bitwise_not(bits[best]), out=uncovered)
        np.bitwise_or(ever, bits[best], out=ever)

    return {
        "n_picks": len(picks),
        "slots_covered": int(_POPCOUNT_LUT[ever].sum()),
        "n_slots": n_slots,
        "capped_exposure": n_capped_picked / len(picks) if picks else 0.0,
        "n_relaxations": n_relax, "n_wave_resets": n_waves,
    }


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

    sig = found["projections_path"].stem
    cache = SIM_CACHE / f"{sig}_{N_SIMS}_{seed}.npz"
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
        sim_results = engine.simulate(N_SIMS)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache, player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
    print(f"sim ready: {sim_results.results_matrix.shape}")

    own_vec = players_df["ownership"].astype(float).to_numpy()
    fp_size = int(gpp.get("external_pool_topn_field_pool_size", 25_000))
    fp_cache = OUT_DIR / f"field_pool_{sig}_{fp_size}_{seed}.npy"
    if fp_cache.exists():
        field_lineups = np.load(fp_cache)
    else:
        t0 = time.time()
        field_lineups = ep.build_topn_field_pool(players_df, own_vec, fp_size, seed)
        np.save(fp_cache, field_lineups)
        print(f"field pool built in {time.time() - t0:.0f}s")
    print(f"field pool: {field_lineups.shape[0]}")

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
    eligible = np.ones(len(alloc_pool.lineups), dtype=bool)
    if floor is not None:
        eligible &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    elig_lineups = [lu for lu, e in zip(alloc_pool.lineups, eligible) if e]
    print(f"eligible candidates: {len(elig_lineups)}")

    name_to_id = dict(zip(players_df["name"], players_df["player_id"].astype(int)))
    cap_pid = int(name_to_id[CAP_PLAYER])
    capped = np.fromiter((cap_pid in lu.player_ids for lu in elig_lineups),
                         dtype=bool, count=len(elig_lineups))
    print(f"{CAP_PLAYER}: {capped.mean() * 100:.1f}% of eligible pool")

    I_pool = ep._lineup_indicator_matrix(elig_lineups, sim_results.player_ids)
    sim_mat = sim_results.results_matrix.astype(np.float32)
    cand = (sim_mat @ I_pool).T
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    fcols = np.array([[col_map[int(p)] for p in row] for row in field_lineups], dtype=np.int32)

    K = int(gpp.get("external_pool_topn_field_samples", 5))
    rank = int(gpp.get("external_pool_topn_rank", 10))
    pct_floor = float(gpp.get("external_pool_topn_percentile_floor", 0.001))

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"])

    for ci, g in enumerate(groups):
        if not g.entries or g.contest_id in done:
            if g.entries:
                print(f"[skip] {g.contest_name}")
            continue
        t0 = time.time()
        k = len(g.entries)
        field_size_g = ep._topn_field_size_for_group(g, fcols.shape[0])
        N = ep._topn_effective_rank(rank, field_size_g, pct_floor)
        rng = np.random.default_rng(seed + ci)
        thr = np.empty((K, N_SIMS), dtype=np.float32)
        for kk in range(K):
            subset = rng.choice(fcols.shape[0], size=field_size_g, replace=False)
            fs = ep._score_field_cols_batched(sim_mat, fcols[subset])
            thr[kk] = np.partition(fs, -N, axis=1)[:, -N]
            del fs

        rows, base = [], None
        for cap in CAPS:
            r = greedy(cand, thr, k, capped, cap)
            if base is None:
                base = r["slots_covered"]
            rows.append({
                "contest_id": g.contest_id, "contest": g.contest_name, "k": k,
                "field_size_g": field_size_g, "N": N, "cap": cap,
                "slots_covered": r["slots_covered"], "n_slots": r["n_slots"],
                "coverage_pct": round(100 * r["slots_covered"] / r["n_slots"], 3),
                "vs_uncapped_pct": round(100 * (r["slots_covered"] / base - 1), 3),
                "realized_exposure": round(r["capped_exposure"], 3),
                "n_relaxations": r["n_relaxations"], "n_wave_resets": r["n_wave_resets"],
            })
            print(f"  {g.contest_name[:28]:<28s} cap={cap:.2f} "
                  f"cov={rows[-1]['coverage_pct']:6.2f}% "
                  f"({rows[-1]['vs_uncapped_pct']:+.2f}% vs uncapped) "
                  f"exp={r['capped_exposure'] * 100:5.1f}%")
        _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        print(f"[{ci + 1}/{len(groups)}] {g.contest_name} k={k} done ({time.time() - t0:.0f}s)")

    t = pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})
    print("\n=== entry-weighted coverage vs exposure cap ===")
    agg = t.groupby("cap").apply(lambda d: pd.Series({
        "coverage_pct": np.average(d["coverage_pct"], weights=d["k"]),
        "vs_uncapped_pct": np.average(d["vs_uncapped_pct"], weights=d["k"]),
        "realized_exposure_pct": 100 * np.average(d["realized_exposure"], weights=d["k"]),
    }), include_groups=False)
    print(agg.round(3).to_string())
    print(f"\nfull table: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
