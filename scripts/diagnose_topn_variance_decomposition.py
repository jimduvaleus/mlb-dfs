"""Which noise source drives topn_coverage's portfolio variance: the SIM
WORLDS a contest is allocated, or the FIELD DRAWS its thresholds come from?

`allocate_contests_topn_coverage`'s `rng_seed` drives both at once:

  * `_SimWorldAllocator(n_sims, rng_seed)`  -> which disjoint sim-world slice
    each contest gets (candidate-side estimation noise)
  * `default_rng(rng_seed + contest_index)` -> which `field_size_g` lineups
    form each of the K threshold draws (threshold-side noise)

So the observed cross-seed portfolio movement is their SUM, and
scripts/diagnose_topn_rung_settling.py measured only the first (it held
thresholds fixed by construction). That matters because the two cost wildly
different amounts to buy down:

  more sim worlds   scales the (n_sims_g x field_size_g) field-score
                    transient LINEARLY -- ~1.5GB per contest at 1x for
                    mini-MAX, and it is the single largest block in the run
  more field draws  do NOT grow that block at all; K only reuses it more
                    times, growing just the bit-planes and thresholds

If threshold noise dominates, the fix is `field_samples` and is nearly free.
If sim-world noise dominates, the fix is `sims_min` and is the expensive one.
Buying the wrong knob wastes the memory budget on the wrong axis.

`field_rng_seed` (added to the allocator, defaulting to `rng_seed` so normal
behaviour is unchanged) lets the two be varied independently, giving a clean
2x2:

  A0  worlds=42   fields=42    reference
  A1  worlds=137  fields=42    ONLY sim-world slices changed
  A2  worlds=42   fields=137   ONLY field draws changed
  A3  worlds=137  fields=137   both (reproduces the earlier cross-seed run)

Each arm is compared to A0 on the measures that survived the identity-overlap
critique -- per-player exposure rho, team-stack rho, mean nearest-neighbour
player overlap -- plus ace exposure and the objective value. Whichever of A1/A2
tracks A3 owns the variance.

Checkpoint / resume per CLAUDE.md. TOPN_VAR_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/diagnose_topn_variance_decomposition.py
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
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
FORCE = os.environ.get("TOPN_VAR_FORCE") == "1"
TRACK_PLAYER = "Tarik Skubal"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_variance_decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
PAIRS_CSV = OUT_DIR / "pairs.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"

ARMS = [
    ("A0_w42_f42",   42, 42),
    ("A1_w137_f42",  137, 42),    # only sim-world slices differ from A0
    ("A2_w42_f137",  42, 137),    # only field draws differ from A0
    ("A3_w137_f137", 137, 137),   # both
]


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
    n_sims = max(int(cfg["simulation"].get("n_sims", 25_000)), ep.topn_total_sims_needed(
        groups, fp_size,
        float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
        int(gpp.get("external_pool_topn_sims_min", 0)),
        float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
        float(gpp.get("external_pool_topn_sims_power", 0.0)),
    ))
    with np.load(SHARED / "sim_cache" / f"{found['projections_path'].stem}_{n_sims}_{base_seed}.npz") as z:
        sim_results = SimulationResults(
            [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
        )
    field_lineups = np.load(
        SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{base_seed}.npy")

    from src.optimization.leverage import compute_generation_ownership_vec
    gen_own = compute_generation_ownership_vec(
        pool.lineups, sim_results, players_df,
        field_size=float(ep.pwin_field_size(groups, floor=int(gpp.get("n_field_lineups", 5_000)))),
        blend_weight=float(gpp.get("external_pool_topn_generated_leverage_weight", 0.0)),
        sharpness=float(gpp.get("external_pool_pwin_sharpness", 0.05)),
    )
    alloc_pool, _ = ep.augment_topn_pool_with_generated(
        pool, players_df, gen_own,
        int(gpp.get("external_pool_topn_generated_pool_size", 0)), base_seed + 1,
    )
    floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
    print(f"sim {sim_results.results_matrix.shape}, alloc pool {len(alloc_pool.lineups)}")

    pids = players_df["player_id"].astype(int).to_numpy()
    team_of = dict(zip(pids, players_df["team"]))
    pos_of = dict(zip(pids, players_df["position"]))
    pid_track = int(players_df.loc[players_df["name"] == TRACK_PLAYER, "player_id"].iloc[0])
    idx_of = {id(lu): i for i, lu in enumerate(alloc_pool.lineups)}

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
    )

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV)["arm"].astype(str))

    rows = []
    for arm, w_seed, f_seed in ARMS:
        if arm in done and (OUT_DIR / f"pick_idx_{arm}.npy").exists():
            print(f"[skip] {arm}")
            continue
        cdone = []
        t0 = time.time()
        alloc = ep.allocate_contests_topn_coverage(
            alloc_pool, sim_results, groups, field_lineups,
            rng_seed=w_seed, field_rng_seed=f_seed, **common,
            progress_cb=lambda i: cdone.append(i)
            if i.get("event") == "contest_done" else None,
        )
        pick_idx = np.array([idx_of[id(lu)] for lu, _ in alloc.portfolio])
        np.save(OUT_DIR / f"pick_idx_{arm}.npy", pick_idx)
        wc = sum(r["worlds_claimed"] for r in cdone)
        ns = sum(r["n_sims_g"] for r in cdone)
        rows.append({
            "arm": arm, "worlds_seed": w_seed, "fields_seed": f_seed,
            "elapsed_s": round(time.time() - t0, 1),
            "track_exposure": round(
                float(np.mean([pid_track in lu.player_ids for lu, _ in alloc.portfolio])), 4),
            "worlds_claimed_pct": round(100 * wc / ns, 3),
            "mean_ceiling": round(float(floor_scores[pick_idx].mean()), 2),
        })
        print(f"  {arm}: {rows[-1]['elapsed_s']}s  {TRACK_PLAYER} "
              f"{rows[-1]['track_exposure'] * 100:.1f}%  "
              f"worlds {rows[-1]['worlds_claimed_pct']}%")
        df = pd.DataFrame(rows)
        if RESULTS_CSV.exists():
            old = pd.read_csv(RESULTS_CSV)
            df = pd.concat([old[~old["arm"].isin(df["arm"])], df], ignore_index=True)
        df.to_csv(RESULTS_CSV, index=False)

    def load(arm):
        return [alloc_pool.lineups[i] for i in np.load(OUT_DIR / f"pick_idx_{arm}.npy")]

    A0 = load("A0_w42_f42")
    e0 = exposures(A0, pids)
    s0 = stack_exposure(A0, team_of, pos_of)
    prs = []
    for arm, _, _ in ARMS[1:]:
        B = load(arm)
        eb = exposures(B, pids)
        used = (e0 + eb) > 0
        sb = stack_exposure(B, team_of, pos_of)
        teams = sorted(set(s0) | set(sb))
        prs.append({
            "vs_A0": arm,
            "changed": {"A1_w137_f42": "sim worlds only",
                        "A2_w42_f137": "field draws only",
                        "A3_w137_f137": "both"}[arm],
            "identity_overlap": round(
                len(set(map(id, A0)) & set(map(id, B))) / len(A0), 3),
            "nn_mean": round(nn_mean(A0, B), 2),
            "expo_rho": round(float(spearmanr(e0[used], eb[used]).statistic), 3),
            "stack_rho": round(float(spearmanr(
                [s0.get(t, 0) for t in teams], [sb.get(t, 0) for t in teams]).statistic), 3),
            "max_player_gap_pp": round(float(np.abs(e0 - eb).max()) * 100, 1),
        })
    pdf = pd.DataFrame(prs)
    pdf.to_csv(PAIRS_CSV, index=False)

    print("\n=== arms ===")
    print(pd.read_csv(RESULTS_CSV).to_string(index=False))
    print("\n=== divergence from A0 (higher rho / nn = more similar) ===")
    print(pdf.to_string(index=False))
    print("\nRead: whichever of A1 (sim worlds) / A2 (field draws) tracks A3's "
          "divergence owns the variance -- and is the knob worth buying.")


if __name__ == "__main__":
    main()
