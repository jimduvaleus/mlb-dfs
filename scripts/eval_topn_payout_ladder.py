"""A/B the payout-weighted rank ladder at production scale: portfolio effect,
wall clock, peak RSS, and CROSS-SEED stability.

Arms (all on one cached sim; `rng_seed` varies the field draws and the
`_SimWorldAllocator` slices, which is exactly the estimator noise
scripts/diagnose_topn_rung_settling.py measured):

  baseline_s42    single top-N bar (shipped behavior)
  ladder1_s42     payout ladder anchored at rank 1 (Fix 1 as specified)
  ladder5_s42     payout ladder floored at rank 5
  baseline_s137   baseline, different seed  -> cross-seed pick overlap
  ladder1_s137    ladder1, different seed   -> cross-seed pick overlap

PREDICTIONS (stated before running):

  P1  the ladder materially changes the portfolio vs the single bar
      (picks replaced well above 0).
  P2  it shifts selection toward higher-ceiling lineups -- the ladder pays
      for WINNING a world, not for scraping the bar.
  P3  ace exposure: MEASUREMENT, not prediction. I do not know the sign. The
      ace is a mean shift that helps clear any bar, so a payout ladder could
      just as easily raise concentration as lower it.
  P4  cost ~2-3x wall clock (R x crossing-bit build + R x per-pick popcount;
      threshold extraction is unchanged), peak RSS staying well under 11GB.
  P5  the KEY one, from the settling diagnostic: cross-seed pick overlap is
      LOWER for ladder1 than for baseline. 96% of ladder1's payout weight
      sits on rungs with full-budget split-half reliability < 0.55, so its
      portfolio should be measurably less reproducible. If overlap is
      instead comparable, the noisy rungs are not actually driving selection
      and the settling concern is overstated.

Peak RSS is sampled by a background thread (50ms): the allocator frees its
big per-contest arrays as it goes, so a single post-hoc reading misses the
peak. ru_maxrss is process-wide and monotonic, so it is reported only as a
loose upper check -- later arms inherit earlier arms' high-water mark.

Checkpoint / resume per CLAUDE.md; TOPN_LADDER_FORCE=1 redoes everything.

Usage
-----
    source venv/bin/activate
    python scripts/eval_topn_payout_ladder.py
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
FORCE = os.environ.get("TOPN_LADDER_FORCE") == "1"
TRACK_PLAYER = "Tarik Skubal"
N_RUNGS = 5

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_payout_ladder"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"

ARMS = [
    ("baseline_s42",  dict(payout_rungs=0), 42),
    ("ladder1_s42",   dict(payout_rungs=N_RUNGS, payout_tightest_rank=1), 42),
    ("ladder5_s42",   dict(payout_rungs=N_RUNGS, payout_tightest_rank=5), 42),
    ("baseline_s137", dict(payout_rungs=0), 137),
    ("ladder1_s137",  dict(payout_rungs=N_RUNGS, payout_tightest_rank=1), 137),
]


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


def _append_and_reload(csv_path: Path, key: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old["arm"].astype(str) != str(key)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


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
    cache = SHARED / "sim_cache" / f"{found['projections_path'].stem}_{n_sims}_{base_seed}.npz"
    with np.load(cache) as z:
        sim_results = SimulationResults(
            [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
        )
    own_vec = players_df["ownership"].astype(float).to_numpy()
    field_lineups = np.load(
        SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{base_seed}.npy"
    )
    print(f"sim {sim_results.results_matrix.shape}, field pool {field_lineups.shape[0]}, "
          f"contests {len(groups)}")

    n_gen = int(gpp.get("external_pool_topn_generated_pool_size", 0))
    alloc_pool = pool
    if n_gen > 0:
        from src.optimization.leverage import compute_generation_ownership_vec
        gen_own = compute_generation_ownership_vec(
            pool.lineups, sim_results, players_df,
            field_size=float(ep.pwin_field_size(
                groups, floor=int(gpp.get("n_field_lineups", 5_000)))),
            blend_weight=float(gpp.get("external_pool_topn_generated_leverage_weight", 0.0)),
            sharpness=float(gpp.get("external_pool_pwin_sharpness", 0.05)),
        )
        alloc_pool, _ = ep.augment_topn_pool_with_generated(
            pool, players_df, gen_own, n_gen, base_seed + 1,
        )
    floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
    proj_scores = ep.compute_pool_proj_scores(alloc_pool.lineups, players_df)
    print(f"alloc pool {len(alloc_pool.lineups)}")

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

    for arm, kwargs, seed in ARMS:
        if arm in done:
            print(f"[skip] {arm}")
            continue
        cdone = []
        print(f"\n--- {arm} (seed {seed}, {kwargs}) ---")
        t0 = time.time()
        with PeakRSS() as peak:
            alloc = ep.allocate_contests_topn_coverage(
                alloc_pool, sim_results, groups, field_lineups,
                rng_seed=seed, **kwargs, **common,
                progress_cb=lambda i: cdone.append(i)
                if i.get("event") == "contest_done" else None,
            )
        elapsed = time.time() - t0
        pick_idx = np.array([idx_of[id(lu)] for lu, _ in alloc.portfolio])
        np.save(OUT_DIR / f"pick_idx_{arm}.npy", pick_idx)
        wc = sum(r["worlds_claimed"] for r in cdone)
        ns = sum(r["n_sims_g"] for r in cdone)
        row = {
            "arm": arm, "seed": seed, "rungs": kwargs.get("payout_rungs", 0),
            "tightest": kwargs.get("payout_tightest_rank", 0),
            "n_picks": len(pick_idx), "elapsed_s": round(elapsed, 1),
            "peak_rss_gb": round(peak.peak, 2),
            "ru_maxrss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
            "track_exposure": round(
                float(np.mean([pid_track in lu.player_ids for lu, _ in alloc.portfolio])), 4),
            "mean_ceiling": round(float(floor_scores[pick_idx].mean()), 2),
            "mean_proj": round(float(proj_scores[pick_idx].mean()), 2),
            "worlds_claimed": wc, "worlds_claimed_pct": round(100 * wc / ns, 3),
            "n_relaxations": sum(r["n_relaxations"] for r in cdone),
            "n_wave_resets": sum(r["n_wave_resets"] for r in cdone),
            "n_unfilled": len(alloc.unfilled),
        }
        print(f"  {elapsed:.0f}s  peak {peak.peak:.2f} GB  {TRACK_PLAYER} "
              f"{row['track_exposure'] * 100:.1f}%  ceiling {row['mean_ceiling']}  "
              f"worlds {row['worlds_claimed_pct']}%")
        _append_and_reload(RESULTS_CSV, arm, [row])

    t = pd.read_csv(RESULTS_CSV).set_index("arm")
    print("\n=== arms ===")
    print(t.to_string())

    def overlap(a, b):
        pa, pb = OUT_DIR / f"pick_idx_{a}.npy", OUT_DIR / f"pick_idx_{b}.npy"
        if not (pa.exists() and pb.exists()):
            return None
        sa, sb = set(np.load(pa).tolist()), set(np.load(pb).tolist())
        return len(sa & sb) / len(sa)

    print("\n=== P1/P2/P3: ladder vs single bar (seed 42) ===")
    for lad in ("ladder1_s42", "ladder5_s42"):
        o = overlap("baseline_s42", lad)
        if o is None or lad not in t.index:
            continue
        b, d = t.loc["baseline_s42"], t.loc[lad]
        print(f"  {lad:<14s} picks changed {100 * (1 - o):5.1f}%  "
              f"ceiling {b.mean_ceiling} -> {d.mean_ceiling}  "
              f"{TRACK_PLAYER} {b.track_exposure * 100:.1f}% -> {d.track_exposure * 100:.1f}%  "
              f"worlds {b.worlds_claimed_pct:.2f}% -> {d.worlds_claimed_pct:.2f}%  "
              f"time {b.elapsed_s:.0f}s -> {d.elapsed_s:.0f}s  "
              f"peak {b.peak_rss_gb:.2f} -> {d.peak_rss_gb:.2f} GB")

    print("\n=== P5: cross-seed reproducibility (42 vs 137) ===")
    for a, b in (("baseline_s42", "baseline_s137"), ("ladder1_s42", "ladder1_s137")):
        o = overlap(a, b)
        if o is not None:
            print(f"  {a.split('_')[0]:<10s} pick overlap {o * 100:5.1f}%  "
                  f"({100 * (1 - o):.1f}% of the portfolio is seed-dependent)")
    print(f"\nfull table: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
