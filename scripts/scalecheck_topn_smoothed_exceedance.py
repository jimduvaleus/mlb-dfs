"""Production-scale cost + portfolio A/B for smoothed exceedance.

scripts/eval_topn_smoothed_exceedance.py answers "is the currency less noisy".
This answers the two operational questions that decide whether it is USABLE:

  1. Cost. Smoothing holds a float32 (n_cand, R, n_slots) array for the whole
     of each contest's pick loop, where the hard path holds a bit-packed one
     (32x smaller), and each pick becomes a BLAS matvec instead of a popcount.
     Against the ~10-11GB working ceiling this repo operates under, that has to
     be measured, not assumed. `field_samples=1` is used on the smoothed arms
     (see allocate_contests_topn_coverage's docstring: the K draws Monte-Carlo
     exactly the threshold noise tau integrates analytically), so the honest
     comparison is hard-at-K=3 (what production runs today) against
     smooth-at-K=1 -- reported alongside smooth-at-K=3 to separate the two
     effects.
  2. What it does to the portfolio. Reports team-stack entropy, max single-team
     and max single-player exposure, and pick overlap vs the hard arm. Per
     [[project-topn-selector-reproducibility]] a seed change alone moves
     per-player exposure ~7pp while team-stack rho stays ~0.95, so read the
     STACK-level numbers and treat single-player deltas as noise at n=1 seed.

This is a cost/behaviour check on ONE slate and ONE seed. It is not evidence
the change pays -- that needs walk-forward per [[project-season-ev-program]].

Checkpoint / resume per CLAUDE.md: one row per arm appended to
outputs/topn_smoothed_scalecheck/results.csv; arms already on disk are skipped.

Usage
-----
    source venv/bin/activate
    python scripts/scalecheck_topn_smoothed_exceedance.py 2>&1 | tee /tmp/sc.log

Env vars
--------
    TOPN_REQ_RAW      slate input dir (default data/raw)
    TOPN_SC_ARMS      comma-separated arms (default "hard,smooth1_k1,smooth1_k3")
    TOPN_SC_FORCE     "1" re-runs arms already in results.csv
"""
import os
import resource
import sys
import threading
import time
from collections import Counter
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
ARMS = os.environ.get("TOPN_SC_ARMS", "hard,smooth1_k1,smooth1_k3").split(",")
FORCE = os.environ.get("TOPN_SC_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_smoothed_scalecheck"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"  # reuse sim/field caches

# arm name -> (smooth_tau_scale, field_samples override or None)
ARM_SPEC = {
    "hard": (0.0, None),
    "smooth0.5_k1": (0.5, 1),
    "smooth1_k1": (1.0, 1),
    "smooth1_k3": (1.0, 3),
    "smooth2_k1": (2.0, 1),
}


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


class PeakRSS:
    """Background sampler: the allocator frees its big per-contest arrays as it
    goes, so only a running sample catches the true high-water mark."""

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


def _stack_profile(picks, players_df) -> dict:
    """Team-stack shape of a portfolio. Entropy is normalized to [0, 1] so it
    is comparable across slates with different team counts -- the same measure
    [[project-pwin-cull-diversity-collapse]] used to show the cull, not the
    selector, was destroying spread."""
    team_of = dict(zip(players_df["player_id"].astype(int), players_df["team"]))
    primary, per_player = [], Counter()
    for lu in picks:
        teams = Counter(team_of.get(int(p), "?") for p in lu.player_ids)
        primary.append(teams.most_common(1)[0][0])
        for p in lu.player_ids:
            per_player[int(p)] += 1
    counts = np.array(list(Counter(primary).values()), dtype=float)
    frac = counts / counts.sum()
    n_teams = len(counts)
    ent = float(-(frac * np.log(frac)).sum() / np.log(n_teams)) if n_teams > 1 else 0.0
    top_player = max(per_player.values()) / len(picks) if picks else 0.0
    return {
        "n_stack_teams": n_teams,
        "stack_entropy": round(ent, 4),
        "max_team_stack_pct": round(100 * float(frac.max()), 1),
        "max_player_pct": round(100 * float(top_player), 1),
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

    fp_size = int(gpp.get("external_pool_topn_field_pool_size", 25_000))
    frac = float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5))
    smin = int(gpp.get("external_pool_topn_sims_min", 0))
    sref = float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0))
    spow = float(gpp.get("external_pool_topn_sims_power", 0.0))
    n_sims = max(int(cfg["simulation"].get("n_sims", 25_000)),
                 ep.topn_total_sims_needed(groups, fp_size, frac, smin, sref, spow))

    cache = SHARED / "sim_cache" / f"{found['projections_path'].stem}_{n_sims}_{seed}.npz"
    if not cache.exists():
        raise SystemExit(
            f"no cached sim at {cache} -- run scripts/eval_topn_smoothed_exceedance.py "
            "first so both scripts grade the same worlds"
        )
    with np.load(cache) as z:
        sim_results = SimulationResults(
            [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
        )
    print(f"sim {sim_results.results_matrix.shape}, contests={len(groups)}, "
          f"pool={len(pool.lineups)}")

    own_vec = players_df["ownership"].astype(float).to_numpy()
    fp_cache = SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{seed}.npy"
    field_lineups = np.load(fp_cache) if fp_cache.exists() else ep.build_topn_field_pool(
        players_df, own_vec, fp_size, seed,
    )
    floor_scores = ep.compute_pool_ceiling_scores(pool, players_df)

    common = dict(
        proj_scores=None,
        proj_score_floor_percentile=float(gpp.get("external_pool_proj_score_pct", 30.0)),
        floor_scores=floor_scores,
        topn_rank=int(gpp.get("external_pool_topn_rank", 10)),
        topn_percentile_floor=float(gpp.get("external_pool_topn_percentile_floor", 0.001)),
        sims_per_contest_fraction=frac, sims_min=smin,
        sims_reference_field_size=sref, sims_power=spow, rng_seed=seed,
    )
    default_k = int(gpp.get("external_pool_topn_field_samples", 3))

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV)["arm"].astype(str))

    idx_of = {id(lu): i for i, lu in enumerate(pool.lineups)}
    for arm in ARMS:
        if arm in done:
            print(f"[skip] {arm}")
            continue
        if arm not in ARM_SPEC:
            raise SystemExit(f"unknown arm {arm!r}; known: {sorted(ARM_SPEC)}")
        tau_scale, k_override = ARM_SPEC[arm]
        K = default_k if k_override is None else k_override
        rows_cb = []
        print(f"\n--- arm: {arm} (tau_scale={tau_scale}, K={K}) ---")
        t0 = time.time()
        with PeakRSS() as peak:
            alloc = ep.allocate_contests_topn_coverage(
                pool, sim_results, groups, field_lineups,
                field_samples=K, smooth_tau_scale=tau_scale,
                progress_cb=lambda i: rows_cb.append(i)
                if i.get("event") == "contest_done" else None,
                **common,
            )
        elapsed = time.time() - t0
        picks = [lu for lu, _ in alloc.portfolio]
        pick_idx = np.array([idx_of[id(lu)] for lu in picks])
        np.save(OUT_DIR / f"pick_idx_{arm}.npy", pick_idx)
        wc = sum(r["worlds_claimed"] for r in rows_cb)
        ns = sum(r["n_sims_g"] for r in rows_cb)
        row = {
            "arm": arm, "tau_scale": tau_scale, "field_samples": K,
            "n_picks": len(picks), "n_unfilled": len(alloc.unfilled),
            "elapsed_s": round(elapsed, 1),
            "peak_rss_gb": round(peak.peak, 2),
            "ru_maxrss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0, 2),
            "worlds_claimed_pct": round(100 * wc / ns, 3) if ns else 0.0,
            "n_relaxations": sum(r["n_relaxations"] for r in rows_cb),
            "n_wave_resets": sum(r["n_wave_resets"] for r in rows_cb),
            **_stack_profile(picks, players_df),
        }
        _append_and_reload(RESULTS_CSV, arm, [row])
        print(f"  {elapsed:.0f}s  peak RSS {peak.peak:.2f} GB  "
              f"worlds {row['worlds_claimed_pct']}%  "
              f"stack entropy {row['stack_entropy']}  "
              f"max team {row['max_team_stack_pct']}%  max player {row['max_player_pct']}%")

    t = pd.read_csv(RESULTS_CSV).set_index("arm")
    print("\n=== scale-check summary ===")
    print(t.to_string())
    if "hard" in t.index:
        base = np.load(OUT_DIR / "pick_idx_hard.npy")
        for arm in t.index:
            f = OUT_DIR / f"pick_idx_{arm}.npy"
            if arm == "hard" or not f.exists():
                continue
            other = np.load(f)
            n = min(len(base), len(other))
            ov = len(set(base.tolist()) & set(other.tolist())) / max(n, 1)
            print(f"  pick-identity overlap vs hard, {arm}: {ov*100:.1f}%  "
                  f"(cf. ~41% for a SEED change alone -- identity is the wrong "
                  f"yardstick, read stack_entropy above)")


if __name__ == "__main__":
    main()
