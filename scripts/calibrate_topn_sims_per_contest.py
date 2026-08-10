"""Calibration probe for topn_coverage's per-contest sim-world budget
(src/api/external_pool.py::_topn_sims_for_field_size).

The allocator currently ships with a flat-fraction placeholder
(external_pool_topn_sims_per_contest_fraction) because the real relationship
between a contest's field size and how many simulated worlds its rank-N
threshold estimate needs to converge is not yet known -- see the design plan
(/home/jduvaleus/.claude/plans/given-external-candidate-lineup-optimized-scone.md,
"Per-contest sim-world subsampling" refinement). This script measures it.

CORRECTNESS NOTE (2026-08-09): the first version of this script used pure
i.i.d. Gaussian noise for both the field pool and the candidate pool, which
produced a degenerate result (every field size "needed" the full sim depth
to converge) -- an artifact of that synthetic data having no correlation
structure to lock onto, not a real property of the statistic. This version
uses REAL archived-slate data throughout: a real players_df/simulation via
tests/bt_core.py's build_slate_context (team-correlated Monte Carlo, not
i.i.d.), the REAL field generator
(src.api.external_pool.build_topn_field_pool -> ContestSimulator.
generate_field, the same ownership-weighted stacked-lineup sampler
production uses -- NOT a synthetic score matrix), and the slate's own real
external candidate pool (ctx["pool"].lineups) as the candidates being ranked
-- exactly what allocate_contests_topn_coverage ranks in production.

Method (mirrors scripts/probe_sim_count_tail_sensitivity.py's nested-subset
approach, applied to topn_coverage's own currency instead of self-play's
ROI): for each of the slate's real per-contest field sizes, build a
ground-truth candidate ranking (by crossing-count against the real field
pool) at full sim depth, then re-rank the same candidates using nested
subsets of increasing size (one fixed permutation, so each level is a strict
superset of the previous -- "simulating more," not "simulating different").
Spearman rank-correlation between a depth level's ranking and the
ground-truth ranking measures convergence; the smallest n_sims level
clearing CONVERGENCE_THRESHOLD is that field size's "sims needed" point. A
field-size -> n_sims_needed power-law is then log-log-fit across those
points to propose external_pool_topn_sims_min / _reference_field_size /
_power values.

Reuses the already-cached n_sims=25,000 Monte Carlo for 07222026 (same slate
probe_sim_count_tail_sensitivity.py uses) -- no new simulation needed.

Checkpoint / resume: each field size's row is appended to
outputs/topn_sims_calibration/results.csv as soon as it's measured, and a
field size already present is skipped on the next invocation (set
TOPN_CALIB_FORCE=1 to redo everything) -- per CLAUDE.md's long-running-
script rule.

Usage
-----
    source venv/bin/activate
    python scripts/calibrate_topn_sims_per_contest.py

Env vars
--------
    TOPN_CALIB_SLATE   archive dir name (default 07222026)
    TOPN_CALIB_FORCE   "1" re-runs field sizes already in results.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from tests.bt_core import build_slate_context, load_real_contests  # noqa: E402

SLATE = os.environ.get("TOPN_CALIB_SLATE", "07222026")
FORCE = os.environ.get("TOPN_CALIB_FORCE") == "1"
SEED = 42
N_SIMS_FULL = 25_000  # matches the cached simulation for SLATE

SIM_LEVELS = [250, 500, 1_000, 2_000, 5_000, 10_000, 25_000]
FIELD_POOL_SIZE = 25_000  # external_pool_topn_field_pool_size default
TOPN_RANK = 10
CONVERGENCE_THRESHOLD = 0.90

OUT_DIR = PROJECT_ROOT / "outputs" / "topn_sims_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def _append_and_reload(csv_path: Path, field_size: int, row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old["field_size"] != field_size]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def _done_field_sizes() -> set:
    if FORCE or not RESULTS_CSV.exists():
        return set()
    return set(pd.read_csv(RESULTS_CSV)["field_size"].unique())


def _crossing_counts(field_scores: np.ndarray, cand_scores: np.ndarray, rank: int) -> np.ndarray:
    """field_scores (n_sims, field_size), cand_scores (M, n_sims) -> (M,)
    crossing count per candidate against the per-world rank-`rank`
    threshold -- exactly allocate_contests_topn_coverage's currency at
    field_samples=1."""
    n = min(rank, field_scores.shape[1])
    thresholds = np.partition(field_scores, -n, axis=1)[:, -n]  # (n_sims,)
    return (cand_scores >= thresholds[None, :]).sum(axis=1)


def main() -> None:
    d = PROJECT_ROOT / "archive" / SLATE
    real = load_real_contests(d)
    sim_cache_dir = PROJECT_ROOT / "outputs" / "self_play_eval" / "sim_cache"

    t0 = time.time()
    ctx = build_slate_context(
        d, SEED, False, real, n_sims=N_SIMS_FULL, sharpness=0.05,
        sim_cache_dir=sim_cache_dir, want_corr=False, want_pwin=False,
    )
    print(f"context build (cached sim): {time.time() - t0:.0f}s")
    players_df, pool, sim_results = ctx["players_df"], ctx["pool"], ctx["sim_results"]
    own_vec = players_df["ownership"].astype(float).to_numpy()
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}

    # Real per-contest field sizes from this slate's actual contests, not
    # invented ones -- capped/deduped to a manageable spread.
    field_sizes = sorted({min(int(c["n_field"]), FIELD_POOL_SIZE) for c in ctx["contests"]})
    if len(field_sizes) > 6:
        idx = np.linspace(0, len(field_sizes) - 1, 6).round().astype(int)
        field_sizes = sorted({field_sizes[i] for i in idx})
    print(f"real field sizes for {SLATE}: {field_sizes}")

    # ONE real field pool, generated via the actual production generator
    # (ContestSimulator.generate_field via build_topn_field_pool) and scored
    # against the FULL sim matrix -- every field_size below is a column
    # subset of this same pool, exactly mirroring
    # allocate_contests_topn_coverage's per-contest field_size_g subsampling.
    t0 = time.time()
    field_lineups = ep.build_topn_field_pool(players_df, own_vec, FIELD_POOL_SIZE, SEED)
    field_pool_scores = ContestSimulator().score_field(
        field_lineups, sim_results.results_matrix, col_map,
    )
    print(f"real field pool ({field_pool_scores.shape[1]:,} lineups, "
          f"generator + scoring): {time.time() - t0:.0f}s")

    # The slate's own real external candidate pool -- exactly what
    # allocate_contests_topn_coverage ranks in production (not a second
    # internally-generated pool).
    cand_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
    print(f"candidate pool: {cand_scores.shape[0]:,} real external lineups")

    rng = np.random.default_rng(SEED)
    # One fixed permutation -> every SIM_LEVELS[i] is a strict superset of
    # SIM_LEVELS[i-1]'s columns (mimics "simulate more," not "simulate
    # different") -- isolates depth effects from RNG-draw effects.
    perm = rng.permutation(N_SIMS_FULL)
    # One fixed field-pool column subset per field size (reused at every
    # depth level, so only sim depth varies -- matches
    # allocate_contests_topn_coverage drawing the field subset once per
    # contest, not once per simulated world).
    field_subset_by_size = {
        fs: rng.choice(FIELD_POOL_SIZE, size=fs, replace=False) for fs in field_sizes
    }

    done = _done_field_sizes()
    per_size_convergence: dict[int, int] = {}
    for field_size in field_sizes:
        if field_size in done:
            prior = pd.read_csv(RESULTS_CSV)
            row = prior[prior["field_size"] == field_size].iloc[0]
            per_size_convergence[field_size] = int(row["n_sims_needed"])
            print(f"[field_size={field_size:,}] already in {RESULTS_CSV.name}, skipping "
                  f"(n_sims_needed={int(row['n_sims_needed']):,})")
            continue

        t0 = time.time()
        subset = field_subset_by_size[field_size]
        field_sub_full = field_pool_scores[:, subset]
        truth = _crossing_counts(field_sub_full, cand_scores, TOPN_RANK)

        n_sims_needed = SIM_LEVELS[-1]
        levels_measured = []
        for n_sims in SIM_LEVELS:
            idx = np.sort(perm[:n_sims])
            level_counts = _crossing_counts(field_sub_full[idx], cand_scores[:, idx], TOPN_RANK)
            corr = spearmanr(truth, level_counts).correlation
            levels_measured.append((n_sims, corr))
            if corr >= CONVERGENCE_THRESHOLD:
                n_sims_needed = n_sims
                break

        elapsed = time.time() - t0
        print(f"[field_size={field_size:,}] {elapsed:.1f}s -> "
              + ", ".join(f"{n}={c:.3f}" for n, c in levels_measured)
              + f"  needed={n_sims_needed:,} (>= {CONVERGENCE_THRESHOLD} Spearman)")
        per_size_convergence[field_size] = n_sims_needed
        _append_and_reload(RESULTS_CSV, field_size, {
            "field_size": field_size, "n_sims_needed": n_sims_needed,
            "final_spearman": levels_measured[-1][1], "elapsed_s": elapsed,
        })

    # Log-log power-law fit: n_sims_needed ~= sims_min * (field_size / ref)^power.
    sizes = np.array(sorted(per_size_convergence))
    needed = np.array([per_size_convergence[s] for s in sizes], dtype=float)
    if len(sizes) >= 2 and np.all(needed > 0):
        log_x, log_y = np.log(sizes), np.log(needed)
        power, log_a = np.polyfit(log_x, log_y, 1)
        ref = float(sizes[0])
        sims_min = float(np.exp(log_a) * ref ** power)
        print("\nSuggested config (fit across field sizes measured above):")
        print(f"  external_pool_topn_sims_min: {sims_min:.0f}")
        print(f"  external_pool_topn_sims_reference_field_size: {ref:.0f}")
        print(f"  external_pool_topn_sims_power: {power:.3f}")
    else:
        print("\nNot enough convergent field sizes to fit a power law yet.")


if __name__ == "__main__":
    main()
