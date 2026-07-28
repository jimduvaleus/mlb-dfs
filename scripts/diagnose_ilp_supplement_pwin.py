"""
Diagnostic: does a generated supplement (ilp or sim_winner) carry a
systematically LOWER p_win_select than the existing external pool, despite
each lineup having won at least one simulated world?

Why this check
---------------
compare_candidate_pools.py's ilp run found augmentation helps the raw
pool's max score occasionally but never helps -- and sometimes hurts --
the SELECTED portfolio's hit99 rate. The working hypothesis: a per-world
argmax lineup is optimal for ONE specific simulated world, not necessarily
a lineup that performs well across the distribution of worlds p_win
actually integrates over, so the selector's own (real) p_win estimate for
these lineups may be unremarkable or poor despite their "this exact
lineup won some simulated world" pedigree. This script tests that
directly rather than assuming it, for whichever generation method is
cached: it recomputes p_win_select for the exact combined pool
compare_candidate_pools.py used (same seed, same n_sims, same field
draws), then reports the external-only vs. supplement-only distributions
side by side.

Confirmed for method=ilp across all 7 test slates (2026-07-28): mean
p_win_select 0.57x-0.68x of the external pool's, and literally 0 of the
supplement lineups landed in the combined pool's top 1% on any slate (106.2
expected under random placement, summed across slates). Testing method=
sim_winner next to see whether the softer rank-softmax objective avoids
the same failure mode.

Uses the supplement + external pool already cached under
outputs/pool_compare/<slate>/<method>/ by compare_candidate_pools.py -- no
new generation, just a fresh p_win pass over what's already on disk.

Usage
-----
    python scripts/diagnose_ilp_supplement_pwin.py --slate 07262026 --method ilp
    python scripts/diagnose_ilp_supplement_pwin.py --slate 07262026 --method sim_winner
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api import external_pool as ep  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402
from compare_candidate_pools import build_players_df, OUT_ROOT  # noqa: E402

_N_SIMS_DEFAULT = 10_000
_SEED_DEFAULT = 0
_SHARPNESS_DEFAULT = 0.05


def diagnose_slate(slate: str, method: str, n_sims: int, seed: int, sharpness: float) -> dict | None:
    d = PROJECT_ROOT / "archive" / slate
    cache_dir = OUT_ROOT / slate / method
    supplement_path = cache_dir / "supplement.json"
    if not supplement_path.exists():
        print(f"  [{slate}/{method}] no cached supplement at {supplement_path} -- skipping "
              f"(run compare_candidate_pools.py --method {method} for this slate first)")
        return None

    print(f"\n=== {slate} ({method}) ===")
    players_df, grids, name_to_id = build_players_df(d)
    found = ep.discover_external_files(str(d))
    valid_ids = set(players_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids)

    supplement_ids = json.loads(supplement_path.read_text())
    supplement = [Lineup(player_ids=ids) for ids in supplement_ids]
    print(f"  external pool: {len(pool.lineups):,}  {method} supplement (cached): {len(supplement):,}")

    # Same dedup rule compare_candidate_pools.py's augment_pool() applies,
    # so the "external" / "supplement" split below matches what selection
    # actually saw -- a supplement lineup that exactly duplicated an
    # existing one wouldn't have been added, and shouldn't be double
    # counted here either.
    seen = {frozenset(int(p) for p in lu.player_ids) for lu in pool.lineups}
    supp_unique = []
    for lu in supplement:
        key = frozenset(int(p) for p in lu.player_ids)
        if key not in seen:
            seen.add(key)
            supp_unique.append(lu)
    combined = list(pool.lineups) + supp_unique
    n_ext = len(pool.lineups)
    print(f"  after de-dup vs pool: {len(supp_unique):,} genuinely new {method} lineups")

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    # Same draw shape as compare_candidate_pools.run_slate: generation slice
    # then selection slice. We only need the SELECTION slice here (p_win is
    # a selection-time question), but must reproduce the same seeded draw
    # sequence to land on the identical select_sims compare_candidate_pools
    # actually used.
    sim_results = engine.simulate(n_sims + n_sims)
    select_sims = SimulationResults(sim_results.player_ids, sim_results.results_matrix[n_sims:])

    ext_pool = ep.ExternalPool(
        lineups=combined, contests={}, n_dropped_unknown_players=0,
        n_dropped_duplicates=0, n_dropped_near_duplicates=0, source_paths=[],
    )
    n_half = select_sims.results_matrix.shape[0] // 2
    lineup_scores = ep.compute_lineup_scores(ext_pool.lineups, select_sims)
    scores_A, scores_B = lineup_scores[:, :n_half], lineup_scores[:, n_half:2 * n_half]
    col_map = {int(p): i for i, p in enumerate(select_sims.player_ids)}
    cs = ContestSimulator()
    own_vec = players_df["ownership"].astype(float).to_numpy()
    field_A = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=100),
                             select_sims.results_matrix[:n_half], col_map)
    field_B = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=101),
                             select_sims.results_matrix[n_half:2 * n_half], col_map)
    exponent = max(1.0, sharpness * 10_000.0)
    # p_win_select is the ranking currency (stage-B draw) -- the one that
    # actually determines who gets picked, so it's the one worth comparing.
    p_win_select = ep.compute_p_win(scores_B, field_B, {"c0": exponent})["c0"]

    ext_vals = p_win_select[:n_ext]
    supp_vals = p_win_select[n_ext:]

    def summarize(vals: np.ndarray) -> dict:
        return {
            "n": len(vals), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)), "p90": float(np.percentile(vals, 90)),
            "p99": float(np.percentile(vals, 99)), "max": float(vals.max()),
        }

    ext_stats = summarize(ext_vals)
    supp_stats = summarize(supp_vals)

    # Where do supplement lineups land in the COMBINED ranking? If they were
    # genuinely as good as their "won some world" pedigree suggests, they
    # should show up disproportionately near the top, not spread through
    # the middle/bottom of the pack.
    order = np.argsort(-p_win_select)
    rank_of = np.empty(len(p_win_select), dtype=int)
    rank_of[order] = np.arange(1, len(p_win_select) + 1)
    supp_ranks = rank_of[n_ext:]
    top1pct_cut = max(1, int(0.01 * len(p_win_select)))
    supp_in_top1pct = int((supp_ranks <= top1pct_cut).sum())
    supp_share_of_pool = len(supp_vals) / len(p_win_select)
    expected_if_random = supp_share_of_pool * top1pct_cut

    print(f"\n  {'':10s} {'n':>6s} {'mean':>10s} {'median':>10s} {'p75':>10s} {'p90':>10s} {'p99':>10s} {'max':>10s}")
    print(f"  {'external':10s} {ext_stats['n']:6d} {ext_stats['mean']:10.2e} {ext_stats['median']:10.2e} "
          f"{ext_stats['p75']:10.2e} {ext_stats['p90']:10.2e} {ext_stats['p99']:10.2e} {ext_stats['max']:10.2e}")
    print(f"  {method:10s} {supp_stats['n']:6d} {supp_stats['mean']:10.2e} {supp_stats['median']:10.2e} "
          f"{supp_stats['p75']:10.2e} {supp_stats['p90']:10.2e} {supp_stats['p99']:10.2e} {supp_stats['max']:10.2e}")
    print(f"\n  {method} lineups in the combined pool's actual top 1% ({top1pct_cut} slots): "
          f"{supp_in_top1pct} observed vs {expected_if_random:.1f} expected if {method} "
          f"lineups were randomly distributed by rank ({method} is {supp_share_of_pool:.1%} of the pool)")

    return {
        "slate": slate, "method": method, "n_ext": n_ext, "n_supp": len(supp_vals),
        "ext_mean": ext_stats["mean"], "supp_mean": supp_stats["mean"],
        "ext_median": ext_stats["median"], "supp_median": supp_stats["median"],
        "ext_p90": ext_stats["p90"], "supp_p90": supp_stats["p90"],
        "supp_in_top1pct": supp_in_top1pct, "expected_if_random": expected_if_random,
        "supp_share_of_pool": supp_share_of_pool,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slate", action="append", default=[])
    p.add_argument("--method", choices=["ilp", "sim_winner"], default="ilp")
    p.add_argument("--n-sims", type=int, default=_N_SIMS_DEFAULT)
    p.add_argument("--seed", type=int, default=_SEED_DEFAULT)
    p.add_argument("--sharpness", type=float, default=_SHARPNESS_DEFAULT)
    args = p.parse_args()
    if not args.slate:
        print("No slates given (use --slate, repeatable).")
        sys.exit(1)

    rows = [r for s in args.slate
            if (r := diagnose_slate(s, args.method, args.n_sims, args.seed, args.sharpness)) is not None]
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n\n=== Summary across slates ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nmean ext p_win   : {df['ext_mean'].mean():.4e}")
    print(f"mean {args.method} p_win : {df['supp_mean'].mean():.4e}")
    print(f"{args.method} lineups actually in top 1%: {df['supp_in_top1pct'].sum()} observed vs "
          f"{df['expected_if_random'].sum():.1f} expected under random rank placement")


if __name__ == "__main__":
    main()
