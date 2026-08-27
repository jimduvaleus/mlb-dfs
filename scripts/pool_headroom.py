"""Is the CANDIDATE POOL the binding constraint on the objectives?

The no-gate ablation showed the ceiling gate is worth 8-22 points, and that its
mechanism is candidate DENSITY rather than selection knowledge: handed an
unbiased 51st-percentile sample, Kelly still climbed to the 92nd ceiling
percentile on its own -- it knew what it wanted, there just wasn't enough of it
in reach. That makes pool composition a GENERATION question, not a selection
one.

This measures, for every saved portfolio, where its lineups sit in the POOL's
own ceiling and ownership distributions, and how much headroom was left above
them. Two failure modes to distinguish:

  PINNED  -- the objective's picks sit at the very top of what the pool offers,
             so it wanted more than generation supplied. Generation is binding.
  SLACK   -- the picks sit well inside the pool's range, so the objective
             stopped for its own reasons. Generation is not binding.

Run per slate; portfolios are matched to pool lineups by roster key.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization import fast_portfolio as fp  # noqa: E402
from analyze_contest_sim_roi import build_slate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--portfolios", action="append", required=True,
                    help="LABEL:path/to/portfolios.csv — repeatable")
    ap.add_argument("--n-candidates", type=int, default=30_000)
    ap.add_argument("--n-anchors", type=int, default=800)
    ap.add_argument("--ceiling-worlds", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    own_pct = players_df["ownership"].fillna(0.0).to_numpy()

    c = fp.FastPortfolioConfig(n_candidates=args.n_candidates,
                               n_anchors=args.n_anchors,
                               ceiling_worlds=args.ceiling_worlds, seed=args.seed)
    print("[1/3] pool")
    cands = fp.build_pool(players_df, engine, own_pct, c,
                          progress=lambda m: print(f"      {m}"))
    C = fp.indicator_matrix(cands, pid_index)
    own_sum = C @ own_pct
    print("[2/3] ceiling")
    ceiling, bits, sim32, bar = fp.lineup_ceilings(engine, C, c)
    key_to_i = {frozenset(int(p) for p in lu.player_ids): i
                for i, lu in enumerate(cands)}
    cr = pd.Series(ceiling).rank(pct=True).to_numpy() * 100
    orr = pd.Series(own_sum).rank(pct=True).to_numpy() * 100
    n_sampler = args.n_candidates
    print(f"      pool {len(cands):,}: ceiling p50 {np.median(ceiling):.1f} "
          f"p99 {np.percentile(ceiling,99):.1f} max {ceiling.max():.1f}; "
          f"ownership p50 {np.median(own_sum):.1f} p99 "
          f"{np.percentile(own_sum,99):.1f} max {own_sum.max():.1f}")
    print(f"      sampler-only ceiling p50 {np.median(ceiling[:n_sampler]):.1f}; "
          f"anchors+mutants p50 {np.median(ceiling[n_sampler:]):.1f}")

    print("[3/3] portfolios")
    rows = []
    for spec in args.portfolios:
        label, path = spec.split(":", 1)
        df = pd.read_csv(PROJECT_ROOT / path)
        for arm, grp in df.groupby("arm", sort=False):
            idx, miss = [], 0
            for ids in grp.sort_values("slot")["player_ids"]:
                k = frozenset(int(x) for x in str(ids).split("|"))
                j = key_to_i.get(k)
                if j is None:
                    miss += 1
                else:
                    idx.append(j)
            if not idx:
                continue
            idx = np.array(idx)
            rows.append({
                "contest": label, "arm": arm, "n": len(idx), "unmatched": miss,
                "ceil_pctile_mean": cr[idx].mean(),
                "ceil_pctile_min": cr[idx].min(),
                "own_pctile_mean": orr[idx].mean(),
                "own_pctile_max": orr[idx].max(),
                "n_pool_above_ceiling": int((ceiling > ceiling[idx].mean()).sum()),
                "n_pool_above_own": int((own_sum > own_sum[idx].mean()).sum()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / f"outputs/pool_headroom_{Path(args.slate).name}.csv",
               index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    main()
