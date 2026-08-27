"""Re-grade saved portfolios across several grading seeds to size the noise band.

A single grading run reports one number per arm, which invites reading a 1-point
gap as a result. It is not: the grade is a Monte Carlo estimate over
`--grade-sims` worlds, and the payout ladder is top-heavy enough that a handful
of tail worlds move it. This re-grades the SAME portfolios against the SAME real
field under independent world draws, so the spread across seeds is a direct
read on how much of any arm-to-arm difference is resolved.

The portfolios are fixed inputs here — nothing is re-selected — so this isolates
grading noise from pool/selection noise.

Usage
-----
    source venv/bin/activate
    python scripts/regrade_portfolios.py \
        --slate archive/08252026 \
        --portfolios outputs/fp4_me/portfolios.csv \
        --entries outputs/contest_sim_roi/entries_sim_roi.csv \
        --payout-table outputs/contest_sim_roi/me_warmup_payouts.txt \
        --entry-fee 333 --seeds 5 --grade-sims 40000
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
from analyze_contest_sim_roi import build_slate  # noqa: E402
import portfolio_grading as pg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--portfolios", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--payout-table", required=True)
    ap.add_argument("--entry-fee", type=float, required=True)
    ap.add_argument("--grade-sims", type=int, default=40_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=5_000)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}

    df = pd.read_csv(args.portfolios)
    mats = {}
    for arm, grp in df.groupby("arm", sort=False):
        g = grp.sort_values("slot")
        A = np.zeros((len(g), len(pid_index)), dtype=np.float32)
        for r, ids in enumerate(g["player_ids"]):
            for p in str(ids).split("|"):
                A[r, pid_index[int(p)]] = 1.0
        mats[arm] = A
    print(f"{len(mats)} arms x {next(iter(mats.values())).shape[0]} lineups")

    Ff, payout, n_field, n_paid = pg.build_field(
        args.entries, args.payout_table, pid_index, name_to_id, len(pid_index))
    print(f"real field {n_field:,} entries, {n_paid:,} paid")

    rois = {a: [] for a in mats}
    for k in range(args.seeds):
        seed = args.seed0 + 1_000 * k
        port, _ = pg.grade_portfolios_multi(
            engine, mats, Ff, payout, args.grade_sims,
            sim_batch=20_000, chunk=500, seed=seed, progress=False)
        for a in mats:
            rois[a].append(pg.summarize(port[a], args.entry_fee)["roi"])
        print(f"  seed {seed} done")

    rows = []
    for a, v in rois.items():
        v = np.array(v) * 100
        rows.append({"arm": a, "mean_roi": v.mean(), "sd": v.std(ddof=1),
                     "min": v.min(), "max": v.max(),
                     "spread": v.max() - v.min()})
    out = pd.DataFrame(rows).sort_values("mean_roi", ascending=False)
    print(f"\n=== portfolio ROI (%) over {args.seeds} independent grading seeds ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    sd = out["sd"].median()
    print(f"\nmedian per-arm sd across seeds: {sd:.2f} points")
    print(f"a gap of ~{2.8 * sd:.1f} points is ~2 sd of a difference "
          f"(sqrt(2) x 2 sd) — treat anything smaller as unresolved")


if __name__ == "__main__":
    main()
