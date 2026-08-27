"""Does the portfolio get chalkier as the contest gets smaller?

The claim under test: with a real payout ladder and a real field size in the
objective, contest-awareness should fall out of the math rather than needing a
knob. A 235-entry contest is won by beating 234 people, so pressing down on
ownership buys little and costs mean; a 47,562-entry contest is won by beating a
crowd, so uniqueness is worth paying for.

Builds the pool and the ceiling ONCE, then varies only the contest context
(registered payout ladder + field size) so nothing but the contest differs
between rows.

Also probes the ownership gate. Gate B caps how chalky a build can get no
matter what the objective wants — if contest-awareness only appears with the
gate relaxed, the gate is fighting the objective and should be reconsidered,
by the same argument that rejected structural caps.

    source venv/bin/activate
    python scripts/contest_awareness_probe.py --slate archive/08252026
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.payout import load_payout_structure, payout_table_to_array  # noqa: E402
from src.optimization import fast_portfolio as fp  # noqa: E402
from analyze_contest_sim_roi import build_slate  # noqa: E402

SIZES = ["dk_skipper_235", "dk_base_hit", "dk_hot_corner", "dk_chin_music_2378",
         "dk_classic_gpp_5001", "dk_bat_flip_11437", "dk_classic_gpp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--n-candidates", type=int, default=30_000)
    ap.add_argument("--n-anchors", type=int, default=800)
    ap.add_argument("--ceiling-worlds", type=int, default=25_000)
    ap.add_argument("--contest-worlds", type=int, default=12_500)
    ap.add_argument("--own-gate-pct", type=float, action="append", default=None,
                    help="repeatable; default 40 (current) and 100 (no own gate)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/contest_awareness.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    gates = args.own_gate_pct or [40.0, 100.0]

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    own_pct = players_df["ownership"].fillna(0.0).to_numpy()

    base = fp.FastPortfolioConfig(
        n_candidates=args.n_candidates, n_anchors=args.n_anchors,
        ceiling_worlds=args.ceiling_worlds, contest_worlds=args.contest_worlds,
        seed=args.seed)
    print("[1/3] pool (built once, reused for every contest)")
    cands = fp.build_pool(players_df, engine, own_pct, base,
                          progress=lambda m: print(f"      {m}"))
    C = fp.indicator_matrix(cands, pid_index)
    own_sum = C @ own_pct
    print(f"[2/3] ceiling over {base.ceiling_worlds:,} worlds")
    ceiling, bits, sim32, bar = fp.lineup_ceilings(engine, C, base)

    cw = base.contest_worlds
    cs = ContestSimulator()
    rows = []
    print("[3/3] sweeping contests x ownership gates")
    for gate_pct in gates:
        cfg_g = fp.FastPortfolioConfig(**{**base.__dict__, "own_gate_pct": gate_pct})
        shortlist, gd = fp.conjunctive_gate(ceiling, own_sum, cfg_g)
        short = [cands[i] for i in shortlist]
        cand_scores = (C[shortlist] @ sim32[:cw].T)
        sl_own = own_sum[shortlist]
        print(f"  own_gate={gate_pct:g}%  shortlist={len(shortlist):,}  "
              f"O*={gd['o_star']:.1f}  shortlist mean own={sl_own.mean():.1f}")
        for name in SIZES:
            st = load_payout_structure(name)
            F = int(st["total_entries"])
            payout = payout_table_to_array(st)
            fee = float(st["entry_fee"])
            t0 = time.perf_counter()
            fl = cs.generate_field(players_df, own_pct, n_lineups=F,
                                   rng_seed=args.seed + 2)
            fsc = cs.score_field(fl, sim32[:cw], pid_index)
            field_sorted = np.sort(fsc, axis=1)
            del fsc
            cand_payout = fp.candidate_payout_matrix(cand_scores, field_sorted, payout)
            arms = {}
            arms["kelly"] = fp.select_kelly(cand_payout, short, cfg_g, fee)[0]
            arms["emax"] = fp.select_emax(cand_payout, short, cfg_g)[0]
            arms["coverage"] = fp.select_coverage(
                cand_payout, bits[shortlist], sl_own, short, cfg_g)[0]
            arms["dr"] = fp.select_dr(cand_scores, field_sorted, payout, short,
                                      cfg_g)[0]
            del field_sorted, cand_payout
            for arm, lus in arms.items():
                A = fp.indicator_matrix(lus, pid_index)
                rows.append({
                    "own_gate_pct": gate_pct, "structure": name, "field": F,
                    "fee": fee, "paid_frac": float((payout > 0).mean()),
                    "top_prize_share": float(payout.max() / payout.sum()),
                    "arm": arm,
                    "mean_own": float((A @ own_pct).mean()),
                    "shortlist_mean_own": float(sl_own.mean()),
                })
            print(f"    {name:<22} F={F:>7,}  " + "  ".join(
                f"{a}={rows[-4 + i]['mean_own']:5.1f}"
                for i, a in enumerate(["kelly", "emax", "coverage", "dr"])))
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / args.out, index=False)
    print(f"\nwrote {args.out}")
    piv = out.pivot_table(index=["own_gate_pct", "field"], columns="arm",
                          values="mean_own")
    print("\n=== mean PROJECTED OWNERSHIP of the selected 150, by field size ===")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
