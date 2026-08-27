"""Grade the arms on a SMALL contest, where contest-awareness should pay off.

Every result so far comes from two large contests (3,335 and 10,170 entries).
That is exactly where a contest-BLIND selector is cheapest, so the measured tie
between Kelly/dR (contest-aware) and coverage (flat across a 63x field-size
range) may be an artifact of never testing where they diverge.

NO REAL SMALL CONTEST EXISTS IN THE ARCHIVE. The four archived standings are
3,336 / 3,336 / 4,458 / 10,174 entries, so a small field has to be constructed.
This sub-samples `field_size` entries from a REAL contest's standings: the
lineup COMPOSITIONS are genuine, human-built entries for this slate, and only
the field's SIZE is synthetic. Repeated over `--draws` independent sub-samples
so no single draw drives the answer.

WHAT THIS ASSUMES, AND IT IS NOT FREE. Sub-sampling a $333 / 3,335-entry
contest down to 235 assumes field composition is size-invariant. It is probably
not: a $25 235-person contest plausibly draws a different mix of max-entry
regulars to casuals. So this measures "do the contest-aware arms win when the
LADDER and the FIELD SIZE are small, holding composition fixed" -- the
mechanism we care about -- and NOT "would they win in a real small contest".
Read it as a mechanism test, not a field-realism test.

ENTRY COUNT MATTERS TOO. 150 lineups into a 235-entry field is 64% of the
contest and would own most of the prize pool; the self-competition term would
dominate everything else. `--portfolio-size` should track what the contest
actually allows.

    source venv/bin/activate
    python scripts/grade_small_contest.py --slate archive/08252026 \
        --structure dk_skipper_235 --portfolio-size 3 --draws 30
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
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.payout import load_payout_structure, payout_table_to_array  # noqa: E402
from src.optimization import fast_portfolio as fp  # noqa: E402
from analyze_contest_sim_roi import build_slate  # noqa: E402
import portfolio_grading as pg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--structure", default="dk_skipper_235")
    ap.add_argument("--entries", default="outputs/contest_sim_roi/entries_sim_roi.csv",
                    help="real standings to sub-sample the field from")
    ap.add_argument("--portfolio-size", type=int, default=3)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--grade-sims", type=int, default=20_000)
    ap.add_argument("--n-candidates", type=int, default=30_000)
    ap.add_argument("--n-anchors", type=int, default=800)
    ap.add_argument("--own-gate-pct", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/small_contest_grade.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    st = load_payout_structure(args.structure)
    F = int(st["total_entries"])
    fee = float(st["entry_fee"])
    payout = payout_table_to_array(st)
    print(f"{args.structure}: {F:,} entries, ${fee:.0f} fee, "
          f"{int((payout>0).sum())} paid, ${payout.sum():,.0f} pool, "
          f"top prize {payout.max()/fee:.0f}x the buy-in")
    if args.portfolio_size > 0.15 * F:
        print(f"  WARNING portfolio is {100*args.portfolio_size/F:.0f}% of the field")

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    own_pct = players_df["ownership"].fillna(0.0).to_numpy()

    c = fp.FastPortfolioConfig(
        n_candidates=args.n_candidates, n_anchors=args.n_anchors,
        field_size=F, portfolio_size=args.portfolio_size,
        own_gate_pct=args.own_gate_pct, seed=args.seed)
    print("[1/4] pool")
    cands = fp.build_pool(players_df, engine, own_pct, c,
                          progress=lambda m: print(f"      {m}"))
    C = fp.indicator_matrix(cands, pid_index)
    own_sum = fp.ownership_currency(C, own_pct, "sum")
    print("[2/4] ceiling")
    ceiling, bits, sim32, bar = fp.lineup_ceilings(engine, C, c)
    shortlist, gd = fp.conjunctive_gate(ceiling, own_sum, c)
    short = [cands[i] for i in shortlist]
    cw = c.contest_worlds
    cand_scores = C[shortlist] @ sim32[:cw].T

    print(f"[3/4] contest context: simulated field of {F:,}")
    cs = ContestSimulator()
    fl = cs.generate_field(players_df, own_pct, n_lineups=F, rng_seed=args.seed + 2)
    field_sorted = np.sort(cs.score_field(fl, sim32[:cw], pid_index), axis=1)
    cand_payout = fp.candidate_payout_matrix(cand_scores, field_sorted, payout)
    arms = {
        "kelly": fp.select_kelly(cand_payout, short, c, fee)[0],
        "emax": fp.select_emax(cand_payout, short, c)[0],
        "coverage": fp.select_coverage(cand_payout, bits[shortlist],
                                       own_sum[shortlist], short, c)[0],
        "dr": fp.select_dr(cand_scores, field_sorted, payout, short, c)[0],
        "gate_then_own": fp.select_gate_then_own(own_sum[shortlist], short, c)[0],
    }
    del field_sorted, cand_payout
    for a, lus in arms.items():
        A = fp.indicator_matrix(lus, pid_index)
        print(f"      {a:<14} mean proj own {(A @ own_pct).mean():6.1f}")

    print(f"[4/4] grading vs {args.draws} sub-sampled real fields of {F:,}")
    Ff_all, _, n_real, _ = pg.build_field(
        args.entries, "outputs/contest_sim_roi/me_warmup_payouts.txt",
        pid_index, name_to_id, len(pid_index))
    if F > n_real:
        raise SystemExit(f"cannot sub-sample {F:,} from a {n_real:,}-entry contest")
    mats = {a: fp.indicator_matrix(l, pid_index) for a, l in arms.items()}
    rng = np.random.default_rng(args.seed + 99)
    rois = {a: [] for a in arms}
    for k in range(args.draws):
        idx = rng.choice(n_real, F, replace=False)
        port, _ = pg.grade_portfolios_multi(
            engine, mats, Ff_all[idx], payout, args.grade_sims,
            sim_batch=20_000, chunk=500, seed=args.seed + 500 + k, progress=False)
        for a in arms:
            rois[a].append(pg.summarize(port[a], fee)["roi"])
        if (k + 1) % 5 == 0:
            print(f"      draw {k+1}/{args.draws}")
    rows = []
    for a, v in rois.items():
        v = np.array(v) * 100
        rows.append({"structure": args.structure, "field": F, "fee": fee,
                     "portfolio_size": args.portfolio_size, "arm": a,
                     "mean_roi": v.mean(), "sd": v.std(ddof=1),
                     "se": v.std(ddof=1) / np.sqrt(len(v))})
    out = pd.DataFrame(rows).sort_values("mean_roi", ascending=False)
    out.to_csv(PROJECT_ROOT / args.out, index=False)
    print(f"\n=== {args.structure}, {args.portfolio_size} entries, "
          f"{args.draws} sub-sampled fields ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
