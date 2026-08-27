"""Should the ceiling gate be denominated in POINTS or in RANK?

The gate's whole job, per the no-gate ablation, is concentrating the shortlist:
the objectives know what they want (Kelly climbs to the 92nd ceiling percentile
from an unbiased 51st-percentile sample) but cannot reach far enough without
help. So WHAT the gate concentrates on is the entire design decision.

It currently concentrates on absolute score -- `p99.9` of the lineup's own
points. But the payout ladder pays RANK, not points. In a quiet world every
score is low, including the field's, so a lineup can win on 150 where an
explosive world needs 210. An absolute-points gate may be discarding lineups
that rank well in exactly the worlds where ranking well is cheap.

Two currencies, both contest-independent (the pool is its own reference, so a
shortlist is still computed once per slate and reused across contests):

  A_abs   p99.9 of the lineup's raw score.                    [current]
  B_rank  the number of worlds in which the lineup clears the
          POOL's OWN p99.5 for that world -- a per-world bar
          that floats with the run environment.

Ground truth is what the objectives actually consume: mean marginal payout
against a real ladder and a simulated field. A currency is better if it ranks
lineups more like the payout does, and if gating on it retains more of the
genuinely top-payout lineups.

Reported per contest, because the answer may depend on ladder shape.
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

_WORLD_OFFSET = 1e6


def mean_payout(C, sim32, field_sorted, payout, world_chunk=1_000):
    """(M,) mean marginal $ per lineup — the ground truth the gate should track.

    Accumulates rather than materialising (M, S): the full payout matrix is
    1.9 GB at 38,650 x 12,500 float32 and only its row-mean is needed here.
    """
    M = C.shape[0]
    S, F = field_sorted.shape
    n_paid = int((payout > 0).sum())
    acc = np.zeros(M, dtype=np.float64)
    for w0 in range(0, S, world_chunk):
        w1 = min(w0 + world_chunk, S)
        c = w1 - w0
        cs = (sim32[w0:w1] @ C.T).astype(np.float64)          # (c, M)
        fs = field_sorted[w0:w1].astype(np.float64)
        offs = (np.arange(c) * _WORLD_OFFSET)[:, None]
        idx = np.searchsorted((fs + offs).ravel(), (cs + offs).ravel(), side="right")
        n_le = idx - np.repeat(np.arange(c) * F, M)
        rank = (F - n_le).reshape(c, M)
        acc += np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0).sum(0)
        del cs, fs, idx, n_le, rank
    return acc / S


def rank_currency(C, sim32, q=99.5, world_chunk=2_000):
    """(M,) count of worlds where the lineup clears the POOL's per-world q.

    Chunked over WORLDS, not candidates: a per-world quantile needs every
    candidate present for that world, which is the opposite axis from the
    absolute-ceiling pass.
    """
    M, S = C.shape[0], sim32.shape[0]
    cnt = np.zeros(M, dtype=np.int64)
    for w0 in range(0, S, world_chunk):
        w1 = min(w0 + world_chunk, S)
        blk = sim32[w0:w1] @ C.T                               # (c, M)
        bar = np.percentile(blk, q, axis=1)                    # per-world bar
        cnt += (blk >= bar[:, None]).sum(axis=0)
        del blk, bar
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--structure", action="append", required=True,
                    help="registered payout structure name — repeatable")
    ap.add_argument("--n-candidates", type=int, default=30_000)
    ap.add_argument("--n-anchors", type=int, default=800)
    ap.add_argument("--worlds", type=int, default=12_500)
    ap.add_argument("--gate-pct", type=float, default=95.0)
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
                               ceiling_worlds=args.worlds, seed=args.seed)
    print("[1/4] pool")
    cands = fp.build_pool(players_df, engine, own_pct, c,
                          progress=lambda m: print(f"      {m}"))
    C = fp.indicator_matrix(cands, pid_index)
    print(f"[2/4] currencies over {args.worlds:,} worlds")
    A_abs, bits, sim32, bar = fp.lineup_ceilings(engine, C, c)
    B_rank = rank_currency(C, sim32)
    sa, sb = pd.Series(A_abs), pd.Series(B_rank)
    print(f"      spearman(A_abs, B_rank) = {sa.corr(sb, method='spearman'):+.4f}")
    k = int(round(len(cands) * (100 - args.gate_pct) / 100.0))
    topA = set(np.argsort(-A_abs)[:k]); topB = set(np.argsort(-B_rank)[:k])
    print(f"      top-{100-args.gate_pct:g}% sets ({k:,} each) overlap "
          f"{len(topA & topB):,} ({100*len(topA & topB)/k:.1f}%)")

    cs = ContestSimulator()
    rows = []
    for name in args.structure:
        st = load_payout_structure(name)
        F = int(st["total_entries"]); payout = payout_table_to_array(st)
        print(f"[3/4] {name}: field {F:,}")
        fl = cs.generate_field(players_df, own_pct, n_lineups=F, rng_seed=args.seed + 2)
        field_sorted = np.sort(cs.score_field(fl, sim32, pid_index), axis=1)
        truth = mean_payout(C, sim32, field_sorted, payout)
        del field_sorted
        st_t = pd.Series(truth)
        rA = sa.corr(st_t, method="spearman"); rB = sb.corr(st_t, method="spearman")
        topT = set(np.argsort(-truth)[:k])
        rows.append({
            "structure": name, "field": F,
            "spearman_A_abs": rA, "spearman_B_rank": rB,
            "recallA": len(topA & topT) / k, "recallB": len(topB & topT) / k,
            "meanpay_topA": truth[list(topA)].mean(),
            "meanpay_topB": truth[list(topB)].mean(),
            "meanpay_pool": truth.mean(),
        })
        print(f"      spearman vs payout — A_abs {rA:+.4f}   B_rank {rB:+.4f}")
        print(f"      top-{100-args.gate_pct:g}% recall of true best — "
              f"A {rows[-1]['recallA']:.3f}  B {rows[-1]['recallB']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / f"outputs/gate_currency_{Path(args.slate).name}.csv",
               index=False)
    print("\n[4/4] summary")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
