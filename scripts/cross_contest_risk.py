"""How correlated is your downside across contests played on the SAME slate?

Every selection objective here optimises ONE contest. None can see that a slate
busting takes every contest down together, so cross-contest correlation is
structurally outside the objective -- which is what makes a constraint (rather
than a better objective) the right tool for it, unlike the within-contest caps
we dropped.

`gamma_out` in mrp/allocator.py already encodes exactly this idea, and its
docstring is explicit that it is "NOT an EV rule ... it is bankroll-variance
control". But it lives only on the marginal-reward multi-contest path; the GPP
selector path builds one portfolio per contest and never allocates across them.

So this measures the exposure first, before anyone builds a knob for it:

  overlap   player-level sharing between the two contests' portfolios
  rho       correlation of per-world NET RETURN between them
  joint     P(both lose), and the left tail of COMBINED bankroll

Per-world returns are what matters, not per-lineup means: two portfolios can
each look fine in expectation and still lose together in the same worlds.
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

_OFF = 1e6


def per_world_net(sc, X, Ffield, payout, fee):
    """(c,) portfolio NET dollars in each world of this chunk."""
    n_field, K = Ffield.shape[0], X.shape[0]
    n_paid = int((payout > 0).sum())
    c = sc.shape[0]
    FS = np.sort((sc @ Ffield.T).astype(np.float64), axis=1)
    XS = (sc @ X.T).astype(np.float64)
    offs = (np.arange(c) * _OFF)[:, None]
    idx = np.searchsorted((FS + offs).ravel(), (XS + offs).ravel(), side="right")
    n_le = idx - np.repeat(np.arange(c) * n_field, K)
    above = (n_field - n_le).reshape(c, K)
    order = np.argsort(-XS, axis=1, kind="stable")
    own = np.empty_like(order)
    np.put_along_axis(own, order, np.broadcast_to(np.arange(K), XS.shape), axis=1)
    rank = above + own
    pay = np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0)
    return pay.sum(axis=1) - fee * K


def load_arm(path, arm, pid_index):
    df = pd.read_csv(PROJECT_ROOT / path)
    g = df[df.arm == arm].sort_values("slot")
    X = np.zeros((len(g), len(pid_index)), dtype=np.float32)
    for r, ids in enumerate(g["player_ids"]):
        for p in str(ids).split("|"):
            X[r, pid_index[int(p)]] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--a", required=True, help="LABEL:portfolios.csv:entries.csv:payout.txt:fee")
    ap.add_argument("--b", required=True, help="same, the second contest on this slate")
    ap.add_argument("--arms", default="kelly,dr,emax,coverage")
    ap.add_argument("--n-sims", type=int, default=60_000)
    ap.add_argument("--sim-batch", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}

    specs = {}
    for tag, spec in (("A", args.a), ("B", args.b)):
        lbl, pf, ent, pay, fee = spec.split(":")
        Ff, payout, n_field, n_paid = pg.build_field(
            ent, pay, pid_index, name_to_id, len(pid_index))
        specs[tag] = {"label": lbl, "pf": pf, "F": Ff, "payout": payout,
                      "fee": float(fee), "n_field": n_field}
        print(f"{tag}: {lbl} — {n_field:,} entries, {n_paid:,} paid, fee ${fee}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = []
    for arm in arms:
        XA = load_arm(specs["A"]["pf"], arm, pid_index)
        XB = load_arm(specs["B"]["pf"], arm, pid_index)
        if len(XA) == 0 or len(XB) == 0:
            print(f"  {arm}: missing from one contest, skipped"); continue
        # composition overlap, the thing gamma_out would cap
        pair = XA @ XB.T                                    # (KA, KB) shared players
        players_a = set(np.where(XA.sum(0) > 0)[0])
        players_b = set(np.where(XB.sum(0) > 0)[0])
        netA = np.zeros(args.n_sims); netB = np.zeros(args.n_sims)
        np.random.seed(args.seed)
        done = 0
        while done < args.n_sims:
            b = min(args.sim_batch, args.n_sims - done)
            sim = engine.simulate(b)
            sc = sim.results_matrix.astype(np.float32)
            for st in range(0, b, args.chunk):
                s = sc[st:st + args.chunk]
                lo, hi = done + st, done + st + s.shape[0]
                netA[lo:hi] = per_world_net(s, XA, specs["A"]["F"],
                                            specs["A"]["payout"], specs["A"]["fee"])
                netB[lo:hi] = per_world_net(s, XB, specs["B"]["F"],
                                            specs["B"]["payout"], specs["B"]["fee"])
            done += b
            del sim, sc
        comb = netA + netB
        costA = specs["A"]["fee"] * len(XA); costB = specs["B"]["fee"] * len(XB)
        rows.append({
            "arm": arm,
            "mean_overlap": float(pair.mean()),
            "max_overlap": float(pair.max()),
            "pct_pairs_ge7": float((pair >= 7).mean()),
            "shared_players": len(players_a & players_b),
            "jaccard": len(players_a & players_b) / max(len(players_a | players_b), 1),
            "rho_worlds": float(np.corrcoef(netA, netB)[0, 1]),
            "p_both_lose": float(((netA < 0) & (netB < 0)).mean()),
            "p_both_lose_indep": float((netA < 0).mean() * (netB < 0).mean()),
            "comb_roi": float(comb.mean() / (costA + costB)),
            "comb_p05": float(np.percentile(comb, 5)),
            "comb_p01": float(np.percentile(comb, 1)),
            "worst_case_cost": -(costA + costB),
        })
        r = rows[-1]
        print(f"  {arm:<9} overlap {r['mean_overlap']:.2f} | shared players "
              f"{r['shared_players']:>3} | rho {r['rho_worlds']:+.3f} | "
              f"P(both lose) {r['p_both_lose']:.3f} vs {r['p_both_lose_indep']:.3f} indep")
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / f"outputs/cross_contest_{Path(args.slate).name}.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()
