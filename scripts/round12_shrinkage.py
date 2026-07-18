"""Round-12: covariance shrinkage sweep — C(lam) = lam*kernel + (1-lam)*dollar.

See plans/round12_shrinkage.md. Reuses round-11 machinery; dollar-space
correlation rebuilt per slate from cached sims via within-pool payout-proxy
ranks (documented approximation).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.gpp_portfolio import DeterminantPortfolioSelector  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402
from scripts.replay_slate import _payout_curve  # noqa: E402
from scripts.round11_kernel_vs_sim import (  # noqa: E402
    CALIB_PATH, CALIB_SLATES, RISK, kernel_corr, load_run, pool_arrays,
)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_SIMS = 25000


def dollar_corr(slate, per_lu, idx):
    z = np.load(sorted(glob.glob(f"outputs/replay/sim_cache/{slate}_*.npz"))[0])
    col = {int(p): k for k, p in enumerate(z["player_ids"])}
    R = z["results_matrix"][:N_SIMS].astype(np.float32)
    M = len(idx)
    S = np.zeros((R.shape[0], M), dtype=np.float32)
    ok = np.ones(M, dtype=bool)
    for i, li in enumerate(idx):
        cs = [col.get(p, -1) for p in per_lu[li]["pids"]]
        if -1 in cs:
            ok[i] = False
            continue
        S[:, i] = R[:, cs].sum(axis=1)
    curve, fee = _payout_curve(int(ok.sum()))
    Sv = S[:, ok]
    n = Sv.shape[1]
    order = np.argsort(-Sv, axis=1)
    ranks = np.empty_like(order)
    rr = np.arange(n, dtype=order.dtype)
    for s in range(Sv.shape[0]):
        ranks[s, order[s]] = rr
    P = curve[ranks].astype(np.float32) - fee
    P -= P.mean(axis=0, keepdims=True)
    std = P.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Pn = P / std
    corr_ok = (Pn.T @ Pn) / P.shape[0]
    corr = np.eye(M, dtype=np.float32)
    oki = np.where(ok)[0]
    corr[np.ix_(oki, oki)] = corr_ok.astype(np.float32)
    return corr


def main():
    coef_d = json.loads(CALIB_PATH.read_text())
    coef = (coef_d["a_player"], coef_d["b_team"], coef_d["g_own"])
    rows = []
    for rd in sorted(glob.glob("outputs/replay/*/blend_full_p150")):
        slate = rd.split("/")[2]
        if slate in CALIB_SLATES:
            continue
        rd = Path(rd)
        try:
            per_lu, pct, std, cfg = load_run(rd)
        except FileNotFoundError:
            continue
        gpp = cfg.get("gpp", {})
        idx, fresh, own_sum = pool_arrays(per_lu, std)
        floor = float(gpp.get("ev_floor", 0.0))
        keep = fresh >= floor
        idx = [li for li, k in zip(idx, keep) if k]
        fresh = fresh[keep]
        own_sum = own_sum[keep]
        if len(idx) < 200:
            continue
        kc = kernel_corr(per_lu, idx, std, own_sum, coef)
        dc = dollar_corr(slate, per_lu, idx)
        cands = [Lineup(player_ids=per_lu[li]["pids"]) for li in idx]
        pos_of = {tuple(sorted(c.player_ids)): i for i, c in enumerate(cands)}
        for lam in LAMBDAS:
            corr = (lam * kc + (1.0 - lam) * dc).astype(np.float32)
            pre = (np.arange(len(idx)), fresh.astype(np.float64), corr)
            sel = DeterminantPortfolioSelector(
                robust_payout=np.zeros((len(idx), 1), dtype=np.float32),
                candidates=cands,
                portfolio_size=150,
                risk=RISK,
                evw_base=float(gpp.get("evw_base", 0.10)),
                evw_max=float(gpp.get("evw_max", 0.40)),
                ev_floor=floor,
                precomputed=pre,
                cash_anchor_fraction=float(gpp.get("cash_anchor_fraction", 0.25)),
            )
            picked = sel.select()
            order = [idx[pos_of[tuple(sorted(lu.player_ids))]] for lu, _ in picked]
            g = [pct[li] for li in order if li in pct]
            owns = [sum(per_lu[li]["own"]) for li in order]
            rows.append({
                "slate": slate, "lam": lam, "n": len(g),
                "pct": float(np.mean(g)),
                "cash": float(np.mean([p >= 0.74 for p in g])),
                "own_first10": float(np.mean(owns[:10])),
                "own_last10": float(np.mean(owns[-10:])),
            })
            print(f"{slate} lam={lam}: pct={rows[-1]['pct']:.3f} cash={rows[-1]['cash']:.2f} "
                  f"own {rows[-1]['own_first10']:.2f}->{rows[-1]['own_last10']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/replay/round12_shrinkage.csv", index=False)
    print("\n=== POOLED by lambda ===")
    print(df.groupby("lam")[["pct", "cash", "own_first10", "own_last10"]].mean().to_string())


if __name__ == "__main__":
    main()
