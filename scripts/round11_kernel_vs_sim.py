"""Round-11: kernel selector (composition-only covariance) vs sim-Det.

Pre-registered in plans/round11_kernel_selector.md. The incumbent
DeterminantPortfolioSelector is reused UNMODIFIED — the experiment swaps
only the `precomputed` correlation matrix: sim-Det's payout correlations
(archived portfolios) vs a structured kernel built from composition:

    cov(i,j) ~= a * SUM_{p in i&j} sigma_p^2        (shared players)
              + b * SUM_t n_it * n_jt               (hitter team overlap)
              + g * c_i * c_j                       (chalk factor; c = centered SUM own)

Coefficients (a,b,g) are fit by --calibrate on CALIB_SLATES (excluded from
judgment): payout-proxy covariances from cached sims — lineup scores ->
within-sample rank -> reference payout curve -> pairwise covariance —
regressed on the three features (nonneg least squares via clipping).

--run executes the head-to-head on the remaining slates: same pool
(fresh-EV survivors), same EV vector, same evw/cash-anchor/risk params from
each run's archived config; only the correlation source differs. Kernel
portfolios are graded against graded_lineups.csv real_pct; the incumbent
arm is the archived risk-tier portfolio (selected_risks membership).
"""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.gpp_portfolio import DeterminantPortfolioSelector  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402
from scripts.replay_slate import _payout_curve  # noqa: E402

CALIB_SLATES = ("06282026", "06052026")
CALIB_PATH = Path("data/processed/kernel_selector_calib.json")
RISK = 1.0  # incumbent comparison tier used throughout rounds 8-9 analyses


def load_run(rd: Path):
    """Pool rows with fresh_ev — the live selection pool — plus metadata."""
    per_lu = {}
    for r in csv.DictReader(open(rd / "candidate_pool_debug.csv")):
        li = int(r["lineup_index"])
        d = per_lu.setdefault(li, {
            "pids": [], "teams": [], "pos": [], "own": [], "fresh": None,
            "sel_risks": r["selected_risks"],
        })
        d["pids"].append(int(r["player_id"]))
        d["teams"].append(r["team"])
        d["pos"].append(r["position"])
        d["own"].append(float(r["ownership"] or 0))
        if r["fresh_ev"].strip():
            d["fresh"] = float(r["fresh_ev"])
    pct = {}
    for r in csv.DictReader(open(rd / "graded_lineups.csv")):
        pct[int(r["lineup_index"])] = float(r["real_pct"])
    std = {}
    proj = pd.read_csv(rd / "projections.csv")
    for r in proj.itertuples(index=False):
        std[int(r.player_id)] = float(getattr(r, "std_dev", 0.0) or 0.0)
    cfg = {}
    import yaml
    with open(rd / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return per_lu, pct, std, cfg


def pool_arrays(per_lu, std):
    """Selection pool (fresh_ev present) -> index list + feature ingredients."""
    idx = sorted(li for li, d in per_lu.items() if d["fresh"] is not None)
    fresh = np.array([per_lu[li]["fresh"] for li in idx])
    own_sum = np.array([sum(d["own"]) for d in (per_lu[li] for li in idx)])
    return idx, fresh, own_sum


def kernel_corr(per_lu, idx, std, own_sum, coef):
    """Structured covariance -> correlation via feature matrix Phi (PSD)."""
    a, b, g = coef
    pid_col: dict[int, int] = {}
    team_col: dict[str, int] = {}
    rows, cols, vals = [], [], []
    c = own_sum - own_sum.mean()
    for i, li in enumerate(idx):
        d = per_lu[li]
        for pid, pos in zip(d["pids"], d["pos"]):
            j = pid_col.setdefault(pid, len(pid_col))
            rows.append(i); cols.append(j)
            vals.append(np.sqrt(max(a, 0.0)) * max(std.get(pid, 5.0), 1e-3))
        tc = defaultdict(int)
        for t, pos in zip(d["teams"], d["pos"]):
            if pos != "P" and t:
                tc[t] += 1
        for t, n in tc.items():
            j = team_col.setdefault(t, len(team_col))
            rows.append(i); cols.append(len(pid_col) + 100000 + j)  # offset later
            vals.append(np.sqrt(max(b, 0.0)) * n)
    n_p = len(pid_col)
    n_t = len(team_col)
    M = len(idx)
    Phi = np.zeros((M, n_p + n_t + 1), dtype=np.float64)
    for r, cc, v in zip(rows, cols, vals):
        j = cc if cc < 100000 else n_p + (cc - 100000 - n_p)
        Phi[r, j] += v
    Phi[:, -1] = np.sqrt(max(g, 0.0)) * c
    norms = np.linalg.norm(Phi, axis=1)
    norms = np.where(norms < 1e-9, 1.0, norms)
    Phin = Phi / norms[:, None]
    corr = (Phin @ Phin.T).astype(np.float32)
    return corr


def pair_features(per_lu, idx, std, own_sum, pairs):
    c = own_sum - own_sum.mean()
    psets = [set(per_lu[li]["pids"]) for li in idx]
    tcs = []
    for li in idx:
        d = per_lu[li]
        tc = defaultdict(int)
        for t, pos in zip(d["teams"], d["pos"]):
            if pos != "P" and t:
                tc[t] += 1
        tcs.append(tc)
    F = np.zeros((len(pairs), 3))
    for k, (i, j) in enumerate(pairs):
        shared = psets[i] & psets[j]
        F[k, 0] = sum(std.get(p, 5.0) ** 2 for p in shared)
        F[k, 1] = sum(n * tcs[j].get(t, 0) for t, n in tcs[i].items())
        F[k, 2] = c[i] * c[j]
    return F


def calibrate():
    coefs = []
    for slate in CALIB_SLATES:
        rd = Path(f"outputs/replay/{slate}/blend_full_p150")
        per_lu, pct, std, cfg = load_run(rd)
        idx, fresh, own_sum = pool_arrays(per_lu, std)
        rng = np.random.default_rng(7)
        sub = rng.choice(len(idx), min(2000, len(idx)), replace=False)
        idx_s = [idx[i] for i in sub]
        own_s = own_sum[sub]
        # cached sims -> lineup scores
        cache = glob.glob(f"outputs/replay/sim_cache/{slate}_*.npz")
        z = np.load(sorted(cache)[0])
        col = {int(p): k for k, p in enumerate(z["player_ids"])}
        R = z["results_matrix"][:4000]
        S = np.zeros((R.shape[0], len(idx_s)), dtype=np.float64)
        ok = np.ones(len(idx_s), dtype=bool)
        for i, li in enumerate(idx_s):
            cs = [col.get(p, -1) for p in per_lu[li]["pids"]]
            if -1 in cs:
                ok[i] = False
                continue
            S[:, i] = R[:, cs].sum(axis=1)
        idx_s = [li for i, li in enumerate(idx_s) if ok[i]]
        own_s = own_s[ok]; S = S[:, ok]
        # PRE-REGISTRATION DEVIATION (documented): the spec said regress on
        # payout covariances, but those are unestimable — split-half
        # reliability r=0.34 at 4k sims (points-space r=0.993). The payout
        # transform destroys pairwise estimability; even the incumbent's
        # 25k-sim correlation matrix is heavily noise-laden. Calibrate on
        # points-space covariance (the estimable object); the head-to-head
        # then tests dollar-space-sim vs points-space-kernel diversification
        # on graded outcomes.
        P = S - S.mean(axis=0, keepdims=True)
        cov = (P.T @ P) / P.shape[0]
        n = S.shape[1]
        pairs_i = rng.integers(0, n, 120000)
        pairs_j = rng.integers(0, n, 120000)
        keep = pairs_i != pairs_j
        pairs = list(zip(pairs_i[keep][:80000], pairs_j[keep][:80000]))
        y = np.array([cov[i, j] for i, j in pairs])
        sub_per = {li: per_lu[li] for li in idx_s}
        F = pair_features(sub_per, idx_s, std, own_s, pairs)
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        coef = np.clip(coef, 0.0, None)
        r2 = 1 - np.sum((F @ coef - y) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"{slate}: n={n} coef a={coef[0]:.4g} b={coef[1]:.4g} g={coef[2]:.4g} R2={r2:.3f}")
        coefs.append(coef)
    final = np.mean(coefs, axis=0)
    CALIB_PATH.write_text(json.dumps({
        "a_player": final[0], "b_team": final[1], "g_own": final[2],
        "calib_slates": CALIB_SLATES, "note": "round-11 kernel covariance fit",
    }, indent=2))
    print("saved", CALIB_PATH, final)


def run_headtohead():
    coef_d = json.loads(CALIB_PATH.read_text())
    coef = (coef_d["a_player"], coef_d["b_team"], coef_d["g_own"])
    out = []
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
        fresh = fresh[keep]; own_sum = own_sum[keep]
        if len(idx) < 200:
            continue
        corr = kernel_corr(per_lu, idx, std, own_sum, coef)
        pool_idx = np.arange(len(idx))
        pre = (pool_idx, fresh.astype(np.float64), corr)
        cands = [Lineup(player_ids=per_lu[li]["pids"]) for li in idx]
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
        order = []
        pos_of = {tuple(sorted(c.player_ids)): i for i, c in enumerate(cands)}
        for lu, _ in picked:
            order.append(idx[pos_of[tuple(sorted(lu.player_ids))]])
        kern = [pct[li] for li in order if li in pct]
        kern_own = [sum(per_lu[li]["own"]) for li in order]
        inc = [pct[li] for li, d in per_lu.items()
               if li in pct and d["sel_risks"] and "1" in str(d["sel_risks"]).split(",")]
        jac = len(set(order) & {li for li, d in per_lu.items()
                                if d["sel_risks"] and "1" in str(d["sel_risks"]).split(",")})
        out.append({
            "slate": slate, "n_pool": len(idx),
            "k_n": len(kern), "k_pct": float(np.mean(kern)),
            "k_cash": float(np.mean([p >= 0.74 for p in kern])),
            "k_own": float(np.mean(kern_own)),
            "k_own_first10": float(np.mean(kern_own[:10])),
            "k_own_last10": float(np.mean(kern_own[-10:])),
            "i_n": len(inc), "i_pct": float(np.mean(inc)) if inc else float("nan"),
            "i_cash": float(np.mean([p >= 0.74 for p in inc])) if inc else float("nan"),
            "overlap": jac,
        })
        r = out[-1]
        print(f"{slate}: pool={r['n_pool']} kernel pct={r['k_pct']:.3f} cash={r['k_cash']:.2f} "
              f"| incumbent pct={r['i_pct']:.3f} cash={r['i_cash']:.2f} | overlap={r['overlap']}/150")
    df = pd.DataFrame(out)
    df.to_csv("outputs/replay/round11_kernel_vs_sim.csv", index=False)
    print("\n=== POOLED (%d slates) ===" % len(df))
    print(f"kernel:    mean_pct={df['k_pct'].mean():.3f} cash={df['k_cash'].mean():.3f} "
          f"own={df['k_own'].mean():.2f} own_first10={df['k_own_first10'].mean():.2f} own_last10={df['k_own_last10'].mean():.2f}")
    print(f"incumbent: mean_pct={df['i_pct'].mean():.3f} cash={df['i_cash'].mean():.3f}")
    print(f"selection overlap: {df['overlap'].mean():.0f}/150 mean")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    if args.run:
        run_headtohead()
