"""Before/after comparison of simulation marginals (work item 4).

Compares the legacy parametric marginals (PCA Exp+Normal mixture for batters,
Gaussian for pitchers) against the market-implied quantile-grid marginals.

Modes
-----
Real mode (default when data/processed/projections_mo_dist.parquet exists):
    Uses the current slate's players and grids; reports per-player moment and
    percentile shifts for every player whose grid validates.

Synthetic mode (--synthetic, or automatic when no dist parquet exists):
    Builds representative batter/pitcher archetypes from typical market rates,
    runs the fetcher's Monte Carlo to produce grids, and compares against the
    parametric marginals those players would have used.

Usage:
    python scripts/compare_marginals.py [--synthetic] [--csv outputs/marginal_comparison.csv]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_market_odds_projections import (  # noqa: E402
    _DK_BATTER, _DK_PITCHER,
    _simulate_batter_distribution, _simulate_pitcher_distribution,
)
from src.models.batter_model import BatterMixtureMarginal, BatterPCAModel  # noqa: E402
from src.models.marginals import EmpiricalQuantileMarginal, GaussianMarginal  # noqa: E402
from src.models.quantile_grids import DIST_FILENAME, load_quantile_grids  # noqa: E402

N_SAMPLE = 200_000

SYNTHETIC_BATTERS = {
    "star slugger": dict(single=1.0, double=0.30, triple=0.02, home_run=0.28,
                         run=1.05, rbi=1.15, walk=0.55, sb=0.05, hbp=0.061),
    "mid-order bat": dict(single=0.9, double=0.25, triple=0.03, home_run=0.15,
                          run=0.70, rbi=0.75, walk=0.40, sb=0.10, hbp=0.044),
    "bottom-order bat": dict(single=0.7, double=0.15, triple=0.02, home_run=0.07,
                             run=0.45, rbi=0.40, walk=0.25, sb=0.05, hbp=0.028),
}
SYNTHETIC_PITCHERS = {
    "ace": dict(outs=18.5, k=8.5, win=0.55, h=4.5, bb=1.6, er=2.0, hbp=0.24),
    "mid-rotation": dict(outs=17.0, k=6.5, win=0.45, h=5.0, bb=2.0, er=2.5, hbp=0.22),
    "back-end starter": dict(outs=15.0, k=4.8, win=0.35, h=5.8, bb=2.6, er=3.1, hbp=0.20),
}


def _stats(draws: np.ndarray) -> dict:
    mu, sd = float(draws.mean()), float(draws.std())
    skew = float(((draws - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    p = np.percentile(draws, [10, 50, 90, 99])
    return dict(mean=mu, std=sd, skew=skew, p10=p[0], p50=p[1], p90=p[2], p99=p[3])


def _analytic_batter(rates: dict) -> tuple[float, float]:
    c = _DK_BATTER
    mean = (rates["single"] * c["single"] + rates["double"] * c["double"]
            + rates["triple"] * c["triple"] + rates["home_run"] * c["home_run"]
            + rates["run"] * c["run"] + rates["rbi"] * c["rbi"]
            + rates["walk"] * c["walk"] + rates["sb"] * c["sb"]
            + rates["hbp"] * c["hbp"])
    var = (rates["single"] * c["single"] ** 2 + rates["double"] * c["double"] ** 2
           + rates["triple"] * c["triple"] ** 2 + rates["home_run"] * c["home_run"] ** 2
           + rates["run"] * c["run"] ** 2 + rates["rbi"] * (1 + rates["rbi"]) * c["rbi"] ** 2
           + rates["walk"] * c["walk"] ** 2 + rates["sb"] * c["sb"] ** 2
           + rates["hbp"] * c["hbp"] ** 2)
    return mean, max(np.sqrt(var), 1.0)


def _analytic_pitcher(rates: dict) -> tuple[float, float]:
    c = _DK_PITCHER
    mean = (rates["outs"] * c["out"] + rates["k"] * c["k"] + rates["win"] * c["win"]
            + rates["er"] * c["er"] + rates["h"] * c["h"] + rates["bb"] * c["bb"]
            + rates["hbp"] * c["hbp"])
    var = (rates["outs"] * c["out"] ** 2 + rates["k"] * c["k"] ** 2
           + rates["win"] * c["win"] ** 2 + rates["er"] * c["er"] ** 2
           + rates["h"] * c["h"] ** 2 + rates["bb"] * c["bb"] ** 2
           + rates["hbp"] * c["hbp"] ** 2)
    return mean, max(np.sqrt(var), 1.0)


def _old_batter_marginal(mean, std, pca_model, score_grid):
    if pca_model is not None:
        w, lam, mu, sigma = pca_model.project(float(mean), float(std))
        if not (w > 0.99 or lam < 0.01):
            return BatterMixtureMarginal(w, lam, mu, sigma, score_grid)
    return GaussianMarginal(mean, std)


def _load_pca():
    pca_path = PROJECT_ROOT / "data/processed/batter_pca_model.npz"
    grid_path = PROJECT_ROOT / "data/processed/batter_score_grid.npy"
    if pca_path.exists() and grid_path.exists():
        return BatterPCAModel.load(str(pca_path)), np.load(grid_path)
    return None, None


def _compare(label, kind, old_marginal, new_grid, rng) -> dict:
    u = rng.random(N_SAMPLE)
    old_draws = old_marginal.ppf(u)
    new_draws = EmpiricalQuantileMarginal(new_grid).ppf(u)
    if kind == "batter":
        old_draws = np.maximum(old_draws, 0)
        new_draws = np.maximum(new_draws, 0)
    row = {"player": label, "kind": kind}
    row.update({f"old_{k}": v for k, v in _stats(old_draws).items()})
    row.update({f"new_{k}": v for k, v in _stats(new_draws).items()})
    return row


def run_synthetic(rng) -> list[dict]:
    pca_model, score_grid = _load_pca()
    rows = []
    for label, rates in SYNTHETIC_BATTERS.items():
        mean, std = _analytic_batter(rates)
        grid, _ = _simulate_batter_distribution(rates, seed=11)
        old = _old_batter_marginal(mean, std, pca_model, score_grid)
        rows.append(_compare(label, "batter", old, grid, rng))
    for label, rates in SYNTHETIC_PITCHERS.items():
        mean, std = _analytic_pitcher(rates)
        grid, _ = _simulate_pitcher_distribution(rates, seed=11)
        rows.append(_compare(label, "pitcher", GaussianMarginal(mean, std), grid, rng))
    return rows


def run_real(rng) -> list[dict]:
    import yaml

    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    proj_path = PROJECT_ROOT / (paths.get("projections") or "data/processed/projections.csv")
    proj_df = pd.read_csv(proj_path)
    if "mu" in proj_df.columns:
        proj_df = proj_df.rename(columns={"mu": "mean", "sigma": "std_dev"})
    dist_path = proj_path.parent / DIST_FILENAME
    grids = load_quantile_grids(str(dist_path), proj_df)
    if not grids:
        print("No validating grids found in real mode.")
        return []

    pca_model, score_grid = _load_pca()
    name_map = dict(zip(proj_df["player_id"].astype(int), proj_df.get("name", proj_df["player_id"])))
    slot_map = (
        dict(zip(proj_df["player_id"].astype(int), proj_df["lineup_slot"]))
        if "lineup_slot" in proj_df.columns else {}
    )
    mean_map = dict(zip(proj_df["player_id"].astype(int), proj_df["mean"].astype(float)))
    std_map = dict(zip(proj_df["player_id"].astype(int), proj_df["std_dev"].astype(float)))

    rows = []
    for pid, grid in grids.items():
        slot = slot_map.get(pid)
        is_pitcher = (slot == 10) if slot is not None else mean_map[pid] > 12.0
        kind = "pitcher" if is_pitcher else "batter"
        if kind == "batter":
            old = _old_batter_marginal(mean_map[pid], std_map[pid], pca_model, score_grid)
        else:
            old = GaussianMarginal(mean_map[pid], std_map[pid])
        rows.append(_compare(str(name_map.get(pid, pid)), kind, old, grid, rng))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic archetype mode")
    ap.add_argument("--csv", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    dist_exists = (PROJECT_ROOT / "data/processed" / DIST_FILENAME).exists()
    if args.synthetic or not dist_exists:
        if not dist_exists and not args.synthetic:
            print(f"(no {DIST_FILENAME} found — running synthetic archetype mode)\n")
        rows = run_synthetic(rng)
    else:
        rows = run_real(rng)
        if not rows:
            rows = run_synthetic(rng)

    df = pd.DataFrame(rows)
    fmt = (
        "{player:<20} {kind:<8} "
        "| mean {old_mean:6.2f} → {new_mean:6.2f} "
        "| std {old_std:5.2f} → {new_std:5.2f} "
        "| skew {old_skew:+5.2f} → {new_skew:+5.2f} "
        "| P10 {old_p10:6.2f} → {new_p10:6.2f} "
        "| P50 {old_p50:6.2f} → {new_p50:6.2f} "
        "| P90 {old_p90:6.2f} → {new_p90:6.2f} "
        "| P99 {old_p99:6.2f} → {new_p99:6.2f}"
    )
    print("Marginal comparison: OLD (parametric) → NEW (market-implied grid)\n")
    for row in rows:
        print(fmt.format(**row))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")


if __name__ == "__main__":
    main()
