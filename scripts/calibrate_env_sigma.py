"""Calibrate the batter-MC latent run-environment sigma (_ENV_SIGMA).

The batter Monte Carlo in fetch_market_odds_projections.py scales all event
rates by a shared mean-one lognormal factor z ~ LogNormal(-sigma^2/2, sigma) so
a batter's counting stats co-move within a game the way they do in real box
scores. This script fits sigma against history:

1. Load per-player per-game batting stats from the Retrosheet cwdaily CSVs
   (data/raw/retrosheet/<year>/daily_<year>.csv — the same source that feeds
   historical_logs.parquet, but before DK-point aggregation).
2. For players with enough games, demean each stat within player (removing the
   skill-level confound) and compute the pooled residual correlation for each
   stat pair — the historical within-player co-movement target.
3. Replicate the fetcher's event model (identical structural couplings:
   R = HR + Poisson excess, RBI = HR + NB(1) excess, all rates scaled by z)
   at league-average per-game rates over a sigma grid, and pick the sigma that
   minimises the squared error against the historical pair correlations.
4. Report the DK-score distribution impact (mean must be unchanged — the grid
   matching tolerance is +/-0.5 — while std and P95/P99 rise).

Usage:
    python scripts/calibrate_env_sigma.py [--min-games 100] [--n-draws 400000]
        [--sigma-max 0.8] [--sigma-step 0.02]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_market_odds_projections import (  # noqa: E402
    _DK_BATTER, _ENV_SIGMA, _simulate_batter_distribution,
)

RETROSHEET_DIR = PROJECT_ROOT / "data" / "raw" / "retrosheet"

# Stats compared, in the vocabulary of the fetcher's rate dict.
STATS = ["single", "double", "home_run", "walk", "run", "rbi"]

# Pairs used for the sigma fit. (home_run, run) and (home_run, rbi) are
# reproduced structurally in the MC (R >= HR, RBI >= HR) rather than through
# z, but they still depend on sigma via the excess terms, so all pairs
# contribute to the fit; the report shows each pair separately.
PAIRS = [
    (a, b) for i, a in enumerate(STATS) for b in STATS[i + 1:]
]


def load_daily_batting(min_games: int) -> pd.DataFrame:
    """Per-player-game batting stats from every cwdaily CSV on disk."""
    paths = sorted(glob.glob(str(RETROSHEET_DIR / "*" / "daily_*.csv")))
    if not paths:
        raise SystemExit(
            f"No cwdaily CSVs found under {RETROSHEET_DIR}. "
            "Run scripts/process_historical.py first."
        )
    cols = ["PLAYER_ID", "SLOT_CT", "SEQ_CT", "B_PA", "B_H", "B_2B", "B_3B",
            "B_HR", "B_R", "B_RBI", "B_BB", "B_SB", "B_HP"]
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=cols)
        df = df[(df["SLOT_CT"].between(1, 9)) & (df["SEQ_CT"] == 1)]
        frames.append(df)
        print(f"  {Path(p).name}: {len(df):,} batter-games")
    daily = pd.concat(frames, ignore_index=True)

    for c in ["B_PA", "B_H", "B_2B", "B_3B", "B_HR", "B_R", "B_RBI",
              "B_BB", "B_SB", "B_HP"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
    daily = daily[daily["B_PA"] >= 1]

    out = pd.DataFrame({
        "player_id": daily["PLAYER_ID"],
        "single": daily["B_H"] - daily["B_2B"] - daily["B_3B"] - daily["B_HR"],
        "double": daily["B_2B"],
        "triple": daily["B_3B"],
        "home_run": daily["B_HR"],
        "walk": daily["B_BB"],
        "run": daily["B_R"],
        "rbi": daily["B_RBI"],
        "sb": daily["B_SB"],
        "hbp": daily["B_HP"],
    })
    games_per_player = out.groupby("player_id")["single"].transform("size")
    out = out[games_per_player >= min_games].reset_index(drop=True)
    print(
        f"Sample: {len(out):,} batter-games from "
        f"{out['player_id'].nunique():,} players with >= {min_games} games."
    )
    return out


def historical_pair_corrs(daily: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Pooled within-player (demeaned) correlation per stat pair."""
    resid = daily[STATS] - daily.groupby("player_id")[STATS].transform("mean")
    corr = resid.corr()
    return {(a, b): float(corr.loc[a, b]) for a, b in PAIRS}


def simulate_pair_corrs(
    rates: dict[str, float], sigma: float, n_draws: int, seed: int = 0
) -> dict[tuple[str, str], float]:
    """Stat-pair correlations under the fetcher's event model at this sigma.

    Mirrors _simulate_batter_distribution's draw structure exactly (shared
    mean-one lognormal z, R = HR + Poisson excess, RBI = HR + NB(1) excess).
    """
    rng = np.random.default_rng(seed)
    if sigma > 0:
        z = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=n_draws)
    else:
        z = np.ones(n_draws)

    draws = {
        "single": rng.poisson(rates["single"] * z),
        "double": rng.poisson(rates["double"] * z),
        "home_run": rng.poisson(rates["home_run"] * z),
        "walk": rng.poisson(rates["walk"] * z),
    }
    hr = draws["home_run"]
    draws["run"] = hr + rng.poisson(
        max(rates["run"] - rates["home_run"], 0.0) * z
    )
    m_excess = max(rates["rbi"] - rates["home_run"], 0.0)
    if m_excess > 0:
        draws["rbi"] = hr + rng.negative_binomial(1, 1.0 / (1.0 + m_excess * z))
    else:
        draws["rbi"] = hr.copy()

    mat = np.column_stack([draws[s].astype(np.float64) for s in STATS])
    corr = np.corrcoef(mat, rowvar=False)
    idx = {s: i for i, s in enumerate(STATS)}
    return {(a, b): float(corr[idx[a], idx[b]]) for a, b in PAIRS}


def score_stats(rates: dict[str, float], sigma: float) -> dict[str, float]:
    """DK-score distribution stats from the actual fetcher MC at this sigma."""
    grid, sd = _simulate_batter_distribution(rates, seed=123, env_sigma=sigma)
    # grid is the P0..P100 percentile grid; its mean approximates the MC mean.
    pct = np.linspace(0, 100, len(grid))
    return {
        "mean": float(np.trapezoid(grid, pct) / 100.0),
        "std": float(sd),
        "p95": float(np.interp(95, pct, grid)),
        "p99": float(np.interp(99, pct, grid)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-games", type=int, default=100,
                    help="minimum games for a player to enter the sample")
    ap.add_argument("--n-draws", type=int, default=400_000,
                    help="MC draws per sigma grid point")
    ap.add_argument("--sigma-max", type=float, default=0.8)
    ap.add_argument("--sigma-step", type=float, default=0.02)
    args = ap.parse_args()

    print("Loading cwdaily batting logs...")
    daily = load_daily_batting(args.min_games)
    hist = historical_pair_corrs(daily)

    # League-average per-game rates over the sample drive the MC.
    rates = {s: float(daily[s].mean()) for s in STATS}
    rates.update({"triple": float(daily["triple"].mean()),
                  "sb": float(daily["sb"].mean()),
                  "hbp": float(daily["hbp"].mean())})
    print("\nLeague-average per-game rates (sample):")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()))

    sigmas = np.arange(0.0, args.sigma_max + 1e-9, args.sigma_step)
    sse = []
    sim_by_sigma = {}
    for sig in sigmas:
        sim = simulate_pair_corrs(rates, float(sig), args.n_draws)
        sim_by_sigma[float(sig)] = sim
        sse.append(sum((sim[p] - hist[p]) ** 2 for p in PAIRS))
    best_i = int(np.argmin(sse))
    best_sigma = float(sigmas[best_i])

    print(f"\n{'pair':<22}{'hist':>8}{'sig=0':>8}"
          f"{'sig=' + format(best_sigma, '.2f'):>10}"
          f"{'current=' + format(_ENV_SIGMA, '.2f'):>14}")
    cur = sim_by_sigma.get(
        float(_ENV_SIGMA), simulate_pair_corrs(rates, _ENV_SIGMA, args.n_draws)
    )
    for p in PAIRS:
        print(f"{p[0] + '-' + p[1]:<22}{hist[p]:>8.3f}"
              f"{sim_by_sigma[0.0][p]:>8.3f}{sim_by_sigma[best_sigma][p]:>10.3f}"
              f"{cur[p]:>14.3f}")

    print(f"\nSSE by sigma (min at sigma={best_sigma:.2f}):")
    for sig, e in zip(sigmas, sse):
        marker = "  <-- best" if float(sig) == best_sigma else ""
        if float(sig) in (0.0, best_sigma, round(float(_ENV_SIGMA), 4)) or e == min(sse):
            print(f"  sigma={sig:.2f}  SSE={e:.5f}{marker}")

    print("\nDK-score distribution impact (league-average batter):")
    base = score_stats(rates, 0.0)
    fit = score_stats(rates, best_sigma)
    curd = score_stats(rates, float(_ENV_SIGMA))
    print(f"  {'':>10}{'mean':>8}{'std':>8}{'P95':>8}{'P99':>8}")
    for name, st in (("sigma=0", base), (f"sigma={best_sigma:.2f}", fit),
                     (f"current={_ENV_SIGMA:.2f}", curd)):
        print(f"  {name:>10}{st['mean']:>8.2f}{st['std']:>8.2f}"
              f"{st['p95']:>8.2f}{st['p99']:>8.2f}")
    dmean = abs(fit["mean"] - base["mean"])
    print(f"\n  mean shift at fitted sigma: {dmean:.3f} "
          f"({'OK' if dmean < 0.5 else 'EXCEEDS'} +/-0.5 grid-match tolerance)")
    print(f"\nRecommended _ENV_SIGMA = {best_sigma:.2f} "
          f"(current default {_ENV_SIGMA:.2f})")


if __name__ == "__main__":
    main()
