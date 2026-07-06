"""Empirical mean calibration factors for market-implied projections.

Single source of truth shared by:

  - scripts/fetch_market_odds_projections.py — applies the factors when
    building projections (batters at the event-rate level, pitchers on the
    whole marginal), so downstream means/grids reflect realized levels;
  - src/api/pipeline.py — divides them back out before ownership
    prediction via restore_fitted_mean_scale(): the ownership heuristic's
    constants and the isotonic calibrator were fitted on pre-calibration
    (hot) means, and the softmax normalization uses absolute std floors, so
    feeding deflated means flattens predicted ownership (~1pp on chalk,
    E[dupes] ~-6%) — measured 2026-07-06.

Factors fitted 2026-07-06 against 30 archived slates (4,791 confirmed
batter-games / 597 pitcher-games, PPD-zeroed team-games excluded):
realized DK points / market-implied mean, flat across projection quintiles.
See the fetcher's _MEAN_CALIB_* comment block and the pool-ceiling memory
notes for full provenance. Set to 1.0 to disable. When the ownership model
is eventually re-fitted on post-2026-07-06 (calibrated) archives, drop the
restore_fitted_mean_scale() calls in the pipeline.
"""
import pandas as pd

MEAN_CALIB_BATTER = 0.867
MEAN_CALIB_PITCHER = 0.946


def restore_fitted_mean_scale(players_df: pd.DataFrame) -> pd.DataFrame:
    """Copy of players_df with the mean calibration divided back out — the
    input scale the ownership constants were fitted on.

    RotoWire-fallback players were never scaled at source and no source
    marker survives the projections merge, so they come out ~15% inflated
    here; they are few (players without usable market odds) and punt-tier.
    """
    df = players_df.copy()
    pitcher = df["position"] == "P"
    df.loc[pitcher, "mean"] = df.loc[pitcher, "mean"] / MEAN_CALIB_PITCHER
    df.loc[~pitcher, "mean"] = df.loc[~pitcher, "mean"] / MEAN_CALIB_BATTER
    return df
