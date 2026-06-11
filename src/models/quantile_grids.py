"""Loading and validation of market-implied score-distribution quantile grids.

The market-odds fetcher writes `projections_mo_dist.parquet` next to the
projections CSV: one row per market-covered player with columns
`player_id`, `mean`, and `q0..q100` (percentiles of a Monte Carlo DK-score
distribution built from the fitted market rates).

Grids are matched to the slate defensively: a grid is only applied when the
player's projected mean in players_df agrees with the grid's mean within a
tolerance. Players whose projection came from a fallback source (RotoWire,
DFF, salary heuristic) or from a stale fetch fail this check and keep their
existing marginal.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIST_FILENAME = "projections_mo_dist.parquet"

# |players_df.mean - grid mean| above this means the projection didn't come
# from the same fetch that produced the grid (fallback source or stale file).
MEAN_MATCH_TOLERANCE = 0.5


def quantile_columns(n_points: int = 101) -> list[str]:
    return [f"q{i}" for i in range(n_points)]


def load_quantile_grids(
    dist_path: str,
    players_df: pd.DataFrame,
    mean_tolerance: float = MEAN_MATCH_TOLERANCE,
) -> dict[int, np.ndarray]:
    """Return {player_id: quantile_grid} for players whose grid validates.

    Returns an empty dict when the file is absent or unreadable — callers
    fall back to the parametric marginals.
    """
    if not dist_path or not os.path.exists(dist_path):
        return {}
    try:
        dist_df = pd.read_parquet(dist_path)
    except Exception as exc:
        logger.warning("Could not read quantile grids %s: %s", dist_path, exc)
        return {}

    q_cols = [c for c in dist_df.columns if c.startswith("q") and c[1:].isdigit()]
    q_cols.sort(key=lambda c: int(c[1:]))
    if len(q_cols) < 2 or "player_id" not in dist_df.columns:
        logger.warning("Quantile grid file %s has unexpected schema; ignoring.", dist_path)
        return {}

    proj_means = dict(zip(
        players_df["player_id"].astype(int), players_df["mean"].astype(float)
    ))

    grids: dict[int, np.ndarray] = {}
    n_stale = 0
    for row in dist_df.itertuples(index=False):
        pid = int(row.player_id)
        proj_mean = proj_means.get(pid)
        if proj_mean is None:
            continue
        grid = np.array([getattr(row, c) for c in q_cols], dtype=np.float64)
        grid_mean = float(getattr(row, "mean", grid.mean()))
        if abs(grid_mean - proj_mean) > mean_tolerance:
            n_stale += 1
            continue
        # Guard against numerical noise in the stored grid.
        grids[pid] = np.maximum.accumulate(grid)

    if n_stale:
        logger.info(
            "Quantile grids: %d player(s) skipped (projection mean does not match "
            "grid — fallback source or stale dist file).", n_stale,
        )
    return grids
