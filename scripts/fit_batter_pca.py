"""
Fit and persist the Phase 4 batter PCA model.

Steps
-----
1. Load data/processed/historical_logs.parquet (produced by process_historical.py).
2. Filter to batter slots (1-9).
3. Build the empirical DK score grid (sorted unique scores observed historically).
4. For each player with ≥ MIN_GAMES appearances, fit a mixture model to their
   per-game DK scores → (w, lam, mu, sigma).
5. Run PCA on the collected (n_players, 4) parameter matrix.
6. Save the PCA model to data/processed/batter_pca_model.npz.
7. Save the score grid to data/processed/batter_score_grid.npy.

Usage
-----
    python scripts/fit_batter_pca.py [--min-games 30]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path when called as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.batter_model import BatterPCAModel, fit_mixture_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HISTORICAL_PATH = project_root / "data" / "processed" / "historical_logs.parquet"
PCA_MODEL_PATH = project_root / "data" / "processed" / "batter_pca_model.npz"
SCORE_GRID_PATH = project_root / "data" / "processed" / "batter_score_grid.npy"


def _load_batter_logs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"player_id", "slot", "dk_points"}
    if not required.issubset(df.columns):
        raise ValueError(f"historical_logs must contain columns {required}. Got: {list(df.columns)}")
    batters = df[df["slot"].between(1, 9)].copy()
    batters["dk_points"] = batters["dk_points"].clip(lower=0)
    log.info("Loaded %d batter game rows for %d unique players.",
             len(batters), batters["player_id"].nunique())
    return batters


def _build_score_grid(dk_points: pd.Series) -> np.ndarray:
    """Sorted array of all unique DK batter scores observed historically."""
    grid = np.sort(dk_points.unique()).astype(float)
    log.info("Score grid: %d unique values, range [%.1f, %.1f].",
             len(grid), grid.min(), grid.max())
    return grid


def _fit_all_players(batters: pd.DataFrame, min_games: int) -> np.ndarray:
    """
    For each player with >= min_games appearances, fit mixture model.
    Returns (n_fitted, 4) array of (w, lam, mu, sigma) vectors.
    """
    params_list = []
    players = batters.groupby("player_id")
    n_attempted = 0

    for player_id, group in players:
        scores = group["dk_points"].values
        if len(scores) < min_games:
            continue
        n_attempted += 1
        result = fit_mixture_params(scores, min_games=min_games)
        if result is None:
            log.debug("Fitting failed for player %s — skipping.", player_id)
            continue
        params_list.append(result)

    log.info("Fitted mixture model for %d / %d qualifying players.",
             len(params_list), n_attempted)

    if not params_list:
        raise RuntimeError(
            "No player fits succeeded. Check that historical_logs.parquet is populated."
        )

    return np.array(params_list, dtype=float)


def main(min_games: int = 30) -> None:
    if not HISTORICAL_PATH.exists():
        log.error("historical_logs.parquet not found at %s. "
                  "Run scripts/process_historical.py first.", HISTORICAL_PATH)
        sys.exit(1)

    # --- Load data ---
    batters = _load_batter_logs(HISTORICAL_PATH)

    # --- Score grid ---
    score_grid = _build_score_grid(batters["dk_points"])

    # --- Fit per-player mixture params ---
    params_matrix = _fit_all_players(batters, min_games=min_games)
    log.info("Parameter matrix shape: %s", params_matrix.shape)
    log.info("Mean params (w, lam, mu, sigma): %s", params_matrix.mean(axis=0).round(4))

    # --- PCA ---
    pca_model = BatterPCAModel()
    pca_model.fit(params_matrix)
    explained = np.linalg.svd(
        (params_matrix - pca_model.mean_) / pca_model.std_,
        compute_uv=False,
    )
    total_var = (explained ** 2).sum()
    top2_var = ((explained[:2] ** 2).sum() / total_var * 100) if total_var > 0 else float("nan")
    log.info("Top-2 PCs explain %.1f%% of normalised variance.", top2_var)

    # --- Save outputs ---
    PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pca_model.save(str(PCA_MODEL_PATH))
    np.save(str(SCORE_GRID_PATH), score_grid)

    log.info("PCA model saved to %s", PCA_MODEL_PATH)
    log.info("Score grid saved to %s", SCORE_GRID_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-games",
        type=int,
        default=30,
        help="Minimum game appearances required to include a player in PCA fitting.",
    )
    args = parser.parse_args()
    main(min_games=args.min_games)
