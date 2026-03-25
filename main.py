#!/usr/bin/env python3
"""
MLB DFS portfolio optimizer — end-to-end pipeline entry point.

Usage:
    python main.py [config.yaml]

Defaults to ``config.yaml`` in the current directory.
Pipeline:
    Load DK slate → derive player metadata → load copula →
    (optionally) load batter PCA model → simulate → construct portfolio → save CSV
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from src.ingestion.dk_slate import DraftKingsSlateIngestor
from src.models.batter_model import BatterPCAModel
from src.models.copula import EmpiricalCopula
from src.optimization.lineup import ROSTER_REQUIREMENTS
from src.optimization.portfolio import PortfolioConstructor
from src.simulation.engine import SimulationEngine
from src.simulation.results import SimulationResults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Slate → players_df
# ---------------------------------------------------------------------------

def _derive_opponent(team: str, game: str) -> str:
    """Return the opponent abbreviation from a 'AWAY@HOME' game string."""
    if not game:
        return ""
    parts = game.split("@")
    if len(parts) != 2:
        return ""
    away, home = parts[0].strip(), parts[1].strip()
    if team == away:
        return home
    if team == home:
        return away
    return ""


def build_players_df(
    slate_df: pd.DataFrame,
    proj_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Produce the ``players_df`` required by SimulationEngine and PortfolioConstructor.

    Derives ``opponent`` from the game string and assigns batting-order ``slot``
    values (1-9 for batters cycling within each team-unit, 10 for pitchers).
    Merges mean/std_dev from *proj_df* when provided; falls back to a
    salary-based heuristic (mean = salary / 400, std_dev = 40 % of mean).

    Required output columns:
        player_id, team, opponent, slot, mean, std_dev, position, salary, game
    """
    df = slate_df.copy()

    # Derive opponent
    df["opponent"] = df.apply(
        lambda r: _derive_opponent(r["team"], r["game"]), axis=1
    )

    # Assign slots: pitchers → 10, batters → 1-9 cycling within (team, opponent)
    is_pitcher = df["position"] == "P"
    df["slot"] = 10
    batter_mask = ~is_pitcher
    df.loc[batter_mask, "slot"] = (
        df[batter_mask]
        .groupby(["team", "opponent"])
        .cumcount()
        .mod(9)
        .add(1)
    )

    # Projections
    if proj_df is not None:
        proj = proj_df.copy().rename(
            columns={"mu": "mean", "sigma": "std_dev"}
        )
        proj_cols = ["player_id", "mean", "std_dev"]
        # Use RotoWire batting-order slots when available
        if "lineup_slot" in proj.columns:
            proj_cols.append("lineup_slot")
        df = df.merge(proj[proj_cols], on="player_id", how="left")
        missing = df["mean"].isna().sum()
        if missing:
            logger.warning(
                "%d player(s) missing from projections CSV; using salary heuristic.",
                missing,
            )
        df.loc[df["mean"].isna(), "mean"] = df.loc[df["mean"].isna(), "salary"] / 400.0
        df.loc[df["std_dev"].isna(), "std_dev"] = df.loc[df["std_dev"].isna(), "mean"] * 0.4
        # Override sequential slot assignment with RotoWire batting order where present
        if "lineup_slot" in df.columns:
            batter_with_slot = (
                ~(df["position"] == "P")
                & df["lineup_slot"].notna()
                & df["lineup_slot"].between(1, 9)
            )
            df.loc[batter_with_slot, "slot"] = df.loc[batter_with_slot, "lineup_slot"].astype(int)
            n_overridden = batter_with_slot.sum()
            if n_overridden:
                logger.info("Applied RotoWire batting-order slots to %d batters.", n_overridden)
    else:
        logger.info("No projections file — using salary-based heuristic (mean = salary / 400).")
        df["mean"] = df["salary"] / 400.0
        df["std_dev"] = df["mean"] * 0.4

    return df[
        ["player_id", "team", "opponent", "slot", "mean", "std_dev", "position", "salary", "game"]
    ]


# ---------------------------------------------------------------------------
# Auto-target computation
# ---------------------------------------------------------------------------

def _compute_auto_target(
    players_df: pd.DataFrame, sim_results: SimulationResults
) -> float:
    """
    Estimate a GPP target score as the 80th-percentile total of a greedy
    high-projection lineup.

    Greedily fills each roster slot with the highest-mean available player,
    then returns p80 of that lineup's simulated totals.  Falls back to a
    proportional row-sum estimate if the greedy fill fails.
    """
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    sorted_df = players_df.sort_values("mean", ascending=False)

    counts: dict[str, int] = {pos: 0 for pos in ROSTER_REQUIREMENTS}
    selected: list[int] = []

    for _, row in sorted_df.iterrows():
        pos = str(row["position"])
        if pos in counts and counts[pos] < ROSTER_REQUIREMENTS[pos]:
            selected.append(int(row["player_id"]))
            counts[pos] += 1
        if len(selected) == 10:
            break

    if len(selected) == 10:
        cols = [col_map[pid] for pid in selected if pid in col_map]
        if len(cols) == 10:
            totals = sim_results.results_matrix[:, cols].sum(axis=1)
            return float(np.percentile(totals, 80))

    # Fallback: proportional scaling of full-slate row sums
    n = len(players_df)
    row_sums = sim_results.results_matrix.sum(axis=1)
    return float(np.percentile(row_sums * 10.0 / n, 80))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_portfolio(
    portfolio: list, players_df: pd.DataFrame
) -> pd.DataFrame:
    """Return a long-form DataFrame with one row per player per lineup."""
    id_to_name = dict(zip(players_df["player_id"], players_df.get("name", players_df["player_id"])))
    id_to_salary = dict(zip(players_df["player_id"], players_df["salary"]))
    id_to_pos = dict(zip(players_df["player_id"], players_df["position"]))
    id_to_team = dict(zip(players_df["player_id"], players_df["team"]))

    rows = []
    for i, (lineup, score) in enumerate(portfolio, start=1):
        total_salary = sum(id_to_salary.get(pid, 0) for pid in lineup.player_ids)
        for pid in lineup.player_ids:
            rows.append(
                {
                    "lineup": i,
                    "p_hit_target": round(score, 4),
                    "player_id": pid,
                    "name": id_to_name.get(pid, str(pid)),
                    "position": id_to_pos.get(pid, ""),
                    "team": id_to_team.get(pid, ""),
                    "salary": id_to_salary.get(pid, 0),
                    "lineup_salary": total_salary,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    sim_cfg = cfg.get("simulation", {})
    opt_cfg = cfg.get("optimizer", {})
    port_cfg = cfg.get("portfolio", {})

    # --- Load slate ---------------------------------------------------------
    dk_path = paths["dk_slate"]
    logger.info("Loading DK slate: %s", dk_path)
    ingestor = DraftKingsSlateIngestor(dk_path)
    slate_df = ingestor.get_slate_dataframe()
    logger.info(
        "Slate loaded: %d players, %d teams",
        len(slate_df),
        slate_df["team"].nunique(),
    )

    # --- Projections (optional) ---------------------------------------------
    proj_df = None
    proj_path = paths.get("projections")
    if proj_path:
        logger.info("Loading projections: %s", proj_path)
        proj_df = pd.read_csv(proj_path)

    # --- Build players_df ---------------------------------------------------
    players_df = build_players_df(slate_df, proj_df)
    # Attach names for output — safe even if 'name' column is absent
    if "name" in slate_df.columns:
        players_df = players_df.merge(
            slate_df[["player_id", "name"]], on="player_id", how="left"
        )

    n_units = players_df.groupby(["team", "opponent"]).ngroups
    logger.info(
        "Player pool: %d players across %d simulation units", len(players_df), n_units
    )

    # --- Load copula --------------------------------------------------------
    copula_path = paths["copula"]
    logger.info("Loading empirical copula: %s", copula_path)
    copula = EmpiricalCopula(copula_path)

    # --- Load batter PCA model (Phase 4+, optional) -------------------------
    pca_model: BatterPCAModel | None = None
    score_grid: np.ndarray | None = None
    pca_path = paths.get("batter_pca_model")
    grid_path = paths.get("batter_score_grid")
    if pca_path and os.path.exists(pca_path) and grid_path and os.path.exists(grid_path):
        logger.info("Loading batter PCA model: %s", pca_path)
        pca_model = BatterPCAModel.load(pca_path)
        score_grid = np.load(grid_path)
    elif pca_path:
        logger.warning(
            "batter_pca_model or batter_score_grid not found — "
            "using Gaussian marginals for batters."
        )

    # --- Simulate -----------------------------------------------------------
    n_sims = int(sim_cfg.get("n_sims", 10_000))
    logger.info("Running %d simulations...", n_sims)
    engine = SimulationEngine(
        copula, players_df, batter_pca_model=pca_model, score_grid=score_grid
    )
    sim_results = engine.simulate(n_sims)
    logger.info(
        "Simulation complete — matrix: %s", sim_results.results_matrix.shape
    )

    # --- Target score -------------------------------------------------------
    target = port_cfg.get("target_score")
    if target is None:
        target = _compute_auto_target(players_df, sim_results)
        logger.info("Auto-computed target score: %.1f DK pts", target)
    else:
        target = float(target)
        logger.info("Using configured target score: %.1f DK pts", target)

    # --- Construct portfolio ------------------------------------------------
    portfolio_size = int(port_cfg.get("size", 20))
    logger.info(
        "Constructing portfolio — size=%d, target=%.1f, chains=%d",
        portfolio_size,
        target,
        opt_cfg.get("n_chains", 250),
    )
    constructor = PortfolioConstructor(
        sim_results=sim_results,
        players_df=players_df,
        target=target,
        portfolio_size=portfolio_size,
        n_chains=int(opt_cfg.get("n_chains", 250)),
        temperature=float(opt_cfg.get("temperature", 0.1)),
        n_steps=int(opt_cfg.get("n_steps", 100)),
        n_workers=int(opt_cfg.get("n_workers", 1)),
        rng_seed=opt_cfg.get("rng_seed"),
    )
    portfolio = constructor.construct()
    logger.info("Portfolio complete: %d lineups selected.", len(portfolio))

    # --- Output -------------------------------------------------------------
    output_dir = paths.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    portfolio_df = _format_portfolio(portfolio, players_df)

    output_path = os.path.join(output_dir, "portfolio.csv")
    portfolio_df.to_csv(output_path, index=False)
    logger.info("Portfolio saved to %s", output_path)

    # Print summary to stdout
    print()
    for lineup_num, group in portfolio_df.groupby("lineup"):
        score = group["p_hit_target"].iloc[0]
        salary = group["lineup_salary"].iloc[0]
        print(
            f"--- Lineup {lineup_num:>2d}  |  "
            f"P(≥ target) = {score:.4f}  |  "
            f"Salary = ${salary:>6,.0f} ---"
        )
        for _, row in group.iterrows():
            print(
                f"  {row['position']:3s}  {str(row['name']):<25s}  "
                f"{row['team']:5s}  ${row['salary']:>6,.0f}"
            )
    print(f"\nPortfolio saved to {output_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB DFS portfolio optimizer")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        print(
            "Copy config.yaml.example to config.yaml and edit the paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    main(args.config)
