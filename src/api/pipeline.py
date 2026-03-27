"""
PipelineRunner — wraps the main.py pipeline with progress callbacks.

Emits callback events at each pipeline stage so the API can forward them
as SSE events to the browser.
"""
import logging
import os
from typing import Callable, Optional

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

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Runs the full DFS portfolio pipeline with progress callbacks.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    progress_cb : callable(stage, data_dict), optional
        Called at each pipeline stage. ``stage`` is a string identifier;
        ``data_dict`` carries stage-specific metadata.
    """

    def __init__(
        self,
        config_path: str,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ):
        self._config_path = config_path
        self._cb = progress_cb or (lambda stage, data: None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """
        Execute the full pipeline.

        Returns
        -------
        list[dict]
            Portfolio serialized as a list of lineup dicts, each with keys:
            lineup_index, p_hit_target, lineup_salary, players (list of dicts).
        """
        cfg = self._load_config()
        paths = cfg.get("paths", {})
        sim_cfg = cfg.get("simulation", {})
        opt_cfg = cfg.get("optimizer", {})
        port_cfg = cfg.get("portfolio", {})

        # --- Load slate --------------------------------------------------
        dk_path = paths["dk_slate"]
        logger.info("Loading DK slate: %s", dk_path)
        ingestor = DraftKingsSlateIngestor(dk_path)
        slate_df = ingestor.get_slate_dataframe()
        n_teams = slate_df["team"].nunique()
        logger.info("Slate loaded: %d players, %d teams", len(slate_df), n_teams)

        # --- Projections (optional) --------------------------------------
        proj_df = None
        proj_path = paths.get("projections")
        if proj_path and os.path.exists(proj_path):
            logger.info("Loading projections: %s", proj_path)
            proj_df = pd.read_csv(proj_path)

        # --- Build players_df -------------------------------------------
        players_df = self._build_players_df(slate_df, proj_df)
        if "name" in slate_df.columns:
            players_df = players_df.merge(
                slate_df[["player_id", "name"]], on="player_id", how="left"
            )

        n_units = players_df.groupby(["team", "opponent"]).ngroups
        self._cb("load_slate", {
            "n_players": len(players_df),
            "n_teams": n_teams,
            "n_units": n_units,
        })

        # --- Load copula -------------------------------------------------
        copula_path = paths["copula"]
        logger.info("Loading empirical copula: %s", copula_path)
        copula = EmpiricalCopula(copula_path)

        # --- Load batter PCA model (optional) ----------------------------
        pca_model: Optional[BatterPCAModel] = None
        score_grid: Optional[np.ndarray] = None
        pca_path = paths.get("batter_pca_model")
        grid_path = paths.get("batter_score_grid")
        if (pca_path and os.path.exists(pca_path)
                and grid_path and os.path.exists(grid_path)):
            logger.info("Loading batter PCA model: %s", pca_path)
            pca_model = BatterPCAModel.load(pca_path)
            score_grid = np.load(grid_path)

        # --- Simulate ---------------------------------------------------
        n_sims = int(sim_cfg.get("n_sims", 10_000))
        logger.info("Running %d simulations...", n_sims)
        self._cb("simulate", {"n_sims": n_sims})
        engine = SimulationEngine(
            copula, players_df, batter_pca_model=pca_model, score_grid=score_grid
        )
        sim_results = engine.simulate(n_sims)
        logger.info("Simulation complete — matrix: %s", sim_results.results_matrix.shape)

        # --- Target score -----------------------------------------------
        target = port_cfg.get("target_score")
        if target is None:
            target_percentile = int(port_cfg.get("target_percentile", 90))
            target = self._compute_auto_target(players_df, sim_results, target_percentile)
            logger.info(
                "Auto-computed target: %.1f DK pts (p%d)", target, target_percentile
            )
            self._cb("compute_target", {"target": target, "percentile": target_percentile})
        else:
            target = float(target)
            logger.info("Using configured target: %.1f DK pts", target)
            self._cb("compute_target", {"target": target, "percentile": None})

        # --- Construct portfolio ----------------------------------------
        portfolio_size = int(port_cfg.get("size", 20))
        logger.info(
            "Constructing portfolio — size=%d, target=%.1f, chains=%d",
            portfolio_size, target, opt_cfg.get("n_chains", 250),
        )
        constructor = PortfolioConstructor(
            sim_results=sim_results,
            players_df=players_df,
            target=target,
            portfolio_size=portfolio_size,
            n_chains=int(opt_cfg.get("n_chains", 250)),
            temperature=float(opt_cfg.get("temperature", 0.1)),
            n_steps=int(opt_cfg.get("n_steps", 100)),
            niter_success=int(opt_cfg.get("niter_success", 25)),
            n_workers=int(opt_cfg.get("n_workers", 1)),
            rng_seed=opt_cfg.get("rng_seed"),
            early_stopping_window=int(opt_cfg.get("early_stopping_window", 25)),
            early_stopping_threshold=float(opt_cfg.get("early_stopping_threshold", 0.001)),
            salary_floor=float(opt_cfg["salary_floor"]) if opt_cfg.get("salary_floor") is not None else None,
        )

        def _on_lineup_complete(lineup_index: int, total: int, score: float) -> None:
            self._cb("optimize_lineup", {
                "lineup_index": lineup_index,
                "total": total,
                "score": round(score, 4),
            })

        portfolio = constructor.construct(on_lineup_complete=_on_lineup_complete)
        logger.info("Portfolio complete: %d lineups.", len(portfolio))

        # --- Serialize --------------------------------------------------
        result = self._serialize_portfolio(portfolio, players_df)
        self._cb("complete", {"portfolio": result, "n_lineups": len(result)})

        # --- Save CSV ---------------------------------------------------
        output_dir = paths.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(portfolio, players_df)
        output_path = os.path.join(output_dir, "portfolio.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Portfolio saved to %s", output_path)

        return result

    # ------------------------------------------------------------------
    # Private helpers (mirrored from main.py)
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self._config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _derive_opponent(team: str, game: str) -> str:
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

    def _build_players_df(
        self,
        slate_df: pd.DataFrame,
        proj_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        df = slate_df.copy()
        df["opponent"] = df.apply(
            lambda r: self._derive_opponent(r["team"], r["game"]), axis=1
        )
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
        if proj_df is not None:
            proj = proj_df.copy().rename(columns={"mu": "mean", "sigma": "std_dev"})
            proj_cols = ["player_id", "mean", "std_dev"]
            if "lineup_slot" in proj.columns:
                proj_cols.append("lineup_slot")
            df = df.merge(proj[proj_cols], on="player_id", how="left")
            if "lineup_slot" in df.columns:
                before = len(df)
                df = df[df["lineup_slot"].notna()].copy()
                excluded = before - len(df)
                if excluded:
                    logger.info("Excluded %d non-starter DK players.", excluded)
                batter_with_slot = (
                    ~(df["position"] == "P")
                    & df["lineup_slot"].notna()
                    & df["lineup_slot"].between(1, 9)
                )
                df.loc[batter_with_slot, "slot"] = df.loc[batter_with_slot, "lineup_slot"].astype(int)
        else:
            logger.info("No projections — using salary-based heuristic.")
            df["mean"] = df["salary"] / 400.0
            df["std_dev"] = df["mean"] * 0.85
        return df[["player_id", "team", "opponent", "slot", "mean", "std_dev", "position", "salary", "game"]]

    @staticmethod
    def _compute_auto_target(
        players_df: pd.DataFrame,
        sim_results: SimulationResults,
        percentile: int,
    ) -> float:
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
                return float(np.percentile(totals, percentile))
        n = len(players_df)
        row_sums = sim_results.results_matrix.sum(axis=1)
        return float(np.percentile(row_sums * 10.0 / n, percentile))

    @staticmethod
    def _serialize_portfolio(
        portfolio: list,
        players_df: pd.DataFrame,
    ) -> list[dict]:
        id_to_name = dict(zip(players_df["player_id"], players_df.get("name", players_df["player_id"])))
        id_to_salary = dict(zip(players_df["player_id"], players_df["salary"]))
        id_to_pos = dict(zip(players_df["player_id"], players_df["position"]))
        id_to_team = dict(zip(players_df["player_id"], players_df["team"]))

        result = []
        for i, (lineup, score) in enumerate(portfolio, start=1):
            total_salary = sum(id_to_salary.get(pid, 0) for pid in lineup.player_ids)
            players = [
                {
                    "player_id": pid,
                    "name": str(id_to_name.get(pid, pid)),
                    "position": id_to_pos.get(pid, ""),
                    "team": id_to_team.get(pid, ""),
                    "salary": id_to_salary.get(pid, 0),
                }
                for pid in lineup.player_ids
            ]
            result.append({
                "lineup_index": i,
                "p_hit_target": round(score, 4),
                "lineup_salary": total_salary,
                "players": players,
            })
        return result

    @staticmethod
    def _format_portfolio_df(
        portfolio: list,
        players_df: pd.DataFrame,
    ) -> pd.DataFrame:
        id_to_name = dict(zip(players_df["player_id"], players_df.get("name", players_df["player_id"])))
        id_to_salary = dict(zip(players_df["player_id"], players_df["salary"]))
        id_to_pos = dict(zip(players_df["player_id"], players_df["position"]))
        id_to_team = dict(zip(players_df["player_id"], players_df["team"]))
        rows = []
        for i, (lineup, score) in enumerate(portfolio, start=1):
            total_salary = sum(id_to_salary.get(pid, 0) for pid in lineup.player_ids)
            for pid in lineup.player_ids:
                rows.append({
                    "lineup": i,
                    "p_hit_target": round(score, 4),
                    "player_id": pid,
                    "name": id_to_name.get(pid, str(pid)),
                    "position": id_to_pos.get(pid, ""),
                    "team": id_to_team.get(pid, ""),
                    "salary": id_to_salary.get(pid, 0),
                    "lineup_salary": total_salary,
                })
        return pd.DataFrame(rows)
