"""
PipelineRunner — wraps the main.py pipeline with progress callbacks.

Emits callback events at each pipeline stage so the API can forward them
as SSE events to the browser.
"""
import json
import logging
import os
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

from src.ingestion.dk_slate import DraftKingsSlateIngestor
from src.models.batter_model import BatterPCAModel
from src.models.copula import EmpiricalCopula
from src.optimization.lineup import ROSTER_REQUIREMENTS, SLOTS
from src.optimization.portfolio import PortfolioConstructor
from src.simulation.engine import SimulationEngine
from src.simulation.results import SimulationResults

logger = logging.getLogger(__name__)


def _extract_upload_tag(filename: str) -> str:
    """Extract the unique prefix from entry filenames.

    'GEDKSalaries.csv' → 'GE'
    """
    return re.sub(r'DKEntries\.csv$', '', filename, flags=re.IGNORECASE)


def _shorten_contest_name(name: str) -> str:
    """Produce a display-friendly contest name.

    'MLB $12K Base Hit [Single Entry]' → 'Base Hit'
    """
    name = re.sub(r'^MLB\s+', '', name)
    name = re.sub(r'^\$[\d.,]+[KkMm]?\s+', '', name)
    name = re.sub(r'\s*\[.*?\]\s*$', '', name)
    return name.strip()


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
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self._config_path = config_path
        self._cb = progress_cb or (lambda stage, data: None)
        self._stop_check = stop_check

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

        # --- Entry file discovery ----------------------------------------
        from src.api.dk_entries import parse_entry_file, scan_entry_files
        raw_dir = os.path.dirname(paths["dk_slate"])
        entry_files = scan_entry_files(raw_dir)
        all_file_entries = [(ef, parse_entry_file(ef)) for ef in entry_files]
        total_entries = sum(len(recs) for _, recs in all_file_entries)
        output_dir = paths.get("output_dir", "outputs")
        if total_entries > 0:
            logger.info(
                "Found %d entry file(s) with %d total entries.",
                len(entry_files), total_entries,
            )

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

        players_df, excl_stats = self._apply_exclusions(players_df)

        n_teams_loaded = players_df["team"].nunique()
        n_batters = int((players_df["position"] != "P").sum())
        n_pitchers = int((players_df["position"] == "P").sum())
        pitcher_counts = players_df[players_df["position"] == "P"].groupby("team").size()
        multi_pitcher_teams = {team: int(cnt) for team, cnt in pitcher_counts.items() if cnt > 1}
        self._cb("load_slate", {
            "n_teams": n_teams_loaded,
            "n_batters": n_batters,
            "n_pitchers": n_pitchers,
            "multi_pitcher_teams": multi_pitcher_teams,
            **excl_stats,
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

        # --- Resolve payout beta and cash line ----------------------------------
        objective = str(opt_cfg.get("objective", "expected_surplus"))
        payout_beta_cfg = opt_cfg.get("payout_beta")
        if objective == "marginal_payout":
            from src.optimization.payout import calibrate_beta, load_payout_structure, get_cash_line_score
            score_dist = self._best_lineup_score_distribution(players_df, sim_results)
            score_percentiles = np.percentile(score_dist, np.arange(1, 101))
            structure = load_payout_structure("dk_classic_gpp")
            payout_cash_line = get_cash_line_score(structure, score_percentiles)
            logger.info("Payout cash line score: %.1f DK pts", payout_cash_line)
            self._cb("calibrate_beta", {"payout_cash_line": round(float(payout_cash_line), 1)})
            if payout_beta_cfg is None:
                payout_beta = calibrate_beta(structure, score_percentiles, payout_cash_line)
                logger.info("Auto-calibrated payout beta: %.2f", payout_beta)
                self._cb("calibrate_beta", {"payout_beta": round(payout_beta, 2), "payout_cash_line": round(float(payout_cash_line), 1)})
            else:
                payout_beta = float(payout_beta_cfg)
        else:
            payout_cash_line = None
            payout_beta = float(payout_beta_cfg) if payout_beta_cfg is not None else 2.5

        # Store simulation artifacts for post-run operations (lineup replacement).
        self._sim_results = sim_results
        self._players_df = players_df
        self._target = target
        self._objective = objective
        self._optimizer_kwargs_replace = dict(
            n_chains=int(opt_cfg.get("n_chains", 250)),
            temperature=float(opt_cfg.get("temperature", 0.1)),
            n_steps=int(opt_cfg.get("n_steps", 100)),
            niter_success=int(opt_cfg.get("niter_success", 25)),
            n_workers=int(opt_cfg.get("n_workers", 1)),
            rng_seed=opt_cfg.get("rng_seed"),
            early_stopping_window=int(opt_cfg.get("early_stopping_window", 25)),
            early_stopping_threshold=float(opt_cfg.get("early_stopping_threshold", 0.001)),
            salary_floor=float(opt_cfg["salary_floor"]) if opt_cfg.get("salary_floor") is not None else None,
            objective=objective,
            payout_beta=payout_beta,
            payout_cash_line=payout_cash_line,
        )

        # --- Construct portfolio ----------------------------------------
        config_size = int(port_cfg.get("size", 20))
        portfolio_size = max(config_size, total_entries) if total_entries > 0 else config_size
        if total_entries > 0 and portfolio_size > config_size:
            logger.info(
                "Entry files require %d lineups; overriding config size %d.",
                total_entries, config_size,
            )
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
            objective=objective,
            payout_beta=payout_beta,
            payout_cash_line=payout_cash_line,
            n_seed_lineups=int(opt_cfg.get("n_seed_lineups", 5)),
        )

        def _on_lineup_complete(lineup_index: int, total: int, score: float, sims_covered: int, sims_remaining: int) -> None:
            self._cb("optimize_lineup", {
                "lineup_index": lineup_index,
                "total": total,
                "score": round(score, 4),
                "sims_covered": sims_covered,
                "sims_remaining": sims_remaining,
            })

        portfolio = constructor.construct(
            on_lineup_complete=_on_lineup_complete,
            stop_check=self._stop_check,
        )
        logger.info("Portfolio complete: %d lineups.", len(portfolio))

        # Store raw artifacts for on-demand upload file writing.
        self._raw_portfolio = portfolio
        self._all_file_entries = all_file_entries
        self._slate_df = slate_df
        self._output_dir = output_dir

        # --- Serialize --------------------------------------------------
        result = self._serialize_portfolio(portfolio, players_df)

        was_stopped = self._stop_check is not None and self._stop_check()

        # --- Save CSV ---------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(portfolio, players_df)
        output_path = os.path.join(output_dir, "portfolio.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Portfolio saved to %s", output_path)

        # --- Generate DK upload files (skipped when stopped) -------------
        if not was_stopped and all_file_entries:
            from src.api.dk_entries import assign_lineups_to_entries, write_upload_files
            assignments = assign_lineups_to_entries(all_file_entries, portfolio)
            paths_written = write_upload_files(
                all_file_entries, assignments, slate_df, output_dir
            )
            self._cb("upload_files", {"n_files": len(paths_written), "paths": paths_written})

            entry_map = self._build_lineup_entry_map(all_file_entries, portfolio)
            for lr in result:
                info = entry_map.get(lr["lineup_index"])
                if info:
                    lr.update(info)
            meta_path = os.path.join(output_dir, "portfolio_entries.json")
            with open(meta_path, "w") as f:
                json.dump({str(k): v for k, v in entry_map.items()}, f)

        # --- Notify (after entry augmentation so SSE payload is complete) -
        if was_stopped:
            logger.info("Run stopped by user after %d lineups.", len(portfolio))
            self._cb("stopped", {"portfolio": result, "n_lineups": len(result)})
        else:
            self._cb("complete", {"portfolio": result, "n_lineups": len(result)})

        return result

    def write_upload_files(self) -> list[str]:
        """Write DK upload CSVs using the portfolio from the last run.

        Returns list of file paths written. Raises RuntimeError if no run
        artifacts are available.
        """
        if not hasattr(self, "_raw_portfolio"):
            raise RuntimeError("No pipeline run artifacts available.")
        if not self._all_file_entries:
            return []
        from src.api.dk_entries import assign_lineups_to_entries, write_upload_files
        assignments = assign_lineups_to_entries(self._all_file_entries, self._raw_portfolio)
        return write_upload_files(
            self._all_file_entries, assignments, self._slate_df, self._output_dir
        )

    def replace_lineup(self, lineup_index: int) -> list[dict]:
        """Delete the lineup at lineup_index (1-based) and generate a replacement.

        The replacement is optimized on the simulation rows NOT consumed by the
        remaining lineups, then appended to the end of the portfolio. The updated
        portfolio.csv and any DK upload files are written automatically.

        Returns
        -------
        list[dict]
            Updated portfolio serialized in the same format as ``run()``.
        """
        from src.optimization.optimizer import BasinHoppingOptimizer

        idx = lineup_index - 1
        deleted_lineup, _ = self._raw_portfolio[idx]
        deleted_player_set = frozenset(deleted_lineup.player_ids)
        remaining = self._raw_portfolio[:idx] + self._raw_portfolio[idx + 1:]

        full_matrix = self._sim_results.results_matrix
        col_map = {pid: i for i, pid in enumerate(self._sim_results.player_ids)}

        # Request enough candidates to find one that differs from the deleted lineup.
        n_candidates = 5

        if self._objective == "marginal_payout":
            best_scores = np.zeros(self._sim_results.n_sims, dtype=np.float64)
            for lineup, _ in remaining:
                cols = [col_map[pid] for pid in lineup.player_ids]
                best_scores = np.maximum(best_scores, full_matrix[:, cols].sum(axis=1))
            optimizer = BasinHoppingOptimizer(
                sim_results=self._sim_results,
                players_df=self._players_df,
                target=self._target,
                best_scores=best_scores,
                **self._optimizer_kwargs_replace,
            )
        else:
            active_mask = np.ones(self._sim_results.n_sims, dtype=bool)
            for lineup, _ in remaining:
                cols = [col_map[pid] for pid in lineup.player_ids]
                active_mask[full_matrix[:, cols].sum(axis=1) >= self._target] = False
            active_sim = SimulationResults(
                player_ids=self._sim_results.player_ids,
                results_matrix=full_matrix[active_mask],
            )
            optimizer = BasinHoppingOptimizer(
                sim_results=active_sim,
                players_df=self._players_df,
                target=self._target,
                **self._optimizer_kwargs_replace,
            )

        candidates = optimizer.optimize_top_k(n_candidates)
        # Pick the best candidate that is not identical to the deleted lineup.
        new_lineup, _ = next(
            ((lu, sc) for lu, sc in candidates if frozenset(lu.player_ids) != deleted_player_set),
            candidates[0],  # fall back to best if all candidates match (very unlikely)
        )
        cols = [col_map[pid] for pid in new_lineup.player_ids]
        full_score = float((full_matrix[:, cols].sum(axis=1) >= self._target).mean())

        self._raw_portfolio = remaining + [(new_lineup, full_score)]
        result = self._serialize_portfolio(self._raw_portfolio, self._players_df)

        os.makedirs(self._output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(self._raw_portfolio, self._players_df)
        output_path = os.path.join(self._output_dir, "portfolio.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Updated portfolio saved to %s", output_path)

        if self._all_file_entries:
            self.write_upload_files()

            entry_map = self._build_lineup_entry_map(self._all_file_entries, self._raw_portfolio)
            for lr in result:
                info = entry_map.get(lr["lineup_index"])
                if info:
                    lr.update(info)
            meta_path = os.path.join(self._output_dir, "portfolio_entries.json")
            with open(meta_path, "w") as f:
                json.dump({str(k): v for k, v in entry_map.items()}, f)

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
            if "slot_confirmed" in proj.columns:
                proj_cols.append("slot_confirmed")
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
        base_cols = ["player_id", "team", "opponent", "slot", "mean", "std_dev", "position", "salary", "game"]
        if "eligible_positions" in df.columns:
            base_cols.append("eligible_positions")
        if "slot_confirmed" in df.columns:
            base_cols.append("slot_confirmed")
        return df[base_cols]

    @staticmethod
    def _apply_exclusions(players_df: pd.DataFrame) -> tuple:
        """Filter players_df based on persisted slate exclusions.

        Returns (filtered_df, excl_stats) where excl_stats has:
          n_teams_excluded, n_batters_ind_excluded, n_pitchers_ind_excluded
        """
        from .slate_exclusions import compute_slate_id, read_exclusions
        empty_stats: dict = {"n_teams_excluded": 0, "n_batters_ind_excluded": 0, "n_pitchers_ind_excluded": 0}
        stored = read_exclusions()
        if not stored.get("slate_id"):
            return players_df, empty_stats

        current_games = [g for g in players_df["game"].dropna().unique().tolist() if g]
        if compute_slate_id(current_games) != stored["slate_id"]:
            from .slate_exclusions import write_exclusions
            new_slate_id = compute_slate_id(current_games)
            write_exclusions(slate_id=new_slate_id, excluded_teams=[], excluded_games=[], excluded_player_ids=[])
            logger.info("DKSalaries.csv slate changed — exclusions reset.")
            return players_df, empty_stats

        excluded_games = set(stored.get("excluded_games", []))
        excluded_teams = set(stored.get("excluded_teams", []))
        excluded_player_ids = set(stored.get("excluded_player_ids", []))
        if not excluded_games and not excluded_teams and not excluded_player_ids:
            return players_df, empty_stats

        # Count individually excluded players by position (before team/game exclusions)
        ind_excl_df = players_df[players_df["player_id"].isin(excluded_player_ids)]
        n_batters_ind_excluded = int((ind_excl_df["position"] != "P").sum())
        n_pitchers_ind_excluded = int((ind_excl_df["position"] == "P").sum())

        pre_n_teams = players_df["team"].nunique()
        mask = (
            players_df["game"].isin(excluded_games) |
            players_df["team"].isin(excluded_teams) |
            players_df["player_id"].isin(excluded_player_ids)
        )
        filtered = players_df[~mask].copy()
        n_removed = len(players_df) - len(filtered)
        if n_removed:
            logger.info("Exclusions removed %d players (%d remain).", n_removed, len(filtered))

        n_teams_excluded = pre_n_teams - filtered["team"].nunique()
        excl_stats = {
            "n_teams_excluded": int(n_teams_excluded),
            "n_batters_ind_excluded": n_batters_ind_excluded,
            "n_pitchers_ind_excluded": n_pitchers_ind_excluded,
        }
        return filtered, excl_stats

    @staticmethod
    def _best_lineup_score_distribution(
        players_df: pd.DataFrame,
        sim_results: "SimulationResults",
    ) -> np.ndarray:
        """Compute the per-sim score totals for the top-mean lineup.

        Returns an (n_sims,) array of lineup totals — one per simulation.
        Used for both auto-target computation and payout beta calibration.
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
                return sim_results.results_matrix[:, cols].sum(axis=1)
        n = len(players_df)
        row_sums = sim_results.results_matrix.sum(axis=1)
        return row_sums * 10.0 / n

    @staticmethod
    def _compute_auto_target(
        players_df: pd.DataFrame,
        sim_results: "SimulationResults",
        percentile: int,
    ) -> float:
        totals = PipelineRunner._best_lineup_score_distribution(players_df, sim_results)
        return float(np.percentile(totals, percentile))

    @staticmethod
    def _pos_label(row: pd.Series) -> str:
        """Return DK-style position string: '3B/SS' for multi-eligible, '3B' for single."""
        ep = row.get("eligible_positions")
        if ep and isinstance(ep, list) and len(ep) > 1:
            return "/".join(ep)
        return str(row["position"])

    @staticmethod
    def _assigned_positions(lineup, player_meta: dict) -> dict:
        """Return {player_id: roster_position} for each player in the lineup.

        Uses the same bipartite matching as the optimizer so the assigned position
        reflects the actual slot the player fills (e.g. 'SS' for a 3B/SS player
        placed at shortstop). Falls back to primary position on failure.
        """
        from src.optimization.optimizer import _compute_slot_assignment
        try:
            _, pidx_to_slot = _compute_slot_assignment(lineup.player_ids, player_meta)
            return {pid: SLOTS[pidx_to_slot[j]] for j, pid in enumerate(lineup.player_ids)}
        except RuntimeError:
            return {pid: player_meta.get(pid, {}).get('position', '') for pid in lineup.player_ids}

    @staticmethod
    def _serialize_portfolio(
        portfolio: list,
        players_df: pd.DataFrame,
    ) -> list[dict]:
        from src.optimization.optimizer import _build_player_meta
        id_to_name = dict(zip(players_df["player_id"], players_df.get("name", players_df["player_id"])))
        id_to_salary = dict(zip(players_df["player_id"], players_df["salary"]))
        id_to_pos = {
            int(r["player_id"]): PipelineRunner._pos_label(r)
            for _, r in players_df.iterrows()
        }
        id_to_team = dict(zip(players_df["player_id"], players_df["team"]))
        id_to_slot = dict(zip(players_df["player_id"], players_df["slot"]))
        id_to_confirmed: dict = {}
        if "slot_confirmed" in players_df.columns:
            id_to_confirmed = dict(zip(players_df["player_id"], players_df["slot_confirmed"].astype(bool)))

        player_meta = _build_player_meta(players_df)

        result = []
        for i, (lineup, score) in enumerate(portfolio, start=1):
            pid_to_assigned = PipelineRunner._assigned_positions(lineup, player_meta)
            total_salary = sum(id_to_salary.get(pid, 0) for pid in lineup.player_ids)
            players = [
                {
                    "player_id": pid,
                    "name": str(id_to_name.get(pid, pid)),
                    "position": id_to_pos.get(pid, ""),
                    "assigned_position": pid_to_assigned.get(pid, id_to_pos.get(pid, "").split('/')[0]),
                    "team": id_to_team.get(pid, ""),
                    "salary": id_to_salary.get(pid, 0),
                    "slot": int(id_to_slot[pid]) if pid in id_to_slot else None,
                    "slot_confirmed": bool(id_to_confirmed[pid]) if pid in id_to_confirmed else False,
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
    def _build_lineup_entry_map(
        all_file_entries: list,
        portfolio: list,
    ) -> dict[int, dict]:
        """Return {lineup_index: {upload_tag, entry_fee, contest_name}} from entry assignments.

        Uses the same fee-descending assignment order as assign_lineups_to_entries.
        """
        flat: list = []
        for file_path, records in all_file_entries:
            for rec in records:
                flat.append((rec.entry_fee_cents, len(flat), file_path, rec))
        flat.sort(key=lambda x: x[0], reverse=True)
        entry_map: dict[int, dict] = {}
        for i, (_, _, file_path, rec) in enumerate(flat):
            if i >= len(portfolio):
                break
            entry_map[i + 1] = {
                "upload_tag": _extract_upload_tag(file_path.name),
                "entry_fee": rec.entry_fee_raw,
                "contest_name": _shorten_contest_name(rec.contest_name),
            }
        return entry_map

    @staticmethod
    def _format_portfolio_df(
        portfolio: list,
        players_df: pd.DataFrame,
    ) -> pd.DataFrame:
        from src.optimization.optimizer import _build_player_meta
        id_to_name = dict(zip(players_df["player_id"], players_df.get("name", players_df["player_id"])))
        id_to_salary = dict(zip(players_df["player_id"], players_df["salary"]))
        id_to_pos = {
            int(r["player_id"]): PipelineRunner._pos_label(r)
            for _, r in players_df.iterrows()
        }
        id_to_team = dict(zip(players_df["player_id"], players_df["team"]))
        id_to_slot = dict(zip(players_df["player_id"], players_df["slot"]))
        id_to_confirmed: dict = {}
        if "slot_confirmed" in players_df.columns:
            id_to_confirmed = dict(zip(players_df["player_id"], players_df["slot_confirmed"].astype(bool)))
        player_meta = _build_player_meta(players_df)
        rows = []
        for i, (lineup, score) in enumerate(portfolio, start=1):
            pid_to_assigned = PipelineRunner._assigned_positions(lineup, player_meta)
            total_salary = sum(id_to_salary.get(pid, 0) for pid in lineup.player_ids)
            for pid in lineup.player_ids:
                rows.append({
                    "lineup": i,
                    "p_hit_target": round(score, 4),
                    "player_id": pid,
                    "name": id_to_name.get(pid, str(pid)),
                    "position": id_to_pos.get(pid, ""),
                    "assigned_position": pid_to_assigned.get(pid, id_to_pos.get(pid, "").split('/')[0]),
                    "team": id_to_team.get(pid, ""),
                    "salary": id_to_salary.get(pid, 0),
                    "lineup_salary": total_salary,
                    "slot": int(id_to_slot[pid]) if pid in id_to_slot else None,
                    "slot_confirmed": bool(id_to_confirmed[pid]) if pid in id_to_confirmed else False,
                })
        return pd.DataFrame(rows)
