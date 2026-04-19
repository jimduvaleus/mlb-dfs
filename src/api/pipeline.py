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

from src.ingestion.factory import get_ingestor
from src.models.batter_model import BatterPCAModel
from src.models.copula import EmpiricalCopula
from src.optimization.lineup import ROSTER_REQUIREMENTS, SLOTS
from src.optimization.portfolio import PortfolioConstructor
from src.platforms.base import Platform
from src.platforms.registry import get_roster, get_slot_eligibility
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

        # --- Platform setup ---------------------------------------------
        platform = Platform(cfg.get("platform", "draftkings"))
        roster_rules = get_roster(platform)
        slot_elig = get_slot_eligibility(platform)
        if platform == Platform.DRAFTKINGS:
            slate_path = paths["dk_slate"]
        else:
            from pathlib import Path as _Path
            from src.ingestion.factory import find_fd_slate
            slate_path = paths.get("fd_slate", "")
            _root = _Path(self._config_path).resolve().parent
            _abs = (_root / slate_path) if slate_path else None
            if _abs is None or not _abs.exists():
                raw_dir = str(_abs.parent) if _abs else str(_root / "data" / "raw")
                discovered = find_fd_slate(raw_dir)
                if discovered:
                    try:
                        slate_path = str(_Path(discovered).relative_to(_root))
                    except ValueError:
                        slate_path = discovered
                    cfg["paths"]["fd_slate"] = slate_path
                    import yaml as _yaml
                    with open(self._config_path, "w") as _f:
                        _yaml.dump(cfg, _f, default_flow_style=False, sort_keys=False)

        # --- Entry file discovery ----------------------------------------
        from src.api.entries_factory import get_entry_handlers
        entry_handlers = get_entry_handlers(platform)
        raw_dir = os.path.dirname(slate_path) if slate_path else ""
        entry_files = entry_handlers["scan"](raw_dir) if raw_dir else []
        all_file_entries = [(ef, entry_handlers["parse"](ef)) for ef in entry_files]
        total_entries = sum(len(recs) for _, recs in all_file_entries)
        output_dir = paths.get("output_dir", "outputs")
        if total_entries > 0:
            logger.info(
                "Found %d entry file(s) with %d total entries.",
                len(entry_files), total_entries,
            )

        # --- Load slate --------------------------------------------------
        logger.info("Loading %s slate: %s", platform.value, slate_path)
        ingestor = get_ingestor(platform, slate_path)
        slate_df = ingestor.get_slate_dataframe()
        n_teams = slate_df["team"].nunique()
        logger.info("Slate loaded: %d players, %d teams", len(slate_df), n_teams)

        # --- Projections (optional) --------------------------------------
        proj_df = None
        if platform == Platform.FANDUEL:
            proj_path = paths.get("fd_projections") or "data/processed/projections_fd.csv"
        else:
            proj_path = paths.get("projections") or "data/processed/projections_dk.csv"
        if proj_path and os.path.exists(proj_path):
            logger.info("Loading projections: %s", proj_path)
            proj_df = pd.read_csv(proj_path)

        # --- Build players_df -------------------------------------------
        players_df = self._build_players_df(slate_df, proj_df)
        if "name" in slate_df.columns:
            players_df = players_df.merge(
                slate_df[["player_id", "name"]], on="player_id", how="left"
            )

        players_df, excl_stats = self._apply_exclusions(players_df, slate_path=slate_path)

        # --- Value cutoff filtering ------------------------------------
        min_p_val = opt_cfg.get("min_pitcher_value")
        min_b_val = opt_cfg.get("min_batter_value")
        n_pitchers_value_excluded = 0
        n_batters_value_excluded = 0

        if min_p_val or min_b_val:
            players_df["_value"] = players_df["mean"] / (players_df["salary"] / 1000.0)
            if min_p_val:
                mask_p = (players_df["position"] == "P") & (players_df["_value"] < min_p_val)
                n_pitchers_value_excluded = int(mask_p.sum())
                players_df = players_df[~mask_p]
            if min_b_val:
                mask_b = (players_df["position"] != "P") & (players_df["_value"] < min_b_val)
                n_batters_value_excluded = int(mask_b.sum())
                players_df = players_df[~mask_b]
            players_df = players_df.drop(columns=["_value"])

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
            "n_pitchers_value_excluded": n_pitchers_value_excluded,
            "n_batters_value_excluded": n_batters_value_excluded,
            **excl_stats,
        })

        # --- Load copula -------------------------------------------------
        copula_path = paths["copula"]
        logger.info("Loading empirical copula: %s", copula_path)
        copula = EmpiricalCopula(copula_path)

        # --- Load batter PCA model (optional) ----------------------------
        # Select platform-specific model and score grid paths.
        # FanDuel uses artifacts built from FD-scored historical data
        # (historical_logs_fd.parquet → fit_batter_pca.py --platform fanduel).
        pca_model: Optional[BatterPCAModel] = None
        score_grid: Optional[np.ndarray] = None
        if platform == Platform.FANDUEL:
            pca_path = (paths.get("batter_pca_model_fd")
                        or "data/processed/batter_pca_model_fd.npz")
            grid_path = (paths.get("batter_score_grid_fd")
                         or "data/processed/batter_score_grid_fd.npy")
        else:
            pca_path = paths.get("batter_pca_model") or "data/processed/batter_pca_model.npz"
            grid_path = (paths.get("batter_score_grid")
                         or "data/processed/batter_score_grid.npy")
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
            target = self._compute_auto_target(
                players_df, sim_results, target_percentile,
                rules=roster_rules, slot_eligibility=slot_elig,
            )
            logger.info(
                "Auto-computed target: %.1f pts (p%d)", target, target_percentile
            )
            self._cb("compute_target", {"target": target, "percentile": target_percentile})
        else:
            target = float(target)
            target_percentile = None
            logger.info("Using configured target: %.1f DK pts", target)
            self._cb("compute_target", {"target": target, "percentile": None})

        # --- Resolve payout beta and cash line ----------------------------------
        objective = str(opt_cfg.get("objective", "expected_surplus"))
        payout_beta_cfg = opt_cfg.get("payout_beta")
        _BETA_MAX = 4.0
        score_dist = self._best_lineup_score_distribution(
            players_df, sim_results, rules=roster_rules, slot_eligibility=slot_elig
        )
        fixed_ref_p90 = float(np.percentile(score_dist, 90))
        fixed_ref_p99 = float(np.percentile(score_dist, 99))
        if objective == "marginal_payout":
            from src.optimization.payout import load_payout_structure, get_cash_line_score
            score_percentiles = np.percentile(score_dist, np.arange(1, 101))
            structure = load_payout_structure("dk_classic_gpp")
            payout_cash_line = get_cash_line_score(structure, score_percentiles)
            logger.info("Payout cash line score: %.1f DK pts", payout_cash_line)
            raw_beta = float(payout_beta_cfg) if payout_beta_cfg is not None else 2.5
            payout_beta = min(raw_beta, _BETA_MAX)
            if payout_beta != raw_beta:
                logger.info("Payout beta capped from %.2f to %.2f", raw_beta, _BETA_MAX)
            logger.info("Payout beta: %.2f", payout_beta)
            self._cb("calibrate_beta", {"payout_beta": round(payout_beta, 2), "payout_cash_line": round(float(payout_cash_line), 1)})
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
            rules=roster_rules,
            slot_eligibility=slot_elig,
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
            payout_coverage_bonus=float(opt_cfg.get("payout_coverage_bonus", 0.0)),
            n_seed_lineups=int(opt_cfg.get("n_seed_lineups", 5)),
            ref_p90=fixed_ref_p90,
            ref_p99=fixed_ref_p99,
            rules=roster_rules,
            slot_eligibility=slot_elig,
        )

        def _on_lineup_complete(
            lineup_index: int, total: int, score: float,
            arg4: int, arg5: int, arg6: Optional[float] = None,
            arg7: Optional[float] = None, arg8: Optional[float] = None,
            arg9: Optional[float] = None,
        ) -> None:
            if objective == "marginal_payout":
                self._cb("optimize_lineup", {
                    "lineup_index": lineup_index,
                    "total": total,
                    "score": round(score, 4),
                    "sims_great": arg4,
                    "sims_good": arg5,
                    "sims_uncovered": arg6,
                    "pct_above_p90": round(arg7, 1) if arg7 is not None else None,
                    "pct_above_p99": round(arg8, 1) if arg8 is not None else None,
                    "pct_above_target": round(arg9, 1) if arg9 is not None else None,
                    "target_percentile": target_percentile,
                    "objective": objective,
                })
            else:
                self._cb("optimize_lineup", {
                    "lineup_index": lineup_index,
                    "total": total,
                    "score": round(score, 4),
                    "sims_covered": arg4,
                    "sims_remaining": arg5,
                    "pct_above_p90": round(arg6, 1) if arg6 is not None else None,
                    "pct_above_p99": round(arg7, 1) if arg7 is not None else None,
                    "pct_above_target": round(arg8, 1) if arg8 is not None else None,
                    "target_percentile": target_percentile,
                    "objective": objective,
                })

        def _on_portfolio_complete(best_scores: np.ndarray) -> None:
            n_sims = len(best_scores)
            covered_mask = best_scores >= target
            covered = best_scores[covered_mask]

            # Build histogram across the full best_scores range (~40 buckets).
            score_min = float(best_scores.min())
            score_max = float(best_scores.max())
            n_buckets = 40
            bucket_size = (score_max - score_min) / n_buckets if score_max > score_min else 1.0
            histogram = []
            for k in range(n_buckets):
                lo = score_min + k * bucket_size
                hi = lo + bucket_size
                count = int(((best_scores >= lo) & (best_scores < hi)).sum())
                histogram.append({
                    "lo": round(lo, 1),
                    "hi": round(hi, 1),
                    "mid": round((lo + hi) / 2, 1),
                    "count": count,
                })

            def _pct(arr: np.ndarray, q: float) -> Optional[float]:
                return round(float(np.percentile(arr, q)), 1) if len(arr) > 0 else None

            self._cb("portfolio_stats", {
                "target": round(target, 1),
                "great_threshold": round(target + 15.0, 1),
                "n_sims": n_sims,
                "covered_count": int(covered_mask.sum()),
                "covered_mean": round(float(covered.mean()), 1) if len(covered) > 0 else None,
                "covered_p50": _pct(covered, 50),
                "covered_p90": _pct(covered, 90),
                "covered_p95": _pct(covered, 95),
                "covered_p99": _pct(covered, 99),
                "overall_p90": round(float(np.percentile(best_scores, 90)), 1),
                "overall_p95": round(float(np.percentile(best_scores, 95)), 1),
                "overall_p99": round(float(np.percentile(best_scores, 99)), 1),
                "histogram": histogram,
            })

        portfolio = constructor.construct(
            on_lineup_complete=_on_lineup_complete,
            on_portfolio_complete=_on_portfolio_complete if objective == "marginal_payout" else None,
            stop_check=self._stop_check,
            target_percentile=target_percentile if objective == "marginal_payout" else None,
        )
        logger.info("Portfolio complete: %d lineups.", len(portfolio))

        # Store raw artifacts for on-demand upload file writing.
        self._raw_portfolio = portfolio
        self._all_file_entries = all_file_entries
        self._entry_handlers = entry_handlers
        self._slate_df = slate_df
        self._output_dir = output_dir
        self._platform = platform

        # --- Serialize --------------------------------------------------
        result = self._serialize_portfolio(portfolio, players_df)

        was_stopped = self._stop_check is not None and self._stop_check()

        # --- Save CSV ---------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(portfolio, players_df)
        output_path = os.path.join(output_dir, f"portfolio_{platform.value}.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Portfolio saved to %s", output_path)

        # --- Generate upload files (skipped when stopped) ----------------
        if not was_stopped and all_file_entries:
            assignments = entry_handlers["assign"](all_file_entries, portfolio)
            paths_written = entry_handlers["write"](
                all_file_entries, assignments, slate_df, output_dir
            )
            self._cb("upload_files", {"n_files": len(paths_written), "paths": paths_written})

            entry_map = self._build_lineup_entry_map(all_file_entries, portfolio)
            for lr in result:
                info = entry_map.get(lr["lineup_index"])
                if info:
                    lr.update(info)
            meta_path = os.path.join(output_dir, f"portfolio_entries_{platform.value}.json")
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
        """Write upload CSVs using the portfolio from the last run.

        Returns list of file paths written. Raises RuntimeError if no run
        artifacts are available.
        """
        if not hasattr(self, "_raw_portfolio"):
            raise RuntimeError("No pipeline run artifacts available.")
        if not self._all_file_entries:
            return []
        assignments = self._entry_handlers["assign"](self._all_file_entries, self._raw_portfolio)
        return self._entry_handlers["write"](
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
        platform_val = self._platform.value if hasattr(self, "_platform") else "draftkings"
        output_path = os.path.join(self._output_dir, f"portfolio_{platform_val}.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Updated portfolio saved to %s", output_path)

        if self._all_file_entries:
            self.write_upload_files()

            entry_map = self._build_lineup_entry_map(self._all_file_entries, self._raw_portfolio)
            for lr in result:
                info = entry_map.get(lr["lineup_index"])
                if info:
                    lr.update(info)
            meta_path = os.path.join(self._output_dir, f"portfolio_entries_{platform_val}.json")
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
    def _apply_exclusions(players_df: pd.DataFrame, slate_path: str = "") -> tuple:
        """Filter players_df based on persisted slate exclusions.

        Returns (filtered_df, excl_stats) where excl_stats has:
          n_teams_excluded, n_batters_ind_excluded, n_pitchers_ind_excluded
        """
        from pathlib import Path as _Path
        from .slate_exclusions import compute_file_fingerprint, compute_slate_id, read_exclusions
        empty_stats: dict = {"n_teams_excluded": 0, "n_batters_ind_excluded": 0, "n_pitchers_ind_excluded": 0}

        current_games = [g for g in players_df["game"].dropna().unique().tolist() if g]
        if not current_games:
            return players_df, empty_stats

        slate_file = _Path(slate_path) if slate_path else None
        fingerprint = compute_file_fingerprint(slate_file)
        slate_id = compute_slate_id(current_games)
        stored = read_exclusions(slate_id, fingerprint)

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
        rules=None,
        slot_eligibility: Optional[dict] = None,
    ) -> np.ndarray:
        """Compute the per-sim score totals for the top-mean lineup.

        Returns an (n_sims,) array of lineup totals — one per simulation.
        Used for both auto-target computation and payout beta calibration.

        When ``rules`` is provided the reference lineup is built by greedily
        filling each roster slot (in order) with the highest-mean eligible
        player not yet selected.  This correctly handles platform-specific
        roster sizes and compound slots (e.g. FD's C/1B and UTIL).
        When ``rules`` is None the legacy DK-hardcoded logic is used.
        """
        col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
        sorted_df = players_df.sort_values("mean", ascending=False)

        if rules is not None:
            pid_list = sorted_df["player_id"].tolist()
            pos_map = dict(zip(sorted_df["player_id"], sorted_df["position"]))
            _slot_elig = slot_eligibility or {}
            used: set[int] = set()
            selected: list[int] = []
            for slot in rules.slots:
                eligible_positions = _slot_elig.get(slot, {slot})
                for pid in pid_list:
                    if pid in used:
                        continue
                    if pos_map.get(pid, "") in eligible_positions:
                        selected.append(pid)
                        used.add(pid)
                        break
            roster_size = rules.roster_size
        else:
            counts: dict[str, int] = {pos: 0 for pos in ROSTER_REQUIREMENTS}
            selected = []
            for _, row in sorted_df.iterrows():
                pos = str(row["position"])
                if pos in counts and counts[pos] < ROSTER_REQUIREMENTS[pos]:
                    selected.append(int(row["player_id"]))
                    counts[pos] += 1
                if len(selected) == 10:
                    break
            roster_size = 10

        if len(selected) == roster_size:
            cols = [col_map[pid] for pid in selected if pid in col_map]
            if len(cols) == roster_size:
                return sim_results.results_matrix[:, cols].sum(axis=1)
        n = len(players_df)
        row_sums = sim_results.results_matrix.sum(axis=1)
        return row_sums * float(roster_size) / n

    @staticmethod
    def _compute_auto_target(
        players_df: pd.DataFrame,
        sim_results: "SimulationResults",
        percentile: int,
        rules=None,
        slot_eligibility: Optional[dict] = None,
    ) -> float:
        totals = PipelineRunner._best_lineup_score_distribution(
            players_df, sim_results, rules=rules, slot_eligibility=slot_eligibility
        )
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

        Players are sorted by number of eligible positions ascending (most
        constrained first) so that single-position players always claim their
        natural slot before multi-eligible players fill in around them.  This
        produces a canonical assignment for display — e.g. a pure 2B player
        always gets the 2B slot even when a 2B/3B teammate is also in the lineup.
        """
        from src.optimization.optimizer import _compute_slot_assignment
        try:
            ids = sorted(
                lineup.player_ids,
                key=lambda pid: len(player_meta.get(pid, {}).get('eligible_positions') or ['']),
            )
            _, pidx_to_slot = _compute_slot_assignment(ids, player_meta)
            return {pid: SLOTS[pidx_to_slot[j]] for j, pid in enumerate(ids)}
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
        id_to_mean = dict(zip(players_df["player_id"], players_df["mean"])) if "mean" in players_df.columns else {}
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
                    "mean": float(id_to_mean[pid]) if pid in id_to_mean else None,
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
        id_to_mean = dict(zip(players_df["player_id"], players_df["mean"])) if "mean" in players_df.columns else {}
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
                    "mean": float(id_to_mean[pid]) if pid in id_to_mean else None,
                    "lineup_salary": total_salary,
                    "slot": int(id_to_slot[pid]) if pid in id_to_slot else None,
                    "slot_confirmed": bool(id_to_confirmed[pid]) if pid in id_to_confirmed else False,
                })
        return pd.DataFrame(rows)
