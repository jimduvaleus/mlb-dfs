"""
PipelineRunner — wraps the main.py pipeline with progress callbacks.

Emits callback events at each pipeline stage so the API can forward them
as SSE events to the browser.
"""
import json
import logging
import os
import re
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

from src.api.dk_entries import _sort_ratio
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
        use_cached_candidates: bool = False,
        use_cached_field: bool = False,
    ):
        self._config_path = config_path
        self._cb = progress_cb or (lambda stage, data: None)
        self._stop_check = stop_check
        self._use_cached_candidates = use_cached_candidates
        self._use_cached_field = use_cached_field

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

        # Clear any stale sweep cache so the UI shows empty while the run is in progress.
        _sweep_cache_clear = os.path.join(output_dir, f"portfolio_sweep_{platform.value}.json")
        try:
            os.remove(_sweep_cache_clear)
        except FileNotFoundError:
            pass

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

        players_df = self._apply_twitter_overrides(players_df, slate_path=slate_path)
        # sim_players_df: "both"-excluded removed — used for simulation + field generation
        # cand_players_df: "candidates"+"both"-excluded removed — used for lineup optimization
        sim_players_df, cand_players_df, excl_stats, game_ppd_pcts = self._apply_exclusions(
            players_df, slate_path=slate_path
        )

        # --- Value cutoff filtering (candidates only) ----------------------
        # sim_players_df is intentionally NOT value-cutoff filtered so that
        # candidate-excluded players retain scoring coverage for field lineups.
        min_p_val = opt_cfg.get("min_pitcher_value")
        min_b_val = opt_cfg.get("min_batter_value")
        n_pitchers_value_excluded = 0
        n_batters_value_excluded = 0

        if min_p_val or min_b_val:
            cand_players_df["_value"] = cand_players_df["mean"] / (cand_players_df["salary"] / 1000.0)
            if min_p_val:
                mask_p = (cand_players_df["position"] == "P") & (cand_players_df["_value"] < min_p_val)
                n_pitchers_value_excluded = int(mask_p.sum())
                cand_players_df = cand_players_df[~mask_p]
            if min_b_val:
                mask_b = (cand_players_df["position"] != "P") & (cand_players_df["_value"] < min_b_val)
                n_batters_value_excluded = int(mask_b.sum())
                cand_players_df = cand_players_df[~mask_b]
            cand_players_df = cand_players_df.drop(columns=["_value"])

        # Stats reflect the candidate pool (what the user is optimizing from)
        n_teams_loaded = cand_players_df["team"].nunique()
        n_batters = int((cand_players_df["position"] != "P").sum())
        n_pitchers = int((cand_players_df["position"] == "P").sum())
        pitcher_counts = cand_players_df[cand_players_df["position"] == "P"].groupby("team").size()
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

        # --- Market-implied quantile grids (optional) ---------------------
        # Written by the market-odds fetcher next to the projections CSV;
        # validated per player against the projected mean so fallback-sourced
        # or stale entries are skipped.
        from src.models.quantile_grids import DIST_FILENAME, load_quantile_grids
        quantile_grids = load_quantile_grids(
            os.path.join(os.path.dirname(proj_path or ""), DIST_FILENAME),
            sim_players_df,
        )
        if quantile_grids:
            logger.info(
                "Market-implied score distributions loaded for %d player(s).",
                len(quantile_grids),
            )

        # --- Simulate ---------------------------------------------------
        n_sims = int(sim_cfg.get("n_sims", 10_000))
        logger.info("Running %d simulations...", n_sims)
        self._cb("simulate", {"n_sims": n_sims, "objective": str(opt_cfg.get("objective", "expected_surplus"))})
        engine = SimulationEngine(
            copula, sim_players_df, batter_pca_model=pca_model, score_grid=score_grid,
            quantile_grids=quantile_grids,
        )
        sim_results = engine.simulate(n_sims)
        logger.info("Simulation complete — matrix: %s", sim_results.results_matrix.shape)

        # --- PPD zeroing ------------------------------------------------
        if game_ppd_pcts:
            # Same seeding convention as field generation: rng_seed when set,
            # else 42 — PPD row selection is deterministic by default.
            sim_results, ppd_stats = self._apply_ppd_to_simulation(
                sim_results, sim_players_df, game_ppd_pcts,
                rng_seed=int(opt_cfg.get("rng_seed") or 42),
            )
            self._cb("ppd_applied", {
                "games": [
                    {"game": g, "ppd_pct": s["ppd_pct"], "n_sims_zeroed": s["n_sims_zeroed"]}
                    for g, s in ppd_stats.items()
                ],
                "n_sims_total": n_sims,
            })

        # --- Target score -----------------------------------------------
        target = port_cfg.get("target_score")
        if target is None:
            target_percentile = int(port_cfg.get("target_percentile", 90))
            target = self._compute_auto_target(
                cand_players_df, sim_results, target_percentile,
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
            cand_players_df, sim_results, rules=roster_rules, slot_eligibility=slot_elig
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

        # --- Compute ownership vector for leverage_surplus objective ----------
        ownership_vector: Optional[np.ndarray] = None
        if objective == "leverage_surplus":
            from src.optimization.ownership import (
                apply_ownership_calibration,
                compute_heuristic_ownership,
                load_ownership_calibrator,
            )
            from src.api.slate_exclusions import (
                compute_slate_id as _compute_slate_id,
                compute_file_fingerprint as _compute_fp,
                read_exclusions as _read_exclusions,
            )
            team_totals = self._load_team_totals(slate_path)
            hr_odds = self._load_hr_fair_odds(slate_path)
            if hr_odds:
                import unicodedata as _ud, re as _re
                def _norm(n: str) -> str:
                    nfkd = _ud.normalize("NFKD", n)
                    return _re.sub(r"[^a-z ]", "", nfkd.encode("ascii", "ignore").decode("ascii").lower()).strip()
                cand_players_df = cand_players_df.copy()
                name_col = "name" if "name" in cand_players_df.columns else None
                cand_players_df["hr_prob"] = (
                    cand_players_df[name_col].apply(lambda n: hr_odds.get(_norm(str(n))))
                    if name_col else np.nan
                )
            # Load per-team ownership reductions from persisted exclusions.
            _team_ownership_reductions: dict = {}
            try:
                _slate_games = [
                    str(g) for g in cand_players_df["game"].dropna().unique()
                    if g
                ]
                if _slate_games and slate_path:
                    from pathlib import Path as _Path
                    _sid = _compute_slate_id(_slate_games)
                    _fp = _compute_fp(_Path(slate_path) if slate_path else None)
                    _team_ownership_reductions = (
                        _read_exclusions(_sid, _fp).get("team_ownership_reductions", {}) or {}
                    )
            except Exception:
                pass
            ownership_vector = compute_heuristic_ownership(
                cand_players_df, team_totals,
                team_ownership_reductions=_team_ownership_reductions or None,
            )
            # Post-hoc isotonic calibration (data/processed/ownership_calibrator.json,
            # built by scripts/fit_ownership_calibrator.py).  Loader returns None when
            # the artifact is missing or fitted under different model constants.
            _calibrator = load_ownership_calibrator()
            if _calibrator is not None:
                ownership_vector = apply_ownership_calibration(
                    ownership_vector, cand_players_df["position"].values, _calibrator
                )
            logger.info(
                "Computed heuristic ownership — model %s%s%s%s, %d players",
                "D" if team_totals else "C",
                "+HR" if hr_odds else "",
                f"+RED({len(_team_ownership_reductions)})" if _team_ownership_reductions else "",
                f"+CAL({_calibrator.get('n_slates')})" if _calibrator is not None else "",
                len(ownership_vector),
            )

        # Store simulation artifacts for post-run operations (lineup replacement).
        self._sim_results = sim_results
        self._players_df = cand_players_df
        self._target = target
        self._objective = objective
        self._ownership_vector = ownership_vector
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
            ownership_vector=ownership_vector,
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

        def _on_portfolio_complete(best_scores: np.ndarray) -> None:
            n_sims_ps = len(best_scores)
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
                "n_sims": n_sims_ps,
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

        _portfolio_sweep_raw: list = []  # populated only by det_ev; safe default for other objectives

        if objective == "leverage_surplus":
            # ----------------------------------------------------------------
            # GPP portfolio: rapid candidate pool + contest simulation +
            # marginal-EV greedy selection.
            # ----------------------------------------------------------------
            from src.optimization.candidate_generator import CandidateGenerator
            from src.optimization.gpp_portfolio import ContestScorer, EVPortfolioSelector

            def _candidate_team_distribution(cands, players_df):
                """Return {team: count} of unambiguous 4/5-hitter primary stacks."""
                pid_team = dict(zip(players_df["player_id"].astype(int), players_df["team"]))
                pid_pos = dict(zip(players_df["player_id"].astype(int), players_df["position"]))
                dist: dict[str, int] = {}
                for lu in cands:
                    th: dict[str, int] = {}
                    for pid in lu.player_ids:
                        if pid_pos.get(int(pid), "") != "P":
                            t = pid_team.get(int(pid), "")
                            if t:
                                th[t] = th.get(t, 0) + 1
                    if not th:
                        continue
                    top = max(th.values())
                    top_teams = [t for t, c in th.items() if c == top]
                    if top >= 4 and len(top_teams) == 1:
                        dist[top_teams[0]] = dist.get(top_teams[0], 0) + 1
                return dist

            gpp_cfg = cfg.get("gpp", {})
            n_candidates = int(gpp_cfg.get("n_candidates", 10_000))
            n_field = int(gpp_cfg.get("n_field_lineups", 5_000))
            n_k = int(gpp_cfg.get("n_field_samples", 3))
            cand_batch = int(gpp_cfg.get("candidate_batch_size", 500))
            max_attempts_mult = int(gpp_cfg.get("max_attempts_multiplier", 50))
            candidate_floor_relief = int(gpp_cfg.get("candidate_floor_relief", 2500))
            salary_floor_gpp = (
                float(opt_cfg["salary_floor"]) if opt_cfg.get("salary_floor") is not None else None
            )

            logger.info(
                "GPP portfolio — %d candidates, N=%d field lineups × K=%d samples, "
                "size=%d",
                n_candidates, n_field, n_k, portfolio_size,
            )

            _gpp_stopped = self._stop_check is not None and self._stop_check
            self._optimal_lineups: list = []  # populated only on fresh (non-cached) runs

            # Compute slate fingerprint for lineup cache (mtime_ns:size)
            from pathlib import Path as _Path
            from src.api.slate_exclusions import compute_file_fingerprint
            from src.api.lineup_cache import (
                load_candidates, save_candidates,
                load_field, save_field,
            )
            _config_root = _Path(self._config_path).resolve().parent
            _slate_fp = compute_file_fingerprint(_config_root / slate_path)

            # Phase 1: generate stacked candidate pool (with optional cache)
            gen = CandidateGenerator(
                cand_players_df, ownership_vector,
                rng_seed=opt_cfg.get("rng_seed"),
                salary_floor=salary_floor_gpp,
            )

            _t_gpp_start = time.perf_counter()
            _preloaded_cands = (
                load_candidates(_slate_fp) if (self._use_cached_candidates and _slate_fp) else None
            )
            if _preloaded_cands is not None:
                # Validate cached candidates against the current eligible player pool.
                # If a player's value has shifted across the cutoff, or exclusions have
                # changed, cached lineups referencing now-ineligible players must be
                # dropped before entering the scoring pipeline.
                _eligible_pids = set(int(p) for p in cand_players_df["player_id"])
                _n_raw = len(_preloaded_cands)
                _preloaded_cands = [
                    lu for lu in _preloaded_cands
                    if all(int(pid) in _eligible_pids for pid in lu.player_ids)
                ]
                _n_dropped = _n_raw - len(_preloaded_cands)
                if _n_dropped:
                    logger.warning(
                        "Dropped %d cached candidate(s) referencing players not in "
                        "current eligible pool (value cutoff or exclusion change?).",
                        _n_dropped,
                    )
                    self._cb("gpp_cache_filtered", {
                        "n_dropped": _n_dropped, "n_remaining": len(_preloaded_cands),
                    })

                n_cached = len(_preloaded_cands)
                if n_cached >= n_candidates:
                    candidates = _preloaded_cands[:n_candidates]
                    logger.info("Using %d candidates from cache.", len(candidates))
                    self._cb("gpp_generate_done", {
                        "n_generated": len(candidates), "from_cache": True,
                        "team_distribution": _candidate_team_distribution(candidates, cand_players_df),
                    })
                else:
                    need = n_candidates - n_cached
                    logger.info(
                        "Partial cache: %d candidates cached, generating %d more.",
                        n_cached, need,
                    )
                    self._cb("gpp_generate_start", {
                        "n_candidates": need, "n_from_cache": n_cached,
                    })
                    new_cands = gen.generate(
                        n_candidates=need,
                        max_attempts_multiplier=max_attempts_mult,
                        floor_relief=candidate_floor_relief,
                        stop_check=self._stop_check,
                        progress_cb=lambda n: self._cb("gpp_generate_progress", {"n": n}),
                    )
                    candidates = _preloaded_cands + new_cands
                    self._cb("gpp_generate_done", {
                        "n_generated": len(candidates),
                        "team_distribution": _candidate_team_distribution(candidates, cand_players_df),
                    })
                    if _slate_fp and not (_gpp_stopped and self._stop_check()):
                        save_candidates(_slate_fp, candidates)

                # Restore optimal lineups from persisted JSON if fingerprints match.
                try:
                    _opt_json = os.path.join(output_dir, f"optimal_lineups_{platform.value}.json")
                    if os.path.exists(_opt_json):
                        with open(_opt_json) as _f:
                            _opt_data = json.load(_f)
                        if _opt_data.get("slate_fingerprint") == _slate_fp and _opt_data.get("lineups"):
                            from src.optimization.lineup import Lineup as _Lineup
                            self._optimal_lineups = [
                                _Lineup(player_ids=[p["player_id"] for p in lr["players"]])
                                for lr in _opt_data["lineups"]
                            ]
                            logger.info(
                                "Restored %d optimal lineups from cache.", len(self._optimal_lineups)
                            )
                except Exception:
                    pass
            else:
                # Optionally seed the first N candidates with ILP-optimal lineups.
                seed_optimal = gpp_cfg.get("seed_optimal_lineups", False)
                _optimal_lineups: list = []

                if seed_optimal and not (_gpp_stopped and self._stop_check()):
                    from src.optimization.optimal_lineups import generate_optimal_lineups

                    batter_teams = sorted(
                        cand_players_df[cand_players_df["position"] != "P"]["team"].unique().tolist()
                    )
                    _batches_per_stack = {5: 35, 4: 25}
                    _stack_sizes = [5, 4]
                    _n_optimal_target = len(batter_teams) * sum(_batches_per_stack.values())

                    self._cb("gpp_optimal_start", {"n_optimal": _n_optimal_target})

                    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

                    _all_batches: dict[tuple, list] = {}
                    _running_count = 0

                    # Phase 1: 5-stack batches (all teams in parallel).
                    with ThreadPoolExecutor() as _executor:
                        _futures = {
                            _executor.submit(
                                generate_optimal_lineups,
                                cand_players_df,
                                n=_batches_per_stack[5],
                                min_uniques=3,
                                min_stack=5,
                                stack_team=tm,
                                salary_floor=salary_floor_gpp,
                            ): (5, tm)
                            for tm in batter_teams
                        }
                        for _fut in _as_completed(_futures):
                            _ss, _tm = _futures[_fut]
                            _all_batches[(_ss, _tm)] = _fut.result()
                            _running_count += len(_all_batches[(_ss, _tm)])
                            self._cb("gpp_optimal_progress", {"n": _running_count, "total": _n_optimal_target})
                            if self._stop_check():
                                for _f in _futures:
                                    _f.cancel()
                                break

                    # Phase 2: 4-stack batches, each constrained to >=3 uniques vs
                    # the 5-stack lineups for the same team.
                    if not self._stop_check():
                        with ThreadPoolExecutor() as _executor:
                            _futures = {
                                _executor.submit(
                                    generate_optimal_lineups,
                                    cand_players_df,
                                    n=_batches_per_stack[4],
                                    min_uniques=3,
                                    min_stack=4,
                                    stack_team=tm,
                                    salary_floor=salary_floor_gpp,
                                    prior_lineups=_all_batches.get((5, tm), []),
                                    min_uniques_vs_prior=3,
                                ): (4, tm)
                                for tm in batter_teams
                            }
                            for _fut in _as_completed(_futures):
                                _ss, _tm = _futures[_fut]
                                _all_batches[(_ss, _tm)] = _fut.result()
                                _running_count += len(_all_batches[(_ss, _tm)])
                                self._cb("gpp_optimal_progress", {"n": _running_count, "total": _n_optimal_target})
                                if self._stop_check():
                                    for _f in _futures:
                                        _f.cancel()
                                    break

                    # Post-hoc collision detection (only 4-stack batches can collide).
                    _seen_keys: dict = {}
                    _collisions: list = []
                    for (_ss, _tm), _lineups in _all_batches.items():
                        for _lu in _lineups:
                            _key = frozenset(_lu.player_ids)
                            if _key in _seen_keys:
                                _collisions.append((_key, _seen_keys[_key], (_ss, _tm)))
                            else:
                                _seen_keys[_key] = (_ss, _tm)

                    # Re-solve to replace colliding lineups in the second owner's batch.
                    for _key, _first, (_ss2, _tm2) in _collisions:
                        _replacements = generate_optimal_lineups(
                            cand_players_df,
                            n=1,
                            min_uniques=3,
                            min_stack=_ss2,
                            stack_team=_tm2,
                            salary_floor=salary_floor_gpp,
                            seen={_key},
                            prior_lineups=_all_batches.get((5, _tm2), []) if _ss2 == 4 else None,
                            min_uniques_vs_prior=3 if _ss2 == 4 else None,
                        )
                        _all_batches[(_ss2, _tm2)] = [
                            _lu for _lu in _all_batches[(_ss2, _tm2)]
                            if frozenset(_lu.player_ids) != _key
                        ] + _replacements

                    _all_optimal: list = [_lu for _lus in _all_batches.values() for _lu in _lus]

                    _mean_map = dict(zip(
                        cand_players_df["player_id"].astype(int),
                        cand_players_df["mean"].astype(float),
                    ))
                    _optimal_lineups = sorted(
                        _all_optimal,
                        key=lambda lu: sum(_mean_map.get(pid, 0.0) for pid in lu.player_ids),
                        reverse=True,
                    )
                    self._cb("gpp_optimal_done", {"n_generated": len(_optimal_lineups)})

                n_random = max(0, n_candidates - len(_optimal_lineups))
                self._cb("gpp_generate_start", {
                    "n_candidates": n_random + len(_optimal_lineups),
                    **({"n_from_optimal": len(_optimal_lineups)} if _optimal_lineups else {}),
                })
                random_cands = gen.generate(
                    n_candidates=n_random,
                    max_attempts_multiplier=max_attempts_mult,
                    floor_relief=candidate_floor_relief,
                    stop_check=self._stop_check,
                    progress_cb=lambda n: self._cb("gpp_generate_progress", {"n": n}),
                )
                candidates = _optimal_lineups + random_cands
                self._cb("gpp_generate_done", {
                    "n_generated": len(candidates),
                    "team_distribution": _candidate_team_distribution(candidates, cand_players_df),
                })
                if _slate_fp and candidates and not (_gpp_stopped and self._stop_check()):
                    save_candidates(_slate_fp, candidates)
                self._optimal_lineups = _optimal_lineups

            logger.info("Candidate pool: %d lineups.", len(candidates))
            logger.info(
                "[TIMING] Candidate phase (load+generate): %.3fs  from_cache=%s",
                time.perf_counter() - _t_gpp_start,
                _preloaded_cands is not None,
            )

            if gpp_cfg.get("dump_candidate_pool", False) and candidates:
                _dump_path = os.path.join(output_dir, "candidate_pool_debug.csv")
                _pid_meta = {
                    int(r["player_id"]): r
                    for r in cand_players_df.to_dict("records")
                }
                import csv as _csv
                with open(_dump_path, "w", newline="") as _f:
                    _w = _csv.writer(_f)
                    _w.writerow(["lineup_index", "player_id", "name", "position", "team", "salary", "mean"])
                    for _li, _lu in enumerate(candidates):
                        for _pid in _lu.player_ids:
                            _m = _pid_meta.get(int(_pid), {})
                            _w.writerow([
                                _li,
                                int(_pid),
                                _m.get("name", ""),
                                _m.get("position", ""),
                                _m.get("team", ""),
                                _m.get("salary", ""),
                                round(float(_m.get("mean", 0)), 3),
                            ])
                logger.info("Candidate pool written to %s", _dump_path)

            if not candidates or (_gpp_stopped and self._stop_check()):
                portfolio = []
            else:
                # Phase 2: score candidates against K simulated opponent fields
                self._cb("gpp_score_start", {
                    "n_candidates": len(candidates),
                    "n_field_lineups": n_field,
                    "n_field_samples": n_k,
                })
                team_totals_gpp = self._load_team_totals(slate_path)
                from src.optimization.payout import load_payout_structure, payout_table_to_array
                _gpp_structure = load_payout_structure("dk_classic_gpp_5001")
                _gpp_payout_arr = payout_table_to_array(_gpp_structure).astype(np.float32)
                _cand_excluded_pids = (
                    set(sim_players_df["player_id"].tolist())
                    - set(cand_players_df["player_id"].tolist())
                )

                _cached_field = (
                    load_field(_slate_fp) if (self._use_cached_field and _slate_fp) else None
                )
                if _cached_field is not None:
                    logger.info("Using %d field samples from cache.", len(_cached_field))
                    self._cb("gpp_field_inject", {"n_field": n_field, "n_k": len(_cached_field)})

                scorer = ContestScorer(
                    sim_results=sim_results,
                    players_df=cand_players_df,
                    field_players_df=sim_players_df,
                    n_field_lineups=n_field,
                    n_field_samples=n_k,
                    payout_arr=_gpp_payout_arr,
                    field_rng_seed=int(opt_cfg.get("rng_seed") or 42),
                    ownership_vec=ownership_vector,
                    team_totals=team_totals_gpp,
                    candidate_batch_size=cand_batch,
                    portfolio_size=portfolio_size,
                    cand_excluded_player_ids=_cand_excluded_pids,
                    preloaded_field=_cached_field,
                )
                _t_score = time.perf_counter()
                candidates, robust_payout = scorer.score_candidates(
                    candidates,
                    stop_check=self._stop_check,
                    progress_cb=lambda done, total: self._cb(
                        "gpp_score_progress",
                        {"batches_done": done, "batches_total": total},
                    ),
                    field_progress_cb=lambda n_done, n_total: self._cb(
                        "gpp_field_progress",
                        {"n_done": n_done, "n_total": n_total},
                    ),
                )
                logger.info(
                    "[TIMING] score_candidates wall time: %.3fs  "
                    "robust_payout shape=%s (%.1f MB)  field_from_cache=%s",
                    time.perf_counter() - _t_score,
                    robust_payout.shape, robust_payout.nbytes / 1e6,
                    _cached_field is not None,
                )
                if _cached_field is None and scorer.last_raw_field_list and _slate_fp:
                    save_field(_slate_fp, scorer.last_raw_field_list)
                self._cb("gpp_score_done", {})

                # Diagnostic: log candidate EV distribution so we can tell if the
                # simulation is producing any +EV lineups at all.
                _ev_means = robust_payout.mean(axis=1)
                _pct_pos = float((_ev_means > 0).mean() * 100)
                logger.info(
                    "Candidate EV distribution (net, post-fee): "
                    "min=$%.3f  p10=$%.3f  p50=$%.3f  p90=$%.3f  max=$%.3f  "
                    "+EV candidates: %.1f%%",
                    float(_ev_means.min()),
                    float(np.percentile(_ev_means, 10)),
                    float(np.percentile(_ev_means, 50)),
                    float(np.percentile(_ev_means, 90)),
                    float(_ev_means.max()),
                    _pct_pos,
                )

                # Phase 2b: EV-guided pool refinement — mutate the top-$EV
                # candidates and score the mutants against the same cached
                # fields, extending the pool along the EV frontier before
                # selection. The selector can only pick lineups the pool
                # contains; sampling alone leaves the EV ceiling to chance.
                refine_rounds = int(gpp_cfg.get("refine_rounds", 2))
                refine_top = int(gpp_cfg.get("refine_top", 150))
                refine_mutants = int(gpp_cfg.get("refine_mutants", 8))
                if (
                    refine_rounds > 0 and refine_top > 0 and refine_mutants > 0
                    and not (_gpp_stopped and self._stop_check())
                ):
                    _t_refine = time.perf_counter()
                    _seen_pool = {
                        frozenset(int(p) for p in lu.player_ids) for lu in candidates
                    }
                    # No single primary-stack team may take more than 25% of
                    # the parent slots, so refinement keeps stack breadth.
                    _parent_team_cap = max(1, int(np.ceil(refine_top * 0.25)))
                    _pid_team_r = dict(zip(
                        cand_players_df["player_id"].astype(int), cand_players_df["team"]
                    ))
                    _pid_pos_r = dict(zip(
                        cand_players_df["player_id"].astype(int), cand_players_df["position"]
                    ))
                    _pid_name_r = dict(zip(
                        cand_players_df["player_id"].astype(int),
                        cand_players_df.get("name", cand_players_df["player_id"].astype(str)),
                    ))

                    def _player_label(pid: int) -> str:
                        return f"{_pid_pos_r.get(pid, '?')} {_pid_name_r.get(pid, pid)}"

                    def _primary_stack_team(lu) -> str:
                        th: dict[str, int] = {}
                        for pid in lu.player_ids:
                            if _pid_pos_r.get(int(pid), "") != "P":
                                t = _pid_team_r.get(int(pid), "")
                                if t:
                                    th[t] = th.get(t, 0) + 1
                        return max(th, key=th.get) if th else ""

                    self._cb("gpp_refine_start", {
                        "rounds": refine_rounds,
                        "top": refine_top,
                        "mutants_per_parent": refine_mutants,
                    })
                    _ev_before = float(robust_payout.mean(axis=1).max())
                    _pool_size_before_refine = len(candidates)

                    # Optional holdout split: parents are ranked and mutants
                    # judged on the train columns only, while the same top-K
                    # set is also measured on held-out columns. Noise-mined
                    # gains regress toward zero on the holdout; real
                    # improvements survive.
                    from src.optimization.refine_stats import (
                        mutant_round_stats, split_sim_columns,
                    )
                    _refine_holdout_frac = float(
                        gpp_cfg.get("refine_holdout_fraction", 0.3)
                    )
                    _train_cols, _hold_cols = split_sim_columns(
                        robust_payout.shape[1], _refine_holdout_frac,
                        int(opt_cfg.get("rng_seed") or 42),
                    )

                    def _refine_ev_means(rp: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
                        """(ranking means, holdout means) for the pool."""
                        if _train_cols is None:
                            return rp.mean(axis=1), None
                        return (
                            rp[:, _train_cols].mean(axis=1),
                            rp[:, _hold_cols].mean(axis=1),
                        )

                    # The frontier depth tracks the selection band: Det-EV
                    # draws a portfolio_size selection from roughly the top
                    # 2x portfolio_size candidates by EV.
                    _top_k = max(20, 2 * portfolio_size)

                    for _round in range(refine_rounds):
                        if _gpp_stopped and self._stop_check():
                            break
                        _ev_means_r, _ev_hold_r = _refine_ev_means(robust_payout)
                        _parents: list = []
                        _parent_idx: list[int] = []
                        _parent_team_counts: dict[str, int] = {}
                        for _ci in np.argsort(_ev_means_r)[::-1]:
                            _lu = candidates[int(_ci)]
                            _pt = _primary_stack_team(_lu)
                            if _pt and _parent_team_counts.get(_pt, 0) >= _parent_team_cap:
                                continue
                            _parent_team_counts[_pt] = _parent_team_counts.get(_pt, 0) + 1
                            _parents.append(_lu)
                            _parent_idx.append(int(_ci))
                            if len(_parents) >= refine_top:
                                break

                        _mutants = gen.generate_mutants(
                            _parents, refine_mutants, _seen_pool,
                            rng_seed=int(opt_cfg.get("rng_seed") or 42) + _round + 1,
                            stop_check=self._stop_check,
                        )
                        if not _mutants:
                            logger.info(
                                "Refine round %d/%d: no new mutants; stopping refinement.",
                                _round + 1, refine_rounds,
                            )
                            break
                        _payout_new = scorer.score_batch(
                            _mutants, stop_check=self._stop_check
                        )

                        # Round telemetry: what did mutation change, and does
                        # it hold up on the held-out columns?
                        _n_before = len(candidates)
                        _mut_evs, _mut_hold = _refine_ev_means(_payout_new)
                        _stats = mutant_round_stats(
                            _parents,
                            [float(_ev_means_r[i]) for i in _parent_idx],
                            _mutants,
                            _mut_evs,
                            _player_label,
                            parent_evs_holdout=(
                                [float(_ev_hold_r[i]) for i in _parent_idx]
                                if _ev_hold_r is not None else None
                            ),
                            mutant_evs_holdout=_mut_hold,
                        )
                        _topk_idx_before = np.argsort(_ev_means_r)[-_top_k:]
                        _topk_before = float(_ev_means_r[_topk_idx_before].mean())
                        _topk_hold_before = (
                            float(_ev_hold_r[_topk_idx_before].mean())
                            if _ev_hold_r is not None else None
                        )

                        candidates = candidates + _mutants
                        robust_payout = np.concatenate(
                            [robust_payout, _payout_new], axis=0
                        )
                        _ev_means_after, _ev_hold_after = _refine_ev_means(robust_payout)
                        _best_ev = float(robust_payout.mean(axis=1).max())
                        _topk_idx_after = np.argsort(_ev_means_after)[-_top_k:]
                        _topk_after = float(_ev_means_after[_topk_idx_after].mean())
                        _topk_hold_after = (
                            float(_ev_hold_after[_topk_idx_after].mean())
                            if _ev_hold_after is not None else None
                        )
                        _n_in_topk = int((_topk_idx_after >= _n_before).sum())
                        logger.info(
                            "Refine round %d/%d: +%d mutants (pool %d), %d beat parent, "
                            "%d in top-%d, top-%d EV $%.3f → $%.3f%s, best swap %s → %s (%+.3f%s)",
                            _round + 1, refine_rounds, len(_mutants), len(candidates),
                            _stats["n_beat_parent"], _n_in_topk, _top_k, _top_k,
                            _topk_before, _topk_after,
                            (f" (holdout ${_topk_hold_before:.3f} → ${_topk_hold_after:.3f})"
                             if _topk_hold_before is not None else ""),
                            ", ".join(_stats["best_swap_out"]),
                            ", ".join(_stats["best_swap_in"]),
                            _stats["best_swap_ev_delta"],
                            (f", holdout {_stats['best_swap_ev_delta_holdout']:+.3f}"
                             if "best_swap_ev_delta_holdout" in _stats else ""),
                        )
                        _refine_event = {
                            "round": _round + 1,
                            "rounds": refine_rounds,
                            "n_parents": len(_parents),
                            "n_mutants": len(_mutants),
                            "pool_size": len(candidates),
                            "best_ev": _best_ev,
                            "n_beat_parent": _stats["n_beat_parent"],
                            "top_k": _top_k,
                            "n_in_topk": _n_in_topk,
                            "topk_ev_before": _topk_before,
                            "topk_ev_after": _topk_after,
                            "best_swap_out": _stats["best_swap_out"],
                            "best_swap_in": _stats["best_swap_in"],
                            "best_swap_ev_delta": _stats["best_swap_ev_delta"],
                            "best_mutant_ev": _stats["best_mutant_ev"],
                        }
                        if _topk_hold_before is not None:
                            _refine_event["holdout_fraction"] = _refine_holdout_frac
                            _refine_event["topk_ev_holdout_before"] = _topk_hold_before
                            _refine_event["topk_ev_holdout_after"] = _topk_hold_after
                            _refine_event["best_swap_ev_delta_holdout"] = _stats.get(
                                "best_swap_ev_delta_holdout"
                            )
                        self._cb("gpp_refine_progress", _refine_event)
                    logger.info(
                        "[TIMING] Refinement wall time: %.3fs  best EV $%.3f → $%.3f  "
                        "pool %d",
                        time.perf_counter() - _t_refine,
                        _ev_before, float(robust_payout.mean(axis=1).max()),
                        len(candidates),
                    )
                    self._cb("gpp_refine_done", {
                        "pool_size": len(candidates),
                        "n_added": len(candidates) - _pool_size_before_refine,
                        "best_ev": float(robust_payout.mean(axis=1).max()),
                        "best_ev_before": _ev_before,
                    })

                if _gpp_stopped and self._stop_check():
                    portfolio = []
                else:
                    # Phase 3: Det-EV portfolio selection
                    _t_select = time.perf_counter()
                    _portfolio_sweep_raw: list[tuple[float, list]] = []  # [(risk, raw_portfolio)]

                    from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
                    # Always sweep risk values 1-5; ignore gpp.risk config for det_ev.
                    _DET_SWEEP_RISKS = [1.0, 2.0, 3.0, 4.0, 5.0]
                    _portfolio_sweep_raw: list[tuple[float, list]] = []

                    logger.info(
                        "Det-EV sweep — portfolio_size=%d, risks=%s",
                        portfolio_size, _DET_SWEEP_RISKS,
                    )

                    for _risk_idx, _sweep_risk in enumerate(_DET_SWEEP_RISKS):
                        if self._stop_check is not None and self._stop_check():
                            break
                        _evw = _sweep_risk * 0.05
                        logger.info(
                            "Det-EV risk %d/%d (risk=%.0f, EVw=%.2f, DEw=%.2f)",
                            _risk_idx + 1, len(_DET_SWEEP_RISKS),
                            _sweep_risk, _evw, 1.0 - _evw,
                        )
                        self._cb("gpp_det_risk_start", {
                            "risk": _sweep_risk,
                            "risk_index": _risk_idx + 1,
                            "total_risks": len(_DET_SWEEP_RISKS),
                        })
                        _det_sel = DeterminantPortfolioSelector(
                            robust_payout=robust_payout,
                            candidates=candidates,
                            portfolio_size=portfolio_size,
                            risk=_sweep_risk,
                        )
                        _det_result = _det_sel.select(
                            progress_cb=lambda data, r=_sweep_risk, ri=_risk_idx: self._cb(
                                "gpp_det_select_progress",
                                {**data, "risk": r, "risk_index": ri + 1,
                                 "total_risks": len(_DET_SWEEP_RISKS)},
                            ),
                        )
                        _portfolio_sweep_raw.append((_sweep_risk, _det_result))

                    # Reorder all sweep portfolios by entry-fee-weighted diversity now,
                    # so the ordering is final from the outset and never needs to be
                    # repeated when a risk level is activated or the sweep is displayed.
                    _sweep_fees = PipelineRunner._extract_sorted_fees(all_file_entries)
                    _portfolio_sweep_raw = [
                        (r, PipelineRunner._reorder_by_diversity(p, _sweep_fees))
                        for r, p in _portfolio_sweep_raw
                    ]

                    # Default active = risk=1 (first entry).
                    gpp_result = _portfolio_sweep_raw[0][1] if _portfolio_sweep_raw else []
                    self._sweep_portfolios_raw = {r: p for r, p in _portfolio_sweep_raw}
                    self._gpp_robust_payout = robust_payout
                    self._gpp_candidates = candidates
                    logger.info(
                        "[TIMING] Det-EV sweep wall time: %.3fs",
                        time.perf_counter() - _t_select,
                    )

                    # Persist sweep to disk so it survives a server restart.
                    _sweep_cache_path = os.path.join(
                        output_dir, f"portfolio_sweep_{platform.value}.json"
                    )
                    try:
                        _sweep_cache_data = {
                            "slate_fingerprint": _slate_fp or "",
                            "active_risk": 1,
                            "sweep": [
                                {
                                    "risk": r,
                                    "lineups": self._serialize_portfolio(
                                        p, players_df, mean_ev_from_score=True
                                    ),
                                }
                                for r, p in _portfolio_sweep_raw
                            ],
                        }
                        os.makedirs(output_dir, exist_ok=True)
                        with open(_sweep_cache_path, "w") as _sc_f:
                            json.dump(_sweep_cache_data, _sc_f)
                        logger.info("Det-EV sweep cache saved to %s", _sweep_cache_path)
                    except Exception as _sc_e:
                        logger.warning("Failed to save sweep cache: %s", _sc_e)

                    portfolio = gpp_result

                    # Compute best_scores for portfolio_stats event.
                    col_map_ps = {pid: i for i, pid in enumerate(sim_results.player_ids)}
                    sim_mat_ps = sim_results.results_matrix.astype(np.float32)
                    best_scores_gpp = np.zeros(sim_mat_ps.shape[0], dtype=np.float64)
                    for lu, _ in portfolio:
                        cols = [col_map_ps[pid] for pid in lu.player_ids if pid in col_map_ps]
                        if len(cols) == 10:
                            lu_scores = sim_mat_ps[:, cols].sum(axis=1).astype(np.float64)
                            np.maximum(best_scores_gpp, lu_scores, out=best_scores_gpp)
                    if not (_gpp_stopped and self._stop_check()):
                        _on_portfolio_complete(best_scores_gpp)

        else:
            # ----------------------------------------------------------------
            # Original PortfolioConstructor path (all other objectives).
            # ----------------------------------------------------------------
            logger.info(
                "Constructing portfolio — size=%d, target=%.1f, chains=%d",
                portfolio_size, target, opt_cfg.get("n_chains", 250),
            )
            constructor = PortfolioConstructor(
                sim_results=sim_results,
                players_df=cand_players_df,
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
                ownership_vector=ownership_vector,
                rules=roster_rules,
                slot_eligibility=slot_elig,
            )

            def _on_lineup_complete(
                lineup_index: int, total: int, score: float,
                arg4: int, arg5: int, arg6: Optional[float] = None,
                arg7: Optional[float] = None, arg8: Optional[float] = None,
                arg9: Optional[float] = None,
            ) -> None:
                if objective in ("marginal_payout",):
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

            portfolio = constructor.construct(
                on_lineup_complete=_on_lineup_complete,
                on_portfolio_complete=_on_portfolio_complete if objective == "marginal_payout" else None,
                stop_check=self._stop_check,
                target_percentile=target_percentile if objective == "marginal_payout" else None,
            )

        logger.info("Portfolio complete: %d lineups.", len(portfolio))

        _portfolio_has_dollar_ev = objective == "leverage_surplus"

        # Store raw artifacts for on-demand upload file writing.
        self._raw_portfolio = portfolio
        self._portfolio_has_dollar_ev = _portfolio_has_dollar_ev
        self._discarded_lineups: set[frozenset] = set()
        self._all_file_entries = all_file_entries
        self._entry_handlers = entry_handlers
        self._slate_df = slate_df
        self._output_dir = output_dir
        self._platform = platform

        # --- Serialize --------------------------------------------------
        result = self._serialize_portfolio(portfolio, players_df, mean_ev_from_score=_portfolio_has_dollar_ev)

        was_stopped = self._stop_check is not None and self._stop_check()

        # --- Save CSV ---------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(portfolio, players_df, mean_ev_from_score=_portfolio_has_dollar_ev)
        output_path = os.path.join(output_dir, f"portfolio_{platform.value}.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Portfolio saved to %s", output_path)

        # --- Generate upload files (skipped when stopped) ----------------
        entry_map: dict = {}
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
        _optimal_result = (
            self._serialize_portfolio(
                [(lu, 0.0) for lu in self._optimal_lineups], players_df
            )
            if getattr(self, "_optimal_lineups", None)
            else []
        )

        # --- Persist optimal lineups alongside portfolio (survives restart) -
        if _optimal_result:
            from pathlib import Path as _Path
            from .slate_exclusions import compute_file_fingerprint as _cfp
            _config_root = _Path(self._config_path).resolve().parent
            _slate_abs = _config_root / slate_path if slate_path else None
            _slate_fp = _cfp(_slate_abs)
            _opt_json_path = os.path.join(output_dir, f"optimal_lineups_{platform.value}.json")
            with open(_opt_json_path, "w") as _f:
                json.dump({"lineups": _optimal_result, "slate_fingerprint": _slate_fp}, _f)
        if was_stopped:
            logger.info("Run stopped by user after %d lineups.", len(portfolio))
            _stopped_sweep_payload = [
                {
                    "risk": r,
                    "lineups": self._serialize_portfolio(p, players_df, mean_ev_from_score=True),
                }
                for r, p in _portfolio_sweep_raw
            ] if _portfolio_sweep_raw else []
            self._cb("stopped", {
                "portfolio": result,
                "n_lineups": len(result),
                "optimal_lineups": _optimal_result,
                "portfolio_sweep": _stopped_sweep_payload,
            })
        else:
            _sweep_payload = [
                {
                    "risk": r,
                    "lineups": self._serialize_portfolio(p, players_df, mean_ev_from_score=True),
                }
                for r, p in _portfolio_sweep_raw
            ] if _portfolio_sweep_raw else []
            # Apply entry meta to all sweep entries (assignment is positional: lineup i → entry i)
            # and update the persisted sweep JSON so entry info survives a server restart.
            if _sweep_payload and entry_map:
                for _se in _sweep_payload:
                    for lr in _se["lineups"]:
                        info = entry_map.get(lr["lineup_index"])
                        if info:
                            lr.update(info)
                _sweep_cache_path = os.path.join(output_dir, f"portfolio_sweep_{platform.value}.json")
                if os.path.exists(_sweep_cache_path):
                    try:
                        with open(_sweep_cache_path) as _scf:
                            _scd = json.load(_scf)
                        for _se in _scd.get("sweep", []):
                            for lr in _se.get("lineups", []):
                                info = entry_map.get(lr["lineup_index"])
                                if info:
                                    lr.update(info)
                        with open(_sweep_cache_path, "w") as _scf:
                            json.dump(_scd, _scf)
                    except Exception as _sce:
                        logger.warning("Failed to update sweep cache with entry meta: %s", _sce)
            self._cb("complete", {
                "portfolio": result,
                "n_lineups": len(result),
                "optimal_lineups": _optimal_result,
                "portfolio_sweep": _sweep_payload,
            })

        return result

    def activate_sweep_risk(self, risk: float) -> list[dict]:
        """Switch the active portfolio to a different det_ev sweep risk level.

        Re-saves the portfolio CSV and (if entry files are available) re-runs
        lineup-to-entry assignment and writes upload CSVs.

        Returns the serialized LineupResult list for the new active portfolio.
        """
        sweep = getattr(self, "_sweep_portfolios_raw", {})
        if not sweep:
            raise RuntimeError("No det_ev sweep portfolios available.")
        # Exact match or closest risk in sweep.
        if risk not in sweep:
            risk = min(sweep.keys(), key=lambda r: abs(r - risk))
        # Portfolios were already diversity-reordered at sweep-build time; use as-is.
        portfolio = sweep[risk]

        self._raw_portfolio = portfolio
        platform_val = self._platform.value if hasattr(self, "_platform") else "draftkings"
        os.makedirs(self._output_dir, exist_ok=True)

        portfolio_df = self._format_portfolio_df(portfolio, self._players_df, mean_ev_from_score=True)
        output_path = os.path.join(self._output_dir, f"portfolio_{platform_val}.csv")
        portfolio_df.to_csv(output_path, index=False)
        logger.info("Activated risk=%.0f portfolio; saved to %s", risk, output_path)

        result = self._serialize_portfolio(portfolio, self._players_df, mean_ev_from_score=True)

        if self._all_file_entries:
            self.write_upload_files()
            entry_map = self._build_lineup_entry_map(self._all_file_entries, portfolio)
            for lr in result:
                info = entry_map.get(lr["lineup_index"])
                if info:
                    lr.update(info)
            meta_path = os.path.join(self._output_dir, f"portfolio_entries_{platform_val}.json")
            with open(meta_path, "w") as f:
                json.dump({str(k): v for k, v in entry_map.items()}, f)

        # Update active_risk in the sweep cache so the choice persists.
        _sweep_cache = os.path.join(self._output_dir, f"portfolio_sweep_{platform_val}.json")
        try:
            if os.path.exists(_sweep_cache):
                with open(_sweep_cache) as _f:
                    _cache = json.load(_f)
                _cache["active_risk"] = risk
                with open(_sweep_cache, "w") as _f:
                    json.dump(_cache, _f)
        except Exception as _e:
            logger.warning("Could not update active_risk in sweep cache: %s", _e)

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
        remaining = self._raw_portfolio[:idx] + self._raw_portfolio[idx + 1:]

        # Accumulate discarded lineups across repeated replacements so no
        # deleted lineup is ever re-inserted into the portfolio.
        if not hasattr(self, '_discarded_lineups'):
            self._discarded_lineups: set[frozenset] = set()
        self._discarded_lineups.add(frozenset(deleted_lineup.player_ids))

        full_matrix = self._sim_results.results_matrix
        col_map = {pid: i for i, pid in enumerate(self._sim_results.player_ids)}

        if self._objective == "leverage_surplus":
            # GPP path: pick next-best candidate from the pre-scored pool.
            if not hasattr(self, '_gpp_candidates') or not hasattr(self, '_gpp_robust_payout'):
                raise RuntimeError("GPP artifacts unavailable — please re-run the portfolio")

            # Build pid-set → candidate-index map.
            cand_pid_sets = [frozenset(lu.player_ids) for lu in self._gpp_candidates]
            pid_set_to_ci = {ps: ci for ci, ps in enumerate(cand_pid_sets)}

            # Excluded = remaining portfolio lineups + all previously discarded lineups.
            remaining_pid_sets = {frozenset(lu.player_ids) for lu, _ in remaining}
            excluded = remaining_pid_sets | self._discarded_lineups

            # Seed best_payout from remaining portfolio lineups.
            n_sims_gpp = self._gpp_robust_payout.shape[1]
            best_payout = np.zeros(n_sims_gpp, dtype=np.float32)
            for lu, _ in remaining:
                ci = pid_set_to_ci.get(frozenset(lu.player_ids))
                if ci is not None:
                    np.maximum(best_payout, self._gpp_robust_payout[ci], out=best_payout)

            remaining_ci = [
                pid_set_to_ci[ps]
                for ps in [frozenset(lu.player_ids) for lu, _ in remaining]
                if ps in pid_set_to_ci
            ]
            deleted_ci = pid_set_to_ci.get(frozenset(deleted_lineup.player_ids))
            discarded_ci = {
                pid_set_to_ci[ps]
                for ps in self._discarded_lineups
                if ps in pid_set_to_ci
            }

            if hasattr(self, '_gpp_selector'):
                new_ci = self._gpp_selector.find_replacement(
                    current_portfolio_indices=remaining_ci,
                    exclude_index=deleted_ci,
                    additional_excluded=discarded_ci,
                )
                logger.info("replace_lineup: MV find_replacement → candidate %d", new_ci)
            else:
                # Backward compat: old artifacts without _gpp_selector — marginal EV fallback.
                avail_payout = self._gpp_robust_payout[
                    np.array(
                        [ci for ci, ps in enumerate(cand_pid_sets) if ps not in excluded],
                        dtype=np.int32,
                    )
                ]
                ev_vals = np.maximum(0.0, avail_payout - best_payout[np.newaxis, :]).mean(axis=1)
                avail_list = [ci for ci, ps in enumerate(cand_pid_sets) if ps not in excluded]
                new_ci = avail_list[int(np.argmax(ev_vals))]
                logger.info(
                    "replace_lineup: no _gpp_selector, using marginal EV fallback → candidate %d",
                    new_ci,
                )

            if new_ci is None or new_ci < 0:
                raise RuntimeError("No available candidates remain after excluding discarded lineups")

            new_lineup = self._gpp_candidates[new_ci]
            full_score = float(self._gpp_robust_payout[new_ci].mean())

        elif self._objective == "marginal_payout":
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
            # Request more candidates to clear the full discard history.
            n_candidates = max(10, len(self._discarded_lineups) + 5)
            candidates = optimizer.optimize_top_k(n_candidates)
            new_lineup, _ = next(
                (
                    (lu, sc) for lu, sc in candidates
                    if frozenset(lu.player_ids) not in self._discarded_lineups
                ),
                candidates[0],
            )
            cols = [col_map[pid] for pid in new_lineup.player_ids]
            full_score = float((full_matrix[:, cols].sum(axis=1) >= self._target).mean())

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
            # Request more candidates to clear the full discard history.
            n_candidates = max(10, len(self._discarded_lineups) + 5)
            candidates = optimizer.optimize_top_k(n_candidates)
            new_lineup, _ = next(
                (
                    (lu, sc) for lu, sc in candidates
                    if frozenset(lu.player_ids) not in self._discarded_lineups
                ),
                candidates[0],
            )
            cols = [col_map[pid] for pid in new_lineup.player_ids]
            full_score = float((full_matrix[:, cols].sum(axis=1) >= self._target).mean())

        _has_ev = getattr(self, "_portfolio_has_dollar_ev", False)
        _fees_re = PipelineRunner._extract_sorted_fees(self._all_file_entries)
        self._raw_portfolio = PipelineRunner._reorder_by_diversity(remaining + [(new_lineup, full_score)], _fees_re)
        result = self._serialize_portfolio(self._raw_portfolio, self._players_df, mean_ev_from_score=_has_ev)

        os.makedirs(self._output_dir, exist_ok=True)
        portfolio_df = self._format_portfolio_df(self._raw_portfolio, self._players_df, mean_ev_from_score=_has_ev)
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
            # Warn if the projections file has players not on the current slate.
            # This happens when projections were fetched before a game was postponed
            # and its teams removed from the DK/FD CSV.  The LEFT join below drops
            # them silently, but surfacing this helps diagnose stale projections.
            _slate_pids = set(pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int))
            _proj_pids  = set(pd.to_numeric(proj["player_id"], errors="coerce").dropna().astype(int))
            _stale = _proj_pids - _slate_pids
            if _stale:
                _stale_names = (
                    proj[proj["player_id"].isin(_stale)]["name"].tolist()
                    if "name" in proj.columns else list(_stale)
                )
                logger.warning(
                    "Projections file has %d player(s) not on the current slate "
                    "(possibly from a postponed game — they will be excluded): %s",
                    len(_stale),
                    ", ".join(str(n) for n in _stale_names[:20]),
                )
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
    def _apply_twitter_overrides(players_df: pd.DataFrame, slate_path: str = "") -> pd.DataFrame:
        """Apply confirmed twitter lineup slot/slot_confirmed overrides to players_df.

        For each team with a confirmed Twitter lineup, batters from that team who are
        NOT in the lineup are dropped entirely (they are scratched). Batters who ARE
        in the lineup get their slot updated and slot_confirmed set to True.
        """
        from pathlib import Path as _Path
        from .twitter_lineups import get_confirmed_team_lineups
        from .slate_exclusions import compute_file_fingerprint
        _fp = compute_file_fingerprint(_Path(slate_path) if slate_path else None)
        confirmed = get_confirmed_team_lineups(_fp)
        if not confirmed:
            return players_df
        df = players_df.copy()
        if "slot_confirmed" not in df.columns:
            df["slot_confirmed"] = False

        # Drop scratched batters: batter from a confirmed-lineup team but not in that lineup
        batter_mask = df["position"] != "P"
        for team, pid_to_slot in confirmed.items():
            scratched = batter_mask & (df["team"] == team) & ~df["player_id"].isin(pid_to_slot)
            df = df[~scratched]
        df = df.copy()

        # Apply confirmed slot and slot_confirmed for each lineup player
        for pid_to_slot in confirmed.values():
            for pid, slot in pid_to_slot.items():
                mask = df["player_id"] == pid
                if mask.any():
                    df.loc[mask, "slot"] = slot
                    df.loc[mask, "slot_confirmed"] = True

        return df

    @staticmethod
    def _apply_exclusions(players_df: pd.DataFrame, slate_path: str = "") -> tuple:
        """Filter players_df based on persisted slate exclusions.

        Returns (sim_players_df, cand_players_df, excl_stats, game_ppd_pcts):
          sim_players_df  — removes only "both"-scoped entities; used for the
                            simulation engine and opponent field generation so
                            "candidate-excluded" players still get scores and can
                            appear in field lineups.
          cand_players_df — removes ("candidates" + "both")-scoped entities;
                            used for our own lineup/candidate optimization.
          excl_stats       — {n_teams_excluded, n_batters_ind_excluded, ...}
          game_ppd_pcts    — {game: pct} for PPD handling
        """
        from pathlib import Path as _Path
        from .slate_exclusions import compute_file_fingerprint, compute_slate_id, read_exclusions
        empty_stats: dict = {"n_teams_excluded": 0, "n_batters_ind_excluded": 0, "n_pitchers_ind_excluded": 0}

        current_games = [g for g in players_df["game"].dropna().unique().tolist() if g]
        if not current_games:
            return players_df, players_df, empty_stats, {}

        slate_file = _Path(slate_path) if slate_path else None
        fingerprint = compute_file_fingerprint(slate_file)
        slate_id = compute_slate_id(current_games)
        stored = read_exclusions(slate_id, fingerprint)

        game_ppd_pcts: dict = {k: v for k, v in stored.get("game_ppd_pcts", {}).items() if v and v > 0}

        # "both" scope — excluded from everything
        both_games = set(stored.get("excluded_games", []))
        both_teams = set(stored.get("excluded_teams", []))
        both_player_ids = set(stored.get("excluded_player_ids", []))

        # "candidates" scope — excluded from our pool only
        cand_games = set(stored.get("candidate_excluded_games", []))
        cand_teams = set(stored.get("candidate_excluded_teams", []))
        cand_player_ids = set(stored.get("candidate_excluded_player_ids", []))

        any_both = bool(both_games or both_teams or both_player_ids)
        any_cand = bool(cand_games or cand_teams or cand_player_ids)

        if not any_both and not any_cand:
            return players_df, players_df, empty_stats, game_ppd_pcts

        # sim_players_df: remove only "both"-scoped entities
        if any_both:
            both_mask = (
                players_df["game"].isin(both_games) |
                players_df["team"].isin(both_teams) |
                players_df["player_id"].isin(both_player_ids)
            )
            sim_df = players_df[~both_mask].copy()
            n_sim_removed = int(both_mask.sum())
            if n_sim_removed:
                logger.info("Both-exclusions removed %d players from sim (%d remain).", n_sim_removed, len(sim_df))
        else:
            sim_df = players_df

        # cand_players_df: remove "candidates" + "both" scoped entities
        all_excl_games = both_games | cand_games
        all_excl_teams = both_teams | cand_teams
        all_excl_pids = both_player_ids | cand_player_ids

        cand_mask = (
            players_df["game"].isin(all_excl_games) |
            players_df["team"].isin(all_excl_teams) |
            players_df["player_id"].isin(all_excl_pids)
        )
        cand_df = players_df[~cand_mask].copy()
        n_cand_removed = int(cand_mask.sum())
        if n_cand_removed:
            logger.info("Exclusions removed %d players from candidate pool (%d remain).", n_cand_removed, len(cand_df))

        # Stats based on candidate pool (what user sees as their optimizable pool)
        ind_excl_df = players_df[players_df["player_id"].isin(all_excl_pids)]
        n_batters_ind_excluded = int((ind_excl_df["position"] != "P").sum())
        n_pitchers_ind_excluded = int((ind_excl_df["position"] == "P").sum())
        pre_n_teams = players_df["team"].nunique()
        n_teams_excluded = pre_n_teams - cand_df["team"].nunique()

        excl_stats = {
            "n_teams_excluded": int(n_teams_excluded),
            "n_batters_ind_excluded": n_batters_ind_excluded,
            "n_pitchers_ind_excluded": n_pitchers_ind_excluded,
        }
        return sim_df, cand_df, excl_stats, game_ppd_pcts

    @staticmethod
    def _apply_ppd_to_simulation(
        sim_results: "SimulationResults",
        players_df: pd.DataFrame,
        game_ppd_pcts: dict,
        rng_seed: Optional[int] = None,
    ) -> tuple:
        """Zero out players from PPD'd games in an independent random fraction of simulations.

        Seeded so identical runs zero identical rows — unseeded PPD draws would
        silently break A/B comparisons (e.g. refinement on/off) on PPD slates.

        Returns (updated SimulationResults, stats dict {game: {ppd_pct, n_sims_zeroed}}).
        """
        matrix = sim_results.results_matrix.copy()
        col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
        rng = np.random.default_rng(rng_seed)
        stats: dict = {}
        for game, pct in game_ppd_pcts.items():
            if not pct or pct <= 0:
                continue
            game_pids = players_df[players_df["game"] == game]["player_id"].tolist()
            cols = [col_map[pid] for pid in game_pids if pid in col_map]
            if not cols:
                continue
            n_ppd = max(1, round(matrix.shape[0] * pct / 100.0))
            idx = rng.choice(matrix.shape[0], size=n_ppd, replace=False)
            matrix[np.ix_(idx, cols)] = 0.0
            stats[game] = {"ppd_pct": pct, "n_sims_zeroed": int(n_ppd)}
            logger.info("PPD applied: %s (%.0f%%) — %d/%d sims zeroed", game, pct, n_ppd, matrix.shape[0])
        return SimulationResults(sim_results.player_ids, matrix), stats

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
    def _load_team_totals(slate_path: str) -> Optional[dict]:
        """Load implied run totals from the archive for the slate date.

        Prefers team_totals.csv (FantasyLabs-derived); then cno_team_totals.csv
        (legacy CNO-derived); falls back to dff_team_totals.csv for older archives.
        Returns {team: implied_total} or None if no file is found.
        """
        import re as _re
        from pathlib import Path as _Path
        # Extract date from DK slate filename: DKSalaries_MM_DD_YYYY or MMDDYYYY patterns
        fname = _Path(slate_path).name if slate_path else ""
        archive_dir = None
        m = _re.search(r'(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})', fname)
        if m:
            mm, dd, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
            archive_dir = _Path(__file__).resolve().parents[2] / "archive" / f"{mm}{dd}{yyyy}"
        if archive_dir is None or not archive_dir.exists():
            # Fall back to slate date from Game Info column in DKSalaries.csv
            slate_file = _Path(slate_path) if slate_path else None
            if slate_file and slate_file.exists():
                try:
                    import csv as _csv
                    with open(slate_file, newline="") as _f:
                        for row in _csv.DictReader(_f):
                            game_info = row.get("Game Info", "")
                            dm = _re.search(r'(\d{2})/(\d{2})/(\d{4})', game_info)
                            if dm:
                                mm, dd, yyyy = dm.group(1), dm.group(2), dm.group(3)
                                archive_dir = _Path(__file__).resolve().parents[2] / "archive" / f"{mm}{dd}{yyyy}"
                                break
                except Exception:
                    pass
        if archive_dir is None:
            return None
        for filename in ("team_totals.csv", "cno_team_totals.csv", "dff_team_totals.csv"):
            totals_path = archive_dir / filename
            if not totals_path.exists():
                continue
            try:
                df = pd.read_csv(totals_path)
                if "team" in df.columns and "implied_total" in df.columns:
                    return dict(zip(df["team"], df["implied_total"].astype(float)))
            except Exception:
                pass
        return None

    @staticmethod
    def _load_hr_fair_odds(slate_path: str) -> dict[str, float]:
        """Load HR fair-implied-probability data from the archive for the slate date.

        Returns {normalised_name: hr_over_0.5_implied_prob} from
        market_odds_fair_odds.json, or an empty dict if unavailable.
        """
        import re as _re
        import unicodedata as _ud
        import json as _json
        from pathlib import Path as _Path

        def _norm(name: str) -> str:
            nfkd = _ud.normalize("NFKD", name)
            ascii_n = nfkd.encode("ascii", "ignore").decode("ascii")
            return _re.sub(r"[^a-z ]", "", ascii_n.lower()).strip()

        fname = _Path(slate_path).name if slate_path else ""
        archive_dir = None
        m = _re.search(r'(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})', fname)
        if m:
            mm, dd, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
            archive_dir = _Path(__file__).resolve().parents[2] / "archive" / f"{mm}{dd}{yyyy}"
        if archive_dir is None or not archive_dir.exists():
            slate_file = _Path(slate_path) if slate_path else None
            if slate_file and slate_file.exists():
                try:
                    import csv as _csv
                    with open(slate_file, newline="") as _f:
                        for row in _csv.DictReader(_f):
                            game_info = row.get("Game Info", "")
                            dm = _re.search(r'(\d{2})/(\d{2})/(\d{4})', game_info)
                            if dm:
                                mm, dd, yyyy = dm.group(1), dm.group(2), dm.group(3)
                                archive_dir = _Path(__file__).resolve().parents[2] / "archive" / f"{mm}{dd}{yyyy}"
                                break
                except Exception:
                    pass
        if archive_dir is None:
            return {}
        odds_path = archive_dir / "market_odds_fair_odds.json"
        if not odds_path.exists():
            return {}
        try:
            with open(odds_path) as _f:
                d = _json.load(_f)
            result: dict[str, float] = {}
            for players in d.get("data", {}).values():
                for name, markets in players.items():
                    for entry in markets.get("Player Home Runs", []):
                        if entry["line"] == 0.5 and entry["outcome"] == "over" and not entry["flagged"]:
                            result[_norm(name)] = float(entry["implied_prob"])
            return result
        except Exception:
            return {}

    @staticmethod
    def _load_cash_threshold() -> float:
        """Compute cash threshold from the DK Classic GPP payout structure.

        Cash threshold = fraction of the field you must beat to finish in a
        paying position.  E.g. top-26% payout → need to beat 74% of the field.
        Falls back to 0.74 if the file is missing or malformed.
        """
        from pathlib import Path as _Path
        import json as _json
        path = _Path(__file__).resolve().parents[2] / "data" / "payout_structures" / "dk_classic_gpp.json"
        try:
            with open(path) as f:
                ps = _json.load(f)
            total = int(ps["total_entries"])
            paying = int(ps["total_payout_positions"])
            return 1.0 - paying / total
        except Exception:
            return 0.74

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
    def _extract_sorted_fees(all_file_entries: list) -> list[int]:
        """Return entry fees in ascending prize_pool/fee ratio order.

        This matches the slot order used by assign_lineups_to_entries, so that
        fees[k] is the actual fee of the k-th entry slot — used as positional
        weights in _reorder_by_diversity.
        """
        flat = []
        for _, records in all_file_entries:
            for rec in records:
                flat.append((_sort_ratio(rec), len(flat), rec.entry_fee_cents))
        flat.sort(key=lambda x: (x[0], -x[2], x[1]))
        return [fee for _, _, fee in flat]

    @staticmethod
    def _reorder_by_diversity(portfolio: list, fees: list[int] | None = None) -> list:
        """Greedy diversity reorder: assigns the most underrepresented lineup to each slot.

        At each step k (filling entry slot with fee slot_fees[k]), selects the remaining
        lineup that maximizes:

            sum over pid: (total_fees * full_freq[pid] - running_fees[pid])

        where total_fees = sum of all entry fees for the slate, full_freq[pid] = fraction
        of portfolio lineups containing pid, and running_fees[pid] = cumulative entry fees
        of slots already filled that include pid.

        Lineups whose players are most "underpaid" relative to their portfolio share
        float to the front.
        """
        if len(portfolio) <= 1:
            return portfolio
        N = len(portfolio)

        slot_fees: list[int] = list(fees) if fees else []
        total_fees = sum(slot_fees)

        if total_fees == 0:
            slot_fees = [1] * N
            total_fees = N

        full_freq: dict[int, float] = {}
        for lineup, _ in portfolio:
            for pid in lineup.player_ids:
                full_freq[pid] = full_freq.get(pid, 0) + 1
        for pid in full_freq:
            full_freq[pid] /= N

        remaining = list(range(N))
        selected: list[int] = []
        running_fees: dict[int, float] = {}

        while remaining:
            k = len(selected)
            slot_fee = slot_fees[k] if k < len(slot_fees) else 0
            best_i, best_score = remaining[0], float('-inf')
            for i in remaining:
                lineup, _ = portfolio[i]
                s = sum(
                    total_fees * full_freq.get(pid, 0) - running_fees.get(pid, 0.0)
                    for pid in lineup.player_ids
                )
                if s > best_score:
                    best_score, best_i = s, i
            selected.append(best_i)
            remaining.remove(best_i)
            for pid in portfolio[best_i][0].player_ids:
                running_fees[pid] = running_fees.get(pid, 0.0) + slot_fee

        return [portfolio[i] for i in selected]

    @staticmethod
    def _upload_display_order(player_ids: list, players_df: pd.DataFrame) -> list:
        """Reorder a lineup's player ids into DK upload column order
        (P,P,C,1B,2B,3B,SS,OF,OF,OF) — the same assignment write_upload_files
        uses — so portfolio cards match the upload_*.csv columns (and the
        Late Swap tab, which pins those columns). Falls back to the given
        order when slot assignment doesn't apply (e.g. FanDuel rosters)."""
        from .dk_entries import assign_players_to_slots
        try:
            return assign_players_to_slots(list(player_ids), players_df)
        except Exception:
            return list(player_ids)

    @staticmethod
    def _serialize_portfolio(
        portfolio: list,
        players_df: pd.DataFrame,
        mean_ev_from_score: bool = False,
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
            ordered_ids = PipelineRunner._upload_display_order(lineup.player_ids, players_df)
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
                for pid in ordered_ids
            ]
            mean_ev: Optional[float] = round(float(score), 4) if mean_ev_from_score else None
            result.append({
                "lineup_index": i,
                "p_hit_target": round(score, 4),
                "lineup_salary": total_salary,
                "mean_ev": mean_ev,
                "players": players,
            })
        return result

    @staticmethod
    def _build_lineup_entry_map(
        all_file_entries: list,
        portfolio: list,
    ) -> dict[int, dict]:
        """Return {lineup_index: {upload_tag, entry_fee, contest_name}} from entry assignments.

        Uses the same ascending prize_pool/fee ratio order as assign_lineups_to_entries.
        """
        flat: list = []
        for file_path, records in all_file_entries:
            for rec in records:
                flat.append((_sort_ratio(rec), len(flat), file_path, rec))
        flat.sort(key=lambda x: (x[0], -x[3].entry_fee_cents, x[1]))
        entry_map: dict[int, dict] = {}
        for i, (ratio, _, file_path, rec) in enumerate(flat):
            if i >= len(portfolio):
                break
            entry_map[i + 1] = {
                "upload_tag": _extract_upload_tag(file_path.name),
                "entry_fee": rec.entry_fee_raw,
                "contest_name": _shorten_contest_name(rec.contest_name),
                "entry_sort_order": ratio if ratio != float("inf") else None,
            }
        return entry_map

    @staticmethod
    def _format_portfolio_df(
        portfolio: list,
        players_df: pd.DataFrame,
        mean_ev_from_score: bool = False,
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
            mean_ev: Optional[float] = round(float(score), 4) if mean_ev_from_score else None
            for pid in PipelineRunner._upload_display_order(lineup.player_ids, players_df):
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
                    "mean_ev": mean_ev,
                    "slot": int(id_to_slot[pid]) if pid in id_to_slot else None,
                    "slot_confirmed": bool(id_to_confirmed[pid]) if pid in id_to_confirmed else False,
                })
        return pd.DataFrame(rows)
