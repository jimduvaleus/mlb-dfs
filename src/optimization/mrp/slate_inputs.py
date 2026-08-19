"""Build pool / players_df / sims for a slate, live or archived.

`tests/bt_core.build_slate_context` already does this, but it lives in tests/,
requires the standings zips (`real`), and additionally computes corr and p_win
that MRP does not use. A live slate has no standings zips, so a runnable entry
point needs its own path.

This is the same external-pool ingest production uses -- `discover_external_files`
-> `parse_lineup_pool` -> `build_external_players_df` -> `SimulationEngine` --
with nothing added.

CALIBRATION DEFAULTS TO ON, matching what production ships (memory
project-external-sim-calibration: batter zero-inflation plus the 0.88 mean
calibration, which together moved lineup mean PIT from 0.405 to 0.497). Note
the MRP archive evaluations were run with calibration OFF, following the
backtest harness's default -- so a live run and those measurements are not on
identical inputs, and a like-for-like comparison must pin this flag explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SlateInputs:
    pool: object
    players_df: object
    sim_results: object
    slate_dir: Path
    n_sims: int
    seed: int
    calibrated: bool


def build_slate_inputs(
    slate_dir: Path,
    n_sims: int = 25_000,
    seed: int = 42,
    calibrated: bool = True,
    sim_cache_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> SlateInputs:
    """Ingest one slate directory and simulate.

    `slate_dir` holds DKSalaries.csv plus the SaberSim export pair
    (lineups_*.csv + its companion projections file) -- true of both
    `data/raw` on a live slate and any `archive/MMDDYYYY`.
    """
    from src.api import external_pool as ep
    from src.api.pipeline import PipelineRunner
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    from src.models.copula import EmpiricalCopula
    from src.simulation.engine import SimulationEngine
    from src.simulation.results import SimulationResults

    slate_dir = Path(slate_dir)
    cfg = yaml.safe_load(open(config_path or (PROJECT_ROOT / "config.yaml")))

    found = ep.discover_external_files(str(slate_dir))
    if not found["lineups_paths"] or not found["projections_path"]:
        raise SystemExit(f"{slate_dir}: no lineups_*.csv / projections CSV pair found")

    slate_df = DraftKingsSlateIngestor(str(slate_dir / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    if not pool.lineups:
        raise SystemExit(f"{slate_dir}: every lineup dropped (unknown player ids)")

    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )

    cache_path = None
    if sim_cache_dir is not None:
        sim_cache_dir = Path(sim_cache_dir)
        sim_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = sim_cache_dir / f"{slate_dir.name}_{n_sims}_{seed}_calib{calibrated}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            return SlateInputs(
                pool=pool, players_df=players_df,
                sim_results=SimulationResults(
                    player_ids=[int(p) for p in z["player_ids"]],
                    results_matrix=z["results_matrix"],
                ),
                slate_dir=slate_dir, n_sims=n_sims, seed=seed, calibrated=calibrated,
            )

    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    grids = ep.build_quantile_grids(
        proj_ext,
        zero_inflate=calibrated, scratch_prob=0.02 if calibrated else 0.0,
        mean_calib_batter=0.88 if calibrated else 1.0, mean_calib_pitcher=1.0,
    )
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    # SimulationEngine.simulate takes no seed argument -- it draws from numpy's
    # global RNG, so seeding is global. Same approach bt_core uses
    # (build_slate_context: np.random.seed(seed) then engine.simulate(n_sims)),
    # kept identical so a cached sim from either path is interchangeable.
    np.random.seed(seed)
    sim_results = engine.simulate(n_sims)

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
    return SlateInputs(pool=pool, players_df=players_df, sim_results=sim_results,
                       slate_dir=slate_dir, n_sims=n_sims, seed=seed,
                       calibrated=calibrated)
