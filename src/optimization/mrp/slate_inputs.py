"""Build pool / players_df / sims for a slate, live or archived.

`tests/bt_core.build_slate_context` already does this, but it lives in tests/,
requires the standings zips (`real`), and additionally computes corr and p_win
that MRP does not use. A live slate has no standings zips, so a runnable entry
point needs its own path.

This is the same external-pool ingest production uses -- `discover_external_files`
-> `parse_lineup_pool` -> `build_external_players_df` -> `SimulationEngine` --
with nothing added.

SIM CALIBRATION IS READ FROM CONFIG, never hardcoded. The four
`gpp.external_pool_*` keys below are the same ones `pipeline.py` reads, so this
path and the UI path simulate identically:

    external_pool_zero_inflate        SHAPE. A DK hitter scores exactly 0 when
                                      he never reaches base and drives nobody
                                      in: 2.19% of the time in the raw grids,
                                      20.6% measured across 10 archived slates.
    external_pool_mean_calib_batter   LOCATION. Realized/grid mean ratio 0.878
                                      for batters (p=0.009). Pitchers measured
                                      0.935 at p=0.30, so left at 1.0 rather
                                      than fitting noise.
    external_pool_scratch_prob        P(a rostered player is scratched).
    external_pool_grid_mean_rescale   LOCATION, per player. Rescales each grid
                                      to that player's "My Proj" when the two
                                      disagree by >20%, so a hand-edited
                                      projection actually reaches the sim (the
                                      dk_*_percentile columns do not follow it).

An earlier version of this file defaulted calibration ON on the belief that
production shipped it. It does not -- the live config and the `flat2000_uc`
production backtest arm ("uc" = uncalibrated) both run with it off. Hardcoding
made this path and the pipeline produce DIFFERENT portfolios from the same
slate, which silently breaks the one thing the CLI is for: `--preassign-from`
runs MRP against a production portfolio, and the two arms have to be built on
the same sim to be comparable at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SimCalibration:
    """The four grid-calibration knobs, exactly as pipeline.py reads them."""

    zero_inflate: bool = False
    scratch_prob: float = 0.02
    mean_calib_batter: float = 1.0
    mean_calib_pitcher: float = 1.0
    grid_mean_rescale: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "SimCalibration":
        g = cfg.get("gpp", {}) or {}
        return cls(
            zero_inflate=bool(g.get("external_pool_zero_inflate", False)),
            scratch_prob=float(g.get("external_pool_scratch_prob", 0.02)),
            mean_calib_batter=float(g.get("external_pool_mean_calib_batter", 1.0)),
            mean_calib_pitcher=float(g.get("external_pool_mean_calib_pitcher", 1.0)),
            grid_mean_rescale=bool(g.get("external_pool_grid_mean_rescale", False)),
        )

    def cache_key(self) -> str:
        """Part of the sim-cache filename: two runs under different calibration
        are different sims and must never share a cached matrix."""
        return (f"z{int(self.zero_inflate)}s{self.scratch_prob:g}"
                f"b{self.mean_calib_batter:g}p{self.mean_calib_pitcher:g}"
                f"r{int(self.grid_mean_rescale)}")

    def describe(self) -> str:
        on = (self.zero_inflate or self.mean_calib_batter != 1.0
              or self.mean_calib_pitcher != 1.0 or self.grid_mean_rescale)
        return (f"{'on' if on else 'off'} (zero_inflate={self.zero_inflate}, "
                f"scratch={self.scratch_prob:g}, batter={self.mean_calib_batter:g}, "
                f"pitcher={self.mean_calib_pitcher:g}, "
                f"grid_mean_rescale={self.grid_mean_rescale})")


@dataclass
class SlateInputs:
    pool: object
    players_df: object
    sim_results: object
    slate_dir: Path
    n_sims: int
    seed: int
    calibration: SimCalibration


def build_slate_inputs(
    slate_dir: Path,
    n_sims: int = 25_000,
    seed: int = 42,
    calibration: Optional[SimCalibration] = None,
    sim_cache_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> SlateInputs:
    """Ingest one slate directory and simulate.

    `slate_dir` holds DKSalaries.csv plus the SaberSim export pair
    (lineups_*.csv + its companion projections file) -- true of both
    `data/raw` on a live slate and any `archive/MMDDYYYY`.

    `calibration` defaults to whatever config.yaml says, so this path matches
    the pipeline. Pass an explicit SimCalibration only to override deliberately.
    """
    from src.api import external_pool as ep
    from src.api.pipeline import PipelineRunner
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    from src.models.copula import EmpiricalCopula
    from src.simulation.engine import SimulationEngine
    from src.simulation.results import SimulationResults

    slate_dir = Path(slate_dir)
    cfg = yaml.safe_load(open(config_path or (PROJECT_ROOT / "config.yaml")))
    calib = calibration if calibration is not None else SimCalibration.from_config(cfg)

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
        cache_path = sim_cache_dir / f"{slate_dir.name}_{n_sims}_{seed}_{calib.cache_key()}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            return SlateInputs(
                pool=pool, players_df=players_df,
                sim_results=SimulationResults(
                    player_ids=[int(p) for p in z["player_ids"]],
                    results_matrix=z["results_matrix"],
                ),
                slate_dir=slate_dir, n_sims=n_sims, seed=seed, calibration=calib,
            )

    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    grids = ep.build_quantile_grids(
        proj_ext,
        zero_inflate=calib.zero_inflate, scratch_prob=calib.scratch_prob,
        mean_calib_batter=calib.mean_calib_batter,
        mean_calib_pitcher=calib.mean_calib_pitcher,
        rescale_to_file_mean=calib.grid_mean_rescale,
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
                       calibration=calib)
