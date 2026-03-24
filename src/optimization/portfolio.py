"""
Portfolio construction for DraftKings MLB DFS.

Phase 5 implementation: iterative greedy lineup selection with simulation row
consumption. Each iteration optimizes the best lineup on the remaining (un-hit)
simulation rows, then removes those rows where the selected lineup already
cleared the target score so that subsequent lineups must cover different upside
scenarios.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.optimization.lineup import Lineup
from src.optimization.optimizer import BasinHoppingOptimizer
from src.simulation.results import SimulationResults

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Iterative greedy portfolio builder with simulation-row consumption.

    Algorithm
    ---------
    1. Optimize the best lineup L_i using only the *active* simulation rows.
    2. Identify the active rows where L_i's total >= target and mark them consumed.
    3. Repeat until ``portfolio_size`` lineups are collected or all rows are consumed.

    Each lineup's final score is reported against the *full* simulation matrix so
    that scores across lineups are directly comparable.

    Parameters
    ----------
    sim_results : SimulationResults
        The Monte Carlo results produced by ``SimulationEngine.simulate()``.
    players_df : pd.DataFrame
        Player pool with columns: player_id, position, salary, team, game.
    target : float
        DraftKings score threshold — same value used by the optimizer.
    portfolio_size : int
        Number of lineups to construct.
    n_chains : int
        Basin-hopping chains per lineup optimisation round (default 250).
    temperature : float
        Metropolis acceptance temperature (default 0.1).
    n_steps : int
        Perturbation steps per chain (default 100).
    n_workers : int
        Worker processes for parallel chains (1 = sequential).
    rng_seed : int, optional
        Base seed for reproducibility. Seed for round i is ``rng_seed + i``.
    """

    def __init__(
        self,
        sim_results: SimulationResults,
        players_df: pd.DataFrame,
        target: float,
        portfolio_size: int,
        n_chains: int = 250,
        temperature: float = 0.1,
        n_steps: int = 100,
        n_workers: int = 1,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.sim_results = sim_results
        self.players_df = players_df
        self.target = target
        self.portfolio_size = portfolio_size
        self._optimizer_kwargs = dict(
            n_chains=n_chains,
            temperature=temperature,
            n_steps=n_steps,
            n_workers=n_workers,
        )
        self._base_seed = rng_seed

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def construct(self) -> List[Tuple[Lineup, float]]:
        """Run greedy portfolio construction.

        Returns
        -------
        List[Tuple[Lineup, float]]
            One ``(lineup, score)`` pair per selected lineup in selection order.
            ``score`` is P(lineup_total >= target) over the *full* simulation
            matrix so values are comparable across all lineups in the portfolio.
        """
        portfolio: List[Tuple[Lineup, float]] = []
        full_matrix = self.sim_results.results_matrix  # shape (n_sims, n_players)
        col_map = {pid: i for i, pid in enumerate(self.sim_results.player_ids)}
        active_mask = np.ones(self.sim_results.n_sims, dtype=bool)

        for i in range(self.portfolio_size):
            n_active = int(active_mask.sum())
            if n_active == 0:
                logger.info(
                    "All simulation rows consumed after %d lineups; "
                    "stopping early (requested %d).",
                    len(portfolio),
                    self.portfolio_size,
                )
                break

            logger.info(
                "Optimizing lineup %d/%d — %d active simulation rows remaining.",
                i + 1,
                self.portfolio_size,
                n_active,
            )

            # Build a SimulationResults view over the active rows only.
            active_sim = SimulationResults(
                player_ids=self.sim_results.player_ids,
                results_matrix=full_matrix[active_mask],
            )

            seed = None if self._base_seed is None else self._base_seed + i
            optimizer = BasinHoppingOptimizer(
                sim_results=active_sim,
                players_df=self.players_df,
                target=self.target,
                rng_seed=seed,
                **self._optimizer_kwargs,
            )
            lineup, _ = optimizer.optimize()

            # Score against the full matrix for comparability.
            cols = [col_map[pid] for pid in lineup.player_ids]
            full_totals = full_matrix[:, cols].sum(axis=1)
            full_score = float((full_totals >= self.target).mean())

            portfolio.append((lineup, full_score))
            logger.info("  Lineup %d score (full): %.4f", i + 1, full_score)

            # Consume active rows where this lineup already hits the target.
            active_indices = np.where(active_mask)[0]
            active_totals = full_matrix[active_mask][:, cols].sum(axis=1)
            hit_mask = active_totals >= self.target
            consumed = int(hit_mask.sum())
            active_mask[active_indices[hit_mask]] = False
            logger.info(
                "  Consumed %d rows; %d remain.",
                consumed,
                int(active_mask.sum()),
            )

        return portfolio
