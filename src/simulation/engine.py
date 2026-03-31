import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.models.copula import EmpiricalCopula
from src.models.marginals import GaussianMarginal
from src.models.batter_model import BatterPCAModel, BatterMixtureMarginal
from src.simulation.results import SimulationResults

class SimulationEngine:
    """
    Core simulation engine for MLB DFS.

    Generates joint player performances by sampling from an empirical copula
    and applying inverse CDFs to each player's projection.

    When a BatterPCAModel and score_grid are supplied, batter slots (1-9) use
    the mixture-distribution marginal (Phase 4).  Pitcher slots (10) always use
    a Gaussian marginal.  Omitting the PCA model falls back to Gaussian for all
    players, preserving backward compatibility with Phases 1-3.
    """

    def __init__(
        self,
        copula: EmpiricalCopula,
        players_df: pd.DataFrame,
        batter_pca_model: Optional[BatterPCAModel] = None,
        score_grid: Optional[np.ndarray] = None,
    ):
        """
        Args:
            copula (EmpiricalCopula): Initialized copula model.
            players_df (pd.DataFrame): Players with columns:
                player_id, team, opponent, slot, mean, std_dev.
            batter_pca_model (BatterPCAModel | None): Fitted PCA model for
                mapping (mu, sigma) projections → mixture parameters.
                When None, Gaussian marginals are used for all slots.
            score_grid (np.ndarray | None): Sorted array of unique historical
                batter DK scores used for mixture PPF discretisation.
                Required when batter_pca_model is provided.
        """
        if batter_pca_model is not None and score_grid is None:
            raise ValueError("score_grid is required when batter_pca_model is provided.")

        self.copula = copula
        self.players_df = players_df.copy().reset_index(drop=True)
        self.batter_pca_model = batter_pca_model
        self.score_grid = score_grid
        self._validate_input()
        
    def _validate_input(self):
        """Basic validation for input DataFrame."""
        required_cols = ['player_id', 'team', 'opponent', 'slot', 'mean', 'std_dev']
        for col in required_cols:
            if col not in self.players_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
    def simulate(self, n_sims: int) -> SimulationResults:
        """
        Run vectorized simulations for all players in the slate.
        
        Args:
            n_sims (int): Number of simulations to run.
            
        Returns:
            SimulationResults: A container for the simulation results.
        """
        # results_matrix will store results for each player across n_sims
        # Shape: (n_sims, n_players)
        n_players = len(self.players_df)
        all_simulated_points = np.zeros((n_sims, n_players))
        
        # We group players into 10-player "units" to match the copula structure.
        # Each unit consists of the 9 batters from a team and the pitcher from their opponent.
        # We use (team, opponent) as the grouping key.
        
        # Note: A game between Team A and Team B will have two units:
        # Unit 1: (Team=A, Opponent=B) - Includes Team A batters and Team B pitcher.
        # Unit 2: (Team=B, Opponent=A) - Includes Team B batters and Team A pitcher.
        
        units = self.players_df.groupby(['team', 'opponent'])
        
        import logging
        _log = logging.getLogger(__name__)

        # Pre-compute marginals for every player so the simulation loop does
        # no per-player object construction or PCA projection work at runtime.
        marginals: List[Any] = [None] * n_players
        is_batter_flags: List[bool] = [False] * n_players
        for row_idx, player in self.players_df.iterrows():
            slot = int(player['slot'])
            is_batter = (1 <= slot <= 9)
            is_batter_flags[row_idx] = is_batter
            if is_batter and self.batter_pca_model is not None:
                w, lam, mu, sigma = self.batter_pca_model.project(
                    float(player['mean']), float(player['std_dev'])
                )
                if w > 0.99 or lam < 0.01:
                    _log.warning(
                        "Degenerate mixture for player %s (w=%.4f, lam=%.4f); "
                        "falling back to Gaussian", player['player_id'], w, lam
                    )
                    marginals[row_idx] = GaussianMarginal(player['mean'], player['std_dev'])
                else:
                    marginals[row_idx] = BatterMixtureMarginal(w, lam, mu, sigma, self.score_grid)
            else:
                marginals[row_idx] = GaussianMarginal(player['mean'], player['std_dev'])

        for (team, opponent), unit_group in units:
            # Sample joint quantiles from the copula for this 10-player unit
            # sampled_quantiles shape: (n_sims, 10)
            sampled_quantiles = self.copula.sample(n_sims)

            slots = unit_group['slot'].values
            indices = unit_group.index.values
            for i in range(len(indices)):
                slot = int(slots[i])
                if not (1 <= slot <= 10):
                    continue
                row_idx = indices[i]
                q = sampled_quantiles[:, slot - 1]
                simulated_points = marginals[row_idx].ppf(q)
                if is_batter_flags[row_idx]:
                    simulated_points = np.maximum(simulated_points, 0)
                all_simulated_points[:, row_idx] = simulated_points
                
        # Wrap the results in the SimulationResults container
        return SimulationResults(self.players_df['player_id'].tolist(), all_simulated_points)
