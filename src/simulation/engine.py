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
        
        for (team, opponent), unit_group in units:
            print(f"Simulating unit: {team} vs {opponent}")
            print(f"Players in unit: {unit_group['player_id'].tolist()}")
            print(f"Slots in unit: {unit_group['slot'].tolist()}")
            # Sample joint quantiles from the copula for this 10-player unit
            # Each sample is a (10,) vector of quantiles.
            # sampled_quantiles shape: (n_sims, 10)
            sampled_quantiles = self.copula.sample(n_sims)
            
            for idx, player in unit_group.iterrows():
                slot = int(player['slot'])
                
                # Check if the slot is within the copula's 1-10 range
                if not (1 <= slot <= 10):
                    # For non-standard slots, we can fallback to independent sampling (not implemented here)
                    continue
                
                # Extract the sampled quantiles for this player's specific slot
                # Slot is 1-indexed, so we subtract 1 for 0-based NumPy indexing.
                q = sampled_quantiles[:, slot - 1]
                
                # Choose marginal: mixture for batters (slot 1-9) when a PCA
                # model is available, Gaussian otherwise and for pitchers.
                is_batter = (1 <= slot <= 9)
                if is_batter and self.batter_pca_model is not None:
                    w, lam, mu, sigma = self.batter_pca_model.project(
                        float(player['mean']), float(player['std_dev'])
                    )
                    # Safety net: degenerate mixture params produce a
                    # near-constant PPF; fall back to Gaussian.
                    if w > 0.99 or lam < 0.01:
                        import logging
                        logging.getLogger(__name__).warning(
                            "Degenerate mixture for player %s (w=%.4f, lam=%.4f); "
                            "falling back to Gaussian", player['player_id'], w, lam
                        )
                        marginal = GaussianMarginal(player['mean'], player['std_dev'])
                    else:
                        marginal = BatterMixtureMarginal(w, lam, mu, sigma, self.score_grid)
                else:
                    marginal = GaussianMarginal(player['mean'], player['std_dev'])
                simulated_points = marginal.ppf(q)

                # Batters cannot score negative in DK; pitchers can (earned runs)
                if is_batter:
                    simulated_points = np.maximum(simulated_points, 0)
                
                # Store the results in the overall matrix
                # idx is the row index in self.players_df
                all_simulated_points[:, idx] = simulated_points
                
        # Wrap the results in the SimulationResults container
        return SimulationResults(self.players_df['player_id'].tolist(), all_simulated_points)
