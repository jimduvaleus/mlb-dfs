import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.models.copula import EmpiricalCopula
from src.models.marginals import GaussianMarginal
from src.simulation.results import SimulationResults

class SimulationEngine:
    """
    Core simulation engine for MLB DFS.
    
    Generates joint player performances by sampling from an empirical copula
    and applying inverse CDF (Gaussian marginals) to each player's projection.
    """
    
    def __init__(self, copula: EmpiricalCopula, players_df: pd.DataFrame):
        """
        Initialize the engine with a copula model and a slate of players.
        
        Args:
            copula (EmpiricalCopula): Initialized copula model.
            players_df (pd.DataFrame): DataFrame containing players and their projections.
                                        Expected columns:
                                        - player_id: Unique identifier for the player.
                                        - team: Player's team abbreviation.
                                        - opponent: Opponent's team abbreviation.
                                        - slot: Projected batting slot (1-9) or pitcher slot (10).
                                        - mean: Projected mean DraftKings points.
                                        - std_dev: Projected standard deviation of points.
        """
        self.copula = copula
        self.players_df = players_df.copy().reset_index(drop=True)
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
                
                # Apply the Gaussian Inverse CDF to the quantiles
                marginal = GaussianMarginal(player['mean'], player['std_dev'])
                simulated_points = marginal.ppf(q)
                
                # DFS points are typically non-negative
                simulated_points = np.maximum(simulated_points, 0)
                
                # Store the results in the overall matrix
                # idx is the row index in self.players_df
                all_simulated_points[:, idx] = simulated_points
                
        # Wrap the results in the SimulationResults container
        return SimulationResults(self.players_df['player_id'].tolist(), all_simulated_points)
