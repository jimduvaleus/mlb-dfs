import pandas as pd
import numpy as np
import os
from typing import List

class SimulationResults:
    """
    Container for simulation results.
    
    Stores a matrix of simulated points for each player across all simulations.
    """
    
    def __init__(self, player_ids: List[int], results_matrix: np.ndarray):
        """
        Args:
            player_ids (List[int]): List of player IDs corresponding to columns in results_matrix.
            results_matrix (np.ndarray): Matrix of shape (n_sims, n_players) with simulated points.
        """
        self.player_ids = player_ids
        self.results_matrix = results_matrix
        self.n_sims, self.n_players = results_matrix.shape
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts results to a long-form DataFrame.
        Columns: [simulation_id, player_id, dk_points]
        """
        df = pd.DataFrame(self.results_matrix, columns=self.player_ids)
        df.index.name = 'simulation_id'
        df = df.reset_index().melt(id_vars='simulation_id', var_name='player_id', value_name='dk_points')
        return df
    
    def save_to_parquet(self, output_path: str):
        """
        Saves the results to a parquet file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.to_dataframe().to_parquet(output_path, index=False)
        print(f"Simulation results saved to {output_path}")

    def get_player_stats(self) -> pd.DataFrame:
        """
        Calculate summary statistics for each player.
        """
        df = pd.DataFrame(self.results_matrix, columns=self.player_ids)
        stats = df.agg(['mean', 'std', 'min', 'max', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), lambda x: np.percentile(x, 99)])
        stats.index = ['mean', 'std', 'min', 'max', 'p25', 'p75', 'p99']
        return stats.T
