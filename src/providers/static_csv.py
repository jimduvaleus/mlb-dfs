import pandas as pd
from typing import Tuple
from src.providers.base import ProjectionProvider

class StaticCSVProvider(ProjectionProvider):
    def __init__(self, csv_filepath: str):
        self.projections = self._load_projections(csv_filepath)

    def _load_projections(self, csv_filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_filepath)
            if not all(col in df.columns for col in ['player_id', 'mu', 'sigma']):
                raise ValueError("CSV must contain 'player_id', 'mu', and 'sigma' columns.")
            df = df.set_index('player_id')
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Projection CSV file not found at {csv_filepath}")
        except ValueError:
            raise # Re-raise the original ValueError for missing columns
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred loading projections from CSV: {e}")

    def get_projections(self, player_id: str) -> Tuple[float, float]:
        if player_id not in self.projections.index:
            raise ValueError(f"Player ID '{player_id}' not found in projections.")
        
        player_data = self.projections.loc[player_id]
        mu = float(player_data['mu'])
        sigma = float(player_data['sigma'])

        return mu, sigma
