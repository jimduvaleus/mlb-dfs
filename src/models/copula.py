import logging

import pandas as pd
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class EmpiricalCopula:
    """
    An empirical copula implementation for MLB DFS simulations.

    The copula captures the joint dependency structure of player performances
    (quantiles) across a game-team (9 batters + 1 opposing pitcher).

    Rows come in pairs: each historical game contributes one row per team.
    Sampling whole games (both rows together) preserves the dependence between
    the two halves of a game — a team's batters and its own pitcher (who lives
    in the opposite row), and the shared game-level run environment.
    """

    def __init__(self, copula_path: str):
        """
        Initialize the copula by loading historical quantile data.

        Args:
            copula_path (str): Path to the empirical_copula.parquet file.
        """
        self.copula_data = pd.read_parquet(copula_path)
        self.n_observations = len(self.copula_data)
        self._paired_games = self._build_paired_games()

    def _build_paired_games(self) -> Optional[np.ndarray]:
        """
        Build a (n_games, 2, 10) array of quantile-row pairs, one pair per
        historical game where both team rows survived copula construction.

        Returns None when the data has no (game_id, team_id) MultiIndex
        (e.g., legacy artifacts), in which case game-level sampling is
        unavailable and callers fall back to independent row sampling.
        """
        idx = self.copula_data.index
        if not isinstance(idx, pd.MultiIndex) or 'game_id' not in (idx.names or []):
            return None

        sizes = self.copula_data.groupby(level='game_id').size()
        paired_ids = sizes[sizes == 2].index
        n_unpaired = int((sizes != 2).sum())
        if n_unpaired:
            logger.info(
                "Copula: %d games lack a complete row pair and are excluded "
                "from game-level sampling (%d paired games available).",
                n_unpaired, len(paired_ids),
            )
        if len(paired_ids) == 0:
            return None

        game_level = self.copula_data.index.get_level_values('game_id')
        sub = self.copula_data[game_level.isin(paired_ids)].sort_index()
        return sub.values.reshape(-1, 2, sub.shape[1])

    def sample(self, n_sims: int, context_filter: Optional[Dict] = None) -> np.ndarray:
        """
        Vectorized sampling from the empirical copula.

        Args:
            n_sims (int): Number of joint quantile vectors to sample.
            context_filter (Optional[Dict]): Future hook for stratification (e.g., {'venue': 'Coors Field'}).

        Returns:
            np.ndarray: A matrix of shape (n_sims, 10) where each row is a joint
                        quantile vector for [Slot 1, ..., Slot 9, Opposing Pitcher].
        """
        # Current implementation: simple bootstrap sampling from the empirical distribution
        # Stratification can be implemented here by filtering self.copula_data

        target_df = self.copula_data
        if context_filter:
            # Placeholder for stratification: filter data based on context
            # Example: if context_filter = {'is_high_total': True}
            # target_df = self.copula_data[self.copula_data['is_high_total'] == True]
            pass

        if len(target_df) == 0:
            raise ValueError("No data available for the given context_filter.")

        # Randomly select row indices with replacement
        indices = np.random.choice(len(target_df), size=n_sims, replace=True)

        # Return as a NumPy array for vectorized processing in the simulation engine
        return target_df.iloc[indices].values

    def sample_games(self, n_sims: int) -> np.ndarray:
        """
        Vectorized sampling of whole games from the empirical copula.

        Each draw bootstraps one historical game and returns both of its team
        rows, preserving cross-row (within-game) dependence.

        Returns:
            np.ndarray: Array of shape (n_sims, 2, 10). [:, 0, :] and [:, 1, :]
                        are the joint quantile vectors of the two opposing
                        team units of the same sampled game. Team order is
                        randomized per draw so the historical row ordering
                        (alphabetical team_id) imparts no systematic bias.
        """
        if self._paired_games is None:
            # No pairing information available — degrade to two independent draws.
            return np.stack([self.sample(n_sims), self.sample(n_sims)], axis=1)

        indices = np.random.choice(len(self._paired_games), size=n_sims, replace=True)
        pairs = self._paired_games[indices]  # fancy indexing copies, safe to mutate
        flip = np.random.random(n_sims) < 0.5
        pairs[flip] = pairs[flip][:, ::-1, :]
        return pairs
