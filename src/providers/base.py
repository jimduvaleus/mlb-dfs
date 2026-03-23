from abc import ABC, abstractmethod
from typing import Tuple

class ProjectionProvider(ABC):
    @abstractmethod
    def get_projections(self, player_id: str) -> Tuple[float, float]:
        """
        Abstract method to get projection (mean and standard deviation) for a given player.

        Args:
            player_id (str): The unique identifier for the player.

        Returns:
            Tuple[float, float]: A tuple containing the mean (mu) and standard deviation (sigma) of the projection.
        """
        pass
