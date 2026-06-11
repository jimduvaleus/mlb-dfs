
import numpy as np
from scipy.stats import norm


class EmpiricalQuantileMarginal:
    """Marginal defined by a precomputed quantile grid (e.g. P0..P100 of a
    market-implied Monte Carlo score distribution).

    ppf() linearly interpolates between grid points, so the simulated
    distribution reproduces the grid's full shape — skew, zero-inflation
    plateaus, and tail mass — rather than a parametric approximation.
    """

    def __init__(self, quantile_grid: np.ndarray):
        grid = np.asarray(quantile_grid, dtype=np.float64)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("quantile_grid must be a 1-D array with >= 2 points.")
        if np.any(np.diff(grid) < 0):
            raise ValueError("quantile_grid must be non-decreasing.")
        self.grid = grid
        self._q_points = np.linspace(0.0, 1.0, len(grid))

    def ppf(self, q: np.ndarray) -> np.ndarray:
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Quantile (q) must be between 0 and 1 inclusive.")
        return np.interp(q, self._q_points, self.grid)


class GaussianMarginal:
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("Standard deviation (sigma) must be positive.")
        self.mu = mu
        self.sigma = sigma

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse of CDF) at q.
        Args:
            q (np.ndarray): Lower tail probabilities.
        Returns:
            np.ndarray: The values at which the cumulative distribution function is equal to q.
        """
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Quantile (q) must be between 0 and 1 inclusive.")
        # Clip quantiles to avoid infinity at 0 and 1
        q_clipped = np.clip(q, 0.0001, 0.9999)
        return norm.ppf(q_clipped, loc=self.mu, scale=self.sigma)
