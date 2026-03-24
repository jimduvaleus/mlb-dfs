
import numpy as np
from scipy.stats import norm

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
