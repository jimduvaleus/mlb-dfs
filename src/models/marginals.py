
import numpy as np
from scipy.stats import norm

class GaussianMarginal:
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("Standard deviation (sigma) must be positive.")
        self.mu = mu
        self.sigma = sigma

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of CDF) at q.
        Args:
            q (float): Lower tail probability.
        Returns:
            float: The value at which the cumulative distribution function is equal to q.
        """
        if not (0 <= q <= 1):
            raise ValueError("Quantile (q) must be between 0 and 1 inclusive.")
        return norm.ppf(q, loc=self.mu, scale=self.sigma)
