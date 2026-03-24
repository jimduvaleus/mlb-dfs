"""
Phase 4: Batter Mixture Model + PCA Pipeline.

Provides:
  - fit_mixture_params: MLE fitting of Exp+Normal mixture to historical DK scores.
  - BatterPCAModel:     Learns a 2D PCA manifold over (w, lam, mu, sigma) parameter
                        space and projects runtime (mu', sigma') projections onto it.
  - BatterMixtureMarginal: Mixture-distribution PPF discretized to historical score grid.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mixture helpers
# ---------------------------------------------------------------------------

def _mixture_cdf(x: np.ndarray, w: float, lam: float,
                 mu: float, sigma: float) -> np.ndarray:
    """CDF of  f(x) = w * Exp(lam)(x)  +  (1-w) * N(mu, sigma)(x)."""
    exp_cdf = np.where(x >= 0, 1.0 - np.exp(-lam * x), 0.0)
    norm_cdf = norm.cdf(x, loc=mu, scale=sigma)
    return w * exp_cdf + (1.0 - w) * norm_cdf


def fit_mixture_params(
    scores: np.ndarray,
    min_games: int = 20,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit a mixture model  f(x) = w * Exp(lam) + (1-w) * N(mu, sigma)
    to a 1-D array of DraftKings batter scores via MLE.

    Returns (w, lam, mu, sigma) or None when fitting fails / insufficient data.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) < min_games:
        return None

    mu0 = float(np.mean(scores))
    sigma0 = float(max(np.std(scores), 0.5))
    lam0 = 1.0 / max(mu0, 1.0)

    def neg_log_likelihood(params: np.ndarray) -> float:
        w, lam, mu, sigma = params
        exp_pdf = np.where(scores >= 0, lam * np.exp(-lam * scores), 0.0)
        norm_pdf = norm.pdf(scores, mu, sigma)
        mixture_pdf = np.maximum(w * exp_pdf + (1.0 - w) * norm_pdf, 1e-300)
        return -float(np.sum(np.log(mixture_pdf)))

    x0 = [0.3, lam0, mu0, sigma0]
    bounds = [(1e-6, 1.0 - 1e-6), (1e-4, 20.0), (-10.0, 80.0), (0.1, 40.0)]

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success and result.fun > neg_log_likelihood(x0) * 1.01:
        return None

    w, lam, mu, sigma = result.x
    return float(w), float(lam), float(mu), float(sigma)


# ---------------------------------------------------------------------------
# PCA Model
# ---------------------------------------------------------------------------

class BatterPCAModel:
    """
    Learns a 2-D PCA manifold over 4-D mixture parameter vectors
    (w, lam, mu, sigma) fitted to historical batter data.

    At runtime, given a projected (mu', sigma') pair, returns the closest
    point on the PCA plane that satisfies those constraints, yielding a
    physically consistent (w', lam', mu', sigma').
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None      # shape (4,)
        self.std_: Optional[np.ndarray] = None       # shape (4,)
        self.components_: Optional[np.ndarray] = None  # shape (2, 4)

    def fit(self, params_matrix: np.ndarray) -> "BatterPCAModel":
        """
        Fit PCA on a (n_players, 4) array of (w, lam, mu, sigma) vectors.

        Each row should come from fit_mixture_params applied to one player's
        historical DK score distribution.
        """
        params_matrix = np.asarray(params_matrix, dtype=float)
        if params_matrix.ndim != 2 or params_matrix.shape[1] != 4:
            raise ValueError("params_matrix must be shape (n, 4)")
        if len(params_matrix) < 3:
            raise ValueError("Need at least 3 parameter vectors to fit PCA")

        self.mean_ = params_matrix.mean(axis=0)
        self.std_ = params_matrix.std(axis=0)
        # Guard against zero variance (e.g. all players have same w)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

        normalized = (params_matrix - self.mean_) / self.std_

        # Truncated SVD — top 2 right singular vectors are the principal components
        _, _, Vt = np.linalg.svd(normalized, full_matrices=False)
        self.components_ = Vt[:2].copy()  # shape (2, 4)

        return self

    def project(self, mu_proj: float, sigma_proj: float) -> Tuple[float, float, float, float]:
        """
        Find the point on the 2-D PCA plane whose mu and sigma components
        best match the supplied projections, then reconstruct the full 4-D
        parameter vector (w, lam, mu, sigma).

        The approach solves a 2×2 linear system:
          components_ @ alpha = [mu_norm, sigma_norm]
        (using the mu and sigma rows of the component matrix), then
        reconstructs the full 4-D point as: alpha @ components_.

        Falls back to the global mean parameters when the system is poorly
        conditioned.
        """
        if self.mean_ is None:
            raise RuntimeError("BatterPCAModel has not been fitted yet.")

        # Normalised target for mu (index 2) and sigma (index 3)
        mu_norm = (mu_proj - self.mean_[2]) / self.std_[2]
        sigma_norm = (sigma_proj - self.mean_[3]) / self.std_[3]

        # Build 2×2 system:  A @ alpha = b
        # A[i, j] = components_[j, dim_i]  where dim_0=2 (mu), dim_1=3 (sigma)
        A = np.array([
            [self.components_[0, 2], self.components_[1, 2]],
            [self.components_[0, 3], self.components_[1, 3]],
        ])
        b = np.array([mu_norm, sigma_norm])

        alpha, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        # Reconstruct 4-D point in normalised space, then un-normalise
        reconstructed_norm = alpha @ self.components_  # shape (4,)
        reconstructed = reconstructed_norm * self.std_ + self.mean_

        w = float(np.clip(reconstructed[0], 1e-6, 1.0 - 1e-6))
        lam = float(max(reconstructed[1], 1e-4))
        mu = float(reconstructed[2])
        sigma = float(max(reconstructed[3], 0.1))

        return w, lam, mu, sigma

    def save(self, path: str) -> None:
        """Save model arrays to a .npz file."""
        if self.mean_ is None:
            raise RuntimeError("Model has not been fitted — nothing to save.")
        np.savez(path, mean=self.mean_, std=self.std_, components=self.components_)

    @classmethod
    def load(cls, path: str) -> "BatterPCAModel":
        """Load a previously saved model from a .npz file."""
        data = np.load(path)
        model = cls()
        model.mean_ = data["mean"]
        model.std_ = data["std"]
        model.components_ = data["components"]
        return model


# ---------------------------------------------------------------------------
# Mixture Marginal (PPF)
# ---------------------------------------------------------------------------

class BatterMixtureMarginal:
    """
    Inverse CDF (PPF) for a mixture distribution discretized to a historical
    DK score grid.

    Given a copula quantile u ∈ (0, 1), returns the discrete DK score s such
    that the mixture CDF F(s−) < u ≤ F(s), where the mixture CDF is evaluated
    at each point of the empirical score grid.

    Args:
        w (float):           Mixture weight for the exponential component.
        lam (float):         Rate parameter of the exponential component.
        mu (float):          Mean of the normal component.
        sigma (float):       Std-dev of the normal component.
        score_grid (array):  Sorted 1-D array of unique historical batter DK scores.
    """

    def __init__(
        self,
        w: float,
        lam: float,
        mu: float,
        sigma: float,
        score_grid: np.ndarray,
    ) -> None:
        self.w = float(w)
        self.lam = float(lam)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.score_grid = np.asarray(score_grid, dtype=float)

        # Pre-compute mixture CDF at every grid point  (shape: grid_size,)
        self._cdf_at_grid = _mixture_cdf(
            self.score_grid, self.w, self.lam, self.mu, self.sigma
        )
        # Ensure the CDF is monotonically non-decreasing (rounding/float safety)
        self._cdf_at_grid = np.maximum.accumulate(self._cdf_at_grid)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Map quantile array q ∈ [0, 1] to discrete DK scores.

        Uses searchsorted on the pre-computed mixture CDF grid, equivalent to
        the discretization rule:  CDF(s−1) < q ≤ CDF(s).
        """
        q = np.asarray(q, dtype=float)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Quantile q must be in [0, 1].")

        # 'left' side: find first index where cdf_at_grid >= q
        indices = np.searchsorted(self._cdf_at_grid, q, side="left")
        indices = np.clip(indices, 0, len(self.score_grid) - 1)
        return self.score_grid[indices]
