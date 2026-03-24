"""
Tests for Phase 4: Batter Mixture Model + PCA Pipeline.
"""

import os
import tempfile

import numpy as np
import pytest

from src.models.batter_model import (
    BatterMixtureMarginal,
    BatterPCAModel,
    _mixture_cdf,
    fit_mixture_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_score_grid():
    """Minimal integer DK score grid."""
    return np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float)


@pytest.fixture
def realistic_scores(rng):
    """Simulate realistic batter DK scores: many zeros, right-skewed."""
    zeros = np.zeros(60)
    low = rng.uniform(1, 6, size=80)
    mid = rng.uniform(6, 12, size=40)
    high = rng.uniform(15, 40, size=20)
    return np.concatenate([zeros, low, mid, high])


@pytest.fixture
def params_matrix(rng):
    """Fake (n_players, 4) parameter matrix for PCA fitting."""
    n = 50
    w = rng.uniform(0.1, 0.5, size=n)
    lam = rng.uniform(0.05, 0.5, size=n)
    mu = rng.uniform(3.0, 15.0, size=n)
    sigma = rng.uniform(2.0, 8.0, size=n)
    return np.column_stack([w, lam, mu, sigma])


# ---------------------------------------------------------------------------
# _mixture_cdf tests
# ---------------------------------------------------------------------------

class TestMixtureCDF:
    def test_cdf_at_minus_inf_is_zero(self):
        result = _mixture_cdf(np.array([-1000.0]), 0.3, 0.2, 7.0, 3.0)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_cdf_at_plus_inf_is_one(self):
        result = _mixture_cdf(np.array([1000.0]), 0.3, 0.2, 7.0, 3.0)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_cdf_is_monotone(self):
        x = np.linspace(-5, 50, 200)
        cdf = _mixture_cdf(x, 0.3, 0.2, 7.0, 3.0)
        assert np.all(np.diff(cdf) >= -1e-12), "CDF must be non-decreasing"

    def test_cdf_at_zero_uses_exp_part(self):
        """At x=0, the exponential CDF is 0 so only the normal contributes."""
        cdf = _mixture_cdf(np.array([0.0]), w=0.5, lam=0.1, mu=0.0, sigma=1.0)
        # Normal(0,1).cdf(0) = 0.5, so mixture = 0.5*0 + 0.5*0.5 = 0.25
        assert cdf[0] == pytest.approx(0.25, abs=1e-6)

    def test_vectorised_output_shape(self):
        x = np.linspace(0, 30, 100)
        cdf = _mixture_cdf(x, 0.3, 0.2, 7.0, 3.0)
        assert cdf.shape == (100,)


# ---------------------------------------------------------------------------
# fit_mixture_params tests
# ---------------------------------------------------------------------------

class TestFitMixtureParams:
    def test_returns_four_tuple(self, realistic_scores):
        result = fit_mixture_params(realistic_scores, min_games=10)
        assert result is not None
        assert len(result) == 4

    def test_params_in_valid_ranges(self, realistic_scores):
        w, lam, mu, sigma = fit_mixture_params(realistic_scores, min_games=10)
        assert 0 < w < 1
        assert lam > 0
        assert sigma > 0

    def test_returns_none_for_too_few_games(self):
        scores = np.array([0.0, 5.0, 10.0])
        result = fit_mixture_params(scores, min_games=20)
        assert result is None

    def test_fitted_mu_is_close_to_data_mean(self, rng):
        """Mixture mean ≈ w/lam + (1-w)*mu, roughly near the data mean."""
        scores = rng.normal(loc=8.0, scale=3.0, size=200).clip(min=0)
        result = fit_mixture_params(scores, min_games=50)
        assert result is not None
        w, lam, mu, sigma = result
        mixture_mean = w / lam + (1 - w) * mu
        assert abs(mixture_mean - 8.0) < 5.0  # generous tolerance for MLE

    def test_deterministic_for_same_input(self, realistic_scores):
        r1 = fit_mixture_params(realistic_scores.copy(), min_games=10)
        r2 = fit_mixture_params(realistic_scores.copy(), min_games=10)
        assert r1 == r2


# ---------------------------------------------------------------------------
# BatterPCAModel tests
# ---------------------------------------------------------------------------

class TestBatterPCAModel:
    def test_fit_returns_self(self, params_matrix):
        model = BatterPCAModel()
        result = model.fit(params_matrix)
        assert result is model

    def test_components_shape(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        assert model.components_.shape == (2, 4)

    def test_mean_std_shape(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        assert model.mean_.shape == (4,)
        assert model.std_.shape == (4,)

    def test_fit_requires_at_least_3_rows(self):
        with pytest.raises(ValueError, match="at least 3"):
            BatterPCAModel().fit(np.random.randn(2, 4))

    def test_fit_requires_4_columns(self):
        with pytest.raises(ValueError, match="shape"):
            BatterPCAModel().fit(np.random.randn(10, 3))

    def test_project_returns_four_tuple(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        result = model.project(mu_proj=8.0, sigma_proj=4.0)
        assert len(result) == 4

    def test_project_w_in_unit_interval(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        w, lam, mu, sigma = model.project(8.0, 4.0)
        assert 0 < w < 1

    def test_project_lam_positive(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        _, lam, _, _ = model.project(8.0, 4.0)
        assert lam > 0

    def test_project_sigma_positive(self, params_matrix):
        model = BatterPCAModel().fit(params_matrix)
        _, _, _, sigma = model.project(8.0, 4.0)
        assert sigma > 0

    def test_project_mu_reflects_input(self, params_matrix):
        """
        The reconstructed mu should be close to the input mu_proj because
        PCA is constrained to satisfy the mu and sigma targets.
        """
        model = BatterPCAModel().fit(params_matrix)
        _, _, mu, _ = model.project(mu_proj=10.0, sigma_proj=5.0)
        # The PCA projection solves for the exact mu/sigma match in normalized
        # space; check it rounds back reasonably
        assert abs(mu - 10.0) < 3.0

    def test_raises_before_fitting(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            BatterPCAModel().project(8.0, 4.0)

    def test_save_load_roundtrip(self, params_matrix, tmp_path):
        model = BatterPCAModel().fit(params_matrix)
        path = str(tmp_path / "pca.npz")
        model.save(path)
        loaded = BatterPCAModel.load(path)

        np.testing.assert_array_almost_equal(model.mean_, loaded.mean_)
        np.testing.assert_array_almost_equal(model.std_, loaded.std_)
        np.testing.assert_array_almost_equal(model.components_, loaded.components_)

    def test_save_raises_before_fitting(self, tmp_path):
        with pytest.raises(RuntimeError, match="not been fitted"):
            BatterPCAModel().save(str(tmp_path / "pca.npz"))


# ---------------------------------------------------------------------------
# BatterMixtureMarginal tests
# ---------------------------------------------------------------------------

class TestBatterMixtureMarginal:
    def test_ppf_returns_values_in_score_grid(self, simple_score_grid, rng):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        q = rng.uniform(0, 1, size=500)
        scores = marginal.ppf(q)
        assert set(scores).issubset(set(simple_score_grid))

    def test_ppf_monotone(self, simple_score_grid):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        q = np.linspace(0, 1, 100)
        scores = marginal.ppf(q)
        assert np.all(np.diff(scores) >= 0), "PPF must be non-decreasing"

    def test_ppf_extremes(self, simple_score_grid):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        low = marginal.ppf(np.array([0.001]))
        high = marginal.ppf(np.array([0.999]))
        assert low[0] == simple_score_grid[0]
        assert high[0] == simple_score_grid[-1]

    def test_ppf_raises_on_invalid_quantile(self, simple_score_grid):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            marginal.ppf(np.array([1.5]))

    def test_ppf_shape_preserved(self, simple_score_grid, rng):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        q = rng.uniform(0, 1, size=(200,))
        assert marginal.ppf(q).shape == (200,)

    def test_higher_mu_shifts_distribution(self, simple_score_grid, rng):
        """Higher projected mu should shift the score distribution upward."""
        q = rng.uniform(0, 1, size=1000)
        low_marginal = BatterMixtureMarginal(0.3, 0.15, 3.0, 2.0, simple_score_grid)
        high_marginal = BatterMixtureMarginal(0.3, 0.05, 15.0, 5.0, simple_score_grid)
        assert high_marginal.ppf(q).mean() > low_marginal.ppf(q).mean()

    def test_cdf_at_grid_is_monotone(self, simple_score_grid):
        marginal = BatterMixtureMarginal(0.3, 0.1, 7.0, 3.0, simple_score_grid)
        assert np.all(np.diff(marginal._cdf_at_grid) >= 0)


# ---------------------------------------------------------------------------
# Integration: BatterPCAModel → BatterMixtureMarginal → ppf
# ---------------------------------------------------------------------------

class TestBatterModelIntegration:
    def test_end_to_end_ppf(self, params_matrix, simple_score_grid, rng):
        pca = BatterPCAModel().fit(params_matrix)
        w, lam, mu, sigma = pca.project(mu_proj=8.0, sigma_proj=4.0)
        marginal = BatterMixtureMarginal(w, lam, mu, sigma, simple_score_grid)

        q = rng.uniform(0, 1, size=500)
        scores = marginal.ppf(q)

        assert scores.shape == (500,)
        assert set(scores).issubset(set(simple_score_grid))
        assert scores.mean() >= 0

    def test_different_projections_yield_different_distributions(
        self, params_matrix, simple_score_grid, rng
    ):
        pca = BatterPCAModel().fit(params_matrix)
        q = rng.uniform(0, 1, size=2000)

        w1, l1, m1, s1 = pca.project(3.0, 2.0)
        w2, l2, m2, s2 = pca.project(15.0, 6.0)
        scores_low = BatterMixtureMarginal(w1, l1, m1, s1, simple_score_grid).ppf(q)
        scores_high = BatterMixtureMarginal(w2, l2, m2, s2, simple_score_grid).ppf(q)

        assert scores_high.mean() > scores_low.mean()
