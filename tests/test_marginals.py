import pytest
from src.models.marginals import GaussianMarginal
from scipy.stats import norm

def test_gaussian_marginal_init():
    marginal = GaussianMarginal(mu=10, sigma=2)
    assert marginal.mu == 10
    assert marginal.sigma == 2

def test_gaussian_marginal_init_invalid_sigma():
    with pytest.raises(ValueError, match=r"Standard deviation \(sigma\) must be positive."):
        GaussianMarginal(mu=10, sigma=0)
    with pytest.raises(ValueError, match=r"Standard deviation \(sigma\) must be positive."):
        GaussianMarginal(mu=10, sigma=-1)

def test_gaussian_marginal_ppf():
    marginal = GaussianMarginal(mu=10, sigma=2)
    
    # Test with known values
    assert marginal.ppf(0.5) == pytest.approx(10.0)
    assert marginal.ppf(norm.cdf(12, loc=10, scale=2)) == pytest.approx(12.0)
    assert marginal.ppf(norm.cdf(8, loc=10, scale=2)) == pytest.approx(8.0)

def test_gaussian_marginal_ppf_invalid_q():
    marginal = GaussianMarginal(mu=10, sigma=2)
    with pytest.raises(ValueError, match=r"Quantile \(q\) must be between 0 and 1 inclusive."):
        marginal.ppf(-0.1)
    with pytest.raises(ValueError, match=r"Quantile \(q\) must be between 0 and 1 inclusive."):
        marginal.ppf(1.1)
