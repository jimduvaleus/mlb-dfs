"""Tests for DeterminantPortfolioSelector."""
import numpy as np
import pytest

from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
from src.optimization.lineup import Lineup


def _make_lineup(player_ids: list[int]) -> Lineup:
    return Lineup(player_ids=player_ids)


def _make_selector(robust_payout: np.ndarray, candidates: list[Lineup], portfolio_size: int, risk: float = 5.0):
    return DeterminantPortfolioSelector(
        robust_payout=robust_payout,
        candidates=candidates,
        portfolio_size=portfolio_size,
        risk=risk,
    )


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _identity_payout(n_lineups: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Produce perfectly uncorrelated +EV payout rows (each lineup's return is independent)."""
    base = rng.standard_normal((n_lineups, n_sims)).astype(np.float32)
    # Shift so every lineup has positive mean EV
    base += 2.0
    return base


def _correlated_payout(base: np.ndarray, n_copies: int, noise: float = 0.01) -> np.ndarray:
    """Return base stacked with n_copies nearly identical to base (very high correlation)."""
    rng = np.random.default_rng(99)
    copies = base + rng.standard_normal((n_copies, base.shape[1])).astype(np.float32) * noise
    return np.vstack([base, copies])


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #

def test_returns_requested_portfolio_size():
    rng = np.random.default_rng(0)
    n_lineups, n_sims = 50, 300
    payout = _identity_payout(n_lineups, n_sims, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(n_lineups)]
    sel = _make_selector(payout, candidates, portfolio_size=10)
    result = sel.select()
    assert len(result) == 10


def test_no_duplicates():
    rng = np.random.default_rng(1)
    n_lineups, n_sims = 40, 200
    payout = _identity_payout(n_lineups, n_sims, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(n_lineups)]
    sel = _make_selector(payout, candidates, portfolio_size=15)
    result = sel.select()
    pid_sets = [frozenset(lu.player_ids) for lu, _ in result]
    assert len(pid_sets) == len(set(pid_sets))


def test_empty_when_no_positive_ev():
    """All candidates with negative mean EV → empty portfolio."""
    n_lineups, n_sims = 20, 100
    rng = np.random.default_rng(2)
    payout = rng.standard_normal((n_lineups, n_sims)).astype(np.float32) - 5.0  # all -EV
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(n_lineups)]
    sel = _make_selector(payout, candidates, portfolio_size=5)
    result = sel.select()
    assert result == []


def test_caps_at_pool_size():
    """Requesting more lineups than +EV candidates returns only the pool."""
    rng = np.random.default_rng(3)
    n_lineups, n_sims = 5, 200
    payout = _identity_payout(n_lineups, n_sims, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(n_lineups)]
    sel = _make_selector(payout, candidates, portfolio_size=20)
    result = sel.select()
    assert len(result) <= n_lineups


def test_perfect_correlation_gets_low_det_score():
    """Near-duplicate lineups score near zero on DEn, so only EV drives selection.

    With risk=10 (EVw=0.9, DEw=0.1) and one very high EV lineup plus many
    near-duplicates, the near-duplicates may still enter but their DEn is near 0.
    With risk=0 (DEw=0.9) the near-duplicates should rarely be picked over
    a genuinely independent lineup.
    """
    rng = np.random.default_rng(4)
    n_sims = 500

    # One high-EV independent lineup
    high_ev = rng.standard_normal((1, n_sims)).astype(np.float32) + 5.0

    # Many near-duplicates of high_ev (very high correlation, same EV level)
    near_dup = high_ev + rng.standard_normal((10, n_sims)).astype(np.float32) * 0.001

    # A few genuinely independent lower-EV lineups
    independent = rng.standard_normal((5, n_sims)).astype(np.float32) + 2.0

    payout = np.vstack([high_ev, near_dup, independent])
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(len(payout))]

    # risk=0: diversity-focused — should prefer independent lineups over near-duplicates
    sel_diverse = _make_selector(payout, candidates, portfolio_size=5, risk=0)
    result_diverse = sel_diverse.select()
    # The 5 independent lineups (indices 0, 11-15) should dominate over near-duplicates
    selected_idx = {candidates.index(lu) for lu, _ in result_diverse}
    n_near_dup_selected = sum(1 <= i <= 10 for i in selected_idx)
    # Expect at most 1-2 near-duplicates to slip through (scoring rounding)
    assert n_near_dup_selected <= 2, (
        f"Too many near-duplicates selected at risk=0: {n_near_dup_selected}"
    )


def test_risk_weights():
    """EVw/DEw formula: risk=1 → EVw=0.05, risk=3 → EVw=0.25, risk=5 → EVw=0.45."""
    rng = np.random.default_rng(5)
    payout = _identity_payout(20, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(20)]

    for risk, expected_evw in [(1, 0.05), (3, 0.25), (5, 0.45)]:
        sel = DeterminantPortfolioSelector(payout, candidates, portfolio_size=3, risk=risk)
        assert abs(sel._evw - expected_evw) < 1e-9, f"risk={risk}: EVw={sel._evw}, expected {expected_evw}"
        assert abs(sel._dew - (1.0 - expected_evw)) < 1e-9


def test_progress_callback_called():
    rng = np.random.default_rng(6)
    payout = _identity_payout(20, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(20)]
    calls = []
    sel = _make_selector(payout, candidates, portfolio_size=5)
    sel.select(progress_cb=lambda d: calls.append(d))
    assert len(calls) == 5
    assert calls[0]["step"] == 1
    assert calls[-1]["step"] == 5
    for d in calls:
        assert 0.0 <= d["partial_var"] <= 1.0
        assert d["n_remaining"] >= 0


def test_stop_check_respected():
    rng = np.random.default_rng(7)
    payout = _identity_payout(30, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(30)]
    call_count = {"n": 0}

    def stop_after_three():
        call_count["n"] += 1
        return call_count["n"] > 3

    sel = _make_selector(payout, candidates, portfolio_size=20)
    result = sel.select(stop_check=stop_after_three)
    assert len(result) < 20
