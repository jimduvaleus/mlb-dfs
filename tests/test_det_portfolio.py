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
    """EVw/DEw formula: risk=1 → EVw=0.10, risk=3 → EVw=0.25, risk=5 → EVw=0.40."""
    rng = np.random.default_rng(5)
    payout = _identity_payout(20, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(20)]

    for risk, expected_evw in [(1, 0.1), (3, 0.25), (5, 0.4)]:
        sel = DeterminantPortfolioSelector(payout, candidates, portfolio_size=3, risk=risk)
        assert abs(sel._evw - expected_evw) < 1e-9, f"risk={risk}: EVw={sel._evw}, expected {expected_evw}"
        assert abs(sel._dew - (1.0 - expected_evw)) < 1e-9


def test_risk_weights_custom_base_max():
    """Custom evw_base/evw_max are linearly interpolated across risk 1-5."""
    rng = np.random.default_rng(5)
    payout = _identity_payout(20, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(20)]

    for risk, expected_evw in [(1, 0.2), (3, 0.5), (5, 0.8)]:
        sel = DeterminantPortfolioSelector(
            payout, candidates, portfolio_size=3, risk=risk, evw_base=0.2, evw_max=0.8,
        )
        assert abs(sel._evw - expected_evw) < 1e-9, f"risk={risk}: EVw={sel._evw}, expected {expected_evw}"


def test_ev_floor_default_is_point_two():
    """Default ev_floor of $0.20 culls candidates below that EV, matching the old hardcoded value."""
    rng = np.random.default_rng(8)
    n_sims = 200
    above = rng.standard_normal((3, n_sims)).astype(np.float32) + 5.0  # EV ~5.0
    below = rng.standard_normal((3, n_sims)).astype(np.float32) * 0.01 + 0.1  # EV ~0.1, below floor
    payout = np.vstack([above, below])
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(len(payout))]
    sel = _make_selector(payout, candidates, portfolio_size=10)
    result = sel.select()
    selected_idx = {candidates.index(lu) for lu, _ in result}
    assert selected_idx == {0, 1, 2}


def test_ev_floor_custom_value():
    """A lower ev_floor admits candidates the default $0.20 floor would have culled."""
    rng = np.random.default_rng(8)
    n_sims = 200
    above = rng.standard_normal((3, n_sims)).astype(np.float32) + 5.0
    below = rng.standard_normal((3, n_sims)).astype(np.float32) * 0.01 + 0.1
    payout = np.vstack([above, below])
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(len(payout))]
    sel = DeterminantPortfolioSelector(
        payout, candidates, portfolio_size=10, ev_floor=0.05,
    )
    result = sel.select()
    assert len(result) == 6


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
        assert 0.0 <= d["distance"] <= 1.0
        assert d["n_remaining"] >= 0


def test_score_is_linear_not_quadratic():
    """score = evw*EVn + dew*DEn (linear), not sqrt((evw*EVn)^2+(dew*DEn)^2).

    Engineered EVn/DEn pair where the two combination rules pick a
    different second lineup: candidate 1 has the higher remaining EV but
    is strongly correlated with the anchor (EVn=1.0, DEn=0.05); candidate 2
    has lower EV but is anti-correlated with the anchor (EVn=0.6, DEn=0.6).
    Linear scores candidate 2 higher (0.6 vs 0.525); the quadratic
    combination this replaced scored candidate 1 higher (0.501 vs 0.424) —
    this guards against reverting to it.
    """
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(3)]
    pool_idx = np.arange(3)
    pool_ev_vals = np.array([100.0, 10.0, 6.0])  # candidate 0 anchors (highest EV)
    corr_matrix = np.array([
        [1.0, 0.9, -0.2],
        [0.9, 1.0, 0.0],
        [-0.2, 0.0, 1.0],
    ], dtype=np.float32)
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=2,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert picked_idx == {0, 2}, f"expected anchor(0) + candidate 2, got {picked_idx}"


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
