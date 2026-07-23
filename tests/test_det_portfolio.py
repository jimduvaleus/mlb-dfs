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
    """EVw/DEw/HedgeW formula: risk=1 → EVw=0.10, risk=3 → EVw=0.25, risk=5 → EVw=0.40.

    DEw/HedgeW split the remaining (1-EVw) as 0.9/0.1, and the three
    weights always sum to 1.
    """
    rng = np.random.default_rng(5)
    payout = _identity_payout(20, 100, rng)
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(20)]

    for risk, expected_evw in [(1, 0.1), (3, 0.25), (5, 0.4)]:
        sel = DeterminantPortfolioSelector(payout, candidates, portfolio_size=3, risk=risk)
        diversity_w = 1.0 - expected_evw
        assert abs(sel._evw - expected_evw) < 1e-9, f"risk={risk}: EVw={sel._evw}, expected {expected_evw}"
        assert abs(sel._dew - 0.9 * diversity_w) < 1e-9
        assert abs(sel._hedge_w - 0.1 * diversity_w) < 1e-9
        assert abs(sel._evw + sel._dew + sel._hedge_w - 1.0) < 1e-9
        assert sel._hedge_w <= sel._dew


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

    Engineered EVn/DEn pair (DEn = 1 - r^2, both r positive, so the
    positive-redundancy distance and its predecessor agree with a single
    already-selected anchor) where the two combination rules pick a
    different second lineup: candidate 1 has the higher remaining EV but
    is strongly correlated with the anchor (r=sqrt(0.95) -> EVn=1.0,
    DEn=0.05); candidate 2 has lower EV but is only moderately correlated
    (r=sqrt(0.4) -> EVn=0.6, DEn=0.6). Linear scores candidate 2 higher
    (0.6 vs 0.525); the quadratic combination this replaced scored
    candidate 1 higher (0.501 vs 0.424) — this guards against reverting
    to it.
    """
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(3)]
    pool_idx = np.arange(3)
    pool_ev_vals = np.array([100.0, 10.0, 6.0])  # candidate 0 anchors (highest EV)
    r1, r2 = np.sqrt(0.95), np.sqrt(0.4)  # -> DEn = 1-r^2 = 0.05, 0.6
    corr_matrix = np.array([
        [1.0, r1, r2],
        [r1, 1.0, 0.0],
        [r2, 0.0, 1.0],
    ], dtype=np.float32)
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=2,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert picked_idx == {0, 2}, f"expected anchor(0) + candidate 2, got {picked_idx}"


def test_hedge_cannot_cancel_a_duplicate():
    """A candidate that duplicates one already-selected lineup and hedges
    another must NOT tie with a candidate that's merely moderately distant
    from both, even though a naive aggregate (sum/average) of the two
    relationships gives both the same 0.5 average distance.

    Two anchors (0, 1) are forced to be picked first by EV dominance. Of
    the remaining two candidates (equal EV, so DEn alone decides): X
    duplicates anchor 0 (r=1.0) and perfectly hedges anchor 1 (r=-1.0);
    Y is moderately correlated with both (r=0.5, r=-0.5) — chosen so a
    plain aggregate-sum distance ties X and Y at 0.5, but positive-
    redundancy distance does not: redundancy_X = max(1,0)^2 + max(-1,0)^2
    = 1.0 -> DEn=0.0 (correctly crushed); redundancy_Y = max(0.5,0)^2 +
    max(-0.5,0)^2 = 0.25 -> DEn=0.75 (correctly preferred).
    """
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(4)]
    pool_idx = np.arange(4)
    pool_ev_vals = np.array([100.0, 90.0, 10.0, 10.0])  # X=idx2, Y=idx3, equal EV
    corr_matrix = np.array([
        [1.0, 0.0, 1.0, 0.5],
        [0.0, 1.0, -1.0, -0.5],
        [1.0, -1.0, 1.0, 0.0],
        [0.5, -0.5, 0.0, 1.0],
    ], dtype=np.float32)
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=3,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert picked_idx == {0, 1, 3}, f"expected both anchors + Y(3), got {picked_idx}"


def test_one_strong_overlap_beats_two_moderate_ones():
    """A candidate with one strong overlap to an already-selected lineup
    must be preferred over one with two moderate overlaps to two different
    already-selected lineups, when both have equal EV — true incremental
    coverage against an (approximately) orthogonal existing portfolio is
    1 - sum(r^2), which penalizes concentrated overlap less than spread
    overlap of the same total squared magnitude only when the spread one's
    sum of squares is larger, as engineered here: P has r=0.6 to one
    already-selected lineup (redundancy=0.36 -> DEn=0.64); R has r=0.5 to
    two already-selected lineups (redundancy=0.5 -> DEn=0.5).
    """
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(4)]
    pool_idx = np.arange(4)
    pool_ev_vals = np.array([100.0, 90.0, 10.0, 10.0])  # P=idx2, R=idx3, equal EV
    corr_matrix = np.array([
        [1.0, 0.0, 0.6, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.6, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 1.0],
    ], dtype=np.float32)
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=3,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert picked_idx == {0, 1, 2}, f"expected both anchors + P(2), got {picked_idx}"


def test_hedge_breaks_ties_between_equally_nonredundant_candidates():
    """Among candidates that are otherwise equally non-redundant (DEn tied
    at 1.0) and equal EV, the one that hedges an already-selected lineup
    must be preferred — the HedgeN tie-breaker this test targets.

    Two anchors (0, 1) are forced to be picked first. Of the remaining two
    equal-EV candidates: H hedges anchor 1 (r=-0.5, redundancy still 0 since
    negatives don't count toward it -> DEn=1.0 same as N); N has zero
    correlation to both anchors (DEn=1.0). DEn alone can't distinguish them;
    HedgeN must.
    """
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(4)]
    pool_idx = np.arange(4)
    pool_ev_vals = np.array([100.0, 90.0, 10.0, 10.0])  # H=idx2, N=idx3, equal EV
    corr_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, -0.5, 0.0],
        [0.0, -0.5, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=3,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert picked_idx == {0, 1, 2}, f"expected both anchors + H(2), got {picked_idx}"


def test_hedge_bonus_cannot_rescue_a_duplicate_even_with_many_hedges():
    """A near-duplicate candidate must still lose even if it ALSO hedges
    many other already-selected lineups heavily -- the uncapped raw hedge
    sum would otherwise grow with however many lineups are hedged (mirror
    image of the dilution bug fixed for redundancy), potentially
    overwhelming hedge_w*1.0's intended ceiling. Capping HedgeN at 1.0 is
    what keeps hedge_w<=dew a real guarantee regardless of portfolio size --
    confirmed load-bearing, not just theoretical, by temporarily removing
    the cap during test design: with 12+ hedged anchors the uncapped
    variant actually flips the winner to the duplicate (X), while the
    capped code never does even at 20 anchors. 15 is used here for margin.

    15 anchors are forced to be picked first (highest EV). Candidate X
    duplicates anchor 0 (r=1.0) and hedges all 14 other anchors strongly
    (r=-0.95 each; uncapped that's 14*0.95^2=12.6). Candidate Y is merely
    uncorrelated with everything (DEn=1.0, HedgeN=0). Y must still win.
    """
    n_anchors = 15
    candidates = [_make_lineup(list(range(i * 10, i * 10 + 10))) for i in range(n_anchors + 2)]
    pool_idx = np.arange(n_anchors + 2)
    # Anchors get strictly decreasing EV (always outrank X/Y); X and Y tie at 10.
    pool_ev_vals = np.array([100.0 - i for i in range(n_anchors)] + [10.0, 10.0])
    corr_matrix = np.eye(n_anchors + 2, dtype=np.float32)
    x, y = n_anchors, n_anchors + 1
    corr_matrix[0, x] = corr_matrix[x, 0] = 1.0        # X duplicates anchor 0
    for a in range(1, n_anchors):
        corr_matrix[a, x] = corr_matrix[x, a] = -0.95  # X hedges every other anchor hard
    sel = DeterminantPortfolioSelector(
        robust_payout=None, candidates=candidates, portfolio_size=n_anchors + 1,
        evw_base=0.5, evw_max=0.5, risk=3.0,
        precomputed=(pool_idx, pool_ev_vals, corr_matrix),
    )
    result = sel.select()
    picked_idx = {candidates.index(lu) for lu, _ in result}
    assert y in picked_idx, f"expected Y({y}) to win over duplicate-but-heavily-hedging X({x}), got {picked_idx}"
    assert x not in picked_idx, f"X({x}) should have been rescued only if the hedge cap were missing, got {picked_idx}"


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
