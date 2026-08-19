"""Tests for the smoothed-exceedance estimator of dR.

Two things have to hold for smoothing to be legitimate rather than merely
different:

  1. CONVERGENCE -- as tau -> 0 the smoothed estimator must reproduce the exact
     rank-lookup one. That is what makes it an estimator of the SAME quantity.
  2. VARIANCE REDUCTION -- at the derived width it must be more stable across
     independent field draws than the hard indicator. That is the only reason
     to use it.

Reliability is not validity (memory project-smoothed-exceedance): a perfectly
stable WRONG statistic scores 1.0 on (2). Test (1) is what stops that -- it
pins the smoothed statistic to the exact one rather than letting it drift
toward plain mean score.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from src.optimization.mrp.delta_reward import ContestDeltaState
from src.optimization.payout import load_payout_structure, payout_table_to_array


def _top_heavy(total=400):
    arr = np.zeros(total, dtype=np.float64)
    for a, b, amt in [(0, 1, 5000.0), (1, 2, 1500.0), (2, 4, 600.0),
                      (4, 8, 200.0), (8, 20, 50.0), (20, 60, 10.0)]:
        arr[a:b] = amt
    return arr


def _fixture(seed, S=400, F=400, M=60):
    rng = np.random.default_rng(seed)
    field = np.sort(rng.normal(120.0, 18.0, size=(S, F)).astype(np.float32), axis=1)
    cand = rng.normal(126.0, 18.0, size=(M, S)).astype(np.float32)
    return field, cand


def test_converges_to_the_exact_estimator_as_tau_goes_to_zero():
    """Depth 0, where no self-competition shift is involved, so the two paths
    should agree to within the logistic's approximation of a step."""
    payout = _top_heavy()
    field, cand = _fixture(0)

    exact = ContestDeltaState(cand, field, payout).marginal_gains()
    tiny = ContestDeltaState(cand, field, payout, smooth_tau_scale=1e-6).marginal_gains()

    np.testing.assert_allclose(tiny, exact, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("tau_scale", [0.5, 1.0])
def test_smoothed_still_ranks_dR_and_not_plain_mean_score(tau_scale):
    """The drift check. Reliability is not validity: as tau grows the ranking
    collapses toward field-blind mean score, which measured ANTI-correlated
    with results (project-external-pool-currency-comparison). So the test is
    not "rho vs exact is high" in isolation -- it is that the smoothed
    statistic stays far closer to exact dR than to mean score."""
    payout = _top_heavy()
    field, cand = _fixture(1)

    exact = ContestDeltaState(cand, field, payout).marginal_gains()
    smooth = ContestDeltaState(cand, field, payout,
                               smooth_tau_scale=tau_scale).marginal_gains()

    rho_exact = spearmanr(exact, smooth).statistic
    rho_mean = spearmanr(cand.mean(axis=1), smooth).statistic
    assert rho_exact > 0.93, f"drifted from exact dR (rho={rho_exact:.3f})"
    assert rho_exact > rho_mean + 0.15, (
        f"tau={tau_scale} is collapsing toward mean score: "
        f"rho_exact={rho_exact:.3f} rho_mean={rho_mean:.3f}"
    )


def test_smoothed_reduces_cross_field_variance_on_the_tight_rungs():
    """The actual claim. Score the same candidates against INDEPENDENT field
    draws and compare how much the two estimators' rankings move."""
    payout = _top_heavy()
    rng = np.random.default_rng(7)
    S, F, M = 600, 500, 80
    cand = rng.normal(126.0, 18.0, size=(M, S)).astype(np.float32)

    def rankings(tau_scale):
        out = []
        for d in range(4):
            f = np.sort(rng.normal(120.0, 18.0, size=(S, F)).astype(np.float32), axis=1)
            out.append(ContestDeltaState(cand, f, payout,
                                         smooth_tau_scale=tau_scale).marginal_gains())
            del f
        return out

    def mean_pairwise_rho(rs):
        vals = [spearmanr(rs[i], rs[j]).statistic
                for i in range(len(rs)) for j in range(i + 1, len(rs))]
        return float(np.mean(vals))

    hard = mean_pairwise_rho(rankings(0.0))
    smooth = mean_pairwise_rho(rankings(1.0))
    assert smooth > hard, f"smoothing did not stabilise the ranking: {smooth:.3f} vs {hard:.3f}"


def test_near_twin_is_devalued_below_an_independent_lineup_under_smoothing():
    """The REACHABLE case. The pool cull removes every 9/10 pair, so the worst
    real collision is a highly-correlated-but-distinct lineup -- which is what
    the self-competition term has to catch."""
    payout = _top_heavy()
    rng = np.random.default_rng(11)
    S, F = 800, 300
    field = np.sort(rng.normal(120.0, 18.0, size=(S, F)).astype(np.float32), axis=1)

    strong = rng.normal(140.0, 16.0, size=S).astype(np.float32)
    twin = (strong + rng.normal(0.0, 1.6, size=S)).astype(np.float32)   # rho ~ 0.99
    other = (strong.mean() + rng.normal(0.0, 16.0, size=S)).astype(np.float32)
    cand = np.vstack([strong, twin, other])

    st = ContestDeltaState(cand, field, payout, smooth_tau_scale=1.0)
    before = st.marginal_gains()
    st.commit(0)
    after = st.marginal_gains()

    assert after[1] / before[1] < after[2] / before[2], (
        "the near-twin must lose more of its value than an independent lineup"
    )


def test_exact_duplicates_are_penalised_harder_by_the_exact_path():
    """A documented LIMITATION, pinned so it cannot drift unnoticed.

    Exact ties are measure-zero in continuous score space, so smoothing models a
    duplicate only as a half-rank displacement, while the exact path splits the
    prize outright. On a steep table the exact path is much harsher. This is
    unreachable in production -- `_find_near_duplicate_removals` culls every
    9/10 pair and the allocator masks re-selection -- but if a future change
    makes exact ties reachable under smoothing, this is the trade being made."""
    payout = _top_heavy()
    rng = np.random.default_rng(11)
    S, F = 500, 300
    field = np.sort(rng.normal(120.0, 18.0, size=(S, F)).astype(np.float32), axis=1)
    strong = rng.normal(140.0, 16.0, size=S).astype(np.float32)
    cand = np.vstack([strong, strong.copy(), rng.normal(140.0, 16.0, size=S).astype(np.float32)])

    retained = {}
    for name, ts in (("exact", 0.0), ("smooth", 1.0)):
        st = ContestDeltaState(cand, field, payout, smooth_tau_scale=ts)
        before = st.marginal_gains()
        assert before[0] == pytest.approx(before[1]), "twins must start equal"
        st.commit(0)
        retained[name] = st.marginal_gains()[1] / before[1]

    assert retained["exact"] < 0.5, "exact path should roughly halve a duplicate"
    assert retained["smooth"] < 1.0, "smoothing must still penalise it somewhat"
    assert retained["exact"] < retained["smooth"], "documented direction of the gap"


def test_marginal_gains_still_decline_under_smoothing():
    payout = _top_heavy()
    field, cand = _fixture(3, S=500)
    st = ContestDeltaState(cand, field, payout, smooth_tau_scale=1.0)
    deltas = []
    for _ in range(8):
        g = st.marginal_gains()
        j = int(np.argmax(g))
        deltas.append(g[j])
        st.commit(j)
    assert all(b <= a + 1e-6 for a, b in zip(deltas, deltas[1:])), deltas


def test_tier_smoothing_retains_only_small_arrays():
    """The point of building it during __init__: the multi-GB field array must
    be droppable straight afterwards."""
    payout = payout_table_to_array(load_payout_structure("dk_four_seamer_2972"))
    rng = np.random.default_rng(5)
    S, F, M = 200, 3000, 20
    field = np.sort(rng.normal(120.0, 18.0, size=(S, F)).astype(np.float32), axis=1)
    cand = rng.normal(126.0, 18.0, size=(M, S)).astype(np.float32)

    st = ContestDeltaState(cand, field, payout, smooth_tau_scale=1.0)
    sm = st.smoothing
    assert sm.thr.shape == (sm.n_tiers, S)
    assert sm.thr.nbytes + sm.slope.nbytes + sm.tau.nbytes < field.nbytes / 10
