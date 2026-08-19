"""Contract test: the fast incremental dR must equal brute-force R(Su{j}) - R(S).

`delta_reward.py` is an optimisation, not an approximation. Everything it does
-- precomputed field ranks, incremental own_above/own_ties counters, the sparse
demotion cells -- has to reproduce, exactly, what you get by re-running the
reference `portfolio_reward` from scratch on S u {j}.

If a test here fails, the optimisation is wrong. `marginal_reward.py` (itself
pinned to bt_core.grade_portfolio by tests/test_mrp_marginal_reward.py) is the
reference.
"""
import numpy as np
import pytest

from src.optimization.mrp.delta_reward import ContestDeltaState
from src.optimization.mrp.marginal_reward import portfolio_reward


def _payout_arr(total: int = 120) -> np.ndarray:
    arr = np.zeros(total, dtype=np.float64)
    for start, end, amt in [(0, 1, 1000.0), (1, 2, 400.0), (2, 4, 200.0),
                            (4, 8, 80.0), (8, 16, 25.0), (16, 45, 5.0)]:
        arr[start:end] = amt
    return arr


def _fixture(seed, S=60, F=90, M=25, tie_heavy=False):
    rng = np.random.default_rng(seed)
    if tie_heavy:
        field = rng.integers(100, 112, size=(S, F)).astype(np.float32)
        cand = rng.integers(100, 112, size=(M, S)).astype(np.float32)
    else:
        field = rng.normal(120.0, 20.0, size=(S, F)).astype(np.float32)
        cand = rng.normal(128.0, 20.0, size=(M, S)).astype(np.float32)
    return np.sort(field, axis=1), cand


@pytest.mark.parametrize("tie_heavy", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_delta_matches_bruteforce_at_every_depth(seed, tie_heavy):
    """The whole contract, checked for every candidate at every portfolio depth."""
    payout = _payout_arr()
    field_sorted, cand = _fixture(seed, tie_heavy=tie_heavy)
    M = cand.shape[0]

    state = ContestDeltaState(cand, field_sorted, payout)
    picked: list[int] = []

    for _depth in range(4):
        fast = state.marginal_gains()

        base = (portfolio_reward(cand[picked], field_sorted, payout)
                if picked else 0.0)
        assert state.reward() == pytest.approx(base, abs=1e-9)

        brute = np.empty(M, dtype=np.float64)
        for j in range(M):
            brute[j] = portfolio_reward(cand[picked + [j]], field_sorted, payout) - base
        np.testing.assert_allclose(fast, brute, rtol=1e-9, atol=1e-9)

        nxt = int(np.argmax(fast))
        state.commit(nxt)
        picked.append(nxt)


def test_demotion_term_is_actually_negative_somewhere():
    """Guard against a vacuous pass: if the sparse cells were always empty the
    test above would still succeed while measuring nothing."""
    payout = _payout_arr()
    field_sorted, cand = _fixture(3, S=200, F=60, M=30)

    state = ContestDeltaState(cand, field_sorted, payout)
    state.commit(int(np.argmax(state.marginal_gains())))
    cells = state._demotion_cells()

    assert cells is not None, "no demotion cells -- the term is untested"
    _, _, delta_gt, delta_eq = cells
    assert (delta_gt < 0).any(), "being outscored by a new entry should cost dollars"
    assert (delta_eq < 0).any(), "being TIED by a new entry should cost dollars too"


def test_duplicate_candidate_gains_less_after_its_twin_is_taken():
    """The mechanism that replaces the diversity heuristic: an exact duplicate
    of an already-selected lineup must lose most of its marginal value."""
    payout = _payout_arr()
    rng = np.random.default_rng(17)
    S, F = 300, 80
    field_sorted = np.sort(rng.normal(120.0, 20.0, size=(S, F)).astype(np.float32), axis=1)

    strong = rng.normal(140.0, 18.0, size=S).astype(np.float32)
    other = rng.normal(140.0, 18.0, size=S).astype(np.float32)
    cand = np.vstack([strong, strong.copy(), other])   # row 1 duplicates row 0

    state = ContestDeltaState(cand, field_sorted, payout)
    before = state.marginal_gains()
    assert before[0] == pytest.approx(before[1]), "twins must start equal"

    state.commit(0)
    after = state.marginal_gains()

    assert after[1] < before[1], "the twin must be devalued once its twin is in"
    assert after[1] < after[2], "and must now rank below an independent lineup"


def test_reward_is_gross_and_additive_over_entries():
    payout = _payout_arr()
    field_sorted, cand = _fixture(5, S=80, F=70, M=12)
    state = ContestDeltaState(cand, field_sorted, payout)

    total = 0.0
    for _ in range(5):
        g = state.marginal_gains()
        j = int(np.argmax(g))
        total += g[j]
        state.commit(j)

    assert state.reward() == pytest.approx(total, abs=1e-9), (
        "telescoping sum of marginal gains must equal R(S)"
    )
