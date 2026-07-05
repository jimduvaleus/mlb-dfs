"""Tests for ContestScorer and payout kernel."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.gpp_portfolio import (
    ContestScorer,
    _build_dilutable_lookup,
    _build_payout_lookup,
    _compute_payout_from_sorted_field,
    _payout_cumsum,
)
from src.optimization.lineup import Lineup
from src.optimization.ownership import compute_heuristic_ownership
from src.simulation.results import SimulationResults


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

def _make_player(pid, pos, salary, team, game, mean=20.0):
    row = {
        "player_id": pid,
        "name": f"P{pid}",
        "position": pos,
        "salary": salary,
        "team": team,
        "game": game,
        "mean": mean,
        "std_dev": 5.0,
        "slot": 9 if pos == "P" else 1,
    }
    parts = game.split("@")
    row["opponent"] = parts[1] if team == parts[0] else parts[0]
    return row


@pytest.fixture
def players_df():
    rows = [
        # Game A@B
        _make_player(1,  "P",  8000, "B", "A@B", 25.0),
        _make_player(2,  "P",  7000, "B", "A@B", 22.0),
        _make_player(20, "P",  8500, "A", "A@B", 21.0),
        _make_player(5,  "C",  4000, "A", "A@B", 18.0),
        _make_player(7,  "1B", 4200, "A", "A@B", 19.0),
        _make_player(9,  "2B", 4100, "A", "A@B", 17.0),
        _make_player(11, "3B", 3900, "A", "A@B", 16.0),
        _make_player(13, "SS", 3800, "A", "A@B", 15.0),
        _make_player(15, "OF", 4000, "A", "A@B", 20.0),
        _make_player(19, "OF", 3600, "A", "A@B", 18.0),
        _make_player(16, "OF", 4000, "B", "A@B", 18.0),
        # Game C@D
        _make_player(3,  "P",  7500, "D", "C@D", 24.0),
        _make_player(4,  "P",  6500, "D", "C@D", 21.0),
        _make_player(21, "P",  7500, "C", "C@D", 20.0),
        _make_player(6,  "C",  3800, "C", "C@D", 18.0),
        _make_player(8,  "1B", 3800, "C", "C@D", 17.0),
        _make_player(10, "2B", 3800, "C", "C@D", 16.0),
        _make_player(12, "3B", 3800, "C", "C@D", 15.0),
        _make_player(14, "SS", 3800, "C", "C@D", 14.0),
        _make_player(17, "OF", 3800, "C", "C@D", 19.0),
        _make_player(22, "OF", 3800, "C", "C@D", 17.0),
        _make_player(23, "OF", 3600, "C", "C@D", 16.0),
        _make_player(18, "OF", 3800, "D", "C@D", 18.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sim_results(players_df):
    rng = np.random.default_rng(42)
    pids = players_df["player_id"].tolist()
    matrix = rng.uniform(0, 40, size=(500, len(pids))).astype(np.float32)
    return SimulationResults(player_ids=pids, results_matrix=matrix)


@pytest.fixture
def ownership_vec(players_df):
    return compute_heuristic_ownership(players_df)


@pytest.fixture
def candidates(players_df, ownership_vec):
    """Generate a small candidate pool for tests."""
    from src.optimization.candidate_generator import CandidateGenerator
    gen = CandidateGenerator(players_df, ownership_vec, rng_seed=0)
    return gen.generate(n_candidates=50)


# ------------------------------------------------------------------ #
#  Numba kernel unit tests (_compute_payout_from_sorted_field)        #
# ------------------------------------------------------------------ #

def _make_test_payout_arr(n_entries=100, first_place=100.0, pay_positions=30, min_cash=6.0):
    """Simple test payout array: top 1 pays first_place, positions 2..pay_positions pay min_cash."""
    arr = np.zeros(n_entries, dtype=np.float32)
    arr[0] = first_place
    arr[1:pay_positions] = min_cash
    return arr


def _kernel(cand, field, lookup, dilute_lookup=None, dupe_scale=None):
    """Invoke the Numba kernel with cumsum lookups and optional dupe penalty."""
    cs = _payout_cumsum(lookup)
    dil = _payout_cumsum(dilute_lookup) if dilute_lookup is not None else np.zeros_like(cs)
    scale = (
        dupe_scale.astype(np.float32) if dupe_scale is not None
        else np.ones(cand.shape[0], dtype=np.float32)
    )
    return _compute_payout_from_sorted_field(
        np.ascontiguousarray(cand), np.ascontiguousarray(field), cs, dil, scale
    )


def _python_payout_dollar_ref(cand_scores, field_sorted, payout_lookup):
    """Pure-Python reference: tie-splitting mean over the lo..hi slot band."""
    BATCH, n_sims = cand_scores.shape
    out = np.zeros((BATCH, n_sims), dtype=np.float32)
    for b in range(BATCH):
        for s in range(n_sims):
            score = float(cand_scores[b, s])
            lo = int(np.searchsorted(field_sorted[s], score, side="left"))
            hi = int(np.searchsorted(field_sorted[s], score, side="right"))
            out[b, s] = payout_lookup[lo:hi + 1].mean()
    return out


def test_payout_kernel_matches_reference():
    rng = np.random.default_rng(0)
    BATCH, n_sims, N = 10, 200, 50
    gross_arr = _make_test_payout_arr()
    payout_lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = rng.uniform(0, 50, (BATCH, n_sims)).astype(np.float32)
    field = np.sort(rng.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)

    result = _kernel(cand, field, payout_lookup)
    expected = _python_payout_dollar_ref(cand, field, payout_lookup)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_payout_kernel_tie_splitting():
    """Integer scores force ties; payout must be the mean over the tie band."""
    rng = np.random.default_rng(3)
    BATCH, n_sims, N = 8, 150, 40
    gross_arr = _make_test_payout_arr()
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = rng.integers(0, 10, (BATCH, n_sims)).astype(np.float32)
    field = np.sort(rng.integers(0, 10, (n_sims, N)).astype(np.float32), axis=1)

    result = _kernel(cand, field, lookup)
    expected = _python_payout_dollar_ref(cand, field, lookup)

    np.testing.assert_allclose(result, expected, atol=1e-5)
    # Sanity: ties actually occurred (otherwise this test is vacuous).
    assert any(
        np.searchsorted(field[s], cand[0, s], side="right")
        > np.searchsorted(field[s], cand[0, s], side="left")
        for s in range(n_sims)
    )


def test_payout_kernel_tie_with_whole_field():
    """Candidate tied with every field lineup → mean of the entire lookup."""
    n_sims, N = 20, 30
    gross_arr = _make_test_payout_arr()
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = np.full((1, n_sims), 7.0, dtype=np.float32)
    field = np.full((n_sims, N), 7.0, dtype=np.float32)
    result = _kernel(cand, field, lookup)
    np.testing.assert_allclose(result, lookup.mean(), atol=1e-4)


def test_payout_kernel_all_above_field():
    """Candidate that beats entire field in every sim → rank 1 → first-place payout."""
    n_sims, N = 100, 50
    gross_arr = _make_test_payout_arr(first_place=5000.0)
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = np.full((1, n_sims), 999.0, dtype=np.float32)
    field = np.sort(np.random.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    result = _kernel(cand, field, lookup)
    # Beat all N → lookup[N]; bin-averaging of top ranks should be close to first-place prize
    np.testing.assert_allclose(result, lookup[N], atol=1e-3)


def test_payout_kernel_all_below_field():
    """Candidate always below entire field → beat none → lookup[0]."""
    n_sims, N = 100, 50
    gross_arr = _make_test_payout_arr()
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = np.full((1, n_sims), -999.0, dtype=np.float32)
    field = np.sort(np.random.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    result = _kernel(cand, field, lookup)
    np.testing.assert_allclose(result, lookup[0], atol=1e-5)


def test_payout_kernel_dupe_dilution():
    """dupe_scale dilutes only the dilutable (top-band gross) portion."""
    n_sims, N = 50, 40
    gross_arr = _make_test_payout_arr(first_place=1000.0, min_cash=6.0)
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    dilute = _build_dilutable_lookup(gross_arr, N=N, min_gross_payout=100.0)
    # Only the first-place rank pays >= $100 → dilutable mass sits at slot N.
    assert dilute[N] > 0 and np.all(dilute[: N // 2] == 0)

    field = np.sort(
        np.random.default_rng(5).uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1
    )
    winner = np.full((2, n_sims), 999.0, dtype=np.float32)
    scale = np.array([1.0, 0.5], dtype=np.float32)
    result = _kernel(winner, field, lookup, dilute_lookup=dilute, dupe_scale=scale)

    # scale=1.0 row unaffected; scale=0.5 row loses half the dilutable portion.
    np.testing.assert_allclose(result[0], lookup[N], atol=1e-3)
    np.testing.assert_allclose(result[1], lookup[N] - 0.5 * dilute[N], atol=1e-3)

    # A loser (below min cash) is untouched by the penalty.
    loser = np.full((2, n_sims), -999.0, dtype=np.float32)
    result_l = _kernel(loser, field, lookup, dilute_lookup=dilute, dupe_scale=scale)
    np.testing.assert_allclose(result_l[0], result_l[1], atol=1e-6)


def test_payout_kernel_output_shape():
    rng = np.random.default_rng(1)
    BATCH, n_sims, N = 7, 100, 30
    lookup = _build_payout_lookup(_make_test_payout_arr(), N=N, entry_fee=0.0)
    cand = rng.uniform(0, 50, (BATCH, n_sims)).astype(np.float32)
    field = np.sort(rng.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    result = _kernel(cand, field, lookup)
    assert result.shape == (BATCH, n_sims)
    assert result.dtype == np.float32


# ------------------------------------------------------------------ #
#  ContestScorer tests                                                 #
# ------------------------------------------------------------------ #

@pytest.fixture
def test_payout_arr():
    return _make_test_payout_arr(n_entries=200, first_place=5000.0, pay_positions=52, min_cash=6.0)


def test_scorer_output_shape(sim_results, players_df, candidates, ownership_vec, test_payout_arr):
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=10,
    )
    _, result = scorer.score_candidates(candidates[:20])
    assert result.shape == (20, sim_results.n_sims)
    assert result.dtype == np.float32


def test_scorer_values_floor(sim_results, players_df, candidates, ownership_vec, test_payout_arr):
    """Net payout floor is -entry_fee (losing entries pay the entry fee)."""
    entry_fee = 4.0
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    _, result = scorer.score_candidates(candidates[:10])
    assert np.all(result >= -entry_fee - 1e-3), (
        f"Net payout below floor (-${entry_fee}): min observed ${result.min():.2f}"
    )


def test_scorer_values_bounded(sim_results, players_df, candidates, ownership_vec, test_payout_arr):
    """Payout is bounded by the maximum value in the payout table."""
    max_payout = float(test_payout_arr.max())
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    _, result = scorer.score_candidates(candidates[:10])
    assert np.all(result <= max_payout + 1e-3), (
        f"Payout exceeds max (${max_payout:.2f}): max observed ${result.max():.2f}"
    )


def test_scorer_progress_callback(sim_results, players_df, candidates, ownership_vec, test_payout_arr):
    calls = []
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=5,
    )
    scorer.score_candidates(
        candidates[:15],
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    # 15 candidates / batch_size=5 → 3 batches
    assert len(calls) == 3
    assert calls[-1] == (3, 3)


def test_scorer_dupe_penalty_never_increases_ev(
    sim_results, players_df, candidates, ownership_vec, test_payout_arr
):
    """With the penalty on, every payout cell is <= its penalty-off value."""
    common = dict(
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr, ownership_vec=ownership_vec,
        candidate_batch_size=20, field_rng_seed=7,
    )
    scorer_off = ContestScorer(sim_results, players_df, **common)
    scorer_on = ContestScorer(
        sim_results, players_df, dupe_penalty=True,
        dupe_min_gross_payout=100.0, **common,
    )
    _, r_off = scorer_off.score_candidates(candidates[:10])
    _, r_on = scorer_on.score_candidates(candidates[:10])
    assert np.all(r_on <= r_off + 1e-4)
    # The penalty must actually bite somewhere (top-band finishes exist).
    assert (r_on < r_off - 1e-4).any()


def test_scorer_dupe_scale_monotonic_in_ownership(
    sim_results, players_df, candidates, ownership_vec, test_payout_arr
):
    """Chalkier lineups (higher Σlog ownership) must get a smaller top-band scale."""
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        payout_arr=test_payout_arr, ownership_vec=ownership_vec,
        dupe_penalty=True,
        # Isolate the ownership term: salary/stack features off.
        dupe_salary_coef=0.0, dupe_stack_coef=0.0,
    )
    scale = scorer._compute_dupe_scale(candidates[:30])
    assert scale.shape == (30,)
    assert np.all((scale > 0) & (scale <= 1))

    own_map = dict(zip(
        scorer._field_players_df["player_id"].astype(int),
        scorer._field_ownership_vec,
    ))

    def sum_log_own(lu):
        return sum(
            np.log(np.clip(own_map.get(int(p), 0.01), 1e-4, 0.95))
            for p in lu.player_ids
        )

    slo = np.array([sum_log_own(lu) for lu in candidates[:30]])
    # With only the ownership term active, scale is strictly decreasing in
    # Σ log(own): chalkier lineup → more expected dupes → smaller scale.
    order = np.argsort(slo)
    assert np.all(np.diff(scale[order].astype(np.float64)) <= 1e-9)


def test_scorer_rescore_fresh_fields(
    sim_results, players_df, candidates, ownership_vec, test_payout_arr
):
    """rescore_fresh_fields returns fresh-field scores with the right shape."""
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=2,
        payout_arr=test_payout_arr, ownership_vec=ownership_vec,
        candidate_batch_size=20, field_rng_seed=11,
    )
    subset = candidates[:12]
    _, mined = scorer.score_candidates(subset)
    fresh = scorer.rescore_fresh_fields(subset, n_samples=3)

    assert fresh.shape == (12, sim_results.n_sims)
    assert fresh.dtype == np.float32
    # Fresh fields come from disjoint seeds → scores differ from stage 1.
    assert not np.allclose(fresh, mined)
    # Fields were replaced: a follow-up score_batch scores against the same
    # fresh fields and reproduces the rescore output exactly.
    again = scorer.score_batch(subset)
    np.testing.assert_allclose(again, fresh, atol=1e-5)


def test_scorer_different_seeds_vary(sim_results, players_df, candidates, ownership_vec, test_payout_arr):
    """Two scorers with different field seeds should produce different results."""
    scorer1 = ContestScorer(
        sim_results, players_df, n_field_lineups=50, n_field_samples=1,
        payout_arr=test_payout_arr, ownership_vec=ownership_vec,
        field_rng_seed=10, candidate_batch_size=20,
    )
    scorer2 = ContestScorer(
        sim_results, players_df, n_field_lineups=50, n_field_samples=1,
        payout_arr=test_payout_arr, ownership_vec=ownership_vec,
        field_rng_seed=99, candidate_batch_size=20,
    )
    _, r1 = scorer1.score_candidates(candidates[:10])
    _, r2 = scorer2.score_candidates(candidates[:10])
    # Should not be identical (different fields → different percentiles)
    assert not np.allclose(r1, r2)
