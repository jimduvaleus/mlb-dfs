"""Tests for ContestScorer and EVPortfolioSelector."""
import numpy as np
import pandas as pd
import pytest

from src.optimization.gpp_portfolio import (
    ContestScorer,
    EVPortfolioSelector,
    _build_payout_lookup,
    _compute_payout_from_sorted_field,
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


def _python_payout_dollar_ref(cand_scores, field_sorted, payout_lookup):
    """Pure-Python reference for dollar payout computation (direct lookup)."""
    BATCH, n_sims = cand_scores.shape
    N = field_sorted.shape[1]
    out = np.zeros((BATCH, n_sims), dtype=np.float32)
    for b in range(BATCH):
        for s in range(n_sims):
            score = float(cand_scores[b, s])
            lo = int(np.searchsorted(field_sorted[s], score, side="right"))
            out[b, s] = payout_lookup[lo]
    return out


def test_payout_kernel_matches_reference():
    rng = np.random.default_rng(0)
    BATCH, n_sims, N = 10, 200, 50
    gross_arr = _make_test_payout_arr()
    payout_lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = rng.uniform(0, 50, (BATCH, n_sims)).astype(np.float32)
    field = np.sort(rng.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    cand_c = np.ascontiguousarray(cand)
    field_c = np.ascontiguousarray(field)

    result = _compute_payout_from_sorted_field(cand_c, field_c, payout_lookup)
    expected = _python_payout_dollar_ref(cand, field, payout_lookup)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_payout_kernel_all_above_field():
    """Candidate that beats entire field in every sim → rank 1 → first-place payout."""
    n_sims, N = 100, 50
    gross_arr = _make_test_payout_arr(first_place=5000.0)
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = np.full((1, n_sims), 999.0, dtype=np.float32)
    field = np.sort(np.random.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    result = _compute_payout_from_sorted_field(
        np.ascontiguousarray(cand), np.ascontiguousarray(field), lookup
    )
    # Beat all N → lookup[N]; bin-averaging of top ranks should be close to first-place prize
    np.testing.assert_allclose(result, lookup[N], atol=1e-3)


def test_payout_kernel_all_below_field():
    """Candidate always below entire field → beat none → lookup[0]."""
    n_sims, N = 100, 50
    gross_arr = _make_test_payout_arr()
    lookup = _build_payout_lookup(gross_arr, N=N, entry_fee=0.0)
    cand = np.full((1, n_sims), -999.0, dtype=np.float32)
    field = np.sort(np.random.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    result = _compute_payout_from_sorted_field(
        np.ascontiguousarray(cand), np.ascontiguousarray(field), lookup
    )
    np.testing.assert_allclose(result, lookup[0], atol=1e-5)


def test_payout_kernel_output_shape():
    rng = np.random.default_rng(1)
    BATCH, n_sims, N = 7, 100, 30
    payout_arr = _make_test_payout_arr()
    cand = np.ascontiguousarray(rng.uniform(0, 50, (BATCH, n_sims)).astype(np.float32))
    field = np.ascontiguousarray(
        np.sort(rng.uniform(0, 50, (n_sims, N)).astype(np.float32), axis=1)
    )
    result = _compute_payout_from_sorted_field(cand, field, payout_arr)
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



# ------------------------------------------------------------------ #
#  EVPortfolioSelector tests                                          #
# ------------------------------------------------------------------ #

def test_selector_round0_picks_highest_mean_ev():
    """Round 0 must select the candidate with the highest mean payout."""
    rng = np.random.default_rng(0)
    M, n_sims = 10, 500
    robust_payout = rng.uniform(0, 0.1, (M, n_sims)).astype(np.float32)
    # Make candidate 5 clearly dominant.
    robust_payout[5] = 0.09
    stubs = [Lineup(player_ids=list(range(i, i + 10))) for i in range(M)]

    selector = EVPortfolioSelector(robust_payout, stubs, portfolio_size=3)
    result = selector.select()

    assert result[0][0] is stubs[5], "Round 0 should pick the highest-mean candidate"


def test_selector_round1_picks_marginal_maximizer():
    """Round 1 picks marginal EV maximizer, not second-highest mean."""
    n_sims = 1000
    # Candidate 0: wins in sims 0..499 (first half)
    # Candidate 1: wins in sims 0..499 as well (high mean, same overlap as 0 → low marginal)
    # Candidate 2: wins in sims 500..999 (low mean, zero overlap with 0 → high marginal)
    robust_payout = np.zeros((3, n_sims), dtype=np.float32)
    robust_payout[0, :500] = 0.1   # mean = 0.05
    robust_payout[1, :500] = 0.08  # mean = 0.04 (second highest mean)
    robust_payout[2, 500:] = 0.1   # mean = 0.05, but zero overlap with candidate 0

    stubs = [Lineup(player_ids=list(range(i, i + 10))) for i in range(3)]
    selector = EVPortfolioSelector(robust_payout, stubs, portfolio_size=2)
    result = selector.select()

    # Round 0: 0 and 2 are tied at mean=0.05; either could be chosen, but both
    # are better than 1. Round 1 should pick the complement.
    r0 = result[0][0]
    r1 = result[1][0]
    assert r0 is not r1, "Two different lineups must be selected"
    # The two selected lineups should together cover most sims.
    idx0 = stubs.index(r0)
    idx1 = stubs.index(r1)
    best = np.maximum(robust_payout[idx0], robust_payout[idx1])
    assert best.mean() > robust_payout[2].mean(), (
        "Portfolio of 2 should beat any single candidate's mean"
    )


def test_selector_round1_picks_by_mean_ev():
    """Round 1 selects by mean EV on uncovered sims (without beat_rate_fn = all sims)."""
    n_sims = 1000
    robust_payout = np.zeros((3, n_sims), dtype=np.float32)
    # Candidate 0: wins first half strongly (mean=0.1)
    robust_payout[0, :500] = 0.2
    # Candidate 1: same sims as 0, lower EV (mean=0.075)
    robust_payout[1, :500] = 0.15
    # Candidate 2: second half, higher mean than 1 (mean=0.09)
    robust_payout[2, 500:] = 0.18

    stubs = [Lineup(player_ids=list(range(i, i + 10))) for i in range(3)]
    selector = EVPortfolioSelector(robust_payout, stubs, portfolio_size=2)
    result = selector.select()

    assert result[0][0] is stubs[0], "Round 0 picks highest mean"
    assert result[1][0] is stubs[2], (
        "Round 1 picks candidate 2 (higher mean EV) over candidate 1 (lower mean EV)"
    )


def test_selector_portfolio_size(sim_results, players_df, candidates, ownership_vec):
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)
    selector = EVPortfolioSelector(robust_payout, candidates, portfolio_size=5)
    result = selector.select()
    assert len(result) == 5


def test_selector_all_distinct(sim_results, players_df, candidates, ownership_vec):
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)
    selector = EVPortfolioSelector(robust_payout, candidates, portfolio_size=8)
    result = selector.select()
    pid_sets = [frozenset(lu.player_ids) for lu, _ in result]
    assert len(set(pid_sets)) == len(pid_sets), "All portfolio lineups must be distinct"


def test_selector_lineup_ev_non_increasing(sim_results, players_df, candidates, ownership_vec):
    """Each lineup's EV (on uncovered sims) should be non-increasing across rounds."""
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)
    selector = EVPortfolioSelector(robust_payout, candidates, portfolio_size=8)
    result = selector.select()

    evs = [lineup_ev for _, lineup_ev in result]
    for i in range(1, len(evs)):
        assert evs[i] <= evs[i - 1] + 1e-8, (
            f"Lineup EV increased at round {i}: {evs[i-1]:.6f} → {evs[i]:.6f}"
        )


def test_selector_holdout_score(sim_results, players_df, candidates, ownership_vec):
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)
    selector = EVPortfolioSelector(
        robust_payout, candidates, portfolio_size=5,
        holdout_fraction=0.2, rng_seed=42,
    )
    result = selector.select()
    score = selector.holdout_score()

    assert score is not None
    assert score >= 0.0
    assert len(result) == 5


def test_selector_no_holdout_returns_none(sim_results, players_df, candidates, ownership_vec):
    scorer = ContestScorer(
        sim_results, players_df,
        n_field_lineups=30, n_field_samples=1,
        ownership_vec=ownership_vec,
        candidate_batch_size=20,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)
    selector = EVPortfolioSelector(robust_payout, candidates, portfolio_size=3)
    selector.select()
    assert selector.holdout_score() is None


def test_selector_zero_payout_edge_case():
    """When all payouts are zero, selector should still return portfolio_size lineups."""
    M, n_sims = 20, 100
    robust_payout = np.zeros((M, n_sims), dtype=np.float32)
    stubs = [Lineup(player_ids=list(range(i, i + 10))) for i in range(M)]
    selector = EVPortfolioSelector(robust_payout, stubs, portfolio_size=5)
    result = selector.select()
    assert len(result) == 5


# ------------------------------------------------------------------ #
#  Unmapped player_id / stale-cache bug regression tests             #
# ------------------------------------------------------------------ #

def test_scorer_invalid_candidate_zeroed_out(players_df, ownership_vec, test_payout_arr):
    """Candidates with player_ids absent from sim_results must have robust_payout zeroed.

    Replicates the stale-cache scenario: candidates were generated for an old slate
    that included player 9999 (e.g. Bryce Harper with a new DK ID), but sim_results
    was built from the current slate which does NOT include player 9999. Without the
    invalid_mask fix, numpy's -1 index wraps to the last column and inflates that
    candidate's score, potentially selecting it into the portfolio.
    """
    rng = np.random.default_rng(42)
    pids = players_df["player_id"].tolist()
    sim_matrix = rng.uniform(0, 40, size=(200, len(pids))).astype(np.float32)
    sim_results_obj = SimulationResults(player_ids=pids, results_matrix=sim_matrix)

    # Build two valid candidates from the current player pool.
    valid_lu_a = Lineup(player_ids=[1, 20, 5, 7, 9, 11, 13, 15, 19, 16])  # P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
    valid_lu_b = Lineup(player_ids=[2, 3, 6, 8, 10, 12, 14, 17, 22, 18])

    # Build one "stale cache" candidate that references player 9999 (not in sim_results).
    ghost_lu = Lineup(player_ids=[9999, 20, 5, 7, 9, 11, 13, 15, 19, 16])

    candidates_mixed = [valid_lu_a, valid_lu_b, ghost_lu]

    scorer = ContestScorer(
        sim_results_obj, players_df,
        n_field_lineups=30, n_field_samples=1,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=10,
    )
    _, robust_payout = scorer.score_candidates(candidates_mixed)

    # The ghost candidate (index 2) must have been zeroed out entirely.
    assert np.all(robust_payout[2] == 0.0), (
        "robust_payout for stale-cache candidate with unmapped player_id must be all zeros"
    )
    # Valid candidates must have non-zero payout (they can score in the contest).
    assert robust_payout[0].max() > 0.0, "Valid candidate A should have non-zero payout"
    assert robust_payout[1].max() > 0.0, "Valid candidate B should have non-zero payout"


def test_scorer_invalid_candidate_not_selected(players_df, ownership_vec, test_payout_arr):
    """EVPortfolioSelector must never select a stale-cache candidate into the portfolio.

    Even if the ghost candidate's raw column values (before zeroing) would have scored
    high — because numpy's -1 wraps to the last-column player — the invalid_mask fix
    must prevent selection.
    """
    rng = np.random.default_rng(7)
    pids = players_df["player_id"].tolist()
    sim_matrix = rng.uniform(0, 40, size=(200, len(pids))).astype(np.float32)

    # Make the last column (index -1, i.e. index len(pids)-1) very high so the ghost
    # candidate would dominate if -1 wraps were not corrected.
    sim_matrix[:, -1] = 999.0

    sim_results_obj = SimulationResults(player_ids=pids, results_matrix=sim_matrix)

    valid_lu = Lineup(player_ids=[1, 20, 5, 7, 9, 11, 13, 15, 19, 16])
    ghost_lu = Lineup(player_ids=[9999, 3, 6, 8, 10, 12, 14, 17, 22, 18])

    candidates_mixed = [ghost_lu, valid_lu]  # ghost first so it would win without the fix

    scorer = ContestScorer(
        sim_results_obj, players_df,
        n_field_lineups=30, n_field_samples=1,
        payout_arr=test_payout_arr,
        ownership_vec=ownership_vec,
        candidate_batch_size=10,
    )
    _, robust_payout = scorer.score_candidates(candidates_mixed)

    selector = EVPortfolioSelector(robust_payout, candidates_mixed, portfolio_size=1)
    result = selector.select()

    selected_lineup = result[0][0]
    assert selected_lineup is valid_lu, (
        "EVPortfolioSelector must select the valid lineup, not the stale-cache ghost candidate"
    )


