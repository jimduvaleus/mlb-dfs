"""Unit tests for src/optimization/self_play.py (Phase 1 prototype).

The payout math itself (_compute_payout_from_sorted_field / _build_payout_lookup)
is already covered by tests/test_gpp_portfolio.py -- these tests instead cover
what's new here: fixed-size no-replacement opponent buckets, the growing
own-selections field composition (exactly one admission per round -- no
batching, see the module docstring's NO BATCHING note), cross-contest/poaching
mask updates, and the operational round log. Candidate scores are constant
across sims and mutually distinct, so relative ROI ranking each round reduces
to "whoever has the higher raw score wins" regardless of field composition --
that isolates the round-loop bookkeeping from the (already-tested) payout
banding math.
"""
import numpy as np
import pytest

from src.optimization.lineup import Lineup
from src.optimization.self_play import (
    SelfPlaySlateContext,
    _merge_and_sort_field,
    run_contest_precision_refinement,
    run_contest_self_play,
)
from src.simulation.results import SimulationResults


# ------------------------------------------------------------------ #
#  _merge_and_sort_field                                             #
# ------------------------------------------------------------------ #

def test_merge_field_basic():
    opp = np.array([[3.0, 1.0], [5.0, 2.0]], dtype=np.float32)   # (n_sims=2, F=2)
    own = np.array([[2.0], [4.0]], dtype=np.float32)              # (n_sims=2, m=1)
    merged = _merge_and_sort_field(opp, own)
    assert merged.shape == (2, 3)
    np.testing.assert_array_equal(merged[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(merged[1], [2.0, 4.0, 5.0])


def test_merge_field_empty_own():
    opp = np.array([[3.0, 1.0]], dtype=np.float32)
    own = np.zeros((1, 0), dtype=np.float32)
    merged = _merge_and_sort_field(opp, own)
    np.testing.assert_array_equal(merged, [[1.0, 3.0]])


def test_merge_field_empty_opponents():
    opp = np.zeros((1, 0), dtype=np.float32)
    own = np.array([[3.0, 1.0]], dtype=np.float32)
    merged = _merge_and_sort_field(opp, own)
    np.testing.assert_array_equal(merged, [[1.0, 3.0]])


# ------------------------------------------------------------------ #
#  run_contest_self_play                                             #
# ------------------------------------------------------------------ #

def _simple_payout_arr(n=20):
    """Strictly decreasing across the whole array (no flat/zero plateau) so
    that, whatever N a given round bands it down to, every reachable rank
    gets a strictly distinct payout -- a flat region (e.g. a "min cash"
    plateau of many equal/zero payouts) would make several different raw
    scores land in the same payout band and tie, which is a real and correct
    property of the kernel but would make this test's "highest score always
    wins" assumption vacuous rather than a bug when it's violated."""
    return np.linspace(100.0, 1.0, n).astype(np.float64)


def _build_test_ctx() -> SelfPlaySlateContext:
    """8 lineups, n_sims=4, scores constant across sims:
       candidates 0-4: scores 5, 15, 25, 30, 1 (strictly distinct)
       opponents   5-6: scores 10, 20 (fixed, opponent-eligible only)
       spare       7:   never referenced (masks start False for it)
    """
    n_sims = 4
    const = {0: 5.0, 1: 15.0, 2: 25.0, 3: 30.0, 4: 1.0, 5: 10.0, 6: 20.0, 7: 0.0}
    scores = np.array(
        [[const[i]] * n_sims for i in range(8)], dtype=np.float32
    )
    return SelfPlaySlateContext(
        lineups=[None] * 8, source=np.array(["x"] * 8), scores=scores,
        n_external=8, n_sims=n_sims,
    )


def test_self_play_picks_best_remaining_candidate_each_round():
    ctx = _build_test_ctx()
    candidate_mask = np.array([True, True, True, True, True, False, False, False])
    opponent_mask = np.array([False, False, False, False, False, True, True, False])
    rng = np.random.default_rng(0)

    result = run_contest_self_play(
        ctx, contest_id="c1", k=3, field_size=5,  # n_opponents = 5 - 3 = 2 (exactly the pool size)
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng, shortlist_size=300,
    )

    # Highest score wins first: 30 (idx3), then 25 (idx2), then 15 (idx1) --
    # one admission per round, so 3 picks takes exactly 3 rounds.
    assert result.own_idx == [3, 2, 1]

    # Cross-contest mask: picked candidates removed, untouched ones remain.
    np.testing.assert_array_equal(
        candidate_mask, [True, False, False, False, True, False, False, False]
    )
    # Opponents are never replaced -- both stay available throughout.
    np.testing.assert_array_equal(
        opponent_mask, [False, False, False, False, False, True, True, False]
    )

    # Operational log: 2 rows (round 0 isn't logged -- it's always the full
    # rescore, nothing to report). shortlist_size=300 exceeds this pool, so
    # the shortlist never needs replenishing -- both logged rounds are
    # restricted (not full), scoring only the candidates still remaining.
    log = result.round_log
    assert len(log) == 2
    assert log.iloc[0][["round", "full_rescore", "n_scored"]].tolist() == [1, False, 4]
    assert log.iloc[1][["round", "full_rescore", "n_scored"]].tolist() == [2, False, 3]


def test_self_play_fills_full_pool_one_at_a_time():
    ctx = _build_test_ctx()
    candidate_mask = np.array([True, True, True, True, True, False, False, False])
    opponent_mask = np.array([False, False, False, False, False, True, True, False])
    rng = np.random.default_rng(0)

    # k=5 == every available candidate -- no batching means this takes 5
    # rounds (round 0 unlogged, rounds 1-4 logged), not one big round.
    result = run_contest_self_play(
        ctx, contest_id="c2", k=5, field_size=7,
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng,
    )
    assert result.own_idx == [3, 2, 1, 0, 4]  # descending by score: 30,25,15,5,1
    assert len(result.round_log) == 4


def test_self_play_stops_when_candidates_exhausted():
    ctx = _build_test_ctx()
    candidate_mask = np.array([True, True, False, False, False, False, False, False])
    opponent_mask = np.array([False, False, False, False, False, True, True, False])
    rng = np.random.default_rng(0)

    # k=5 requested but only 2 candidates are actually available.
    result = run_contest_self_play(
        ctx, contest_id="c3", k=5, field_size=7,
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng,
    )
    assert sorted(result.own_idx) == [0, 1]


def test_self_play_raises_when_opponent_pool_too_small():
    ctx = _build_test_ctx()
    candidate_mask = np.array([True, True, True, True, True, False, False, False])
    opponent_mask = np.array([False, False, False, False, False, True, False, False])  # only 1 available
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="opponent pool exhausted"):
        run_contest_self_play(
            ctx, contest_id="c4", k=3, field_size=5,  # needs 2 opponents, only 1 available
            payout_arr=_simple_payout_arr(), entry_fee=0.0,
            candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
            rng=rng,
        )


def test_self_play_poached_generated_lineup_loses_opponent_eligibility():
    """A lineup that's both candidate- and opponent-eligible (the poachable
    generated case) must stop being drawable as an opponent once picked."""
    ctx = _build_test_ctx()
    # idx 5 (score 10) is eligible as BOTH a candidate and an opponent --
    # simulates a generated lineup that could be poached into our own portfolio.
    candidate_mask = np.array([False, False, False, False, False, True, False, False])
    opponent_mask = np.array([False, False, False, False, False, True, True, False])
    rng = np.random.default_rng(0)

    result = run_contest_self_play(
        ctx, contest_id="c5", k=1, field_size=2,  # n_opponents = 1
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng,
    )
    assert result.own_idx == [5]
    assert opponent_mask[5] == False  # poached -- no longer opponent-eligible
    assert opponent_mask[6] == True   # untouched opponent still available


# ------------------------------------------------------------------ #
#  run_contest_precision_refinement                                  #
# ------------------------------------------------------------------ #

def _build_refinement_ctx() -> SelfPlaySlateContext:
    """5 lineups, n_sims=4, scores constant across sims:
       idx 0: current (weak) pick, score 5, generated (poachable)
       idx 1: true-best candidate the cheap pass missed, score 30, generated
       idx 2: mediocre candidate, score 10, external
       idx 3-4: opponents, generated, scores 8 and 15
    """
    n_sims = 4
    const = {0: 5.0, 1: 30.0, 2: 10.0, 3: 8.0, 4: 15.0}
    scores = np.array([[const[i]] * n_sims for i in range(5)], dtype=np.float32)
    source = np.array(["generated", "generated", "external", "generated", "generated"])
    # precise_sim is now scored on demand (_precise_scores_for), not a
    # precomputed array -- give each fixture "lineup" a single distinct
    # player whose constant simulated score reproduces the same values as
    # the cheap tier above, so precise-tier scoring in this test agrees
    # with it exactly (matches the original "scores constant across sims"
    # setup this whole test file relies on).
    lineups = [Lineup(player_ids=[i]) for i in range(5)]
    precise_sim = SimulationResults(
        player_ids=[0, 1, 2, 3, 4],
        results_matrix=np.array([[const[i] for i in range(5)]] * n_sims, dtype=np.float64),
    )
    return SelfPlaySlateContext(
        lineups=lineups, source=source, scores=scores,
        n_external=0, n_sims=n_sims,
        precise_sim=precise_sim, precise_n_sims=n_sims,
        promoted_idx=np.empty(0, dtype=np.int64),
    )


def test_refinement_swaps_in_true_best_and_converges():
    ctx = _build_refinement_ctx()
    # idx 0 is "ours"; 1, 2 available as candidates; 3, 4 opponent-eligible.
    candidate_mask = np.array([False, True, True, False, False])
    opponent_mask = np.array([False, False, False, True, True])
    rng = np.random.default_rng(0)

    own_idx, own_roi, log = run_contest_precision_refinement(
        ctx, contest_id="r1", own_idx=[0], own_roi=[1.0],
        final_shortlist_idx=np.array([0, 1, 2]),
        field_size=3, k=1,  # n_opponents = 2 (indices 3, 4 exactly)
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng,
    )

    assert own_idx == [1]  # swapped the weak pick (0) for the true-best one (1)
    assert len(log) == 1   # one swap, then converged (no further improvement)
    assert log.iloc[0]["swapped_out_idx"] == 0
    assert log.iloc[0]["swapped_in_idx"] == 1

    # cross-contest bookkeeping: 0 freed back to the pool, 1 now claimed
    assert candidate_mask[0] == True
    assert candidate_mask[1] == False
    # both are "generated" -- poaching 1 removes its opponent eligibility,
    # freeing 0 (also generated) restores its own
    assert opponent_mask[1] == False
    assert opponent_mask[0] == True


def test_refinement_noop_when_no_precise_scores():
    ctx = _build_refinement_ctx()
    ctx.precise_sim = None
    candidate_mask = np.array([False, True, True, False, False])
    opponent_mask = np.array([False, False, False, True, True])
    rng = np.random.default_rng(0)

    own_idx, own_roi, log = run_contest_precision_refinement(
        ctx, contest_id="r2", own_idx=[0], own_roi=[1.0],
        final_shortlist_idx=np.array([0, 1, 2]),
        field_size=3, k=1,
        payout_arr=_simple_payout_arr(), entry_fee=0.0,
        candidate_mask=candidate_mask, opponent_available_mask=opponent_mask,
        rng=rng,
    )
    assert own_idx == [0]
    assert log.empty
    # masks untouched
    np.testing.assert_array_equal(candidate_mask, [False, True, True, False, False])
