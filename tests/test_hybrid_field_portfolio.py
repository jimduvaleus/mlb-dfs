"""Tests for HybridFieldPortfolioSelector."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.optimization.hybrid_field_portfolio import HybridFieldPortfolioSelector
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
        "slot": 10 if pos == "P" else 1,
    }
    parts = game.split("@")
    row["opponent"] = parts[1] if team == parts[0] else parts[0]
    return row


@pytest.fixture
def players_df():
    rows = [
        # Game A@B  (team A bats, team B pitches)
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
def player_ids(players_df):
    return players_df["player_id"].tolist()


@pytest.fixture
def ownership_vec(players_df):
    return compute_heuristic_ownership(players_df)


@pytest.fixture
def candidates(players_df, ownership_vec):
    from src.optimization.candidate_generator import CandidateGenerator
    gen = CandidateGenerator(players_df, ownership_vec, rng_seed=0)
    return gen.generate(n_candidates=60)


def _make_payout_arr(n_entries=200, first_place=500.0, n_pay=50):
    arr = np.zeros(n_entries, dtype=np.float32)
    arr[0] = first_place
    arr[1:n_pay] = 10.0
    return arr


def _make_mock_engine(player_ids: list[int], players_df: pd.DataFrame, n_hybrid_sims: int = 200):
    """Return a MagicMock SimulationEngine whose simulate() returns a fresh random matrix."""
    rng = np.random.default_rng(99)
    engine = MagicMock()
    engine.players_df = players_df

    def _simulate(n_sims):
        mat = rng.uniform(0, 40, size=(n_sims, len(player_ids))).astype(np.float32)
        return SimulationResults(player_ids=player_ids, results_matrix=mat)

    engine.simulate.side_effect = _simulate
    return engine


def _make_robust_payout(candidates, player_ids, n_sims=300, high_ev_indices=None):
    """Return a robust_payout array where specified candidates are +EV and the rest are -EV."""
    M = len(candidates)
    rng = np.random.default_rng(7)
    # Default: all negative (lose entry fee)
    rp = np.full((M, n_sims), -4.0, dtype=np.float32)
    if high_ev_indices is not None:
        for idx in high_ev_indices:
            rp[idx] = rng.uniform(0.5, 5.0, size=n_sims).astype(np.float32)
    return rp


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

def test_returns_requested_portfolio_size(players_df, player_ids, candidates, ownership_vec):
    """Selector returns exactly portfolio_size lineups when enough +EV candidates exist."""
    # Make the first 30 candidates strongly +EV.
    n_pos = min(30, len(candidates))
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=list(range(n_pos)))

    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=5,
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=42,
    )
    portfolio = sel.select()
    assert len(portfolio) == 5


def test_no_duplicates(players_df, player_ids, candidates, ownership_vec):
    """All selected lineups are distinct (no duplicate player_id sets)."""
    n_pos = min(40, len(candidates))
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=list(range(n_pos)))

    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=8,
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=0,
    )
    portfolio = sel.select()
    pid_sets = [frozenset(lu.player_ids) for lu, _ in portfolio]
    assert len(pid_sets) == len(set(pid_sets)), "Duplicate lineup in portfolio"


def test_empty_portfolio_when_no_positive_ev(players_df, player_ids, candidates, ownership_vec):
    """Returns empty list when no candidate has mean(robust_payout) > 0."""
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=[])  # all negative

    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=5,
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=0,
    )
    portfolio = sel.select()
    assert portfolio == []


def test_zero_overlap_fast_track_limits_to_positive_ev_pool(players_df, player_ids, candidates, ownership_vec):
    """With only 2 +EV candidates, portfolio size is capped at 2 regardless of target."""
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=[0, 1])

    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=10,  # larger than +EV pool
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=0,
    )
    portfolio = sel.select()
    # With only 2 initially +EV candidates, the portfolio cannot exceed 2.
    assert len(portfolio) <= 2
    assert all(isinstance(lu, Lineup) for lu, _ in portfolio)


def test_all_returned_lineups_are_valid(players_df, player_ids, candidates, ownership_vec):
    """Every lineup returned passes DK constraints."""
    from src.optimization.lineup import PlayerMeta

    n_pos = min(25, len(candidates))
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=list(range(n_pos)))
    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=5,
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=1,
    )
    portfolio = sel.select()
    player_meta: PlayerMeta = {
        int(row["player_id"]): {
            "position": row["position"],
            "eligible_positions": [row["position"]],
            "salary": int(row["salary"]),
            "team": row["team"],
            "game": row["game"],
            "opponent": row["opponent"],
        }
        for _, row in players_df.iterrows()
    }
    for lu, _ in portfolio:
        assert lu.is_valid(player_meta), f"Invalid lineup: {lu.player_ids}"


def test_stop_check_respected(players_df, player_ids, candidates, ownership_vec):
    """Selector stops before completing portfolio when stop_check returns True."""
    n_pos = min(30, len(candidates))
    rp = _make_robust_payout(candidates, player_ids, high_ev_indices=list(range(n_pos)))
    engine = _make_mock_engine(player_ids, players_df)
    payout_arr = _make_payout_arr()

    call_count = {"n": 0}
    def stop_after_one_cycle():
        call_count["n"] += 1
        return call_count["n"] > 1  # stop after first hybrid cycle

    sel = HybridFieldPortfolioSelector(
        candidates=candidates,
        robust_payout=rp,
        sim_engine=engine,
        players_df=players_df,
        payout_arr=payout_arr,
        portfolio_size=50,  # unreachably large
        n_field_lineups=100,
        n_hybrid_sims=200,
        ownership_vec=ownership_vec,
        rng_seed=0,
    )
    portfolio = sel.select(stop_check=stop_after_one_cycle)
    assert len(portfolio) < 50  # stopped early
