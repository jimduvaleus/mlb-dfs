"""Tests for LeveragePortfolioSelector (src/optimization/gpp_portfolio.py)."""
import numpy as np
import pytest

from src.optimization.gpp_portfolio import LeveragePortfolioSelector
from src.optimization.lineup import Lineup


def _make_lineup(player_ids: list[int]) -> Lineup:
    return Lineup(player_ids=player_ids)


def _indicator(lineups: list[Lineup], all_ids: list[int]) -> np.ndarray:
    col = {pid: i for i, pid in enumerate(all_ids)}
    I = np.zeros((len(all_ids), len(lineups)), dtype=np.float64)
    for j, lu in enumerate(lineups):
        for pid in lu.player_ids:
            I[col[pid], j] = 1.0
    return I


# ------------------------------------------------------------------ #
#  _admissible_subset (band-widening)                                 #
# ------------------------------------------------------------------ #

class TestAdmissibleSubset:
    def test_widens_band_until_min_candidates_met(self):
        all_ids = list(range(1, 11))
        lineups = [_make_lineup([pid]) for pid in all_ids]
        p_opt = np.array([0.001 * (i + 1) for i in range(10)])  # 0.001 .. 0.010
        I = _indicator(lineups, all_ids)
        sel = LeveragePortfolioSelector(
            candidates=lineups, portfolio_size=3, p_opt=p_opt,
            optimal_ownership=np.zeros(10), leverage_diff=np.zeros(10),
            leverage_ratio=np.zeros(10), player_indicator=I,
            field_size=100.0, target_anchor_c=1.0,
            band_widen_steps=(1.0, 0.5, 0.25, 0.1), min_candidates=5,
        )
        # anchor = 1.0/100 = 0.01.
        #   mult=1.0 -> threshold 0.01 -> only p_opt[9]=0.010            (1, < 5)
        #   mult=0.5 -> threshold 0.005 -> p_opt[4..9]                    (6, >= 5) -> stop
        idx = sel._admissible_subset()
        assert set(idx.tolist()) == {4, 5, 6, 7, 8, 9}

    def test_prefer_leverage_over_sizing_falls_back_to_whole_pool(self):
        all_ids = list(range(1, 6))
        lineups = [_make_lineup([pid]) for pid in all_ids]
        p_opt = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        I = _indicator(lineups, all_ids)
        sel = LeveragePortfolioSelector(
            candidates=lineups, portfolio_size=2, p_opt=p_opt,
            optimal_ownership=np.zeros(5), leverage_diff=np.zeros(5),
            leverage_ratio=np.zeros(5), player_indicator=I,
            field_size=1.0, target_anchor_c=1.0,
            band_widen_steps=(1.0,), min_candidates=1000,
            prefer_leverage_over_sizing=True,
        )
        idx = sel._admissible_subset()
        assert len(idx) == 5  # every candidate has finite p_opt

    def test_prefer_leverage_over_sizing_false_keeps_widest_step_short(self):
        all_ids = list(range(1, 6))
        lineups = [_make_lineup([pid]) for pid in all_ids]
        p_opt = np.array([0.001, 0.002, 0.03, 0.04, 0.05])
        I = _indicator(lineups, all_ids)
        sel = LeveragePortfolioSelector(
            candidates=lineups, portfolio_size=2, p_opt=p_opt,
            optimal_ownership=np.zeros(5), leverage_diff=np.zeros(5),
            leverage_ratio=np.zeros(5), player_indicator=I,
            field_size=1.0, target_anchor_c=0.05,
            band_widen_steps=(1.0, 0.5, 0.1), min_candidates=1000,
            prefer_leverage_over_sizing=False,
        )
        # anchor = 0.05/1.0 = 0.05.
        #   mult=1.0 -> threshold 0.05  -> {4}          (1, < 1000)
        #   mult=0.5 -> threshold 0.025 -> {2,3,4}       (3, < 1000)
        #   mult=0.1 -> threshold 0.005 -> still {2,3,4} (3, < 1000) -- widest step
        idx = sel._admissible_subset()
        assert set(idx.tolist()) == {2, 3, 4}


# ------------------------------------------------------------------ #
#  select()                                                            #
# ------------------------------------------------------------------ #

class TestSelect:
    def _wide_open_selector(self, **overrides):
        """4 single-player candidates, all trivially admissible (huge anchor
        band), so select() tests can isolate the ranking behavior."""
        all_ids = [1, 2, 3, 4]
        lineups = [_make_lineup([pid]) for pid in all_ids]
        I = _indicator(lineups, all_ids)
        defaults = dict(
            candidates=lineups, portfolio_size=2, p_opt=np.full(4, 0.5),
            optimal_ownership=np.zeros(4), leverage_diff=np.zeros(4),
            leverage_ratio=np.zeros(4), player_indicator=I,
            field_size=1.0, target_anchor_c=1e-6,
        )
        defaults.update(overrides)
        return LeveragePortfolioSelector(**defaults)

    def test_coverage_weight_zero_ranks_purely_by_leverage_ratio(self):
        sel = self._wide_open_selector(
            leverage_ratio=np.array([1.0, 3.0, 2.0, 0.5]), coverage_weight=0.0,
        )
        result = sel.select()
        picked_ids = [lu.player_ids[0] for lu, _ in result]
        assert picked_ids == [2, 3]  # leverage_ratio 3.0 then 2.0, ignoring the rest

    def test_coverage_weight_one_ignores_candidate_own_leverage_ratio(self):
        # Only player 1 has positive leverage (so only it gets a nonzero
        # coverage target); its own leverage_ratio is deliberately the
        # WORST of the four so a coverage_weight=1.0 pick can only be
        # explained by the coverage term, not by candidate quality.
        sel = self._wide_open_selector(
            leverage_diff=np.array([5.0, -1.0, -1.0, -1.0]),
            optimal_ownership=np.array([50.0, 50.0, 50.0, 50.0]),
            leverage_ratio=np.array([-10.0, 10.0, 10.0, 10.0]),
            coverage_weight=1.0, portfolio_size=1,
        )
        result = sel.select()
        assert result[0][0].player_ids == [1]

    def test_negative_leverage_player_gets_no_coverage_target_even_with_high_ownership(self):
        # Same optimal_ownership for both, opposite leverage sign -- only
        # the positive-leverage one should drive the coverage term.
        sel = self._wide_open_selector(
            candidates=[_make_lineup([1]), _make_lineup([2])],
            p_opt=np.full(2, 0.5),
            optimal_ownership=np.array([50.0, 50.0]),
            leverage_diff=np.array([5.0, -5.0]),
            leverage_ratio=np.array([0.0, 10.0]),
            player_indicator=_indicator(
                [_make_lineup([1]), _make_lineup([2])], [1, 2]),
            coverage_weight=1.0, portfolio_size=1,
        )
        result = sel.select()
        assert result[0][0].player_ids == [1]

    def test_returns_requested_size_and_no_duplicate_candidates(self):
        rng = np.random.default_rng(0)
        n_players, n_lineups, roster_size = 20, 40, 3
        ids = list(range(1, n_players + 1))
        lineups = [
            _make_lineup(sorted(rng.choice(ids, size=roster_size, replace=False).tolist()))
            for _ in range(n_lineups)
        ]
        I = _indicator(lineups, ids)
        sel = LeveragePortfolioSelector(
            candidates=lineups, portfolio_size=8,
            p_opt=rng.uniform(0.01, 0.5, size=n_lineups),
            optimal_ownership=rng.uniform(0.0, 20.0, size=n_players),
            leverage_diff=rng.uniform(-5.0, 5.0, size=n_players),
            leverage_ratio=rng.uniform(-1.0, 1.0, size=n_players),
            player_indicator=I, field_size=50.0,
        )
        result = sel.select()
        assert len(result) == 8
        assert len({id(lu) for lu, _ in result}) == 8

    def test_empty_admissible_subset_returns_empty_list(self):
        all_ids = [1, 2]
        lineups = [_make_lineup([pid]) for pid in all_ids]
        I = _indicator(lineups, all_ids)
        sel = LeveragePortfolioSelector(
            candidates=lineups, portfolio_size=1, p_opt=np.array([0.001, 0.002]),
            optimal_ownership=np.zeros(2), leverage_diff=np.zeros(2),
            leverage_ratio=np.zeros(2), player_indicator=I,
            field_size=1.0, target_anchor_c=1.0, band_widen_steps=(1.0,),
            min_candidates=1000, prefer_leverage_over_sizing=False,
        )
        assert sel.select() == []
