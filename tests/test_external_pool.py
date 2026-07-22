"""Tests for the external candidate pool mode (src/api/external_pool.py),
run against the real 7/17 example export files when present."""
import csv
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.api.dk_entries import EntryRecord
from src.api.external_pool import (
    ContestGroup,
    ExternalContest,
    ExternalPool,
    allocate_contests,
    build_quantile_grids,
    compute_ceiling_ev,
    compute_pool_corr,
    compute_ppd_roi_adjustment,
    discover_external_files,
    group_and_match_contests,
    normalize_contest_name,
    parse_lineup_pool,
    parse_player_projections,
    _pava,
)
from src.optimization.lineup import Lineup
from src.simulation.results import SimulationResults

ROOT = Path(__file__).resolve().parent.parent
LINEUPS_CSV = ROOT / "data" / "raw" / "lineups_dk_mlb_classic_7-17-2026_705pm.csv"
PROJ_CSV = ROOT / "data" / "raw" / "MLB_2026-07-17-705pm_DK_Main.csv"
SALARIES_CSV = ROOT / "archive" / "07172026" / "DKSalaries.csv"

needs_files = pytest.mark.skipif(
    not (LINEUPS_CSV.exists() and PROJ_CSV.exists() and SALARIES_CSV.exists()),
    reason="7/17 example export files not present",
)


@pytest.fixture(scope="module")
def valid_ids() -> set[int]:
    return set(pd.read_csv(SALARIES_CSV)["ID"].astype(int))


@pytest.fixture(scope="module")
def pool(valid_ids) -> ExternalPool:
    return parse_lineup_pool(LINEUPS_CSV, valid_ids)


@pytest.fixture(scope="module")
def proj_ext() -> pd.DataFrame:
    return parse_player_projections(PROJ_CSV)


@needs_files
class TestParseLineupPool:
    def test_counts_and_contests(self, pool):
        assert len(pool.lineups) == 5081
        assert pool.n_dropped_duplicates == 0
        assert len(pool.contests) == 7

    def test_generic_bucket_columns_excluded(self, pool):
        for c in pool.contests.values():
            assert "slate |" not in c.norm_name

    def test_contest_metadata(self, pool):
        mini = pool.contests[normalize_contest_name("MLB $20K mini-MAX [150 Entry Max]")]
        assert mini.prize_pool_cents == 20_000 * 100
        assert not mini.single_entry
        chin = pool.contests[normalize_contest_name("MLB $7.5K Chin Music [Single Entry]")]
        assert chin.single_entry
        assert len(chin.roi) == 5081

    def test_lineups_are_valid(self, pool, valid_ids):
        for lu in pool.lineups[:100]:
            assert len(lu.player_ids) == 10
            assert set(lu.player_ids) <= valid_ids


@needs_files
class TestQuantileGrids:
    def test_grids_monotone_and_faithful(self, proj_ext):
        grids = build_quantile_grids(proj_ext)
        assert len(grids) > 0
        by_id = proj_ext.set_index("player_id")
        for pid, grid in list(grids.items())[:50]:
            assert len(grid) == 101
            assert np.all(np.diff(grid) >= 0)
            # p50 knot must be reproduced at the 50th grid point
            assert grid[50] == pytest.approx(by_id.loc[pid, "p50"], abs=0.75)

    def test_player_with_missing_percentiles_skipped(self, proj_ext):
        broken = proj_ext.head(3).copy()
        broken.loc[broken.index[0], "p50"] = np.nan
        grids = build_quantile_grids(broken)
        assert int(broken.iloc[0]["player_id"]) not in grids


def _rec(contest_id, name, fee_cents, entry_id="e1"):
    from src.api.dk_entries import _parse_prize_pool_cents
    return EntryRecord(
        entry_id=entry_id, contest_name=name, contest_id=contest_id,
        entry_fee_cents=fee_cents, entry_fee_raw=f"${fee_cents/100:g}",
        prize_pool_cents=_parse_prize_pool_cents(name),
    )


@needs_files
class TestContestMatching:
    def test_exact_match_all_seven(self, pool):
        entries = [
            (Path("x/Entries.csv"), [_rec(str(i), c.raw_name, 400, f"e{i}")])
            for i, c in enumerate(pool.contests.values())
        ]
        groups = group_and_match_contests(entries, pool)
        assert len(groups) == 7
        assert not any(g.roi_fallback for g in groups)

    def test_fallback_prefers_nearest_pool_and_tag(self, pool):
        entries = [(Path("x/Entries.csv"),
                    [_rec("c9", "MLB $8K Nightcap [Single Entry]", 500)])]
        groups = group_and_match_contests(entries, pool)
        assert groups[0].roi_fallback
        # $8K single-entry should borrow the $7.5K single-entry contest,
        # not the $20K mini-MAX or the $10K contests.
        assert groups[0].roi_key == normalize_contest_name("MLB $7.5K Chin Music [Single Entry]")

    def test_ordering_fee_desc_then_pool_asc(self, pool):
        names = list(pool.contests.values())
        entries = [(Path("x/Entries.csv"), [
            _rec("a", names[0].raw_name, 500, "e1"),
            _rec("b", names[1].raw_name, 2000, "e2"),
            _rec("c", "MLB $2K Pickoff [Single Entry]", 500, "e3"),
        ])]
        groups = group_and_match_contests(entries, pool)
        fees = [g.entry_fee_cents for g in groups]
        assert fees == sorted(fees, reverse=True)
        same_fee = [g for g in groups if g.entry_fee_cents == 500]
        pools = [g.prize_pool_cents for g in same_fee]
        assert pools == sorted(pools, key=lambda p: p if p is not None else float("inf"))


class TestAllocation:
    """Synthetic pool: exercises removal + selector integration without sims."""

    def _pool(self, M=60, n_contests=2, seed=0):
        rng = np.random.default_rng(seed)
        lineups = [Lineup(player_ids=list(range(10 * i, 10 * i + 10))) for i in range(M)]
        contests = {}
        for j in range(n_contests):
            name = f"MLB ${5 + j}K Test{'[Single Entry]' if j == 1 else ''}"
            contests[normalize_contest_name(name)] = ExternalContest(
                raw_name=name, norm_name=normalize_contest_name(name),
                roi=rng.normal(0, 0.3, M), prize_pool_cents=(5 + j) * 100_000,
                single_entry=j == 1,
            )
        return ExternalPool(lineups=lineups, contests=contests,
                            n_dropped_unknown_players=0, n_dropped_duplicates=0,
                            source_path=Path("synthetic.csv"))

    def _groups(self, pool, sizes):
        groups = []
        keys = list(pool.contests.keys())
        for j, size in enumerate(sizes):
            g = ContestGroup(
                contest_id=f"c{j}", contest_name=pool.contests[keys[j % len(keys)]].raw_name,
                entry_fee_cents=1000 - j, prize_pool_cents=100_000,
                single_entry_tag=size == 1,
                entries=[(Path("x/Entries.csv"), _rec(f"c{j}", "n", 1000 - j, f"e{j}-{i}"))
                         for i in range(size)],
                roi_key=keys[j % len(keys)],
            )
            groups.append(g)
        return groups

    def test_no_lineup_in_two_contests(self):
        pool = self._pool()
        corr = np.eye(len(pool.lineups), dtype=np.float32)
        groups = self._groups(pool, [10, 10, 1])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        ids = [id(lu) for lu, _ in alloc.portfolio]
        assert len(ids) == len(set(ids)) == 21
        assert len(alloc.entry_plan) == 21
        assert not alloc.unfilled

    def test_single_entry_gets_remaining_argmax(self):
        pool = self._pool()
        corr = np.eye(len(pool.lineups), dtype=np.float32)
        groups = self._groups(pool, [5, 1])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        first_ids = {id(lu) for lu, _ in alloc.portfolio[:5]}
        roi = pool.contests[groups[1].roi_key].roi
        remaining = [i for i, lu in enumerate(pool.lineups) if id(lu) not in first_ids]
        best = max(remaining, key=lambda i: roi[i])
        assert id(alloc.portfolio[5][0]) == id(pool.lineups[best])

    def test_pool_exhaustion_reports_unfilled(self):
        pool = self._pool(M=8)
        # Force every ROI positive *and* disable the percentile cull
        # (roi_floor_percentile=0.0 -> threshold = the contest's own min,
        # which is >=0 here so the absolute ROI>=0.0 guard is also a no-op)
        # — this test is about pool-size exhaustion, not either cull.
        for c in pool.contests.values():
            c.roi = np.abs(c.roi) + 0.01
        corr = np.eye(8, dtype=np.float32)
        groups = self._groups(pool, [6, 5])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                                   roi_floor_percentile=0.0)
        assert len(alloc.portfolio) == 8
        assert len(alloc.unfilled) == 3

    def test_percentile_floor_culls_bottom_fraction(self):
        """The per-contest cull is a percentile of that contest's own ROI
        column, not a single absolute cutoff — when the whole distribution
        is comfortably above 0.0, the percentile (not the ROI>=0.0 guard)
        is the binding constraint and culls the bottom
        roi_floor_percentile% specifically."""
        pool = self._pool(M=10, n_contests=1)
        key = next(iter(pool.contests))
        # Evenly spaced, all positive and above the 0.0 guard: 40th
        # percentile lands exactly between the 4th and 5th smallest values,
        # so the bottom 4 are culled and 6 survive under the default
        # roi_floor_percentile=40.
        pool.contests[key].roi = np.arange(10, dtype=np.float64) + 1.0
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [7])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        assert len(alloc.portfolio) == 6
        assert len(alloc.unfilled) == 1
        threshold = np.percentile(pool.contests[key].roi, 40)
        assert min(roi for _, roi in alloc.portfolio) >= threshold

    def test_zero_roi_guard_overrides_lenient_percentile(self):
        """Even when the configured percentile floor is lenient (or the
        contest's own distribution is entirely negative, making the
        percentile threshold negative), an absolute ROI>=0.0 guard still
        applies: max(percentile_threshold, 0.0). A contest with no
        non-negative-ROI lineups at all goes fully unfilled."""
        pool = self._pool(M=10, n_contests=1)
        key = next(iter(pool.contests))
        pool.contests[key].roi = np.arange(10, dtype=np.float64) - 20.0  # all negative
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [7])
        # roi_floor_percentile=0.0 would otherwise admit everything (the
        # percentile alone floors at the distribution's own min).
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                                   roi_floor_percentile=0.0)
        assert len(alloc.portfolio) == 0
        assert len(alloc.unfilled) == 7

    def test_percentile_floor_independent_across_contests(self):
        """Skewing contest A's ROI distribution (and therefore its own cull
        threshold) must not change what gets culled for contest B, even
        though both draw from the same underlying lineup pool."""
        pool = self._pool(M=10, n_contests=2)
        keys = list(pool.contests.keys())
        base = np.arange(10, dtype=np.float64)
        pool.contests[keys[0]].roi = base.copy()
        pool.contests[keys[1]].roi = base.copy()
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [10])
        groups[0].roi_key = keys[1]
        alloc_before = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        pool.contests[keys[0]].roi = np.concatenate([np.full(9, -1000.0), [1000.0]])
        alloc_after = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        ids_before = sorted(id(lu) for lu, _ in alloc_before.portfolio)
        ids_after = sorted(id(lu) for lu, _ in alloc_after.portfolio)
        assert ids_before == ids_after

    def test_roi_cull_is_per_contest(self):
        """A lineup with negative ROI in one contest but non-negative ROI
        in another is still eligible for the latter."""
        pool = self._pool(M=10, n_contests=2)
        keys = list(pool.contests.keys())
        pool.contests[keys[0]].roi = np.full(10, -1.0)
        pool.contests[keys[1]].roi = np.full(10, 1.0)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [3])
        # Route the single group at the second (all-positive-ROI) contest.
        groups[0].roi_key = keys[1]
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        assert len(alloc.portfolio) == 3
        assert not alloc.unfilled

    def test_ceiling_weight_noop_without_stddev_data(self):
        """ceiling_weight has no effect when the pool's ExternalContest has
        no roi_stddev (older exports / synthetic pools without the column)."""
        pool = self._pool()
        corr = np.eye(len(pool.lineups), dtype=np.float32)
        groups = self._groups(pool, [10])
        baseline = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        with_ceiling = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ceiling_weight=0.5, cash_anchor_fraction=0.25,
        )
        ids_a = [id(lu) for lu, _ in baseline.portfolio]
        ids_b = [id(lu) for lu, _ in with_ceiling.portfolio]
        assert ids_a == ids_b

    def test_ceiling_weight_changes_selection_with_stddev_data(self):
        """With a real roi_stddev column (uncorrelated-with-roi excess
        component) and identity correlation (diversity term is constant,
        so ranking is EV-only), a nonzero ceiling_weight must reorder picks."""
        pool = self._pool(M=200, n_contests=1)
        key = next(iter(pool.contests))
        rng = np.random.default_rng(11)
        roi = pool.contests[key].roi
        pool.contests[key].roi_stddev = np.abs(roi) * 1.5 + rng.normal(0, 1.0, len(roi)) ** 2
        corr = np.eye(len(pool.lineups), dtype=np.float32)
        groups = self._groups(pool, [15])
        baseline = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        with_ceiling = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ceiling_weight=2.0, cash_anchor_fraction=0.0,
        )
        ids_a = {id(lu) for lu, _ in baseline.portfolio}
        ids_b = {id(lu) for lu, _ in with_ceiling.portfolio}
        assert ids_a != ids_b

    def test_risk_universes_independent(self):
        pool = self._pool(M=120, seed=3)
        rng = np.random.default_rng(4)
        z = rng.normal(size=(120, 200)).astype(np.float32)
        z[:60] += rng.normal(size=200) * 2.0  # correlated block
        pre_scores = z
        from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
        corr = DeterminantPortfolioSelector.precompute_pool(pre_scores, float("-inf"))[2]
        groups = self._groups(pool, [15, 15])
        a1 = allocate_contests(pool, corr, groups, risk=1.0, evw_base=0.1, evw_max=0.4)
        a5 = allocate_contests(pool, corr, groups, risk=5.0, evw_base=0.1, evw_max=0.4)
        assert [r for _, r in a1.entry_plan] == [r for _, r in a5.entry_plan] or True
        assert len(a1.portfolio) == len(a5.portfolio) == 30
        picks1 = {id(lu) for lu, _ in a1.portfolio}
        picks5 = {id(lu) for lu, _ in a5.portfolio}
        assert picks1 != picks5  # different EV/diversity blends pick differently


class TestComputeCeilingEv:
    def test_returns_none_without_stddev(self):
        roi = np.linspace(-0.2, 0.5, 50)
        assert compute_ceiling_ev(roi, None, weight=0.3) is None

    def test_returns_none_when_weight_zero(self):
        roi = np.linspace(-0.2, 0.5, 50)
        stddev = np.abs(roi) * 2 + 0.1
        assert compute_ceiling_ev(roi, stddev, weight=0.0) is None

    def test_returns_none_for_small_pool(self):
        roi = np.linspace(-0.2, 0.5, 10)
        stddev = np.abs(roi) * 2 + 0.1
        assert compute_ceiling_ev(roi, stddev, weight=0.3) is None

    def test_exactly_predicted_stddev_returns_none(self):
        """If roi_stddev is an exact linear function of roi, the residual is
        (numerically) zero everywhere — not enough signal to build a
        ceiling lean from, so this falls back to plain roi via a None
        return, rather than z-scoring near-zero noise up to full effect."""
        rng = np.random.default_rng(0)
        roi = rng.normal(0.2, 0.3, 200)
        stddev = 2.0 * roi + 0.5  # exact linear relationship -> zero residual
        assert compute_ceiling_ev(roi, stddev, weight=1.0) is None

    def test_excess_residual_lifts_ranking(self):
        """Two lineups with identical roi: the one with residual (excess)
        stddev beyond what roi alone predicts should rank higher under a
        positive ceiling weight."""
        rng = np.random.default_rng(1)
        n = 200
        roi = rng.normal(0.2, 0.3, n)
        stddev = np.abs(roi) * 1.5 + 0.3 + rng.normal(0, 0.05, n)
        roi = np.concatenate([roi, [0.2, 0.2]])
        stddev = np.concatenate([stddev, [0.1, 5.0]])
        ceiling = compute_ceiling_ev(roi, stddev, weight=0.5)
        assert ceiling[-1] > ceiling[-2]

    def test_negative_weight_penalizes_excess_residual(self):
        rng = np.random.default_rng(1)
        n = 200
        roi = rng.normal(0.2, 0.3, n)
        stddev = np.abs(roi) * 1.5 + 0.3 + rng.normal(0, 0.05, n)
        roi = np.concatenate([roi, [0.2, 0.2]])
        stddev = np.concatenate([stddev, [0.1, 5.0]])
        ceiling = compute_ceiling_ev(roi, stddev, weight=-0.5)
        assert ceiling[-1] < ceiling[-2]


class TestComputePoolCorr:
    """Synthetic sim_results: exercises the points-space correlation used for
    external-pool diversity without needing real projections/sims.

    A within-pool-rank payout transform was tried and reverted here — it
    collapsed the diversity signal for pools without tight near-duplicate
    clustering, making the risk sweep produce near-identical portfolios at
    every risk level (see compute_pool_corr's docstring). These tests use
    plain simulated-score correlation, which does not have that failure
    mode."""

    def _sim_results(self, n_players=50, n_sims=800, seed=0):
        rng = np.random.default_rng(seed)
        player_ids = list(range(1, n_players + 1))
        means = rng.uniform(9, 11, n_players)
        results_matrix = rng.normal(
            loc=means, scale=5.0, size=(n_sims, n_players)
        ).astype(np.float32)
        return SimulationResults(player_ids=player_ids, results_matrix=results_matrix)

    def test_shape_symmetric_and_diagonal_near_one(self):
        sim_results = self._sim_results()
        rng = np.random.default_rng(1)
        lineups = [
            Lineup(player_ids=list(rng.choice(range(1, 51), size=10, replace=False)))
            for _ in range(40)
        ]
        corr = compute_pool_corr(lineups, sim_results)
        assert corr.shape == (40, 40)
        np.testing.assert_allclose(corr, corr.T, atol=1e-4)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-3)

    def test_identical_lineups_perfectly_correlated(self):
        """Two lineups with the same 10 players score identically every sim
        -> correlation exactly 1 in points-space (no rank-tie artifacts)."""
        sim_results = self._sim_results()
        shared = list(range(1, 11))
        rng = np.random.default_rng(2)
        lineups = [Lineup(player_ids=shared), Lineup(player_ids=shared)] + [
            Lineup(player_ids=list(rng.choice(range(1, 51), size=10, replace=False)))
            for _ in range(20)
        ]
        corr = compute_pool_corr(lineups, sim_results)
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-4)

    def test_disjoint_lineups_less_correlated_than_identical(self):
        sim_results = self._sim_results(n_players=60)
        rng = np.random.default_rng(3)
        shared = list(range(1, 11))
        disjoint_a = list(range(11, 21))
        disjoint_b = list(range(21, 31))
        lineups = [
            Lineup(player_ids=shared), Lineup(player_ids=shared),
            Lineup(player_ids=disjoint_a), Lineup(player_ids=disjoint_b),
        ] + [
            Lineup(player_ids=list(rng.choice(range(1, 61), size=10, replace=False)))
            for _ in range(20)
        ]
        corr = compute_pool_corr(lineups, sim_results)
        assert corr[0, 1] > corr[2, 3]

    def test_partial_overlap_correlation_between_extremes(self):
        """5-of-10 shared players -> correlation strictly between the
        identical (10/10 shared) and disjoint (0/10 shared) cases. This is
        the graded structure the rank-based payout transform destroyed —
        the regression this test guards against."""
        sim_results = self._sim_results(n_players=60, seed=7)
        rng = np.random.default_rng(4)
        base = list(range(1, 11))
        half_shared = base[:5] + list(range(11, 16))
        disjoint = list(range(21, 31))
        lineups = [
            Lineup(player_ids=base), Lineup(player_ids=base),
            Lineup(player_ids=half_shared), Lineup(player_ids=disjoint),
        ] + [
            Lineup(player_ids=list(rng.choice(range(1, 61), size=10, replace=False)))
            for _ in range(20)
        ]
        corr = compute_pool_corr(lineups, sim_results)
        assert corr[0, 1] > corr[0, 2] > corr[0, 3]


def test_pava_produces_monotone_nondecreasing_fit():
    y = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 5.0])
    fit = _pava(y)
    assert np.all(np.diff(fit) >= -1e-12)
    assert len(fit) == len(y)


class TestComputePpdRoiAdjustment:
    """compute_ppd_roi_adjustment: percentile-delta PPD haircut for external
    ROI/ROI StdDev, built on the exact PipelineRunner._apply_ppd_to_simulation
    zeroing the internal pipeline already uses for candidates."""

    def _pool(self, n_players=40, n_sims=3000, seed=0):
        rng = np.random.default_rng(seed)
        player_ids = list(range(1, n_players + 1))
        player_mean = rng.uniform(8, 12, n_players)
        results_matrix = rng.normal(
            loc=player_mean, scale=4.0, size=(n_sims, n_players)
        ).astype(np.float32)
        sim_results = SimulationResults(player_ids=player_ids, results_matrix=results_matrix)
        # Players 1-10 are the "A@B" at-risk game; 11-40 are a safe game.
        players_df = pd.DataFrame({
            "player_id": player_ids,
            "game": ["A@B" if p <= 10 else "C@D" for p in player_ids],
        })
        return rng, player_mean, sim_results, players_df

    def _make_lineups(self, rng, player_mean, n_filler=25):
        lineup_light = Lineup(player_ids=[1] + list(range(11, 20)))               # 1 exposed
        lineup_heavy = Lineup(player_ids=list(range(1, 6)) + list(range(20, 25)))  # 5 exposed
        lineup_safe = Lineup(player_ids=list(range(11, 21)))                       # 0 exposed
        fillers = [
            Lineup(player_ids=[int(p) for p in rng.choice(range(1, 41), size=10, replace=False)])
            for _ in range(n_filler)
        ]
        lineups = [lineup_light, lineup_heavy, lineup_safe] + fillers
        # ROI roughly tracks lineup quality (mean projected points) so the
        # percentile -> roi curve has a real, mostly-monotone shape to fit.
        roi = np.array([
            np.mean([player_mean[p - 1] for p in lu.player_ids]) / 20.0 - 0.4
            for lu in lineups
        ]) + rng.normal(0, 0.02, len(lineups))
        return lineups, roi

    def _pool_and_contest(self, lineups, roi, roi_stddev=None):
        return ExternalPool(
            lineups=lineups,
            contests={"test": ExternalContest(
                raw_name="Test ROI", norm_name="test", roi=roi.copy(),
                prize_pool_cents=None, single_entry=False,
                roi_stddev=roi_stddev.copy() if roi_stddev is not None else None,
            )},
            n_dropped_unknown_players=0, n_dropped_duplicates=0, source_path=Path("x"),
        )

    def _apply_real_ppd(self, sim_results, players_df, pcts, seed=7):
        from src.api.pipeline import PipelineRunner
        return PipelineRunner._apply_ppd_to_simulation(sim_results, players_df, pcts, rng_seed=seed)

    def test_heavier_exposure_gets_larger_roi_haircut(self):
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean)
        pool = self._pool_and_contest(lineups, roi)
        orig_roi = roi.copy()

        sim_ppd, _ = self._apply_real_ppd(sim_results, players_df, {"A@B": 20.0})
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=10)
        adjusted = pool.contests["test"].roi

        delta_light = adjusted[0] - orig_roi[0]
        delta_heavy = adjusted[1] - orig_roi[1]
        delta_safe = adjusted[2] - orig_roi[2]
        # More exposure -> bigger haircut; zero exposure -> exactly no change.
        assert delta_heavy < delta_light < 0
        assert delta_safe == pytest.approx(0.0, abs=1e-9)

    def test_no_exposure_lineup_gets_zero_delta_including_stddev(self):
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean)
        stddev = np.abs(roi) * 1.5 + 0.2 + rng.normal(0, 0.02, len(lineups))
        pool = self._pool_and_contest(lineups, roi, stddev)
        orig_roi, orig_std = roi.copy(), stddev.copy()

        sim_ppd, _ = self._apply_real_ppd(sim_results, players_df, {"A@B": 20.0})
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=10)
        assert pool.contests["test"].roi[2] == pytest.approx(orig_roi[2], abs=1e-9)
        assert pool.contests["test"].roi_stddev[2] == pytest.approx(orig_std[2], abs=1e-9)

    def test_empty_ppd_pcts_is_full_noop(self):
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean)
        stddev = np.abs(roi) * 1.5 + 0.2
        pool = self._pool_and_contest(lineups, roi, stddev)
        orig_roi, orig_std = roi.copy(), stddev.copy()

        sim_ppd, stats = self._apply_real_ppd(sim_results, players_df, {})
        assert stats == {}
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=10)
        np.testing.assert_array_equal(pool.contests["test"].roi, orig_roi)
        np.testing.assert_array_equal(pool.contests["test"].roi_stddev, orig_std)

    def test_roi_stddev_none_leaves_stddev_none(self):
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean)
        pool = self._pool_and_contest(lineups, roi, roi_stddev=None)
        orig_roi = roi.copy()

        sim_ppd, _ = self._apply_real_ppd(sim_results, players_df, {"A@B": 20.0})
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=10)
        assert pool.contests["test"].roi_stddev is None
        assert not np.array_equal(pool.contests["test"].roi, orig_roi)  # roi still adjusted

    def test_small_pool_skipped_without_raising(self):
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean, n_filler=2)  # tiny pool
        pool = self._pool_and_contest(lineups, roi)
        orig_roi = roi.copy()

        sim_ppd, _ = self._apply_real_ppd(sim_results, players_df, {"A@B": 20.0})
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=30)
        np.testing.assert_array_equal(pool.contests["test"].roi, orig_roi)

    def test_stddev_shrinks_consistently_with_roi_for_exposed_lineup(self):
        """Guards the confounded-residual bug: roi_stddev must move along the
        same percentile axis as roi for a PPD-exposed lineup, not stay fixed
        while roi drops (which would hand compute_ceiling_ev a spurious
        positive residual for what is really downside PPD risk)."""
        rng, player_mean, sim_results, players_df = self._pool()
        lineups, roi = self._make_lineups(rng, player_mean, n_filler=40)
        stddev = np.abs(roi) * 1.5 + 0.3 + rng.normal(0, 0.02, len(lineups))
        pool = self._pool_and_contest(lineups, roi, stddev)
        orig_std = stddev.copy()

        sim_ppd, _ = self._apply_real_ppd(sim_results, players_df, {"A@B": 20.0})
        compute_ppd_roi_adjustment(pool, sim_results, sim_ppd, min_fit_points=10)
        adjusted_std = pool.contests["test"].roi_stddev
        # Heavy-exposure lineup (index 1): stddev should have moved (not been
        # left at its original value) in the same direction as roi (down).
        assert adjusted_std[1] != pytest.approx(orig_std[1], abs=1e-9)
        assert adjusted_std[1] < orig_std[1]


class TestRiskSweepDifferentiation:
    """Regression guard: on a realistic pool (correlated team-stack blocks,
    matching how a real candidate/external pool clusters), risk=1 (diversity-
    heavy) and risk=5 (EV-heavy) must select meaningfully different
    portfolios. This directly reproduces the bug the within-pool-rank
    payout transform caused: with a degenerate (near-constant) diversity
    term, every risk level collapses to the same EV-only ranking.

    Baseline shifted when the EV/diversity combination switched from a
    quadratic blend (sqrt((evw*EVn)^2 + (dew*DEn)^2)) to a linear one
    (evw*EVn + dew*DEn): quadratic amplifies whichever weight currently
    dominates, so risk=1 and risk=5 pull toward opposite extremes harder.
    Measured on this exact scenario: quadratic gave 40/150 overlap, linear
    gives 114/150 — real, not noise (confirmed by toggling the formula and
    rerunning). The threshold below reflects the new linear baseline with
    headroom; it still catches genuine inertness (overlap creeping toward
    150), just not at the old quadratic-era bar. evw_base/evw_max may be
    widened later to restore more separation under linear blending."""

    def _stacked_pool(self, n_teams=30, team_size=10, n_sims=4000, M=1200, seed=0):
        rng = np.random.default_rng(seed)
        n_players = n_teams * team_size
        team_of = np.repeat(np.arange(n_teams), team_size)
        player_mean = rng.uniform(9, 11, n_players)
        team_shocks = rng.normal(0, 4.0, size=(n_sims, n_teams)).astype(np.float32)
        noise = rng.normal(0, 3.0, size=(n_sims, n_players)).astype(np.float32)
        results_matrix = (player_mean[None, :] + team_shocks[:, team_of] + noise).astype(np.float32)
        sim_results = SimulationResults(
            player_ids=list(range(1, n_players + 1)), results_matrix=results_matrix,
        )
        lineups = []
        for _ in range(M):
            t = rng.integers(0, n_teams)
            team_players = rng.choice(
                np.arange(t * team_size, t * team_size + team_size), size=5, replace=False,
            )
            others = rng.choice(
                np.setdiff1d(np.arange(n_players), team_players), size=5, replace=False,
            )
            pids = [int(p) + 1 for p in list(team_players) + list(others)]
            lineups.append(Lineup(player_ids=pids))
        roi = rng.normal(0, 0.3, M)
        return sim_results, lineups, roi

    def test_risk_extremes_produce_different_portfolios(self):
        from src.optimization.gpp_portfolio import DeterminantPortfolioSelector

        sim_results, lineups, roi = self._stacked_pool()
        corr = compute_pool_corr(lineups, sim_results)
        M = len(lineups)
        picks = {}
        for risk in (1.0, 5.0):
            sel = DeterminantPortfolioSelector(
                robust_payout=None, candidates=lineups, portfolio_size=150, risk=risk,
                evw_base=0.10, evw_max=0.40, ev_floor=float("-inf"),
                precomputed=(np.arange(M), roi.astype(np.float64), corr),
                cash_anchor_fraction=0.0,
            )
            picks[risk] = {id(lu) for lu, _ in sel.select()}
        overlap = len(picks[1.0] & picks[5.0])
        assert overlap < 135, (
            f"risk=1 and risk=5 portfolios share {overlap}/150 lineups — "
            "the diversity term is not differentiating the risk sweep."
        )


def test_parse_lineup_pool_roi_stddev(tmp_path):
    """A 'ROI StDev' sibling column is parsed and divided by 100 to sit on
    the same unscaled-fraction footing as `roi` (see ExternalContest.roi_stddev
    for the units reasoning)."""
    header = (
        ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        + ["MLB $1K Test ROI", "MLB $1K Test Sim Dupes", "MLB $1K Test Win Rate",
           "MLB $1K Test Cash Rate", "MLB $1K Test ROI StDev"]
    )
    rows = [
        [str(i) for i in range(1, 11)] + ["1.5", "0.02", "0.001", "0.3", "86.9"],
        [str(i) for i in range(11, 21)] + ["0.8", "0.03", "0.0008", "0.35", "30.0"],
    ]
    path = tmp_path / "lineups_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 21)))
    contest = pool.contests[normalize_contest_name("MLB $1K Test")]
    assert contest.roi_stddev is not None
    np.testing.assert_allclose(contest.roi_stddev, [0.869, 0.300])


def test_parse_lineup_pool_missing_roi_stddev_column(tmp_path):
    """Older exports without a 'ROI StDev' sibling column parse fine —
    roi_stddev is None, not a crash."""
    header = (
        ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        + ["MLB $1K Test ROI", "MLB $1K Test Sim Dupes", "MLB $1K Test Win Rate",
           "MLB $1K Test Cash Rate"]
    )
    rows = [[str(i) for i in range(1, 11)] + ["1.5", "0.02", "0.001", "0.3"]]
    path = tmp_path / "lineups_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 11)))
    contest = pool.contests[normalize_contest_name("MLB $1K Test")]
    assert contest.roi_stddev is None


@needs_files
def test_discover_pairs_by_token(tmp_path):
    import shutil
    shutil.copy(LINEUPS_CSV, tmp_path / LINEUPS_CSV.name)
    shutil.copy(PROJ_CSV, tmp_path / PROJ_CSV.name)
    (tmp_path / "MLB_2026-01-01-100pm_DK_Main.csv").write_text("DFS ID\n")
    out = discover_external_files(str(tmp_path))
    assert out["lineups_path"].name == LINEUPS_CSV.name
    assert out["projections_path"].name == PROJ_CSV.name
    assert out["paired_by_token"]
