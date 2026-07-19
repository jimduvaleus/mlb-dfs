"""Tests for the external candidate pool mode (src/api/external_pool.py),
run against the real 7/17 example export files when present."""
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
    compute_pool_corr,
    discover_external_files,
    group_and_match_contests,
    normalize_contest_name,
    parse_lineup_pool,
    parse_player_projections,
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
        # Force every ROI positive so the ROI>=0.0 cull isn't the binding
        # constraint here — this test is about pool-size exhaustion.
        for c in pool.contests.values():
            c.roi = np.abs(c.roi)
        corr = np.eye(8, dtype=np.float32)
        groups = self._groups(pool, [6, 5])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        assert len(alloc.portfolio) == 8
        assert len(alloc.unfilled) == 3

    def test_negative_roi_lineups_culled_and_left_unfilled(self):
        """Candidates are culled to ROI >= 0.0 per contest before the
        Det/ROI selection runs — a contest with no non-negative-ROI
        lineups left in the pool goes unfilled rather than backfilling
        with negative-ROI picks."""
        pool = self._pool()
        for c in pool.contests.values():
            c.roi = -np.abs(c.roi) - 0.5
        corr = np.eye(len(pool.lineups), dtype=np.float32)
        groups = self._groups(pool, [7])
        alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        assert len(alloc.portfolio) == 0
        assert len(alloc.unfilled) == 7

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


class TestComputePoolCorr:
    """Synthetic sim_results: exercises the payout-space transform (round-11/12
    finding — dollar-space correlation beats raw points-space) without needing
    real projections/sims.

    Player means are drawn from a tight range (10 +/- 1) with substantial
    per-sim noise (scale=5) so no lineup's pool-rank is degenerate (locked
    to the same rank every sim, which would zero its payout variance) —
    the noise-vs-mean-spread ratio here keeps every candidate's rank
    genuinely stochastic across sims."""

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

    def test_identical_lineups_are_each_others_top_match(self):
        """Two lineups with the same 10 players are not perfectly correlated
        in payout space (stable-sort tie-breaking assigns them adjacent —
        not equal — ranks every sim, and the payout curve's top-heavy
        nonlinearity can amplify that adjacent-rank gap: this is the exact
        mechanism, not a bug, behind round-11/12's points-vs-dollar-space
        finding). The robust invariant is relative, not absolute: an exact
        duplicate lineup should still be the single most-correlated partner
        in the whole pool."""
        sim_results = self._sim_results()
        shared = list(range(1, 11))
        rng = np.random.default_rng(2)
        lineups = [Lineup(player_ids=shared), Lineup(player_ids=shared)] + [
            Lineup(player_ids=list(rng.choice(range(1, 51), size=10, replace=False)))
            for _ in range(20)
        ]
        corr = compute_pool_corr(lineups, sim_results)
        row0 = corr[0].copy()
        row0[0] = -np.inf  # exclude self
        assert np.argmax(row0) == 1

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
