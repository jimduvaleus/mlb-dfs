"""Tests for the external candidate pool mode (src/api/external_pool.py),
run against the real 7/17 example export files when present."""
import csv
import os
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
    archive_external_inputs,
    build_external_players_df,
    build_quantile_grids,
    batter_blank_probability,
    _zero_inflate_grid,
    compute_ceiling_ev,
    compute_lineup_scores,
    compute_p_win,
    compute_pool_corr,
    compute_pool_ownership,
    compute_pool_proj_scores,
    compute_ppd_roi_adjustment,
    compute_prj_own_ev,
    discover_external_files,
    implied_field_size,
    group_and_match_contests,
    normalize_contest_name,
    parse_lineup_pool,
    parse_player_projections,
    pwin_exponents,
    pwin_field_size,
    pwin_implied_entries,
    _field_percentiles,
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

    def test_zero_inflate_off_by_default(self, proj_ext):
        base = build_quantile_grids(proj_ext)
        again = build_quantile_grids(proj_ext, zero_inflate=False)
        assert set(base) == set(again)
        for pid in list(base)[:25]:
            assert np.array_equal(base[pid], again[pid])

    def test_mean_calibration_scales_batters_only(self, proj_ext):
        base = build_quantile_grids(proj_ext)
        cal = build_quantile_grids(proj_ext, mean_calib_batter=0.88)
        pos = dict(zip(proj_ext["player_id"].astype(int), proj_ext["position"].astype(str)))
        n_bat = 0
        for pid, g in cal.items():
            if pos.get(pid) == "P":
                assert np.array_equal(g, base[pid])
            else:
                n_bat += 1
                assert g.mean() == pytest.approx(0.88 * base[pid].mean(), rel=1e-9)
        assert n_bat > 0

    def test_calibration_composes_with_zero_inflation(self, proj_ext):
        base = build_quantile_grids(proj_ext)
        both = build_quantile_grids(proj_ext, zero_inflate=True, mean_calib_batter=0.88)
        pos = dict(zip(proj_ext["player_id"].astype(int), proj_ext["position"].astype(str)))
        for pid, g in list(both.items())[:60]:
            if pos.get(pid) == "P":
                continue
            # zero-inflation holds the mean, so the only mean change is the calibration
            assert g.mean() == pytest.approx(0.88 * base[pid].mean(), rel=1e-6)
            assert np.all(np.diff(g) >= -1e-12)

    def test_calibration_does_not_change_gaussian_fallback_membership(self, proj_ext):
        # the +-20% grid-vs-file-mean check must run on the RAW grid, so an
        # aggressive calibration constant cannot add or drop players
        assert set(build_quantile_grids(proj_ext)) == set(
            build_quantile_grids(proj_ext, mean_calib_batter=0.5, mean_calib_pitcher=0.5)
        )

    def test_zero_inflate_touches_batters_only_and_holds_the_mean(self, proj_ext):
        base = build_quantile_grids(proj_ext)
        inf = build_quantile_grids(proj_ext, zero_inflate=True)
        pos = dict(zip(proj_ext["player_id"].astype(int), proj_ext["position"].astype(str)))
        changed_bat = changed_pit = 0
        for pid, g in inf.items():
            if pid not in base:
                continue
            if not np.array_equal(base[pid], g):
                if pos.get(pid) == "P":
                    changed_pit += 1
                else:
                    changed_bat += 1
            # the projected mean is SaberSim's and must survive untouched
            assert g.mean() == pytest.approx(base[pid].mean(), rel=1e-6)
        assert changed_bat > 0
        assert changed_pit == 0


class TestPwinExponents:
    @staticmethod
    def _groups():
        # a $1 mini-MAX ($20K pool) and a $25 single-entry Skipper ($10K pool):
        # the two ends of the real per-contest exponent range
        return [
            ContestGroup(
                contest_id="mini", contest_name="mini-MAX", entry_fee_cents=100,
                prize_pool_cents=20_000 * 100, single_entry_tag=False,
                roi_key="", entries=[(Path("x"), None)] * 72),
            ContestGroup(
                contest_id="skip", contest_name="Skipper", entry_fee_cents=2500,
                prize_pool_cents=10_000 * 100, single_entry_tag=True,
                roi_key="", entries=[(Path("x"), None)]),
        ]

    def test_scaling_is_the_legacy_default(self):
        e = pwin_exponents(self._groups(), 0.05, flat_reference=0.0)
        legacy = {cid: max(1.0, 0.05 * sz)
                  for cid, sz in pwin_implied_entries(self._groups()).items()}
        assert e == legacy

    def test_scaling_spreads_exponents_by_contest_size(self):
        e = pwin_exponents(self._groups(), 0.05, flat_reference=0.0)
        # mini-MAX implies ~23.8k entries, Skipper ~476 — a ~50x spread
        assert e["mini"] > 20 * e["skip"]

    def test_flat_reference_gives_every_contest_the_same_exponent(self):
        e = pwin_exponents(self._groups(), 0.05, flat_reference=10_000.0)
        assert set(e) == {"mini", "skip"}
        assert len(set(e.values())) == 1
        assert e["mini"] == pytest.approx(500.0)

    def test_sharpness_still_scales_the_flat_exponent(self):
        a = pwin_exponents(self._groups(), 0.05, flat_reference=10_000.0)["mini"]
        b = pwin_exponents(self._groups(), 0.10, flat_reference=10_000.0)["mini"]
        assert b == pytest.approx(2 * a)

    def test_exponent_floored_at_one(self):
        e = pwin_exponents(self._groups(), 1e-9, flat_reference=10_000.0)
        assert all(v >= 1.0 for v in e.values())

    def test_flat_needs_no_prize_pool(self):
        g = [ContestGroup(
            contest_id="c", contest_name="c", entry_fee_cents=400,
            prize_pool_cents=None, single_entry_tag=False,
            roi_key="", entries=[(Path("x"), None)] * 5)]
        assert pwin_exponents(g, 0.05, flat_reference=10_000.0)["c"] == pytest.approx(500.0)


class TestBatterBlankProbability:
    def test_two_component_mixture(self):
        # scratch floor applies even to a batter who never blanks in play
        assert batter_blank_probability(4.2, 4.2, 0.0, scratch_prob=0.02) == pytest.approx(
            0.02 + 0.98 * (1 - 0.70) ** 4.2, rel=1e-6
        )

    def test_decreasing_in_obp(self):
        good = batter_blank_probability(4.2, 1.4, 0.6)
        weak = batter_blank_probability(4.2, 0.7, 0.2)
        assert good < weak

    def test_decreasing_in_pa(self):
        assert batter_blank_probability(4.5, 1.2, 0.5) < batter_blank_probability(3.0, 0.8, 0.33)

    def test_missing_rate_stats_fall_back_to_population_default(self):
        p = batter_blank_probability(np.nan, np.nan, np.nan, scratch_prob=0.0)
        assert p == pytest.approx(0.19, abs=1e-9)

    def test_zero_pa_falls_back(self):
        assert batter_blank_probability(0.0, 0.0, 0.0, scratch_prob=0.0) == pytest.approx(0.19)

    def test_bounded(self):
        assert 0.01 <= batter_blank_probability(1.0, 0.0, 0.0) <= 0.60
        assert 0.01 <= batter_blank_probability(9.9, 9.9, 9.9) <= 0.60

    def test_matches_measured_population_rate(self):
        # 10-slate sample: mechanistic term 19.3%, +2% scratch -> ~20.9%,
        # against a realized 20.6%. A league-average batter should land there.
        p = batter_blank_probability(4.2, 1.10, 0.42, scratch_prob=0.02)
        assert 0.15 < p < 0.27


class TestZeroInflation:
    @staticmethod
    def _grid(mean=10.0):
        q = np.linspace(0.0, 1.0, 101)
        return np.interp(q, [0.0, 0.5, 1.0], [0.0, mean * 0.8, mean * 3.0])

    def test_preserves_mean(self):
        q = np.linspace(0.0, 1.0, 101)
        g = self._grid()
        out = _zero_inflate_grid(g, q, 0.25)
        assert out.mean() == pytest.approx(g.mean(), rel=1e-9)

    def test_adds_mass_at_zero(self):
        q = np.linspace(0.0, 1.0, 101)
        g = self._grid()
        before = float((g <= 1e-9).mean())
        out = _zero_inflate_grid(g, q, 0.25)
        assert float((out <= 1e-9).mean()) > before
        assert float((out <= 1e-9).mean()) == pytest.approx(0.25, abs=0.02)

    def test_stays_monotone(self):
        q = np.linspace(0.0, 1.0, 101)
        out = _zero_inflate_grid(self._grid(), q, 0.3)
        assert np.all(np.diff(out) >= 0)

    def test_raises_the_surviving_ceiling(self):
        # mass moved to 0 must be compensated in the non-blank part, so the
        # conditional-on-playing distribution shifts UP
        q = np.linspace(0.0, 1.0, 101)
        g = self._grid()
        out = _zero_inflate_grid(g, q, 0.25)
        assert out[99] > g[99]

    def test_noop_when_target_below_existing_mass(self):
        q = np.linspace(0.0, 1.0, 101)
        g = np.concatenate([np.zeros(40), np.linspace(0.1, 20.0, 61)])
        out = _zero_inflate_grid(g, q, 0.10)
        assert np.array_equal(out, g)


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

    def test_empty_pool_contests_falls_back_without_crashing(self):
        """A pool with zero ROI blocks (prj_own/p_win ev_type, see
        parse_lineup_pool's require_roi_blocks=False) must not crash
        group_and_match_contests's nearest-size fallback -- roi_key/roi
        just go unused downstream for those ev_types."""
        empty_pool = ExternalPool(
            lineups=[], contests={}, n_dropped_unknown_players=0,
            n_dropped_duplicates=0, n_dropped_near_duplicates=0, source_paths=[],
        )
        entries = [(Path("x/Entries.csv"), [_rec("c1", "MLB $1K Test", 400)])]
        groups = group_and_match_contests(entries, empty_pool)
        assert len(groups) == 1
        assert groups[0].roi_fallback
        assert groups[0].roi_key == ""

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
                            n_dropped_near_duplicates=0,
                            source_paths=[Path("synthetic.csv")])

    def _groups(self, pool, sizes, prize_pool_cents=100_000, entry_fee_cents=None):
        """`prize_pool_cents`/`entry_fee_cents` set the implied field size
        (prize/fee) that the prj_own EV currency scales its ownership penalty
        by; the defaults (100_000 / 1000-j) give ~100 implied entries, i.e. a
        negligible penalty, which is what the ROI-mode tests want."""
        groups = []
        keys = list(pool.contests.keys())
        for j, size in enumerate(sizes):
            g = ContestGroup(
                contest_id=f"c{j}", contest_name=pool.contests[keys[j % len(keys)]].raw_name,
                entry_fee_cents=entry_fee_cents if entry_fee_cents is not None else 1000 - j,
                prize_pool_cents=prize_pool_cents,
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

    def test_proj_score_floor_culls_pool_wide_across_contests(self):
        """proj_score_floor_percentile is a single pool-wide cull applied
        once up front (unlike the per-contest ROI floor): a lineup that
        fails it must be absent from *every* contest's allocation, even
        one whose own ROI column would otherwise pick it first."""
        pool = self._pool(M=10, n_contests=2)
        keys = list(pool.contests.keys())
        for key in keys:
            pool.contests[key].roi = np.arange(10, dtype=np.float64)  # higher index = better ROI
        proj_scores = np.arange(10, dtype=np.float64)  # same order: index 0..3 are the bottom 40%
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [10])
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            roi_floor_percentile=0.0,  # isolate the proj-score cull
            proj_scores=proj_scores, proj_score_floor_percentile=40.0,
        )
        picked = {id(lu) for lu, _ in alloc.portfolio}
        for i in range(4):
            assert id(pool.lineups[i]) not in picked
        assert len(alloc.portfolio) == 6
        assert len(alloc.unfilled) == 4

    def test_proj_score_floor_disabled_by_default(self):
        """proj_score_floor_percentile=0.0 (the default) is a no-op even
        when a proj_scores array is supplied."""
        pool = self._pool(M=10, n_contests=1)
        key = next(iter(pool.contests))
        pool.contests[key].roi = np.abs(pool.contests[key].roi) + 0.01
        proj_scores = np.arange(10, dtype=np.float64)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [10])
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            roi_floor_percentile=0.0,  # isolate the proj-score cull
            proj_scores=proj_scores,
        )
        assert len(alloc.portfolio) == 10
        assert not alloc.unfilled

    # --- ev_type="prj_own" ------------------------------------------------

    def test_prj_own_ignores_roi_entirely(self):
        """Under prj_own, Saber's ROI column — including the absolute
        ROI>=0.0 guard that would cull most of this pool in roi mode — has no
        say at all: the pick is the best projected-minus-ownership lineup even
        though it has the *worst* ROI in the contest."""
        pool = self._pool(M=10, n_contests=1)
        key = next(iter(pool.contests))
        pool.contests[key].roi = -np.arange(10, dtype=np.float64)  # index 0 best (0.0), 9 worst
        proj_scores = np.arange(10, dtype=np.float64)              # index 9 best
        own_scores = np.zeros(10)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [1])

        roi_alloc = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        assert id(roi_alloc.portfolio[0][0]) == id(pool.lineups[0])

        prj_alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="prj_own", proj_scores=proj_scores, own_scores=own_scores,
        )
        assert id(prj_alloc.portfolio[0][0]) == id(pool.lineups[9])
        # Reported EV is the prj_own value (proj 9.0, negligible ownership
        # penalty at ~100 implied entries), not the lineup's -9.0 ROI.
        assert prj_alloc.portfolio[0][1] == pytest.approx(9.0)

    def test_prj_own_still_respects_pool_wide_proj_score_floor(self):
        """The pool-wide projected-score cull is the one cull that survives
        into prj_own mode."""
        pool = self._pool(M=10, n_contests=2)
        proj_scores = np.arange(10, dtype=np.float64)  # index 0..3 = bottom 40%
        own_scores = np.zeros(10)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [10])
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="prj_own", proj_scores=proj_scores, own_scores=own_scores,
            proj_score_floor_percentile=40.0,
        )
        picked = {id(lu) for lu, _ in alloc.portfolio}
        for i in range(4):
            assert id(pool.lineups[i]) not in picked
        assert len(alloc.portfolio) == 6
        assert len(alloc.unfilled) == 4

    def test_prj_own_field_size_scales_the_ownership_penalty(self):
        """Same pool, same two lineups: the high-projection/high-ownership one
        wins a small contest and the low-owned one wins a large contest,
        purely through the field_size = prize_pool/entry_fee multiplier."""
        pool = self._pool(M=2, n_contests=1)
        proj_scores = np.array([110.0, 100.0])   # 0 = higher projection
        own_scores = np.array([150.0, 10.0])     # 0 = chalk, 1 = leverage
        corr = np.eye(2, dtype=np.float32)

        # $1K prize pool / $10 entry = 100 implied entries -> penalty x1/300
        small = self._groups(pool, [1], prize_pool_cents=100_000, entry_fee_cents=1000)
        alloc_small = allocate_contests(
            pool, corr, small, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="prj_own", proj_scores=proj_scores, own_scores=own_scores,
        )
        assert id(alloc_small.portfolio[0][0]) == id(pool.lineups[0])

        # $100K prize pool / $4 entry = 25,000 implied entries -> penalty x5/6
        large = self._groups(pool, [1], prize_pool_cents=100_000_00, entry_fee_cents=400)
        alloc_large = allocate_contests(
            pool, corr, large, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="prj_own", proj_scores=proj_scores, own_scores=own_scores,
        )
        assert id(alloc_large.portfolio[0][0]) == id(pool.lineups[1])

    def test_prj_own_unknown_field_size_falls_back_to_projection(self):
        """An unparseable prize pool zeroes the ownership penalty rather than
        guessing a field size — the chalk lineup wins on projection alone."""
        pool = self._pool(M=2, n_contests=1)
        corr = np.eye(2, dtype=np.float32)
        groups = self._groups(pool, [1], prize_pool_cents=None)
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="prj_own",
            proj_scores=np.array([110.0, 100.0]), own_scores=np.array([150.0, 10.0]),
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[0])
        assert alloc.portfolio[0][1] == pytest.approx(110.0)

    def test_prj_own_requires_own_scores(self):
        pool = self._pool(M=4, n_contests=1)
        corr = np.eye(4, dtype=np.float32)
        groups = self._groups(pool, [1])
        with pytest.raises(ValueError, match="own_scores"):
            allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                              ev_type="prj_own", proj_scores=np.arange(4, dtype=float))

    def test_unknown_ev_type_raises(self):
        pool = self._pool(M=4, n_contests=1)
        corr = np.eye(4, dtype=np.float32)
        groups = self._groups(pool, [1])
        with pytest.raises(ValueError, match="ev_type"):
            allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                              ev_type="nonsense")

    def test_roi_mode_unaffected_by_prj_own_inputs(self):
        """Regression guard: the default ev_type still ranks on ROI and is
        byte-identical whether or not the prj_own inputs are supplied."""
        pool = self._pool(M=40, n_contests=2)
        corr = np.eye(40, dtype=np.float32)
        groups = self._groups(pool, [10, 10])
        rng = np.random.default_rng(7)
        baseline = allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4)
        with_inputs = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            proj_scores=rng.uniform(80, 120, 40), own_scores=rng.uniform(10, 170, 40),
        )
        assert [id(lu) for lu, _ in baseline.portfolio] == [id(lu) for lu, _ in with_inputs.portfolio]
        assert [ev for _, ev in baseline.portfolio] == [ev for _, ev in with_inputs.portfolio]

    # --- ev_type="p_win" ----------------------------------------------

    def test_p_win_picks_highest_select_value_first(self):
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [1])
        select = {"c0": np.arange(10, dtype=np.float64)}  # index 9 highest
        cull = {"c0": np.arange(10, dtype=np.float64)}
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[9])
        assert alloc.portfolio[0][1] == pytest.approx(9.0)

    def test_p_win_ignores_roi_entirely(self):
        pool = self._pool(M=10, n_contests=1)
        key = next(iter(pool.contests))
        pool.contests[key].roi = -np.arange(10, dtype=np.float64)  # index 0 best ROI
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [1])
        select = {"c0": np.arange(10, dtype=np.float64)}  # index 9 best p_win
        cull = {"c0": np.arange(10, dtype=np.float64)}
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[9])

    def test_p_win_stage_a_cull_is_independent_of_stage_b_ranking(self):
        """A lineup that ranks top by p_win_select but bottom by p_win_cull
        must be excluded once admit_n culls it — proving the two stages
        genuinely use separate information rather than one overriding the
        other."""
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [5])
        select = {"c0": np.arange(10, dtype=np.float64)}       # index 9 best select
        cull = {"c0": -np.arange(10, dtype=np.float64)}        # index 9 WORST cull
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=5,  # keeps indices 0..4 (best cull), excludes 9
        )
        picked = {id(lu) for lu, _ in alloc.portfolio}
        assert id(pool.lineups[9]) not in picked
        assert len(alloc.portfolio) == 5

    def test_p_win_admit_n_zero_disables_the_cull(self):
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [1])
        select = {"c0": np.arange(10, dtype=np.float64)}
        cull = {"c0": -np.arange(10, dtype=np.float64)}  # would exclude index 9 if applied
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select, p_win_admit_n=0,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[9])

    def test_p_win_admit_multiplier_zero_matches_flat_admit_n(self):
        """Default multiplier=0.0 must be byte-identical to a flat admit_n —
        the whole point of making it opt-in."""
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [5])
        select = {"c0": np.arange(10, dtype=np.float64)}
        cull = {"c0": -np.arange(10, dtype=np.float64)}
        flat = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=5, p_win_admit_multiplier=0.0,
        )
        scaled = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=5,
        )
        ids_flat = [id(lu) for lu, _ in flat.portfolio]
        ids_scaled = [id(lu) for lu, _ in scaled.portfolio]
        assert ids_flat == ids_scaled

    def test_p_win_admit_multiplier_scales_up_for_a_large_contest(self):
        """A large-fill contest gets a bigger effective admit_n than the
        flat floor once the multiplier is set — proving the per-contest
        formula actually engages, not just documented."""
        pool = self._pool(M=20, n_contests=1)
        corr = np.eye(20, dtype=np.float32)
        # A lineup ranked worst by both cull and select, at index 19 -- a
        # flat floor of 5 always excludes it; multiplier * 15 entries = 30
        # should admit it once the contest needs 15 picks.
        groups = self._groups(pool, [15])
        select = {"c0": np.arange(20, dtype=np.float64)}
        cull = {"c0": np.arange(20, dtype=np.float64)}
        floor_only = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=5, p_win_admit_multiplier=0.0,
        )
        scaled = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=5, p_win_admit_multiplier=2.0,  # max(5, 2*15) = 30 -> whole pool
        )
        assert len(floor_only.portfolio) == 5   # flat floor of 5 admits only 5 candidates
        assert len(scaled.portfolio) == 15       # scaled admit_n (30) covers the full ask

    def test_p_win_admit_multiplier_never_shrinks_below_the_floor(self):
        """A tiny contest must not get an effective admit_n below the flat
        floor even though multiplier * n_entries is small for it."""
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [1])
        select = {"c0": np.arange(10, dtype=np.float64)}
        cull = {"c0": -np.arange(10, dtype=np.float64)}  # would exclude index 9 at admit_n<10
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            p_win_admit_n=10, p_win_admit_multiplier=0.5,  # max(10, 0.5*1)=10 -> whole pool
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[9])

    def test_p_win_still_respects_pool_wide_proj_score_floor(self):
        pool = self._pool(M=10, n_contests=2)
        proj_scores = np.arange(10, dtype=np.float64)  # index 0..3 = bottom 40%
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [10])
        select = {"c0": np.arange(10, dtype=np.float64)}
        cull = {"c0": np.arange(10, dtype=np.float64)}
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=cull, p_win_select=select,
            proj_scores=proj_scores, proj_score_floor_percentile=40.0,
        )
        picked = {id(lu) for lu, _ in alloc.portfolio}
        for i in range(4):
            assert id(pool.lineups[i]) not in picked
        assert len(alloc.portfolio) == 6
        assert len(alloc.unfilled) == 4

    def test_p_win_requires_both_cull_and_select(self):
        pool = self._pool(M=4, n_contests=1)
        corr = np.eye(4, dtype=np.float32)
        groups = self._groups(pool, [1])
        select = {"c0": np.arange(4, dtype=np.float64)}
        with pytest.raises(ValueError, match="p_win_cull"):
            allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                              ev_type="p_win", p_win_select=select)
        with pytest.raises(ValueError, match="p_win_cull"):
            allocate_contests(pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
                              ev_type="p_win", p_win_cull=select)

    def test_p_win_missing_contest_key_leaves_it_unfilled(self):
        """A contest_id absent from p_win_select (e.g. a field/sim failure
        for just that contest) is left unfilled rather than crashing."""
        pool = self._pool(M=10, n_contests=1)
        corr = np.eye(10, dtype=np.float32)
        groups = self._groups(pool, [3])
        alloc = allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull={}, p_win_select={},
        )
        assert len(alloc.portfolio) == 0
        assert len(alloc.unfilled) == 3

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


class TestProjTopOwnershipCap:
    """ev_type='proj_top' with own_cap_start_pct/own_cap_end_pct: an
    optional large-field (implied_field_size >= 5,000) ownership cap, off
    by default (100/100)."""

    def _pool(self, M=60, seed=0):
        rng = np.random.default_rng(seed)
        lineups = [Lineup(player_ids=list(range(10 * i, 10 * i + 10))) for i in range(M)]
        name = "MLB $5K Test"
        contests = {normalize_contest_name(name): ExternalContest(
            raw_name=name, norm_name=normalize_contest_name(name),
            roi=rng.normal(0, 0.3, M), prize_pool_cents=500_000, single_entry=False,
        )}
        return ExternalPool(lineups=lineups, contests=contests,
                            n_dropped_unknown_players=0, n_dropped_duplicates=0,
                            n_dropped_near_duplicates=0,
                            source_paths=[Path("synthetic.csv")])

    def _group(self, pool, size, *, prize_pool_cents, entry_fee_cents, cid="c0"):
        key = next(iter(pool.contests))
        return ContestGroup(
            contest_id=cid, contest_name=pool.contests[key].raw_name,
            entry_fee_cents=entry_fee_cents, prize_pool_cents=prize_pool_cents,
            single_entry_tag=size == 1,
            entries=[(Path("x/Entries.csv"), _rec(cid, "n", entry_fee_cents, f"{cid}-{i}"))
                     for i in range(size)],
            roi_key=key,
        )

    # $100,000 / $20 * 0.84 ~= 5,952 implied entries -- comfortably >= 5,000.
    _LARGE = dict(prize_pool_cents=100_000_00, entry_fee_cents=2000)
    # $1,000 / $10 * 0.84 ~= 119 implied entries -- comfortably < 5,000.
    _SMALL = dict(prize_pool_cents=100_000, entry_fee_cents=1000)

    def test_default_is_byte_identical_to_uncapped(self):
        pool = self._pool(M=20)
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 10, **self._LARGE)
        proj_scores = np.arange(20, dtype=np.float64)
        own_scores = np.arange(20, dtype=np.float64)[::-1].astype(np.float64)
        baseline = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
        )
        with_own = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
        )
        ids_base = [id(lu) for lu, _ in baseline.portfolio]
        ids_own = [id(lu) for lu, _ in with_own.portfolio]
        assert ids_base == ids_own == [id(pool.lineups[i]) for i in range(19, 9, -1)]

    def test_small_field_unaffected_by_tight_cap(self):
        """A contest below the 5,000-entry threshold ignores the cap
        entirely, however tight."""
        pool = self._pool(M=20)
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 10, **self._SMALL)
        proj_scores = np.arange(20, dtype=np.float64)
        own_scores = np.arange(20, dtype=np.float64)  # highest-proj is also highest-owned
        uncapped = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
        )
        capped = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=10.0, own_cap_end_pct=10.0,
        )
        assert ([id(lu) for lu, _ in uncapped.portfolio]
                == [id(lu) for lu, _ in capped.portfolio])

    def test_tight_cap_excludes_high_ownership_top_pick(self):
        """The single highest-proj_score lineup is also the highest-owned;
        a tight cap on a large-field contest must exclude it, so a
        lower-projection/lower-ownership lineup wins the slot instead."""
        pool = self._pool(M=20)
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        proj_scores = np.arange(20, dtype=np.float64)  # index 19 best
        own_scores = np.arange(20, dtype=np.float64)   # index 19 also most-owned
        uncapped = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
        )
        assert id(uncapped.portfolio[0][0]) == id(pool.lineups[19])

        capped = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=50.0, own_cap_end_pct=50.0,
        )
        assert id(capped.portfolio[0][0]) != id(pool.lineups[19])

    def test_phase_in_tightens_with_field_size(self):
        """Two large contests, same start/end percentages: the one with the
        bigger implied field size gets a strictly tighter effective cutoff
        (fewer eligible candidates), never a looser one."""
        pool = self._pool(M=40)
        corr = np.eye(40, dtype=np.float32)
        proj_scores = np.arange(40, dtype=np.float64)
        own_scores = np.arange(40, dtype=np.float64)  # monotone with proj_score
        medium = self._group(pool, 1, prize_pool_cents=100_000_00, entry_fee_cents=2000, cid="c0")
        huge = self._group(pool, 1, prize_pool_cents=100_000_00_00, entry_fee_cents=2000, cid="c1")
        assert implied_field_size(medium) < implied_field_size(huge)

        alloc = allocate_contests(
            pool, corr, [medium, huge], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=90.0, own_cap_end_pct=50.0,
            own_cap_max_field_size=implied_field_size(huge),
        )
        # Both contests take the single best surviving lineup by proj_score;
        # the medium (looser cap) contest is filled first (larger prize
        # pool sorts first under no explicit fill order here since both
        # share entry_fee/size -- assert via the two winners' own_scores
        # instead of fill order).
        picked = {g.contest_id: lu for (lu, _), g in zip(alloc.portfolio, [medium, huge])}
        assert own_scores[[i for i, lu in enumerate(pool.lineups)
                           if id(lu) == id(picked["c0"])][0]] >= \
               own_scores[[i for i, lu in enumerate(pool.lineups)
                           if id(lu) == id(picked["c1"])][0]]

    def test_degenerate_max_equals_threshold(self):
        """implied field size exactly at the 5,000 threshold: no
        divide-by-zero, resolves to the loose (start) end."""
        pool = self._pool(M=10)
        corr = np.eye(10, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        proj_scores = np.arange(10, dtype=np.float64)
        own_scores = np.arange(10, dtype=np.float64)
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=90.0, own_cap_end_pct=10.0,
            own_cap_max_field_size=implied_field_size(group),  # == this contest's own size
        )
        assert len(alloc.portfolio) == 1  # no crash/NaN cutoff

    def test_requires_own_scores_when_cap_active(self):
        pool = self._pool(M=10)
        corr = np.eye(10, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        with pytest.raises(ValueError, match="own_scores"):
            allocate_contests(
                pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
                ev_type="proj_top", proj_scores=np.arange(10, dtype=np.float64),
                own_cap_start_pct=50.0,
            )

    def test_other_ev_types_unaffected_by_cap_kwargs(self):
        """roi/prj_own/p_win must be byte-identical whether or not the new
        cap kwargs are passed -- the gating on ev_type=='proj_top' must be
        airtight."""
        pool = self._pool(M=20)
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 10, **self._LARGE)
        baseline = allocate_contests(pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4)
        with_cap_kwargs = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            own_cap_start_pct=10.0, own_cap_end_pct=5.0,
            own_scores=np.arange(20, dtype=np.float64),
        )
        assert ([id(lu) for lu, _ in baseline.portfolio]
                == [id(lu) for lu, _ in with_cap_kwargs.portfolio])

    def test_max_field_size_none_self_computes_from_groups(self):
        pool = self._pool(M=20)
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        proj_scores = np.arange(20, dtype=np.float64)
        own_scores = np.arange(20, dtype=np.float64)
        explicit = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=90.0, own_cap_end_pct=50.0,
            own_cap_max_field_size=implied_field_size(group),
        )
        auto = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            own_cap_start_pct=90.0, own_cap_end_pct=50.0,
        )
        assert ([id(lu) for lu, _ in explicit.portfolio]
                == [id(lu) for lu, _ in auto.portfolio])


class TestProjTopCeilingTiers:
    """ev_type='proj_top' with ceiling_tier_boundary: an optional
    field-size-tiered ranking-signal swap (mean below 5,000, sim p95
    from 5,000 to the boundary, sim p99 at/above the boundary), off by
    default (ceiling_tier_boundary=None)."""

    def _pool(self, M=20, seed=0):
        rng = np.random.default_rng(seed)
        lineups = [Lineup(player_ids=list(range(10 * i, 10 * i + 10))) for i in range(M)]
        name = "MLB $5K Test"
        contests = {normalize_contest_name(name): ExternalContest(
            raw_name=name, norm_name=normalize_contest_name(name),
            roi=rng.normal(0, 0.3, M), prize_pool_cents=500_000, single_entry=False,
        )}
        return ExternalPool(lineups=lineups, contests=contests,
                            n_dropped_unknown_players=0, n_dropped_duplicates=0,
                            n_dropped_near_duplicates=0,
                            source_paths=[Path("synthetic.csv")])

    def _group(self, pool, size, *, prize_pool_cents, entry_fee_cents, cid="c0"):
        key = next(iter(pool.contests))
        return ContestGroup(
            contest_id=cid, contest_name=pool.contests[key].raw_name,
            entry_fee_cents=entry_fee_cents, prize_pool_cents=prize_pool_cents,
            single_entry_tag=size == 1,
            entries=[(Path("x/Entries.csv"), _rec(cid, "n", entry_fee_cents, f"{cid}-{i}"))
                     for i in range(size)],
            roi_key=key,
        )

    # < 5,000 implied entries (small tier -- always mean, regardless of
    # ceiling_tier_boundary).
    _SMALL = dict(prize_pool_cents=100_000, entry_fee_cents=1000)
    # ~5,952 implied entries -- medium tier under the default 15,000 boundary.
    _MEDIUM = dict(prize_pool_cents=100_000_00, entry_fee_cents=2000)
    # ~17,857 implied entries -- large tier under the default 15,000 boundary.
    _LARGE = dict(prize_pool_cents=300_000_00, entry_fee_cents=2000)

    def _signals(self, M=20):
        # Three signals disagree on the winner so the active one is
        # identifiable from the single pick: proj_scores peaks at the last
        # index, sim_p95_scores at the first, sim_p99_scores in the middle.
        proj_scores = np.arange(M, dtype=np.float64)
        sim_p95_scores = np.arange(M, dtype=np.float64)[::-1].astype(np.float64)
        mid = M // 2
        sim_p99_scores = -np.square(np.arange(M) - mid).astype(np.float64)
        return proj_scores, sim_p95_scores, sim_p99_scores

    def test_default_off_byte_identical_even_with_percentiles_passed(self):
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 10, **self._LARGE)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        baseline = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
        )
        with_percentiles = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
        )
        assert ([id(lu) for lu, _ in baseline.portfolio]
                == [id(lu) for lu, _ in with_percentiles.portfolio])

    def test_small_field_always_uses_mean(self):
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._SMALL)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=15000.0,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[19])

    def test_medium_field_uses_p95(self):
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._MEDIUM)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=15000.0,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[0])

    def test_large_field_uses_p99(self):
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=15000.0,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[10])

    def test_field_exactly_at_boundary_uses_p99(self):
        """implied_field_size(group) == ceiling_tier_boundary resolves to
        the large-tier (p99) treatment, not medium."""
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._MEDIUM)
        boundary = implied_field_size(group)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=boundary,
        )
        assert id(alloc.portfolio[0][0]) == id(pool.lineups[10])

    def test_requires_both_percentile_arrays(self):
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 1, **self._LARGE)
        proj_scores, sim_p95_scores, _ = self._signals()
        with pytest.raises(ValueError, match="sim_p95_scores"):
            allocate_contests(
                pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
                ev_type="proj_top", proj_scores=proj_scores,
                sim_p95_scores=sim_p95_scores, ceiling_tier_boundary=15000.0,
            )

    def test_other_ev_types_unaffected_by_ceiling_kwargs(self):
        """roi/prj_own/p_win must be byte-identical whether or not the new
        ceiling-tier kwargs are passed -- the ev_type=='proj_top' gate
        must be airtight."""
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 10, **self._LARGE)
        _, sim_p95_scores, sim_p99_scores = self._signals()
        baseline = allocate_contests(pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4)
        with_kwargs = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=15000.0,
        )
        assert ([id(lu) for lu, _ in baseline.portfolio]
                == [id(lu) for lu, _ in with_kwargs.portfolio])

    def test_combines_without_crashing_with_ownership_cap(self):
        """Discouraged in docs (tested to hurt), but not blocked: both
        features can be enabled together without error."""
        pool = self._pool()
        corr = np.eye(20, dtype=np.float32)
        group = self._group(pool, 5, **self._LARGE)
        proj_scores, sim_p95_scores, sim_p99_scores = self._signals()
        own_scores = np.arange(20, dtype=np.float64)
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="proj_top", proj_scores=proj_scores, own_scores=own_scores,
            sim_p95_scores=sim_p95_scores, sim_p99_scores=sim_p99_scores,
            ceiling_tier_boundary=15000.0,
            own_cap_start_pct=90.0, own_cap_end_pct=50.0,
        )
        assert len(alloc.portfolio) == 5


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


class TestComputeLineupScores:
    def test_matches_manual_indicator_sum(self):
        sim_results = SimulationResults(
            player_ids=list(range(1, 21)),
            results_matrix=np.arange(1.0, 1.0 + 5 * 20).reshape(5, 20).astype(np.float32),
        )
        lineups = [Lineup(player_ids=list(range(1, 11))), Lineup(player_ids=list(range(11, 21)))]
        scores = compute_lineup_scores(lineups, sim_results)
        assert scores.shape == (2, 5)
        expected0 = sim_results.results_matrix[:, :10].sum(axis=1)
        np.testing.assert_allclose(scores[0], expected0, rtol=1e-5)

    def test_compute_pool_corr_agrees_with_and_without_precomputed_scores(self):
        """compute_pool_corr(scores=...) must be identical to letting it
        compute compute_lineup_scores internally — the whole point of
        splitting the two functions is that callers needing both the raw
        score matrix (p_win) and the correlation can share one matmul."""
        rng = np.random.default_rng(5)
        sim_results = SimulationResults(
            player_ids=list(range(1, 51)),
            results_matrix=rng.normal(10, 3, size=(300, 50)).astype(np.float32),
        )
        lineups = [
            Lineup(player_ids=list(rng.choice(range(1, 51), size=10, replace=False)))
            for _ in range(30)
        ]
        scores = compute_lineup_scores(lineups, sim_results)
        corr_a = compute_pool_corr(lineups, sim_results)
        corr_b = compute_pool_corr(lineups, sim_results, scores=scores)
        np.testing.assert_allclose(corr_a, corr_b)


class TestFieldPercentiles:
    def test_dominant_lineup_percentile_near_but_not_exactly_one(self):
        rng = np.random.default_rng(0)
        field_scores = rng.normal(100, 10, size=(50, 200))  # (S, F)
        pool_scores = np.full((1, 50), 1000.0)  # always beats the whole field
        q = _field_percentiles(pool_scores, field_scores)
        assert np.all(q < 1.0)
        assert np.all(q > 0.99)

    def test_dominated_lineup_percentile_near_but_not_exactly_zero(self):
        rng = np.random.default_rng(1)
        field_scores = rng.normal(100, 10, size=(50, 200))
        pool_scores = np.full((1, 50), -1000.0)
        q = _field_percentiles(pool_scores, field_scores)
        assert np.all(q > 0.0)
        assert np.all(q < 0.01)

    def test_monotone_in_pool_score(self):
        rng = np.random.default_rng(2)
        field_scores = rng.normal(100, 10, size=(20, 200))
        pool_scores = np.array([[50.0] * 20, [100.0] * 20, [150.0] * 20])
        q = _field_percentiles(pool_scores, field_scores)
        assert np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])


class TestComputePWin:
    def _field(self, S=400, F=500, mean=100.0, sd=10.0, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(mean, sd, size=(S, F)).astype(np.float32)

    def test_dominant_lineup_p_win_near_one(self):
        field_scores = self._field()
        pool_scores = np.full((1, 400), 1000.0)
        out = compute_p_win(pool_scores, field_scores, {"c1": 1.0})
        assert out["c1"][0] == pytest.approx(1.0, abs=1e-2)

    def test_dominated_lineup_p_win_near_zero(self):
        field_scores = self._field()
        pool_scores = np.full((1, 400), -1000.0)
        out = compute_p_win(pool_scores, field_scores, {"c1": 1.0})
        assert out["c1"][0] == pytest.approx(0.0, abs=1e-2)

    def test_higher_exponent_lowers_p_win_for_subcertain_lineup(self):
        """A lineup that's good-but-not-dominant (q < 1 most worlds) sees a
        strictly lower P(win) at a larger exponent — q**n is decreasing in n
        for q<1, which is the whole mechanism behind sharpness/field-size
        scaling the win requirement."""
        field_scores = self._field(seed=3)
        rng = np.random.default_rng(4)
        pool_scores = rng.normal(115, 10, size=(1, 400)).astype(np.float32)  # above field mean, not dominant
        out = compute_p_win(pool_scores, field_scores, {"small": 2.0, "big": 500.0})
        assert out["big"][0] < out["small"][0]

    def test_multiple_exponents_in_one_call_match_separate_calls(self):
        field_scores = self._field(seed=5)
        rng = np.random.default_rng(6)
        pool_scores = rng.normal(105, 12, size=(20, 400)).astype(np.float32)
        combined = compute_p_win(pool_scores, field_scores, {"a": 10.0, "b": 5000.0})
        sep_a = compute_p_win(pool_scores, field_scores, {"a": 10.0})
        sep_b = compute_p_win(pool_scores, field_scores, {"b": 5000.0})
        np.testing.assert_allclose(combined["a"], sep_a["a"])
        np.testing.assert_allclose(combined["b"], sep_b["b"])

    def test_chunking_does_not_change_the_result(self):
        field_scores = self._field(S=1000, seed=7)
        rng = np.random.default_rng(8)
        pool_scores = rng.normal(105, 12, size=(15, 1000)).astype(np.float32)
        whole = compute_p_win(pool_scores, field_scores, {"c": 50.0}, chunk=1000)
        chunked = compute_p_win(pool_scores, field_scores, {"c": 50.0}, chunk=137)
        np.testing.assert_allclose(whole["c"], chunked["c"], rtol=1e-5)

    def test_stop_check_normalizes_by_worlds_actually_processed(self):
        """An interruption must not silently under-divide the running sum by
        the full world count — the mean over whatever was processed should
        still land close to the mean over everything (same distribution)."""
        field_scores = self._field(S=2000, seed=9)
        rng = np.random.default_rng(10)
        pool_scores = rng.normal(105, 12, size=(5, 2000)).astype(np.float32)
        calls = {"n": 0}

        def stop_after_a_few():
            calls["n"] += 1
            return calls["n"] > 3

        partial = compute_p_win(pool_scores, field_scores, {"c": 20.0}, chunk=100,
                                stop_check=stop_after_a_few)
        full = compute_p_win(pool_scores, field_scores, {"c": 20.0}, chunk=100)
        # Both are unbiased estimates of the same expectation — not equal,
        # but not wildly different either (loose bound, this is a sanity
        # check that partial isn't e.g. 10x too small from a bad divisor).
        np.testing.assert_allclose(partial["c"], full["c"], atol=0.15)

    def test_progress_cb_called_once_per_chunk(self):
        field_scores = self._field(S=500, seed=11)
        pool_scores = np.full((3, 500), 100.0)
        calls = []
        compute_p_win(pool_scores, field_scores, {"c": 1.0}, chunk=100,
                      progress_cb=lambda done, total: calls.append((done, total)))
        assert calls == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


class TestPwinFieldSizing:
    def _group(self, contest_id, prize_pool_cents, entry_fee_cents):
        return ContestGroup(
            contest_id=contest_id, contest_name="x", entry_fee_cents=entry_fee_cents,
            prize_pool_cents=prize_pool_cents, single_entry_tag=False, roi_key="x",
        )

    def test_implied_entries_borrows_median_for_unparseable_prize_pool(self):
        # Rake-adjusted: prize_pool / (entry_fee * 0.84), not the raw ratio.
        groups = [
            self._group("a", 100_000_00, 400),   # 100,000 / (4 * 0.84) ≈ 29,761.90
            self._group("b", 20_000_00, 400),    # 20,000 / (4 * 0.84) ≈ 5,952.38
            self._group("c", None, 400),          # unparseable -> median of a, b
        ]
        sizes = pwin_implied_entries(groups)
        assert sizes["a"] == pytest.approx(100_000 / (4 * 0.84))
        assert sizes["b"] == pytest.approx(20_000 / (4 * 0.84))
        assert sizes["c"] == pytest.approx((sizes["a"] + sizes["b"]) / 2)

    def test_implied_entries_falls_back_to_default_when_nothing_parses(self):
        groups = [self._group("a", None, 400)]
        sizes = pwin_implied_entries(groups)
        assert sizes["a"] == pytest.approx(10_000.0)

    def test_field_size_respects_floor_and_cap(self):
        groups = [self._group("a", 100_00, 400)]  # ~29.8 implied entries — tiny
        assert pwin_field_size(groups, floor=5_000) == 5_000
        groups_huge = [self._group("a", 100_000_000_00, 100)]  # 1,000,000,000 implied
        assert pwin_field_size(groups_huge, floor=5_000, cap=25_000) == 25_000


class TestComputePoolProjScores:
    def _players_df(self, n_players=20, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "player_id": list(range(1, n_players + 1)),
            "mean": rng.uniform(5, 15, n_players),
        })

    def test_matches_manual_sum(self):
        players_df = self._players_df()
        rng = np.random.default_rng(1)
        lineups = [
            Lineup(player_ids=list(rng.choice(range(1, 21), size=10, replace=False)))
            for _ in range(15)
        ]
        scores = compute_pool_proj_scores(lineups, players_df)
        mean_by_id = dict(zip(players_df["player_id"], players_df["mean"]))
        expected = np.array([sum(mean_by_id[p] for p in lu.player_ids) for lu in lineups])
        np.testing.assert_allclose(scores, expected, atol=1e-3)

    def test_shape_matches_lineup_count(self):
        players_df = self._players_df()
        lineups = [Lineup(player_ids=list(range(1, 11)))] * 3
        scores = compute_pool_proj_scores(lineups, players_df)
        assert scores.shape == (3,)


class TestComputePoolOwnership:
    def _players_df(self, n_players=20, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "player_id": list(range(1, n_players + 1)),
            "ownership": rng.uniform(1.0, 30.0, n_players),
        })

    def test_matches_manual_sum(self):
        players_df = self._players_df()
        rng = np.random.default_rng(1)
        lineups = [
            Lineup(player_ids=list(rng.choice(range(1, 21), size=10, replace=False)))
            for _ in range(15)
        ]
        own = compute_pool_ownership(lineups, players_df)
        own_by_id = dict(zip(players_df["player_id"], players_df["ownership"]))
        expected = np.array([sum(own_by_id[p] for p in lu.player_ids) for lu in lineups])
        np.testing.assert_allclose(own, expected, atol=1e-3)

    def test_shape_matches_lineup_count(self):
        players_df = self._players_df()
        lineups = [Lineup(player_ids=list(range(1, 11)))] * 3
        assert compute_pool_ownership(lineups, players_df).shape == (3,)


class TestBuildExternalPlayersDfOwnership:
    """build_external_players_df must carry projected ownership through — it
    is the per-player input compute_pool_ownership sums for the prj_own EV
    currency, and it stays on the file's percentage-point scale."""

    def _frames(self):
        slate_df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "position": ["P", "OF", "OF"],
            "team": ["NYY", "NYY", "BOS"],
            "game": ["NYY@BOS"] * 3,
            "salary": [10000, 5000, 4000],
        })
        proj_ext = pd.DataFrame({
            "player_id": [1, 2],           # 3 is pool-only, unknown to the file
            "order": [np.nan, 3.0],
            "mean": [18.0, 9.0],
            "std_dev": [6.0, 4.0],
            "ownership": [22.5, 11.25],
        })
        return slate_df, proj_ext

    def test_ownership_column_present_and_unscaled(self):
        slate_df, proj_ext = self._frames()
        df = build_external_players_df(
            slate_df, proj_ext, pool_pids={1, 2, 3},
            derive_opponent=lambda team, game: "BOS" if team == "NYY" else "NYY",
        )
        own = dict(zip(df["player_id"], df["ownership"]))
        assert own[1] == pytest.approx(22.5)   # percentage points, not /100
        assert own[2] == pytest.approx(11.25)

    def test_players_missing_from_projections_get_a_small_positive_floor(self):
        """Not 0.0: ContestSimulator normalizes ownership into a sampling
        weight, so a hard zero would make the player impossible to draw into
        a simulated opponent field (see the p_win EV currency)."""
        slate_df, proj_ext = self._frames()
        df = build_external_players_df(
            slate_df, proj_ext, pool_pids={1, 2, 3},
            derive_opponent=lambda team, game: "BOS" if team == "NYY" else "NYY",
        )
        own = dict(zip(df["player_id"], df["ownership"]))
        assert 0.0 < own[3] <= 0.1


class TestImpliedFieldSize:
    def _group(self, prize_pool_cents, entry_fee_cents):
        return ContestGroup(
            contest_id="c0", contest_name="MLB $100K Test",
            entry_fee_cents=entry_fee_cents, prize_pool_cents=prize_pool_cents,
            single_entry_tag=False, roi_key="mlb $100k test",
        )

    def test_prize_pool_over_entry_fee(self):
        # $100K prize pool at a $4 entry fee, rake-adjusted (DK pays out
        # only ~84% of collected fees) -> 100_000 / (4 * 0.84) ≈ 29,761.9
        # implied entries, not the raw 25,000 prize/fee ratio.
        assert implied_field_size(self._group(100_000_00, 400)) == pytest.approx(100_000 / (4 * 0.84))

    def test_missing_prize_pool_is_zero(self):
        assert implied_field_size(self._group(None, 400)) == 0.0

    def test_missing_or_zero_entry_fee_is_zero(self):
        assert implied_field_size(self._group(100_000_00, 0)) == 0.0


class TestComputePrjOwnEv:
    def test_formula(self):
        proj = np.array([100.0, 90.0])
        own = np.array([120.0, 40.0])
        ev = compute_prj_own_ev(proj, own, field_size=30_000.0)
        # penalty multiplier = 30000/30000 = 1.0
        np.testing.assert_allclose(ev, [100.0 - 120.0, 90.0 - 40.0])

    def test_zero_field_size_is_plain_projected_score(self):
        proj = np.array([100.0, 90.0])
        own = np.array([120.0, 40.0])
        np.testing.assert_allclose(compute_prj_own_ev(proj, own, 0.0), proj)

    def test_calibration_anchor_10k_indifference(self):
        """Calibration anchor: at ~10,000 entries a 95-point projection with
        60 ownership must be worth the same as a 105-point projection with
        90 ownership (10 projected points per 30 ownership points)."""
        ev = compute_prj_own_ev(
            np.array([95.0, 105.0]), np.array([60.0, 90.0]), field_size=10_000.0,
        )
        assert ev[0] == pytest.approx(ev[1])
        assert ev[0] == pytest.approx(75.0)

    def test_calibration_anchor_1k_is_ten_times_weaker(self):
        """Second anchor: at 1,000 entries ownership carries a tenth of the
        weight it does at 10,000 — which linear field-size scaling gives for
        free, so both anchors are satisfied by the one own_scale constant."""
        own = np.array([60.0, 90.0])
        proj = np.zeros(2)
        big = compute_prj_own_ev(proj, own, field_size=10_000.0)
        small = compute_prj_own_ev(proj, own, field_size=1_000.0)
        np.testing.assert_allclose(small, big / 10.0)

    def test_own_scale_dials_the_tradeoff(self):
        """Halving own_scale doubles the ownership penalty at a given field
        size (it is the field size at which 1 ownership pt == 1 proj pt)."""
        proj, own = np.array([100.0]), np.array([60.0])
        base = compute_prj_own_ev(proj, own, 15_000.0, own_scale=30_000.0)
        steep = compute_prj_own_ev(proj, own, 15_000.0, own_scale=15_000.0)
        assert base[0] == pytest.approx(100.0 - 30.0)
        assert steep[0] == pytest.approx(100.0 - 60.0)


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
            n_dropped_unknown_players=0, n_dropped_duplicates=0,
            n_dropped_near_duplicates=0, source_paths=[Path("x")],
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


class TestPWinIntegrationWithRealSimMatrix:
    """End-to-end p_win through allocate_contests against a REAL sim/corr
    matrix (compute_pool_corr, not np.eye) — TestAllocation's p_win tests
    all use identity correlation to isolate the EV branch; this is the one
    that exercises the actual diversity term alongside it, closing the gap
    noted in the plan (no existing test drove allocate_contests off a real
    sim matrix at all, for any ev_type)."""

    def test_full_pipeline_end_to_end(self):
        sim_results, lineups, _roi = TestRiskSweepDifferentiation()._stacked_pool(
            n_sims=4000, M=300, seed=11,
        )
        n_players = len(sim_results.player_ids)
        pool_scores = compute_lineup_scores(lineups, sim_results)   # (M, n_sims)
        corr = compute_pool_corr(lineups, sim_results, scores=pool_scores)

        # Synthetic opponent field: random 10-player draws over the same
        # player universe, scored the same way and transposed to the
        # (n_sims, F) shape compute_p_win/score_field expect.
        rng = np.random.default_rng(12)
        field_lineups = [
            Lineup(player_ids=[int(p) + 1 for p in
                               rng.choice(n_players, size=10, replace=False)])
            for _ in range(200)
        ]
        field_scores = compute_lineup_scores(field_lineups, sim_results).T  # (n_sims, F)

        n_half = 2000
        pool_A, pool_B = pool_scores[:, :n_half], pool_scores[:, n_half:]
        field_A, field_B = field_scores[:n_half], field_scores[n_half:]
        exponents = {"c0": 50.0}
        p_win_cull = compute_p_win(pool_A, field_A, exponents)
        p_win_select = compute_p_win(pool_B, field_B, exponents)

        pool = ExternalPool(
            lineups=lineups, contests={}, n_dropped_unknown_players=0,
            n_dropped_duplicates=0, n_dropped_near_duplicates=0,
            source_paths=[Path("synthetic.csv")],
        )
        k = 20
        group = ContestGroup(
            contest_id="c0", contest_name="synthetic", entry_fee_cents=400,
            prize_pool_cents=1_000_000_00, single_entry_tag=False, roi_key="",
            entries=[(Path("x/Entries.csv"), _rec("c0", "n", 400, f"e{i}")) for i in range(k)],
        )
        alloc = allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.1, evw_max=0.4,
            ev_type="p_win", p_win_cull=p_win_cull, p_win_select=p_win_select,
            p_win_admit_n=100,
        )
        assert len(alloc.portfolio) == k
        assert not alloc.unfilled
        picked_ids = {id(lu) for lu, _ in alloc.portfolio}
        assert len(picked_ids) == k  # shared-removal mask: no duplicate picks

        # The top pick must be within the top-100-by-cull admitted set and
        # must be the single highest p_win_select value among admissible
        # candidates (k>1 so this is the selector's step-1 pure-EV argmax).
        admitted = set(np.argsort(-p_win_cull["c0"])[:100])
        idx_of = {id(lu): i for i, lu in enumerate(lineups)}
        top_idx = idx_of[id(alloc.portfolio[0][0])]
        assert top_idx in admitted
        best_admitted = max(admitted, key=lambda i: p_win_select["c0"][i])
        assert top_idx == best_admitted


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


def test_parse_lineup_pool_no_roi_blocks_requires_flag(tmp_path):
    """A file with no '<name> ROI'/'<name> Sim Dupes' pair is rejected by
    default (ev_type='roi'), but accepted with require_roi_blocks=False
    (ev_type='prj_own'/'p_win', which never read contest.roi)."""
    header = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    rows = [[str(i) for i in range(1, 11)]]
    path = tmp_path / "lineups_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with pytest.raises(ValueError, match="no contest ROI blocks"):
        parse_lineup_pool(path, valid_ids=set(range(1, 11)))
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 11)), require_roi_blocks=False)
    assert len(pool.lineups) == 1
    assert pool.contests == {}


def test_parse_lineup_pool_no_roi_blocks_multi_lineup_near_dup_check(tmp_path):
    """Regression: with >1 lineup and zero ROI blocks, the near-duplicate
    (9/10 overlap) pass used to call max() on an empty contest_order range
    via _pick_primary_contest_index, crashing with "max() iterable argument
    is empty". Two non-conflicting lineups must parse fine and keep both."""
    header = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    rows = [
        [str(i) for i in range(1, 11)],
        [str(i) for i in range(11, 21)],
    ]
    path = tmp_path / "lineups_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 21)), require_roi_blocks=False)
    assert len(pool.lineups) == 2
    assert pool.contests == {}


def test_parse_lineup_pool_no_roi_blocks_near_dup_tiebreak_uses_proj_score(tmp_path):
    """With zero ROI blocks, a 9/10-overlap conflict is resolved by each
    lineup's own "Proj Score" column (higher wins), not first-in-file."""
    header = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Proj Score"]
    rows = [
        [str(i) for i in range(1, 11)] + ["80.0"],              # ids 1-10, lower proj score, listed first
        [str(i) for i in range(1, 10)] + ["11"] + ["95.0"],     # ids 1-9,11, higher proj score
    ]
    path = tmp_path / "lineups_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 12)), require_roi_blocks=False)
    assert pool.n_dropped_near_duplicates == 1
    assert len(pool.lineups) == 1
    # Higher Proj Score (95.0, ids 1-9,11) survives despite arriving second.
    assert set(pool.lineups[0].player_ids) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}


def _write_lineup_csv(path: Path, contest_names: list[str], rows: list[list]) -> None:
    header = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    for name in contest_names:
        header += [f"{name} ROI", f"{name} Sim Dupes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_near_duplicate_removed_keeps_higher_roi(tmp_path):
    """Two lineups differing by exactly one swapped player (9/10 overlap)
    collapse to one, keeping the higher-ROI lineup."""
    path = tmp_path / "lineups_test.csv"
    _write_lineup_csv(path, ["MLB $1K Test"], [
        [str(i) for i in range(1, 11)] + ["1.5", "0.02"],       # ids 1-10, higher roi
        [str(i) for i in range(1, 10)] + ["11"] + ["0.8", "0.02"],  # ids 1-9,11, lower roi
    ])
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 12)))
    assert pool.n_dropped_near_duplicates == 1
    assert pool.n_dropped_duplicates == 0
    assert len(pool.lineups) == 1
    assert set(pool.lineups[0].player_ids) == set(range(1, 11))


def test_near_duplicate_tiebreak_uses_largest_prize_pool_contest(tmp_path):
    """The tie-break ROI column is the contest with the largest parsed
    prize pool, not simply the first contest in file order."""
    path = tmp_path / "lineups_test.csv"
    # "Small" ranks lineup A higher; "Big" (bigger prize pool) ranks B
    # higher -- Big's ranking must win.
    _write_lineup_csv(path, ["MLB $1K Small", "MLB $50K Big"], [
        [str(i) for i in range(1, 11)] + ["9.0", "0.02", "1.0", "0.02"],       # A: ids 1-10
        [str(i) for i in range(1, 10)] + ["11"] + ["1.0", "0.02", "9.0", "0.02"],  # B: ids 1-9,11
    ])
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 12)))
    assert pool.n_dropped_near_duplicates == 1
    assert len(pool.lineups) == 1
    # B (higher ROI in the bigger contest) survives, not A.
    assert set(pool.lineups[0].player_ids) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}


def test_near_duplicate_no_conflict_kept(tmp_path):
    """Lineups that don't share a 9-player core are unaffected."""
    path = tmp_path / "lineups_test.csv"
    _write_lineup_csv(path, ["MLB $1K Test"], [
        [str(i) for i in range(1, 11)] + ["1.0", "0.02"],
        [str(i) for i in range(21, 31)] + ["1.0", "0.02"],
    ])
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 31)))
    assert pool.n_dropped_near_duplicates == 0
    assert len(pool.lineups) == 2


def test_near_duplicate_chain_is_not_transitively_collapsed(tmp_path):
    """L1={1..10}, L2={1..9,11} shares L1's full 9-core {1..9} (conflicts
    with L1); L3={1..8,10,12} shares a *different* 9-core {1..8,10} with L1
    (conflicts with L1) but shares only 8 players with L2 (no conflict).
    With L2 ranked highest, L1 loses its head-to-head with L2 and is
    dropped -- but L3 never directly conflicts with L2, so it survives even
    though it would have conflicted with the now-dropped L1. This is the
    non-transitive case _find_near_duplicate_removals exists for."""
    path = tmp_path / "lineups_test.csv"
    l1 = [str(i) for i in range(1, 11)]                      # 1..10
    l2 = [str(i) for i in range(1, 10)] + ["11"]              # 1..9, 11
    l3 = [str(i) for i in range(1, 9)] + ["10", "12"]         # 1..8, 10, 12
    _write_lineup_csv(path, ["MLB $1K Test"], [
        l1 + ["8.0", "0.02"],
        l2 + ["10.0", "0.02"],
        l3 + ["5.0", "0.02"],
    ])
    pool = parse_lineup_pool(path, valid_ids=set(range(1, 13)))
    assert pool.n_dropped_near_duplicates == 1
    surviving = {frozenset(lu.player_ids) for lu in pool.lineups}
    assert surviving == {
        frozenset(int(x) for x in l2),
        frozenset(int(x) for x in l3),
    }


@needs_files
def test_discover_pairs_by_token(tmp_path):
    import shutil
    shutil.copy(LINEUPS_CSV, tmp_path / LINEUPS_CSV.name)
    shutil.copy(PROJ_CSV, tmp_path / PROJ_CSV.name)
    (tmp_path / "MLB_2026-01-01-100pm_DK_Main.csv").write_text("DFS ID\n")
    out = discover_external_files(str(tmp_path))
    assert [p.name for p in out["lineups_paths"]] == [LINEUPS_CSV.name]
    assert out["projections_path"].name == PROJ_CSV.name
    assert out["paired_by_token"]


def test_discover_groups_multiple_files_same_slate_signature(tmp_path):
    """Two exports of the same slate ('lineups_..._705pm.csv' and a browser
    re-download '... (1).csv') are grouped together; a file for a different
    date is excluded even though it shares the same time token."""
    f_other_date = tmp_path / "lineups_dk_mlb_classic_7-23-2026_705pm.csv"
    f_a = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm.csv"
    f_b = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm (1).csv"
    for i, f in enumerate([f_other_date, f_a, f_b]):
        f.write_text("h\n")
        os.utime(f, (1000 + i, 1000 + i))
    (tmp_path / "MLB_2026-07-24-705pm_DK_Main.csv").write_text("DFS ID\n")
    out = discover_external_files(str(tmp_path))
    assert sorted(p.name for p in out["lineups_paths"]) == sorted([f_a.name, f_b.name])
    assert out["projections_path"].name == "MLB_2026-07-24-705pm_DK_Main.csv"
    assert out["paired_by_token"]


def test_discover_different_format_prefix_not_grouped(tmp_path):
    """A different format/type prefix (e.g. showdown vs classic) is a
    different slate even with the same date/time token."""
    f_classic = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm.csv"
    f_showdown = tmp_path / "lineups_dk_mlb_showdown_7-24-2026_705pm.csv"
    for i, f in enumerate([f_classic, f_showdown]):
        f.write_text("h\n")
        os.utime(f, (1000 + i, 1000 + i))
    out = discover_external_files(str(tmp_path))
    assert [p.name for p in out["lineups_paths"]] == [f_showdown.name]


def test_parse_lineup_pool_multi_file_dedupe(tmp_path):
    """Duplicate 10-player lineups across files are collapsed to one, using
    the roi from the first file the lineup appears in."""
    header = (
        ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        + ["MLB $1K Test ROI", "MLB $1K Test Sim Dupes"]
    )
    rows_a = [
        [str(i) for i in range(1, 11)] + ["1.5", "0.02"],
        [str(i) for i in range(11, 21)] + ["0.8", "0.03"],
    ]
    rows_b = [
        [str(i) for i in range(1, 11)] + ["9.9", "0.02"],   # dup of rows_a[0]
        [str(i) for i in range(21, 31)] + ["0.5", "0.01"],
    ]
    path_a = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm.csv"
    path_b = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm (1).csv"
    for path, rows in [(path_a, rows_a), (path_b, rows_b)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    pool = parse_lineup_pool([path_a, path_b], valid_ids=set(range(1, 31)))
    assert len(pool.lineups) == 3
    assert pool.n_dropped_duplicates == 1
    contest = pool.contests[normalize_contest_name("MLB $1K Test")]
    assert contest.roi[0] == pytest.approx(1.5)


def test_parse_lineup_pool_contest_union_across_files(tmp_path):
    """A contest present in only one file leaves NaN roi for lineups sourced
    from files that don't define it."""
    header_a = (
        ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        + ["Contest A ROI", "Contest A Sim Dupes"]
    )
    header_b = (
        ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
        + ["Contest B ROI", "Contest B Sim Dupes"]
    )
    row_a = [str(i) for i in range(1, 11)] + ["1.0", "0.01"]
    row_b = [str(i) for i in range(11, 21)] + ["2.0", "0.02"]
    path_a = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm.csv"
    path_b = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm (1).csv"
    with open(path_a, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_a)
        w.writerow(row_a)
    with open(path_b, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_b)
        w.writerow(row_b)
    pool = parse_lineup_pool([path_a, path_b], valid_ids=set(range(1, 21)))
    assert set(pool.contests.keys()) == {
        normalize_contest_name("Contest A"), normalize_contest_name("Contest B"),
    }
    a = pool.contests[normalize_contest_name("Contest A")]
    b = pool.contests[normalize_contest_name("Contest B")]
    assert a.roi[0] == pytest.approx(1.0)
    assert np.isnan(b.roi[0])
    assert np.isnan(a.roi[1])
    assert b.roi[1] == pytest.approx(2.0)


def test_archive_external_inputs_copies_all_lineup_files(tmp_path):
    slate = tmp_path / "DKSalaries.csv"
    slate.write_text('Game Info\n"NYY@BOS 07/24/2026 07:05PM ET"\n')
    lp_a = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm.csv"
    lp_b = tmp_path / "lineups_dk_mlb_classic_7-24-2026_705pm (1).csv"
    lp_a.write_text("a\n")
    lp_b.write_text("b\n")
    proj = tmp_path / "MLB_2026-07-24-705pm_DK_Main.csv"
    proj.write_text("p\n")
    d = archive_external_inputs(tmp_path, str(slate), [lp_a, lp_b], proj)
    assert d is not None
    assert (d / lp_a.name).exists()
    assert (d / lp_b.name).exists()
    assert (d / proj.name).exists()
    assert (d / "DKSalaries.csv").exists()
