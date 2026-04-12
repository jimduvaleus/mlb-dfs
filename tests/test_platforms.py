"""
Tests for the src/platforms/ abstraction layer (Phase 1).

Coverage:
- DK constants match the legacy values in src/utils/scoring.py and lineup.py
- FD batter/pitcher point calculations
- Roster slot counts and salary caps
- Registry factory functions return the correct objects
- FD slot eligibility mapping is complete and correct
"""

import src.utils.scoring as legacy_scoring
from src.optimization.lineup import (
    MAX_HITTERS_PER_TEAM as LEGACY_MAX_HITTERS,
    MIN_GAMES as LEGACY_MIN_GAMES,
    ROSTER_REQUIREMENTS as LEGACY_ROSTER_REQS,
    SALARY_CAP as LEGACY_SALARY_CAP,
)
from src.platforms.base import Platform, RosterRules, ScoringRules
from src.platforms.draftkings import DK_ROSTER, DK_SCORING
from src.platforms.fanduel import FD_ROSTER, FD_SCORING, FD_SLOT_ELIGIBILITY
from src.platforms.registry import get_roster, get_scoring, get_slot_eligibility


# ---------------------------------------------------------------------------
# DK scoring — parity with legacy src/utils/scoring.py
# ---------------------------------------------------------------------------

class TestDKScoringParity:
    def test_single(self):
        assert DK_SCORING.single == legacy_scoring.BATTER_SINGLE

    def test_double(self):
        assert DK_SCORING.double == legacy_scoring.BATTER_DOUBLE

    def test_triple(self):
        assert DK_SCORING.triple == legacy_scoring.BATTER_TRIPLE

    def test_home_run(self):
        assert DK_SCORING.home_run == legacy_scoring.BATTER_HOME_RUN

    def test_rbi(self):
        assert DK_SCORING.rbi == legacy_scoring.BATTER_RBI

    def test_run(self):
        assert DK_SCORING.run == legacy_scoring.BATTER_RUN

    def test_walk(self):
        assert DK_SCORING.walk == legacy_scoring.BATTER_WALK

    def test_hbp(self):
        assert DK_SCORING.hbp == legacy_scoring.BATTER_HBP

    def test_sb(self):
        assert DK_SCORING.sb == legacy_scoring.BATTER_SB

    def test_pitcher_win(self):
        assert DK_SCORING.win == legacy_scoring.PITCHER_WIN

    def test_pitcher_er(self):
        assert DK_SCORING.er == legacy_scoring.PITCHER_ER

    def test_pitcher_so(self):
        assert DK_SCORING.so == legacy_scoring.PITCHER_SO

    def test_pitcher_ip(self):
        assert DK_SCORING.ip == legacy_scoring.PITCHER_IP

    def test_pitcher_h(self):
        assert DK_SCORING.h == legacy_scoring.PITCHER_H

    def test_pitcher_bb(self):
        assert DK_SCORING.bb == legacy_scoring.PITCHER_BB

    def test_pitcher_hb(self):
        assert DK_SCORING.hb == legacy_scoring.PITCHER_HB

    def test_pitcher_cg(self):
        assert DK_SCORING.cg == legacy_scoring.PITCHER_CG

    def test_pitcher_cgs(self):
        assert DK_SCORING.cgs == legacy_scoring.PITCHER_CGS

    def test_pitcher_nh(self):
        assert DK_SCORING.nh == legacy_scoring.PITCHER_NH


# ---------------------------------------------------------------------------
# DK roster — parity with legacy src/optimization/lineup.py constants
# ---------------------------------------------------------------------------

class TestDKRosterParity:
    def test_salary_cap(self):
        assert DK_ROSTER.salary_cap == LEGACY_SALARY_CAP

    def test_max_hitters_per_team(self):
        assert DK_ROSTER.max_hitters_per_team == LEGACY_MAX_HITTERS

    def test_min_games(self):
        assert DK_ROSTER.min_games == LEGACY_MIN_GAMES

    def test_slot_counts_match_requirements(self):
        """Slot list must expand to exactly the same counts as ROSTER_REQUIREMENTS."""
        from collections import Counter
        slot_counts = Counter(DK_ROSTER.slots)
        assert dict(slot_counts) == LEGACY_ROSTER_REQS

    def test_roster_size(self):
        assert DK_ROSTER.roster_size == 10


# ---------------------------------------------------------------------------
# DK batter_points / pitcher_points helpers
# ---------------------------------------------------------------------------

class TestDKScoringHelpers:
    def test_batter_hr_rbi_run(self):
        pts = DK_SCORING.batter_points(home_run=1, rbi=1, run=1)
        assert pts == 10 + 2 + 2  # HR + RBI + R

    def test_pitcher_6ip_7k_no_win(self):
        pts = DK_SCORING.pitcher_points(ip=6, so=7)
        assert pts == 6 * 2.25 + 7 * 2  # 13.5 + 14 = 27.5

    def test_batter_single_sb(self):
        pts = DK_SCORING.batter_points(single=2, sb=1)
        assert pts == 2 * 3 + 5  # 11


# ---------------------------------------------------------------------------
# FD scoring values
# ---------------------------------------------------------------------------

class TestFDScoringValues:
    def test_single(self):
        assert FD_SCORING.single == 3

    def test_double(self):
        assert FD_SCORING.double == 6

    def test_triple(self):
        assert FD_SCORING.triple == 9

    def test_home_run(self):
        assert FD_SCORING.home_run == 12

    def test_rbi(self):
        assert FD_SCORING.rbi == 3.5

    def test_run(self):
        assert FD_SCORING.run == 3.2

    def test_walk(self):
        assert FD_SCORING.walk == 3

    def test_hbp(self):
        assert FD_SCORING.hbp == 3

    def test_sb(self):
        assert FD_SCORING.sb == 6

    def test_pitcher_win(self):
        assert FD_SCORING.win == 6

    def test_pitcher_qs(self):
        assert FD_SCORING.qs == 4

    def test_pitcher_er(self):
        assert FD_SCORING.er == -3

    def test_pitcher_so(self):
        assert FD_SCORING.so == 3

    def test_pitcher_ip(self):
        assert FD_SCORING.ip == 3

    def test_pitcher_no_h_bonus(self):
        assert FD_SCORING.h == 0.0

    def test_pitcher_no_bb_penalty(self):
        assert FD_SCORING.bb == 0.0

    def test_pitcher_no_hb_penalty(self):
        assert FD_SCORING.hb == 0.0

    def test_pitcher_no_cg_bonus(self):
        assert FD_SCORING.cg == 0.0

    def test_pitcher_no_nh_bonus(self):
        assert FD_SCORING.nh == 0.0


class TestFDScoringHelpers:
    def test_batter_hr_rbi_run(self):
        pts = FD_SCORING.batter_points(home_run=1, rbi=1, run=1)
        assert pts == 12 + 3.5 + 3.2

    def test_pitcher_win_qs_6ip_7k(self):
        pts = FD_SCORING.pitcher_points(win=1, qs=1, ip=6, so=7)
        assert pts == 6 + 4 + 6 * 3 + 7 * 3  # 6+4+18+21 = 49

    def test_pitcher_er_penalty(self):
        pts = FD_SCORING.pitcher_points(er=3)
        assert pts == 3 * -3  # -9


# ---------------------------------------------------------------------------
# FD roster
# ---------------------------------------------------------------------------

class TestFDRoster:
    def test_salary_cap(self):
        assert FD_ROSTER.salary_cap == 35_000.0

    def test_max_hitters_per_team(self):
        assert FD_ROSTER.max_hitters_per_team == 4

    def test_roster_size(self):
        assert FD_ROSTER.roster_size == 9

    def test_slots_contain_util(self):
        assert 'UTIL' in FD_ROSTER.slots

    def test_slots_contain_c1b(self):
        assert 'C/1B' in FD_ROSTER.slots

    def test_one_pitcher_slot(self):
        assert FD_ROSTER.slots.count('P') == 1

    def test_three_of_slots(self):
        assert FD_ROSTER.slots.count('OF') == 3


# ---------------------------------------------------------------------------
# FD slot eligibility
# ---------------------------------------------------------------------------

class TestFDSlotEligibility:
    def test_p_slot(self):
        assert FD_SLOT_ELIGIBILITY['P'] == {'P'}

    def test_c1b_slot(self):
        assert FD_SLOT_ELIGIBILITY['C/1B'] == {'C', '1B'}

    def test_util_includes_all_non_pitcher(self):
        util = FD_SLOT_ELIGIBILITY['UTIL']
        for pos in ('C', '1B', '2B', '3B', 'SS', 'OF'):
            assert pos in util, f"{pos} should be UTIL-eligible"
        assert 'P' not in util

    def test_all_fd_slots_have_eligibility(self):
        for slot in FD_ROSTER.slots:
            assert slot in FD_SLOT_ELIGIBILITY, f"Missing eligibility for slot {slot!r}"


# ---------------------------------------------------------------------------
# Registry factory functions
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_scoring_dk(self):
        assert get_scoring(Platform.DRAFTKINGS) is DK_SCORING

    def test_get_scoring_fd(self):
        assert get_scoring(Platform.FANDUEL) is FD_SCORING

    def test_get_roster_dk(self):
        assert get_roster(Platform.DRAFTKINGS) is DK_ROSTER

    def test_get_roster_fd(self):
        assert get_roster(Platform.FANDUEL) is FD_ROSTER

    def test_get_slot_eligibility_dk_exact_match(self):
        se = get_slot_eligibility(Platform.DRAFTKINGS)
        # Every DK slot is its own single eligibility
        for slot in DK_ROSTER.slots:
            assert slot in se
            assert se[slot] == {slot}

    def test_get_slot_eligibility_fd(self):
        se = get_slot_eligibility(Platform.FANDUEL)
        assert se is FD_SLOT_ELIGIBILITY
