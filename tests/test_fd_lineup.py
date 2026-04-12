"""
Tests for platform-aware Lineup.is_valid() (Phase 4).

Coverage:
- Valid FD 9-player lineups pass
- Invalid FD lineups fail (roster size, salary cap, max hitters, min games,
  pitcher-batter conflict)
- C/1B slot accepts C players and 1B players
- UTIL slot accepts every non-pitcher position
- The 'both pitchers same team' constraint is naturally skipped for FD
  (only 1 P slot — never two pitchers to compare)
- Default call (no rules/slot_eligibility args) retains DK behaviour
- DK lineups still validate correctly after Phase 4 changes
"""

import pytest
from src.optimization.lineup import Lineup
from src.platforms.base import Platform
from src.platforms.draftkings import DK_ROSTER
from src.platforms.fanduel import FD_ROSTER, FD_SLOT_ELIGIBILITY
from src.platforms.registry import get_slot_eligibility


# ---------------------------------------------------------------------------
# Player metadata helpers
# ---------------------------------------------------------------------------

def _player(position, salary, team, opponent, game, eligible_positions=None):
    return {
        'position': position,
        'eligible_positions': eligible_positions or [position],
        'salary': salary,
        'team': team,
        'opponent': opponent,
        'game': game,
    }


def _meta(*players):
    """Build a PlayerMeta dict from (pid, ...) tuples."""
    return {pid: _player(*args) for pid, *args in players}


# ---------------------------------------------------------------------------
# Canonical FD player pool
#
# 9-player valid base lineup (total salary = 32 200 ≤ 35 000):
#   slot P     → pid 1  (PHI, opp ARI)
#   slot C/1B  → pid 2  C player (NYY, opp TB)
#   slot 2B    → pid 3  (MIL, opp WSH)
#   slot 3B    → pid 4  (CLE, opp MIN)
#   slot SS    → pid 5  (WSH, opp MIL)
#   slot OF    → pid 6  (NYY, opp TB)
#   slot OF    → pid 7  (TB, opp NYY)
#   slot OF    → pid 8  (ATL, opp MIA)
#   slot UTIL  → pid 9  OF (BOS, opp STL)
# ---------------------------------------------------------------------------

FD_META = _meta(
    (1,  'P',  8000, 'PHI', 'ARI', 'ARI@PHI'),
    (2,  'C',  3500, 'NYY', 'TB',  'NYY@TB'),
    (3,  '2B', 3000, 'MIL', 'WSH', 'WSH@MIL'),
    (4,  '3B', 3000, 'CLE', 'MIN', 'CLE@MIN'),
    (5,  'SS', 3000, 'WSH', 'MIL', 'WSH@MIL'),
    (6,  'OF', 3200, 'NYY', 'TB',  'NYY@TB'),
    (7,  'OF', 3200, 'TB',  'NYY', 'NYY@TB'),
    (8,  'OF', 3000, 'ATL', 'MIA', 'ATL@MIA'),
    (9,  'OF', 3300, 'BOS', 'STL', 'BOS@STL'),
    # Extra players for swap / negative tests
    (10, '1B', 3000, 'SEA', 'HOU', 'SEA@HOU'),  # 1B player for C/1B slot test
    (11, '2B', 3000, 'NYY', 'TB',  'NYY@TB'),   # extra NYY batter
    (12, 'SS', 3000, 'NYY', 'TB',  'NYY@TB'),   # extra NYY batter
    (13, '3B', 3000, 'NYY', 'TB',  'NYY@TB'),   # extra NYY batter
    (14, 'C',  3000, 'NYY', 'TB',  'NYY@TB'),   # extra NYY batter (C in UTIL)
    (15, '1B', 3000, 'MIN', 'CLE', 'CLE@MIN'),  # 1B for salary-over test fill
    (16, 'P',  8000, 'ARI', 'PHI', 'ARI@PHI'),  # second pitcher (same team check)
    (17, 'P',  8000, 'PHI', 'ARI', 'ARI@PHI'),  # second pitcher same team as pid 1
    (18, 'P',  8000, 'NYY', 'TB',  'NYY@TB'),   # pitcher who opposes batters in NYY
    (19, '3B', 3000, 'MIN', 'CLE', 'CLE@MIN'),  # 3B to replace pid 4
    (20, 'C',  3000, 'CHC', 'STL', 'CHC@STL'),  # C to test UTIL when C/1B has 1B
    (21, 'OF', 3000, 'NYY', 'TB',  'NYY@TB'),   # 5th NYY batter
)

BASE_LINEUP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def valid_fd_lineup():
    return Lineup(list(BASE_LINEUP_IDS))


# Salary sum helper
def _total_salary(pids, meta):
    return sum(meta[p]['salary'] for p in pids)


# ---------------------------------------------------------------------------
# Valid FD lineups
# ---------------------------------------------------------------------------

class TestFDValidLineup:
    def test_base_lineup_is_valid(self):
        assert valid_fd_lineup().is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_base_lineup_salary_within_cap(self):
        assert _total_salary(BASE_LINEUP_IDS, FD_META) <= FD_ROSTER.salary_cap

    def test_1b_fills_c1b_slot(self):
        """Replace the C player (pid 2) with a 1B player (pid 10)."""
        ids = [1, 10, 3, 4, 5, 6, 7, 8, 9]
        assert Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_c_fills_util_when_c1b_has_1b(self):
        """
        1B player in C/1B slot, C player in UTIL slot.
        UTIL accepts C, so bipartite matching should resolve this.
        """
        # pid 10 = 1B (fills C/1B), pid 20 = C (fills UTIL)
        ids = [1, 10, 3, 4, 5, 6, 7, 8, 20]
        assert Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_2b_fills_util(self):
        """Replace OF in UTIL slot (pid 9) with a 2B player."""
        # Need a 2nd 2B player not already in the lineup.
        # pid 3 is already in as 2B slot, so add another 2B in UTIL.
        # Build a lineup where the UTIL is filled by a 2B from a different slot.
        # Swap pid 9 (OF/UTIL) for pid 3's position, then put pid 3 in UTIL.
        # Simplest: put pid 3 in UTIL and add a new OF for the 2B slot.
        # But there's no standalone extra OF+2B available in the meta.
        # Easier: use pid 19 (3B) in UTIL by replacing pid 4 with pid 19 and
        # putting a 3B in UTIL. Actually let me just add to FD_META here.
        extra_meta = {**FD_META,
                      30: _player('2B', 3000, 'STL', 'BOS', 'BOS@STL')}
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 30]  # pid 30 is 2B in UTIL slot
        assert Lineup(ids).is_valid(extra_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_ss_fills_util(self):
        extra_meta = {**FD_META,
                      31: _player('SS', 3000, 'STL', 'BOS', 'BOS@STL')}
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 31]
        assert Lineup(ids).is_valid(extra_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_3b_fills_util(self):
        extra_meta = {**FD_META,
                      32: _player('3B', 3000, 'STL', 'BOS', 'BOS@STL')}
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 32]
        assert Lineup(ids).is_valid(extra_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_1b_fills_util(self):
        extra_meta = {**FD_META,
                      33: _player('1B', 3000, 'STL', 'BOS', 'BOS@STL')}
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 33]
        assert Lineup(ids).is_valid(extra_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_exactly_4_batters_same_team_is_valid(self):
        """FD max is 4 — a lineup with exactly 4 from one team should pass."""
        # NYY: pid 2 (C), pid 6 (OF), pid 11 (2B), pid 12 (SS) → 4 NYY batters
        ids = [1, 2, 11, 4, 12, 6, 7, 8, 9]
        # Lineup: P=1, C/1B=2(C), 2B=11, 3B=4, SS=12, OF=6,7,8, UTIL=9
        assert Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_c1b_compound_position_player(self):
        """A player whose position is C/1B (eligible for both C and 1B) fills C/1B slot."""
        extra_meta = {**FD_META,
                      40: _player('C', 3500, 'SEA', 'HOU', 'SEA@HOU',
                                  eligible_positions=['C', '1B'])}
        ids = [1, 40, 3, 4, 5, 6, 7, 8, 9]
        assert Lineup(ids).is_valid(extra_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)


# ---------------------------------------------------------------------------
# Invalid FD lineups
# ---------------------------------------------------------------------------

class TestFDInvalidLineup:
    def test_10_players_fails(self):
        """FD roster is 9; a 10-player lineup must be rejected."""
        ids = BASE_LINEUP_IDS + [10]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_8_players_fails(self):
        ids = BASE_LINEUP_IDS[:8]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_duplicate_player_fails(self):
        ids = [1, 1, 3, 4, 5, 6, 7, 8, 9]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_salary_over_cap_fails(self):
        """Total salary > 35 000 must fail."""
        # Override pid 1 salary to blow the cap
        expensive_meta = {**FD_META,
                          1: {**FD_META[1], 'salary': 30000}}
        assert not Lineup(BASE_LINEUP_IDS).is_valid(
            expensive_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY
        )

    def test_5_batters_same_team_fails(self):
        """FD max is 4 hitters per team; 5 must fail."""
        # NYY batters: pid 2 (C), 6 (OF), 11 (2B), 12 (SS), 21 (OF)
        ids = [1, 2, 11, 4, 12, 6, 21, 8, 9]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_only_one_game_fails(self):
        """Lineup must span ≥ 2 games."""
        same_game_meta = {pid: {**data, 'game': 'NYY@TB'}
                         for pid, data in FD_META.items()}
        assert not Lineup(BASE_LINEUP_IDS).is_valid(
            same_game_meta, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY
        )

    def test_pitcher_opposes_batter_fails(self):
        """Pitcher whose opponent is a batter's team must be rejected."""
        # pid 18 is a NYY pitcher; NYY batters (pid 2, 6) are in the lineup
        ids = [18, 2, 3, 4, 5, 6, 7, 8, 9]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)

    def test_pitcher_not_in_slot_fails(self):
        """Lineup with no valid pitcher cannot fill the P slot."""
        # Replace pitcher (pid 1) with a non-pitcher (pid 10, 1B)
        ids = [10, 2, 3, 4, 5, 6, 7, 8, 9]
        assert not Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)


# ---------------------------------------------------------------------------
# FD-specific constraint: 1 pitcher, so 'both pitchers same team' never fires
# ---------------------------------------------------------------------------

class TestFDOnePitcherConstraint:
    def test_single_pitcher_same_team_as_batters_is_not_blocked_by_two_pitcher_rule(self):
        """
        The 'both P from same team' constraint guards with len(pitcher_teams)==2.
        With one FD pitcher that guard is never reached — the lineup is
        rejected only if the pitcher actually opposes batters (separate check).
        pid 17 is PHI pitcher, and no PHI batter is in this lineup.
        """
        ids = [17, 2, 3, 4, 5, 6, 7, 8, 9]  # pid 17: PHI pitcher
        # No PHI batters in the lineup → should be valid
        assert Lineup(ids).is_valid(FD_META, rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY)


# ---------------------------------------------------------------------------
# Salary floor enforcement
# ---------------------------------------------------------------------------

class TestFDSalaryFloor:
    def test_below_floor_fails(self):
        floor = _total_salary(BASE_LINEUP_IDS, FD_META) + 1
        assert not valid_fd_lineup().is_valid(
            FD_META, salary_floor=floor,
            rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY
        )

    def test_at_floor_passes(self):
        floor = _total_salary(BASE_LINEUP_IDS, FD_META)
        assert valid_fd_lineup().is_valid(
            FD_META, salary_floor=floor,
            rules=FD_ROSTER, slot_eligibility=FD_SLOT_ELIGIBILITY
        )


# ---------------------------------------------------------------------------
# Backward compatibility: DK defaults
# ---------------------------------------------------------------------------

class TestDKDefaultsUnchanged:
    """
    Calling is_valid() with no rules/slot_eligibility must produce exactly the
    same results as before Phase 4 (DK Classic constraints).
    """

    # Pitchers oppose ARI and WSH; no batter in the lineup is from ARI or WSH,
    # so the pitcher-opposes-batter constraint is satisfied.
    DK_META = _meta(
        (101, 'P',  9000, 'PHI', 'ARI', 'ARI@PHI'),   # PHI opp ARI
        (102, 'P',  8000, 'MIL', 'WSH', 'WSH@MIL'),   # MIL opp WSH
        (103, 'C',  3500, 'NYY', 'TB',  'NYY@TB'),
        (104, '1B', 3500, 'TB',  'NYY', 'NYY@TB'),
        (105, '2B', 3500, 'CLE', 'MIN', 'CLE@MIN'),
        (106, '3B', 3500, 'MIN', 'CLE', 'CLE@MIN'),
        (107, 'SS', 3500, 'BOS', 'SEA', 'BOS@SEA'),
        (108, 'OF', 3500, 'ATL', 'SD',  'ATL@SD'),
        (109, 'OF', 3500, 'SEA', 'BOS', 'BOS@SEA'),
        (110, 'OF', 3500, 'SD',  'ATL', 'ATL@SD'),
    )
    VALID_DK_IDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

    def test_valid_dk_lineup_passes_with_defaults(self):
        assert Lineup(self.VALID_DK_IDS).is_valid(self.DK_META)

    def test_valid_dk_lineup_passes_with_explicit_dk_rules(self):
        se = get_slot_eligibility(Platform.DRAFTKINGS)
        assert Lineup(self.VALID_DK_IDS).is_valid(
            self.DK_META, rules=DK_ROSTER, slot_eligibility=se
        )

    def test_10_player_dk_lineup_fails_with_9_players(self):
        """Passing 9 players to a DK lineup (needs 10) should fail."""
        assert not Lineup(self.VALID_DK_IDS[:9]).is_valid(self.DK_META)

    def test_dk_salary_cap_enforced(self):
        expensive_meta = {**self.DK_META,
                          101: {**self.DK_META[101], 'salary': 49000}}
        assert not Lineup(self.VALID_DK_IDS).is_valid(expensive_meta)

    def test_dk_max_5_hitters_same_team(self):
        """DK allows up to 5; a 6th from the same team must fail."""
        extra = {**self.DK_META,
                 111: _player('OF', 3000, 'NYY', 'TB', 'NYY@TB')}
        # NYY would then have: pid 102(P-not-counted), 103(C), pid 111(OF) + ???
        # Let's construct a lineup where NYY has 6 non-pitcher players.
        # Replace PHI pitcher with a NYY one, and stack 6 NYY batters.
        # Easier: just mutate meta so one team has 6 batters.
        many_nyy = {
            i: _player(i, pos, 3000, 'NYY', 'TB', 'NYY@TB')
            for i, pos in enumerate(
                ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF'], start=200
            )
        }
        many_nyy[209] = _player('OF', 3000, 'BOS', 'STL', 'BOS@STL')
        many_nyy[210] = _player('P',  9000, 'PHI', 'ARI', 'ARI@PHI')
        many_nyy[211] = _player('P',  8000, 'MIL', 'WSH', 'WSH@MIL')
        ids = [210, 211, 200, 201, 202, 203, 204, 205, 206, 209]
        # 6 NYY non-pitchers (200-205 + OF 206 = 7 but we only have 7 slots)
        # Actually: 200(C),201(1B),202(2B),203(3B),204(SS),205(OF),206(OF) = 7 NYY batters
        # But DK only allows 8 non-pitcher slots. With 7 NYY batters → > 5 → fail
        assert not Lineup(ids).is_valid(many_nyy)

    def test_dk_two_pitchers_same_team_fails(self):
        """DK requires 2 pitchers from different teams."""
        same_team_meta = {
            **self.DK_META,
            102: {**self.DK_META[102], 'team': 'PHI'},
        }
        assert not Lineup(self.VALID_DK_IDS).is_valid(same_team_meta)

    def test_fd_lineup_fails_dk_validation(self):
        """A 9-player FD lineup must fail DK validation (wrong roster size)."""
        assert not valid_fd_lineup().is_valid(FD_META)
