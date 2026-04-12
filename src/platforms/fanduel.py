"""
FanDuel platform constants.

Scoring (from FanDuel MLB scoring rules):
  Batters: 1B=3, 2B=6, 3B=9, HR=12, RBI=3.5, R=3.2, BB=3, SB=6, HBP=3
  Pitchers: W=6, QS=4, ER=-3, SO=3, IP=3  (no CG/NH/hit-against bonuses)

Roster: 9 players — P×1, C/1B×1, 2B×1, 3B×1, SS×1, OF×3, UTIL×1 (any non-P)
Salary cap: $35,000 | Max hitters per team: 4
"""

from src.platforms.base import Platform, RosterRules, ScoringRules

FD_SCORING = ScoringRules(
    # Batter
    single=3,
    double=6,
    triple=9,
    home_run=12,
    rbi=3.5,
    run=3.2,
    walk=3,
    hbp=3,
    sb=6,
    # Pitcher (h/bb/hb/cg/cgs/nh intentionally 0 — FD has no such bonuses)
    win=6,
    er=-3,
    so=3,
    ip=3,
    qs=4,
    h=0.0,
    bb=0.0,
    hb=0.0,
    cg=0.0,
    cgs=0.0,
    nh=0.0,
)

# Roster — 9 players; UTIL accepts any non-pitcher position.
FD_ROSTER = RosterRules(
    slots=('P', 'C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL'),
    salary_cap=35_000.0,
    max_hitters_per_team=4,
    min_games=2,
)

# Maps each FD slot label to the set of player positions that may fill it.
# Compound labels (C/1B, UTIL) expand to multiple eligible positions.
# UTIL can be filled by any non-pitcher.
FD_SLOT_ELIGIBILITY: dict[str, set[str]] = {
    'P':    {'P'},
    'C/1B': {'C', '1B'},
    '2B':   {'2B'},
    '3B':   {'3B'},
    'SS':   {'SS'},
    'OF':   {'OF'},
    'UTIL': {'C', '1B', '2B', '3B', 'SS', 'OF'},
}
