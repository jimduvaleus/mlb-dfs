"""
DraftKings platform constants — exact numeric parity with src/utils/scoring.py
and the legacy module-level constants in src/optimization/lineup.py.

DO NOT change these values without also updating src/utils/scoring.py and
rebuilding the historical copula (see HISTORICAL PIPELINE WARNING in
src/utils/scoring.py and Phase 9 of the FanDuel Platform Plan).
"""

from src.platforms.base import Platform, RosterRules, ScoringRules

# Scoring — mirrors src/utils/scoring.py exactly.
DK_SCORING = ScoringRules(
    # Batter
    single=3,
    double=5,
    triple=8,
    home_run=10,
    rbi=2,
    run=2,
    walk=2,
    hbp=2,
    sb=5,
    # Pitcher
    win=4,
    er=-2,
    so=2,
    ip=2.25,
    h=-0.6,
    bb=-0.6,
    hb=-0.6,
    cg=2.5,
    cgs=2.5,
    nh=5,
    qs=0.0,
)

# Roster — mirrors ROSTER_REQUIREMENTS / SALARY_CAP / MAX_HITTERS_PER_TEAM /
# MIN_GAMES in src/optimization/lineup.py exactly.
DK_ROSTER = RosterRules(
    slots=('P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'),
    salary_cap=50_000.0,
    max_hitters_per_team=5,
    min_games=2,
)
