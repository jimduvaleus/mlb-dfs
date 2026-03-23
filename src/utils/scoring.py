"""
DraftKings MLB Scoring Constants and Functions.
"""

# Batter Scoring
BATTER_SINGLE = 3
BATTER_DOUBLE = 5
BATTER_TRIPLE = 8
BATTER_HOME_RUN = 10
BATTER_RBI = 2
BATTER_RUN = 2
BATTER_WALK = 2
BATTER_HBP = 2
BATTER_SB = 5

# Pitcher Scoring
PITCHER_WIN = 4
PITCHER_ER = -2
PITCHER_SO = 2
PITCHER_IP = 2.25
PITCHER_H = -0.6
PITCHER_BB = -0.6
PITCHER_HB = -0.6
PITCHER_CG = 2.5
PITCHER_CGS = 2.5
PITCHER_NH = 5

def calculate_batter_points(single=0, double=0, triple=0, hr=0, rbi=0, run=0, walk=0, hbp=0, sb=0):
    """
    Calculate DraftKings points for a batter.
    """
    return (
        single * BATTER_SINGLE +
        double * BATTER_DOUBLE +
        triple * BATTER_TRIPLE +
        hr * BATTER_HOME_RUN +
        rbi * BATTER_RBI +
        run * BATTER_RUN +
        walk * BATTER_WALK +
        hbp * BATTER_HBP +
        sb * BATTER_SB
    )

def calculate_pitcher_points(win=0, er=0, so=0, ip=0, h=0, bb=0, hb=0, cg=0, cgs=0, nh=0):
    """
    Calculate DraftKings points for a pitcher.
    """
    return (
        win * PITCHER_WIN +
        er * PITCHER_ER +
        so * PITCHER_SO +
        ip * PITCHER_IP +
        h * PITCHER_H +
        bb * PITCHER_BB +
        hb * PITCHER_HB +
        cg * PITCHER_CG +
        cgs * PITCHER_CGS +
        nh * PITCHER_NH
    )
