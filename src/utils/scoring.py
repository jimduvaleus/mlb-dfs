"""
DraftKings MLB Scoring Constants and Functions.

# HISTORICAL PIPELINE WARNING
#
# The constants in this module are imported directly by
# src/ingestion/retrosheet_parser.py to compute DK fantasy points that are
# stored in data/processed/historical_logs.parquet.  That parquet file is the
# training source for the empirical copula (build_copula.py) and the batter PCA
# model (fit_batter_pca.py).
#
# Changing ANY constant here — even a single decimal — silently corrupts the
# copula and all downstream simulation results without raising an error.
#
# RULES:
#   1. Do NOT modify these constants for FanDuel support.  FD scoring lives in
#      src/platforms/fanduel.py and must never touch this file until Phase 9
#      of the FanDuel Platform Plan.
#   2. tests/test_scoring_contract.py pins every constant to its exact value.
#      If that test fails, you have introduced a breaking change to the
#      historical pipeline.
#   3. If you need to add a new platform's scoring, create a new module under
#      src/platforms/ and do NOT import from this file.
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
