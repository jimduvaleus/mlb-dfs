import pandas as pd
import numpy as np
from typing import Optional
from src.utils.scoring import (
    BATTER_SINGLE, BATTER_DOUBLE, BATTER_TRIPLE, BATTER_HOME_RUN,
    BATTER_RBI, BATTER_RUN, BATTER_WALK, BATTER_HBP, BATTER_SB,
    PITCHER_WIN, PITCHER_ER, PITCHER_SO, PITCHER_IP,
    PITCHER_H, PITCHER_BB, PITCHER_HB, PITCHER_CG, PITCHER_CGS, PITCHER_NH,
)
from src.platforms.base import ScoringRules


class RetrosheetParser:
    """
    Parser for Retrosheet data processed by Chadwick tools.

    Both process_batting_stats and process_pitching_stats accept an optional
    `rules: ScoringRules` parameter (Phase 9 addition).  When omitted they
    fall back to the legacy DraftKings constants so the existing DK pipeline
    is unchanged.
    """

    @staticmethod
    def process_batting_stats(
        df: pd.DataFrame,
        rules: Optional[ScoringRules] = None,
    ) -> pd.DataFrame:
        """
        Process batting statistics and calculate fantasy points.

        Expected columns in df:
        - H (Hits)
        - D (Doubles)
        - T (Triples)
        - HR (Home Runs)
        - RBI (Runs Batted In)
        - R (Runs Scored)
        - BB (Walks)
        - HBP (Hit By Pitch — batting stat)
        - SB (Stolen Bases)

        Args:
            rules: Platform-specific scoring weights.  Defaults to DraftKings
                   constants when None.
        """
        df = df.copy()
        df['1B'] = df['H'] - df['D'] - df['T'] - df['HR']

        if rules is not None:
            df['dk_points'] = (
                df['1B'] * rules.single +
                df['D'] * rules.double +
                df['T'] * rules.triple +
                df['HR'] * rules.home_run +
                df['RBI'] * rules.rbi +
                df['R'] * rules.run +
                df['BB'] * rules.walk +
                df['HBP'] * rules.hbp +
                df['SB'] * rules.sb
            )
        else:
            df['dk_points'] = (
                df['1B'] * BATTER_SINGLE +
                df['D'] * BATTER_DOUBLE +
                df['T'] * BATTER_TRIPLE +
                df['HR'] * BATTER_HOME_RUN +
                df['RBI'] * BATTER_RBI +
                df['R'] * BATTER_RUN +
                df['BB'] * BATTER_WALK +
                df['HBP'] * BATTER_HBP +
                df['SB'] * BATTER_SB
            )
        return df

    @staticmethod
    def process_pitching_stats(
        df: pd.DataFrame,
        starters_only: bool = True,
        rules: Optional[ScoringRules] = None,
    ) -> pd.DataFrame:
        """
        Process pitching statistics and calculate fantasy points.

        Expected columns in df:
        - GS (Games Started: 1 if starter, 0 otherwise)
        - W (Win: 1 or 0)
        - IP (Innings Pitched as outs recorded — Chadwick cwbox outputs outs)
        - ER (Earned Runs)
        - SO (Strikeouts) — note: the rename in process_historical maps P_SO → SO
          via the K column alias; callers must rename appropriately.
        - H (Hits Against)
        - BB (Walks Against)
        - HB (Hit Batters — pitching stat)
        - CG (Complete Game: 1 or 0)

        Args:
            starters_only: If True, filter to rows where GS == 1 before scoring.
            rules: Platform-specific scoring weights.  Defaults to DraftKings
                   constants when None.

        Quality Start detection
        -----------------------
        A Quality Start (QS) is awarded when IP_outs >= 18 (>= 6 innings) AND
        ER <= 3.  The QS column is always computed and stored in the output df;
        for DraftKings (rules=None or rules.qs == 0) it contributes 0 points.
        """
        df = df.copy()

        if starters_only:
            df = df[df['GS'] == 1].reset_index(drop=True)

        # Chadwick cwbox outputs IP as total outs recorded (e.g. 37 outs = 12.1 innings).
        # Convert to decimal innings.
        df['IP_dec'] = df['IP'] / 3.0

        # Quality Start: >= 6 full innings (18 outs) AND <= 3 earned runs.
        # Stored as int (1/0) so it can be used directly in point calculations.
        df['QS'] = ((df['IP'] >= 18) & (df['ER'] <= 3)).astype(int)

        if rules is not None:
            df['dk_points'] = (
                df['W'] * rules.win +
                df['ER'] * rules.er +
                df['SO'] * rules.so +
                df['IP_dec'] * rules.ip +
                df['H'] * rules.h +
                df['BB'] * rules.bb +
                df['HB'] * rules.hb +
                df['CG'] * rules.cg +
                df['QS'] * rules.qs
            )
        else:
            df['dk_points'] = (
                df['W'] * PITCHER_WIN +
                df['ER'] * PITCHER_ER +
                df['SO'] * PITCHER_SO +
                df['IP_dec'] * PITCHER_IP +
                df['H'] * PITCHER_H +
                df['BB'] * PITCHER_BB +
                df['HB'] * PITCHER_HB +
                df['CG'] * PITCHER_CG
            )
        return df
