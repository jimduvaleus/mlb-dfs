import pandas as pd
import numpy as np
from src.utils.scoring import (
    BATTER_SINGLE, BATTER_DOUBLE, BATTER_TRIPLE, BATTER_HOME_RUN,
    BATTER_RBI, BATTER_RUN, BATTER_WALK, BATTER_HBP, BATTER_SB,
    PITCHER_WIN, PITCHER_ER, PITCHER_SO, PITCHER_IP,
    PITCHER_H, PITCHER_BB, PITCHER_HB, PITCHER_CG, PITCHER_CGS, PITCHER_NH,
)


class RetrosheetParser:
    """
    Parser for Retrosheet data processed by Chadwick tools.
    """

    @staticmethod
    def process_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batting statistics and calculate DraftKings points.

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
        """
        df = df.copy()
        df['1B'] = df['H'] - df['D'] - df['T'] - df['HR']
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
    def process_pitching_stats(df: pd.DataFrame, starters_only: bool = True) -> pd.DataFrame:
        """
        Process pitching statistics and calculate DraftKings points.

        Expected columns in df:
        - GS (Games Started: 1 if starter, 0 otherwise)
        - W (Win: 1 or 0)
        - IP (Innings Pitched as outs recorded — Chadwick cwbox outputs outs)
        - ER (Earned Runs)
        - K (Strikeouts)
        - H (Hits Against)
        - BB (Walks Against)
        - HB (Hit Batters — pitching stat)
        - CG (Complete Game: 1 or 0)

        Args:
            starters_only: If True, filter to rows where GS == 1 before scoring.
        """
        df = df.copy()

        if starters_only:
            df = df[df['GS'] == 1].reset_index(drop=True)

        # Chadwick cwbox outputs IP as total outs recorded (e.g. 37 outs = 12.1 innings).
        # Convert to decimal innings.
        df['IP_dec'] = df['IP'] / 3.0

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
