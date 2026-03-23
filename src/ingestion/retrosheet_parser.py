import pandas as pd
import numpy as np
from src.utils.scoring import calculate_batter_points, calculate_pitcher_points

class RetrosheetParser:
    """
    Parser for Retrosheet data processed by Chadwick tools.
    """

    @staticmethod
    def process_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batting statistics and calculate DraftKings points.
        
        Expected columns in df:
        - player_id
        - game_id
        - H (Hits)
        - D (Doubles)
        - T (Triples)
        - HR (Home Runs)
        - RBI (Runs Batted In)
        - R (Runs Scored)
        - BB (Walks)
        - HBP (Hit By Pitch)
        - SB (Stolen Bases)
        """
        # Calculate Singles (H - D - T - HR)
        df['1B'] = df['H'] - df['D'] - df['T'] - df['HR']
        
        # Calculate DK points
        df['dk_points'] = df.apply(
            lambda row: calculate_batter_points(
                single=row['1B'],
                double=row['D'],
                triple=row['T'],
                hr=row['HR'],
                rbi=row['RBI'],
                run=row['R'],
                walk=row['BB'],
                hbp=row['HBP'],
                sb=row['SB']
            ),
            axis=1
        )
        return df

    @staticmethod
    def process_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process pitching statistics and calculate DraftKings points.
        
        Expected columns in df:
        - player_id
        - game_id
        - W (Win: 1 or 0)
        - IP (Innings Pitched - Note: Chadwick might output this as outs or 12.1 style)
        - ER (Earned Runs)
        - K (Strikeouts)
        - H (Hits Against)
        - BB (Walks Against)
        - HB (Hit Batters)
        """
        # Convert IP to decimal if it's in 12.1/12.2 format
        def convert_ip(ip):
            if isinstance(ip, str) and '.' in ip:
                parts = ip.split('.')
                return float(parts[0]) + (float(parts[1]) / 3.0)
            return float(ip)

        df['IP_dec'] = df['IP'].apply(convert_ip)
        
        # Calculate DK points
        df['dk_points'] = df.apply(
            lambda row: calculate_pitcher_points(
                win=row['W'],
                er=row['ER'],
                so=row['K'],
                ip=row['IP_dec'],
                h=row['H'],
                bb=row['BB'],
                hb=row['HB']
            ),
            axis=1
        )
        return df
