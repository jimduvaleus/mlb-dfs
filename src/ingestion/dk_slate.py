import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Player:
    player_id: int
    name: str
    position: str
    salary: float
    team: str
    roster_position: str
    game: str = ""


class DraftKingsSlateIngestor:
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.slate_df = self._load_and_parse_csv()

    def _load_and_parse_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_filepath)

        # Standardize column names
        df.rename(columns={
            "ID": "player_id",
            "Name": "name",
            "Position": "position",
            "Roster Position": "roster_position",
            "Salary": "salary",
            "TeamAbbrev": "team",
            "Game Info": "game_info",
            "Name + ID": "name_plus_id" # Keep this for potential future use or validation
        }, inplace=True)

        # Data cleaning and validation
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        if df['salary'].isnull().any():
            raise ValueError("Missing or invalid salaries found in CSV.")

        # Normalize positions: take primary position from multi-eligible
        # strings (e.g. "1B/2B" → "1B") and map DK-specific labels.
        position_map = {'SP': 'P', 'RP': 'P'}
        df['position'] = (
            df['position']
            .str.split('/')
            .str[0]
            .replace(position_map)
        )

        # Basic validation for positions
        valid_positions = {'P', 'C', '1B', '2B', '3B', 'SS', 'OF'}
        if not df['position'].apply(lambda x: x in valid_positions).all():
            raise ValueError("Unexpected position strings found in CSV.")

        # Extract game ID from "Game Info" column (e.g. "LAD @ SD 03/20/2026 ..." -> "LAD@SD")
        if 'game_info' in df.columns:
            def _extract_game(info: str) -> str:
                tokens = str(info).split()
                if len(tokens) >= 3 and tokens[1] == '@':
                    return f"{tokens[0]}@{tokens[2]}"
                return ""
            df['game'] = df['game_info'].apply(_extract_game)
        else:
            df['game'] = ""

        return df[['player_id', 'name', 'position', 'roster_position', 'salary', 'team', 'game']]

    def get_players(self) -> List[Player]:
        players = []
        for _, row in self.slate_df.iterrows():
            players.append(Player(
                player_id=row['player_id'],
                name=row['name'],
                position=row['position'],
                salary=row['salary'],
                team=row['team'],
                roster_position=row['roster_position'],
                game=row['game'],
            ))
        return players

    def get_slate_dataframe(self) -> pd.DataFrame:
        return self.slate_df
