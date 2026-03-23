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
            "Name + ID": "name_plus_id" # Keep this for potential future use or validation
        }, inplace=True)

        # Data cleaning and validation
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        if df['salary'].isnull().any():
            raise ValueError("Missing or invalid salaries found in CSV.")

        # Basic validation for positions (can be expanded)
        valid_positions = {'P', 'C', '1B', '2B', '3B', 'SS', 'OF'}
        if not df['position'].apply(lambda x: x in valid_positions).all():
             # Log or handle unexpected positions more gracefully in a real app
            # For now, we'll just raise an error as per the design to validate
            raise ValueError("Unexpected position strings found in CSV.")

        return df[['player_id', 'name', 'position', 'roster_position', 'salary', 'team']]

    def get_players(self) -> List[Player]:
        players = []
        for _, row in self.slate_df.iterrows():
            players.append(Player(
                player_id=row['player_id'],
                name=row['name'],
                position=row['position'],
                salary=row['salary'],
                team=row['team'],
                roster_position=row['roster_position']
            ))
        return players

    def get_slate_dataframe(self) -> pd.DataFrame:
        return self.slate_df
