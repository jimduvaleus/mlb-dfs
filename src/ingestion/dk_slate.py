import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Player:
    player_id: int
    name: str
    position: str
    eligible_positions: List[str] = field(default_factory=list)
    salary: float = 0.0
    team: str = ""
    opponent: str = ""
    roster_position: str = ""
    game: str = ""
    fd_player_id: str = ""  # FD-specific: full "slate_id-player_id" string for upload


class BaseSlateIngestor(ABC):
    @abstractmethod
    def get_slate_dataframe(self) -> pd.DataFrame:
        """Return a standardized player DataFrame for this slate."""
        ...

    @abstractmethod
    def get_players(self) -> List[Player]:
        """Return a list of Player objects for this slate."""
        ...


class DraftKingsSlateIngestor(BaseSlateIngestor):
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

        # Parse all eligible positions, mapping DK-specific labels.
        position_map = {'SP': 'P', 'RP': 'P'}

        def _parse_positions(raw: str) -> List[str]:
            tokens = str(raw).strip().split('/')
            mapped = [position_map.get(t, t) for t in tokens]
            seen: set = set()
            result: List[str] = []
            for t in mapped:
                if t not in seen:
                    seen.add(t)
                    result.append(t)
            return result

        df['eligible_positions'] = df['position'].apply(_parse_positions)
        df['position'] = df['eligible_positions'].str[0]

        # Basic validation for positions
        valid_positions = {'P', 'C', '1B', '2B', '3B', 'SS', 'OF'}
        if not df['position'].apply(lambda x: x in valid_positions).all():
            raise ValueError("Unexpected position strings found in CSV.")

        # Extract game ID from "Game Info" column.
        # Handles both "LAD @ SD 03/20/2026 ..." and "DET@SD 03/27/2026 ..." formats.
        if 'game_info' in df.columns:
            def _extract_game(info: str) -> str:
                tokens = str(info).split()
                if not tokens:
                    return ""
                first = tokens[0]
                if '@' in first:
                    # Format: "DET@SD 03/27/2026 ..."
                    return first
                if len(tokens) >= 3 and tokens[1] == '@':
                    # Format: "LAD @ SD 03/20/2026 ..."
                    return f"{tokens[0]}@{tokens[2]}"
                return ""
            df['game'] = df['game_info'].apply(_extract_game)
        else:
            df['game'] = ""

        df['opponent'] = ""

        return df[['player_id', 'name', 'position', 'eligible_positions', 'roster_position', 'salary', 'team', 'opponent', 'game']]

    def get_players(self) -> List[Player]:
        players = []
        for _, row in self.slate_df.iterrows():
            players.append(Player(
                player_id=row['player_id'],
                name=row['name'],
                position=row['position'],
                eligible_positions=row['eligible_positions'],
                salary=row['salary'],
                team=row['team'],
                opponent=row.get('opponent', ''),
                roster_position=row['roster_position'],
                game=row['game'],
            ))
        return players

    def get_slate_dataframe(self) -> pd.DataFrame:
        return self.slate_df
