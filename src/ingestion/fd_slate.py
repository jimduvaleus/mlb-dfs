"""
FanDuel slate ingestor.

FanDuel exports a single CSV file that serves as both salary list and upload
template.  Its layout is unconventional:

  Columns 0-12  (entry_id … UTIL)   Upload template rows, one per entry.
  Column  13                         Always empty.
  Column  14+                        Player pool — but offset: the column
                                     header row appears several rows *below*
                                     the top of the file (at the row where
                                     column 14 == "Player ID + Player Name").

Parsing strategy
----------------
1. Read the entire file with no header (dtype=str to preserve leading zeros
   and mixed types).
2. Scan column 14 to find the player-pool header row.
3. Slice the raw DataFrame from that row downwards, columns 14+.
4. Promote the first slice row to column headers.
5. Drop trailing empty rows, then rename / coerce / validate.

For uploads (Phase 5): only columns 0-12 of the original file are needed.
The player pool section (column 14+) is the salary/projection source.
"""

from __future__ import annotations

import pandas as pd
from typing import List

from src.ingestion.dk_slate import BaseSlateIngestor, Player

# Column index where the player-pool section starts in the raw file.
_PLAYER_SECTION_START_COL = 14

# Sentinel that marks the player-pool header row in column 14.
_PLAYER_HEADER_SENTINEL = "Player ID + Player Name"

# Positions valid in both FD and our internal model.
_VALID_POSITIONS = {"P", "C", "1B", "2B", "3B", "SS", "OF"}


class FanDuelSlateIngestor(BaseSlateIngestor):
    """
    Parses a FanDuel MLB salary/upload-template CSV into a standardized
    player DataFrame.

    The ingestor preserves:
      - player_id  (int)   — numeric suffix of the FD "Id" field
      - fd_player_id (str) — full "slate_id-player_id" string, needed for
                             upload file generation in Phase 5
      - name, position, eligible_positions, roster_position
      - salary, team, opponent, game
    """

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.slate_df = self._load_and_parse_csv()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_player_header_row(raw: pd.DataFrame) -> int:
        """Return the 0-based row index of the player-pool header row."""
        if raw.shape[1] <= _PLAYER_SECTION_START_COL:
            raise ValueError(
                f"Could not locate player-pool header ('{_PLAYER_HEADER_SENTINEL}') "
                f"in column {_PLAYER_SECTION_START_COL} of the FanDuel CSV — "
                f"file only has {raw.shape[1]} columns."
            )
        col = raw.iloc[:, _PLAYER_SECTION_START_COL]
        matches = col[col == _PLAYER_HEADER_SENTINEL].index.tolist()
        if not matches:
            raise ValueError(
                f"Could not locate player-pool header ('{_PLAYER_HEADER_SENTINEL}') "
                f"in column {_PLAYER_SECTION_START_COL} of the FanDuel CSV."
            )
        return matches[0]

    @staticmethod
    def _parse_fd_positions(raw: str) -> List[str]:
        """
        Split a FD position string on '/' and return unique tokens.
        e.g. 'C/1B' -> ['C', '1B'],  'OF' -> ['OF'],  'P' -> ['P']
        """
        tokens = [t.strip() for t in str(raw).split("/") if t.strip()]
        seen: set = set()
        result: List[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    # ------------------------------------------------------------------
    # Main parsing
    # ------------------------------------------------------------------

    def _load_and_parse_csv(self) -> pd.DataFrame:
        # Read the whole file as strings.
        # The FD CSV has variable column counts per row: the upload-template
        # section (cols 0-12) has 15 fields but the player-pool section starts
        # at col 14 and extends to col 31 (18 player columns).  Supplying
        # `names=range(N)` with N > the maximum row width causes pandas to pad
        # all shorter rows with empty strings, avoiding the "Expected X fields,
        # saw Y" error that both engines raise when row widths differ.
        _TOTAL_COLS = 35  # safely wider than the actual max of 32
        raw = pd.read_csv(
            self.csv_filepath,
            header=None,
            names=range(_TOTAL_COLS),
            dtype=str,
            keep_default_na=False,
        )

        # Locate and extract the player-pool section.
        header_row_idx = self._find_player_header_row(raw)
        player_section = (
            raw.iloc[header_row_idx:, _PLAYER_SECTION_START_COL:]
            .reset_index(drop=True)
        )

        # Promote the first row to column headers.
        player_section.columns = player_section.iloc[0].tolist()
        player_section = player_section.iloc[1:].reset_index(drop=True)

        # Drop rows where the sentinel column is empty (padding rows).
        player_section = player_section[
            player_section[_PLAYER_HEADER_SENTINEL].notna()
            & (player_section[_PLAYER_HEADER_SENTINEL].str.strip() != "")
        ].reset_index(drop=True)

        # Rename to our internal schema.
        player_section = player_section.rename(columns={
            "Id":               "fd_player_id",
            "Nickname":         "name",
            "Position":         "fd_position",
            "Salary":           "salary",
            "Game":             "game",
            "Team":             "team",
            "Opponent":         "opponent",
            "Roster Position":  "fd_roster_position",
        })

        # player_id: numeric suffix of fd_player_id ("128874-16960" → 16960).
        player_section["player_id"] = (
            player_section["fd_player_id"].str.split("-").str[-1].astype(int)
        )

        # Salary: coerce to numeric.
        player_section["salary"] = pd.to_numeric(
            player_section["salary"], errors="coerce"
        )
        if player_section["salary"].isnull().any():
            raise ValueError("Missing or invalid salaries found in FD CSV.")

        # Parse positions from the Position column (may be compound, e.g. 'C/1B').
        player_section["eligible_positions"] = player_section["fd_position"].apply(
            self._parse_fd_positions
        )
        player_section["position"] = player_section["eligible_positions"].str[0]

        # Validate that all primary positions are recognised.
        invalid_mask = ~player_section["position"].isin(_VALID_POSITIONS)
        if invalid_mask.any():
            bad = player_section.loc[invalid_mask, "position"].unique().tolist()
            raise ValueError(f"Unexpected position strings in FD CSV: {bad}")

        # roster_position: the raw FD 'Roster Position' field (e.g. 'OF/UTIL',
        # 'C/1B/UTIL', 'P') is preserved as-is for Phase-5 upload use.
        player_section["roster_position"] = player_section["fd_roster_position"]

        player_section["game_start_time"] = ""

        return player_section[[
            "player_id",
            "fd_player_id",
            "name",
            "position",
            "eligible_positions",
            "roster_position",
            "salary",
            "team",
            "opponent",
            "game",
            "game_start_time",
        ]]

    # ------------------------------------------------------------------
    # Public interface (BaseSlateIngestor)
    # ------------------------------------------------------------------

    def get_slate_dataframe(self) -> pd.DataFrame:
        return self.slate_df

    def get_players(self) -> List[Player]:
        players = []
        for _, row in self.slate_df.iterrows():
            players.append(Player(
                player_id=row["player_id"],
                name=row["name"],
                position=row["position"],
                eligible_positions=row["eligible_positions"],
                salary=row["salary"],
                team=row["team"],
                opponent=row["opponent"],
                roster_position=row["roster_position"],
                game=row["game"],
                fd_player_id=row["fd_player_id"],
            ))
        return players
