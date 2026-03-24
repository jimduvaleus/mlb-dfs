import pandas as pd
import numpy as np
import os
import glob
import subprocess
from src.ingestion.retrosheet_parser import RetrosheetParser

RETROSHEET_DIR = "data/raw/retrosheet"
PROCESSED_DIR = "data/processed"


def run_chadwick_cwevent(year, input_dir):
    """
    Runs cwevent to process Retrosheet event files into a CSV with a header row.
    The -n flag causes cwevent to emit field names as the first row.
    """
    output_file = os.path.join(input_dir, f"events_{year}.csv")
    event_files = glob.glob(os.path.join(input_dir, f"{year}*.EVN"))
    if not event_files:
        print(f"No .EVN files found for year {year} in {input_dir}")
        return None

    command = [
        "cwevent",
        "-y", str(year),
        "-n",        # include header row with field names
        "-f", "0-96",
    ] + event_files

    print(f"Running cwevent for {year}...")
    with open(output_file, "w") as outfile:
        subprocess.run(command, stdout=outfile, check=True)
    print(f"cwevent output saved to {output_file}")
    return output_file


def run_chadwick_cwbox(year, input_dir):
    """
    Runs cwbox to generate boxscore data for batting and pitching.
    """
    batting_output = os.path.join(input_dir, f"box_batting_{year}.csv")
    pitching_output = os.path.join(input_dir, f"box_pitching_{year}.csv")

    event_files = glob.glob(os.path.join(input_dir, f"{year}*.EVN"))
    if not event_files:
        print(f"No .EVN files found for year {year} in {input_dir}")
        return None, None

    batting_command = [
        "cwbox",
        "-y", str(year),
        "-f", "playerid,gameid,AB,H,D,T,HR,RBI,R,BB,SO,SB,HBP,SH,SF",
        "-P",
    ] + event_files

    pitching_command = [
        "cwbox",
        "-y", str(year),
        "-f", "playerid,gameid,W,L,G,GS,CG,SHO,SV,IP,H,R,ER,HR,BB,IBB,SO,HB,BK,WP",
        "-p",
    ] + event_files

    print(f"Running cwbox for batting for {year}...")
    with open(batting_output, "w") as outfile:
        subprocess.run(batting_command, stdout=outfile, check=True)

    print(f"Running cwbox for pitching for {year}...")
    with open(pitching_output, "w") as outfile:
        subprocess.run(pitching_command, stdout=outfile, check=True)

    return batting_output, pitching_output


def _get_batting_order(event_file):
    """
    Extract (game_id, player_id, team_id, slot) from cwevent output.

    Retrosheet game IDs are formatted as HHHyyyymmddG where the first three
    characters are the home team code.  The AWAY_TEAM_ID column carries the
    visiting team code, and BAT_HOME_ID (0=visitor, 1=home) tells us which
    side a batter belongs to.
    """
    events = pd.read_csv(event_file, low_memory=False)

    home_team = events["GAME_ID"].str[:3]
    events["bat_team_id"] = np.where(
        events["BAT_HOME_ID"] == 1,
        home_team,
        events["AWAY_TEAM_ID"],
    )
    events["pit_team_id"] = np.where(
        events["BAT_HOME_ID"] == 1,
        events["AWAY_TEAM_ID"],
        home_team,
    )

    # Batting order: one slot per (game, batter) — take first occurrence
    batting_order = (
        events[["GAME_ID", "BAT_ID", "bat_team_id", "BAT_LINEUP_ID"]]
        .rename(columns={
            "GAME_ID": "game_id",
            "BAT_ID": "player_id",
            "bat_team_id": "team_id",
            "BAT_LINEUP_ID": "slot",
        })
        .drop_duplicates(["game_id", "player_id"])
    )

    # Pitcher team mapping: needed to assign the opposing starter later
    pitcher_teams = (
        events[["GAME_ID", "PIT_ID", "pit_team_id"]]
        .rename(columns={"GAME_ID": "game_id", "PIT_ID": "player_id"})
        .drop_duplicates(["game_id", "player_id"])
    )

    return batting_order, pitcher_teams


def assign_slots(batting_df, pitching_df, event_file):
    """
    Combine batting and pitching stats into a historical_logs DataFrame.

    Batters receive slots 1–9 from their batting order position.
    The opposing starting pitcher receives slot 10 for each (game, team) pair.

    Returns a DataFrame with columns:
        game_id, team_id, player_id, slot, dk_points
    """
    batting_order, pitcher_teams = _get_batting_order(event_file)

    # Normalise cwbox column names to match cwevent names
    batting_df = batting_df.rename(columns={"playerid": "player_id", "gameid": "game_id"})
    pitching_df = pitching_df.rename(columns={"playerid": "player_id", "gameid": "game_id"})

    # --- Batter rows (slots 1–9) ---
    batting_merged = batting_df.merge(
        batting_order[["game_id", "player_id", "team_id", "slot"]],
        on=["game_id", "player_id"],
        how="inner",
    )

    # --- Pitcher rows (slot 10) ---
    # pitching_df has already been filtered to starters (GS == 1) by
    # RetrosheetParser.process_pitching_stats(starters_only=True).
    starters = pitching_df.merge(pitcher_teams, on=["game_id", "player_id"], how="inner")

    # For each (game, batting_team) pair, find the starter from the opposing team.
    game_teams = batting_merged[["game_id", "team_id"]].drop_duplicates()
    opp_pitchers = game_teams.merge(
        starters[["game_id", "player_id", "pit_team_id", "dk_points"]],
        on="game_id",
    )
    opp_pitchers = opp_pitchers[
        opp_pitchers["team_id"] != opp_pitchers["pit_team_id"]
    ].copy()
    opp_pitchers["slot"] = 10
    opp_pitchers = opp_pitchers[["game_id", "team_id", "player_id", "slot", "dk_points"]]

    batting_rows = batting_merged[["game_id", "team_id", "player_id", "slot", "dk_points"]]
    return pd.concat([batting_rows, opp_pitchers], ignore_index=True)


def process_historical_data(year):
    """
    Main function to process historical Retrosheet data for a given year.
    Produces data/processed/historical_logs.parquet with columns:
        game_id, team_id, player_id, slot, dk_points
    """
    year_dir = os.path.join(RETROSHEET_DIR, str(year))
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Run cwevent (needed for batting order and team assignment)
    event_file = run_chadwick_cwevent(year, year_dir)
    if event_file is None:
        print(f"Skipping {year}: no event files found.")
        return

    # Step 2: Run cwbox for per-player stats
    batting_file, pitching_file = run_chadwick_cwbox(year, year_dir)
    if batting_file is None or pitching_file is None:
        print(f"Skipping {year}: cwbox failed.")
        return

    # Step 3: Score batting and pitching
    batting_df = RetrosheetParser.process_batting_stats(pd.read_csv(batting_file))
    pitching_df = RetrosheetParser.process_pitching_stats(
        pd.read_csv(pitching_file), starters_only=True
    )

    # Step 4: Assign slots and combine into historical_logs
    logs = assign_slots(batting_df, pitching_df, event_file)

    output_path = os.path.join(PROCESSED_DIR, "historical_logs.parquet")
    logs.to_parquet(output_path, index=False)
    print(f"Historical logs ({len(logs)} rows) saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process historical Retrosheet data for a given year.")
    parser.add_argument("year", type=int, help="Year to process (e.g., 2023)")
    args = parser.parse_args()
    process_historical_data(args.year)
