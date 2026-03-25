import pandas as pd
import os
import glob
import subprocess
from src.ingestion.retrosheet_parser import RetrosheetParser

RETROSHEET_DIR = "data/raw/retrosheet"
PROCESSED_DIR = "data/processed"


def run_chadwick_cwdaily(year, input_dir):
    """
    Runs cwdaily to produce a per-player per-game stats CSV.
    Handles both .EVN (NL) and .EVA (AL) event files.
    Runs with cwd=input_dir so cwdaily can locate the TEAM<year> file.
    """
    abs_input_dir = os.path.abspath(input_dir)
    output_file = os.path.join(abs_input_dir, f"daily_{year}.csv")

    evn_files = glob.glob(os.path.join(abs_input_dir, f"{year}*.EVN"))
    eva_files = glob.glob(os.path.join(abs_input_dir, f"{year}*.EVA"))
    event_files = sorted(evn_files + eva_files)
    if not event_files:
        print(f"No .EVN/.EVA files found for year {year} in {abs_input_dir}")
        return None

    basenames = [os.path.basename(f) for f in event_files]
    command = ["cwdaily", "-y", str(year), "-n"] + basenames

    print(f"Running cwdaily for {year} ({len(basenames)} event files)...")
    with open(output_file, "w") as outfile:
        subprocess.run(command, stdout=outfile, check=True, cwd=abs_input_dir)
    print(f"cwdaily output saved to {output_file}")
    return output_file


def _prep_batting_df(daily_df):
    """
    Filter cwdaily output to batter rows (SLOT_CT 1–9, SEQ_CT 1) and rename
    columns to match RetrosheetParser.process_batting_stats() expectations.
    SEQ_CT == 1 keeps the player who started in each batting order slot;
    pinch-hitters and mid-game substitutes (SEQ_CT > 1) are excluded.
    DK counts both regular (B_BB) and intentional walks (B_IBB).
    """
    batters = daily_df[
        daily_df["SLOT_CT"].between(1, 9) & (daily_df["SEQ_CT"] == 1)
    ].copy()
    batters["BB"] = batters["B_BB"] + batters["B_IBB"]
    return batters.rename(columns={
        "PLAYER_ID": "player_id",
        "GAME_ID":   "game_id",
        "TEAM_ID":   "team_id",
        "SLOT_CT":   "slot",
        "B_H":       "H",
        "B_2B":      "D",
        "B_3B":      "T",
        "B_HR":      "HR",
        "B_RBI":     "RBI",
        "B_R":       "R",
        "B_HP":      "HBP",
        "B_SB":      "SB",
    })


def _prep_pitching_df(daily_df):
    """
    Rename cwdaily columns to match RetrosheetParser.process_pitching_stats()
    expectations. Filtering to starters is left to process_pitching_stats().
    TEAM_ID becomes pit_team_id so assign_slots can match opposing pitchers.
    """
    return daily_df.rename(columns={
        "PLAYER_ID": "player_id",
        "GAME_ID":   "game_id",
        "TEAM_ID":   "pit_team_id",
        "P_GS":      "GS",
        "P_W":       "W",
        "P_OUT":     "IP",   # outs recorded; RetrosheetParser divides by 3.0
        "P_ER":      "ER",
        "P_SO":      "SO",
        "P_H":       "H",
        "P_BB":      "BB",
        "P_HP":      "HB",
        "P_CG":      "CG",
    })


def assign_slots(batting_df, pitching_df):
    """
    Combine per-game batting and pitching rows into historical_logs format.

    batting_df  – output of _prep_batting_df after process_batting_stats:
                  columns include game_id, team_id, player_id, slot, dk_points
    pitching_df – output of _prep_pitching_df after process_pitching_stats
                  (starters_only=True): columns include game_id, player_id,
                  pit_team_id, GS, dk_points

    Returns DataFrame with columns: game_id, team_id, player_id, slot, dk_points
    """
    # --- Batter rows (slots 1–9) ---
    batting_rows = batting_df[["game_id", "team_id", "player_id", "slot", "dk_points"]].copy()

    # --- Pitcher rows (slot 10) ---
    # For each (game, batting_team), assign the opposing starter as slot 10.
    game_teams = batting_rows[["game_id", "team_id"]].drop_duplicates()

    starters = pitching_df[["game_id", "player_id", "pit_team_id", "dk_points"]].copy()

    opp_pitchers = game_teams.merge(starters, on="game_id")
    opp_pitchers = opp_pitchers[
        opp_pitchers["team_id"] != opp_pitchers["pit_team_id"]
    ].copy()
    opp_pitchers["slot"] = 10
    opp_pitchers = opp_pitchers[["game_id", "team_id", "player_id", "slot", "dk_points"]]

    return pd.concat([batting_rows, opp_pitchers], ignore_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process historical Retrosheet data for one or more years."
    )
    parser.add_argument(
        "years",
        type=int,
        nargs="+",
        help="One or more years to process (e.g., 2022 2023 2024 2025)",
    )
    args = parser.parse_args()

    all_logs = []
    for year in args.years:
        year_dir = os.path.join(RETROSHEET_DIR, str(year))
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        daily_file = run_chadwick_cwdaily(year, year_dir)
        if daily_file is None:
            print(f"Skipping {year}: no event files found.")
            continue

        daily_df = pd.read_csv(daily_file)

        batting_df = _prep_batting_df(daily_df)
        batting_df = RetrosheetParser.process_batting_stats(batting_df)

        pitching_df = _prep_pitching_df(daily_df)
        pitching_df = RetrosheetParser.process_pitching_stats(pitching_df, starters_only=True)

        logs = assign_slots(batting_df, pitching_df)
        print(f"  {year}: {len(logs)} rows")
        all_logs.append(logs)

    if not all_logs:
        print("No data processed — check that .EVN/.EVA files exist under data/raw/retrosheet/<year>/")
        raise SystemExit(1)

    combined = pd.concat(all_logs, ignore_index=True)
    output_path = os.path.join(PROCESSED_DIR, "historical_logs.parquet")
    combined.to_parquet(output_path, index=False)
    print(f"\nHistorical logs ({len(combined)} rows across {len(all_logs)} year(s)) saved to {output_path}")
