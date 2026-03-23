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
    Runs cwevent to process Retrosheet event files into a CSV.
    """
    output_file = os.path.join(input_dir, f"events_{year}.csv")
    event_files = glob.glob(os.path.join(input_dir, f"{year}*.EVN"))
    if not event_files:
        print(f"No .EVN files found for year {year} in {input_dir}")
        return None

    # cwevent -y [year] -f 0-96 [files...] > events.csv
    command = [
        "cwevent",
        "-y", str(year),
        "-f", "0-96", # Output all standard fields
    ] + event_files

    print(f"Running cwevent for {year}...")
    with open(output_file, "w") as outfile:
        subprocess.run(command, stdout=outfile, check=True)
    print(f"cwevent output saved to {output_file}")
    return output_file

def run_chadwick_cwbox(year, input_dir):
    """
    Runs cwbox to generate boxscore data for pitching and batting.
    """
    # Output paths
    batting_output = os.path.join(input_dir, f"box_batting_{year}.csv")
    pitching_output = os.path.join(input_dir, f"box_pitching_{year}.csv")

    event_files = glob.glob(os.path.join(input_dir, f"{year}*.EVN"))
    if not event_files:
        print(f"No .EVN files found for year {year} in {input_dir}")
        return None, None

    # cwbox -y [year] -f "playerid, gameid, AB, H, D, T, HR, RBI, R, BB, SO, SB, HBP, SH, SF" > box_batting.csv
    batting_command = [
        "cwbox",
        "-y", str(year),
        "-f", "playerid,gameid,AB,H,D,T,HR,RBI,R,BB,SO,SB,HBP,SH,SF", # Select batting stats
        "-P", # Batting stats
    ] + event_files

    # cwbox -y [year] -f "playerid, gameid, W, L, G, GS, CG, SHO, SV, IP, H, R, ER, HR, BB, IBB, SO, HB, BK, WP" -p > box_pitching.csv
    pitching_command = [
        "cwbox",
        "-y", str(year),
        "-f", "playerid,gameid,W,L,G,GS,CG,SHO,SV,IP,H,R,ER,HR,BB,IBB,SO,HB,BK,WP", # Select pitching stats
        "-p", # Pitching stats
    ] + event_files

    print(f"Running cwbox for batting for {year}...")
    with open(batting_output, "w") as outfile:
        subprocess.run(batting_command, stdout=outfile, check=True)
    print(f"cwbox batting output saved to {batting_output}")

    print(f"Running cwbox for pitching for {year}...")
    with open(pitching_output, "w") as outfile:
        subprocess.run(pitching_command, stdout=outfile, check=True)
    print(f"cwbox pitching output saved to {pitching_output}")

    return batting_output, pitching_output

def process_historical_data(year):
    """
    Main function to process historical Retrosheet data for a given year.
    """
    year_dir = os.path.join(RETROSHEET_DIR, str(year))
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Run cwevent (optional for now, as we mainly use cwbox for stats)
    # event_file_path = run_chadwick_cwevent(year, year_dir)

    # Step 2: Run cwbox to get batting and pitching stats
    batting_file, pitching_file = run_chadwick_cwbox(year, year_dir)

    # Step 3: Process batting stats
    if batting_file and os.path.exists(batting_file):
        batting_df = pd.read_csv(batting_file)
        batting_df = RetrosheetParser.process_batting_stats(batting_df)
        batting_output_path = os.path.join(PROCESSED_DIR, f"historical_batting_{year}.parquet")
        batting_df.to_parquet(batting_output_path, index=False)
        print(f"Processed batting data saved to {batting_output_path}")
    else:
        print(f"No batting data processed for {year}")

    # Step 4: Process pitching stats
    if pitching_file and os.path.exists(pitching_file):
        pitching_df = pd.read_csv(pitching_file)
        pitching_df = RetrosheetParser.process_pitching_stats(pitching_df)
        pitching_output_path = os.path.join(PROCESSED_DIR, f"historical_pitching_{year}.parquet")
        pitching_df.to_parquet(pitching_output_path, index=False)
        print(f"Processed pitching data saved to {pitching_output_path}")
    else:
        print(f"No pitching data processed for {year}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process historical Retrosheet data for a given year.")
    parser.add_argument("year", type=int, help="Year to process (e.g., 2023)")
    args = parser.parse_args()
    process_historical_data(args.year)
