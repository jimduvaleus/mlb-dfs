"""
Fetch today's MLB schedule and report any doubleheader teams.

Thin CLI wrapper around src/api/mlb_schedule.py for manual debugging — the
server calls that module's functions directly (a single fast JSON request,
no subprocess needed).

Usage
-----
    python scripts/fetch_mlb_schedule.py
    python scripts/fetch_mlb_schedule.py --date 2026-06-24
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.mlb_schedule import fetch_schedule, save_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    data = fetch_schedule(args.date)
    save_schedule(data)
    print(json.dumps(data, indent=2))
    if data["doubleheader_teams"]:
        print(f"\nDoubleheader teams: {', '.join(data['doubleheader_teams'])}", file=sys.stderr)
    else:
        print("\nNo doubleheaders detected.", file=sys.stderr)


if __name__ == "__main__":
    main()
