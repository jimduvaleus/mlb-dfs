#!/usr/bin/env python3
"""Inject a missed lineup notification directly into twitter_lineups.json.

Usage
-----
Paste body interactively (end with Ctrl-D on a blank line):
    python scripts/inject_lineup.py

Read from a file:
    python scripts/inject_lineup.py -f lineup.txt

Pass body as a single argument (use $'...' for newlines in bash):
    python scripts/inject_lineup.py --body $'Giants 6/8\n\nSchmitt LF\n...'
"""

import argparse
import sys
import uuid
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.api.config_io import read_config
from src.api.slate_exclusions import compute_file_fingerprint
from src.api.twitter_lineups import (
    match_player_name,
    parse_notification_body,
    upsert_twitter_lineup,
)


def _load_team_hitters(team: str) -> list[dict]:
    cfg = read_config()
    slate_df = pd.read_csv(cfg.paths.dk_slate)
    slate_df.columns = [c.lower().replace(" ", "_") for c in slate_df.columns]
    for c in list(slate_df.columns):
        if "teamabbrev" in c:
            slate_df = slate_df.rename(columns={c: "team"})
        elif c == "id":
            slate_df = slate_df.rename(columns={c: "player_id"})
    rows = slate_df[(slate_df["team"] == team) & (slate_df["position"] != "P")]
    return [
        {
            "player_id": int(r["player_id"]),
            "name": str(r["name"]),
            "team": str(r["team"]),
            "position": str(r["position"]),
            "salary": int(r["salary"]),
        }
        for _, r in rows.iterrows()
    ]


def _pick(prompt: str, options: list[dict]) -> dict | None:
    """Ask the user to pick one candidate or skip."""
    print(f"\n  {prompt}")
    for i, c in enumerate(options, 1):
        print(f"    [{i}] {c['name']}  {c['position']}  ${c['salary']:,}  [{c['match_confidence']}]")
    print(f"    [0] Skip (save as placeholder)")
    while True:
        raw = input("  Choice: ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Enter 0–{len(options)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject a missed lineup notification.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", help="Path to a text file containing the notification body")
    group.add_argument("--body", help="Notification body as a single string argument")
    args = parser.parse_args()

    if args.file:
        body = Path(args.file).read_text()
    elif args.body:
        body = args.body
    else:
        print("Paste the notification body below. End with Ctrl-D on a blank line.\n")
        try:
            body = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    body = body.strip()
    if not body:
        print("Empty body — nothing to do.")
        sys.exit(1)

    # Parse
    team, raw_slots, is_updated = parse_notification_body(body)
    if team is None:
        print("ERROR: Could not identify team from body. Check the team name spelling.")
        sys.exit(1)

    print(f"\nTeam: {team}  |  slots found: {len(raw_slots)}  |  is_updated: {is_updated}")

    if not raw_slots:
        print("ERROR: No batter slots extracted. Check that player lines follow 'Name POS' format.")
        sys.exit(1)

    team_hitters = _load_team_hitters(team)
    if not team_hitters:
        print(f"WARNING: {team} not found on the current DK slate — lineup will be saved but won't affect this run.")

    # Resolve each slot
    save_slots: list[dict] = []
    ambiguous = False
    print()
    for raw in raw_slots:
        candidates = match_player_name(raw["name"], team_hitters)
        if len(candidates) == 1:
            c = candidates[0]
            print(f"  Slot {raw['slot']:1d}  {raw['name']:<20s}  →  {c['name']}  [{c['match_confidence']}]")
            save_slots.append({"slot": raw["slot"], "player_id": c["player_id"], "name": c["name"]})
        elif len(candidates) == 0:
            print(f"  Slot {raw['slot']:1d}  {raw['name']:<20s}  →  NO MATCH (not in slate)")
            save_slots.append({"slot": raw["slot"], "player_id": None, "name": raw["name"]})
            ambiguous = True
        else:
            print(f"  Slot {raw['slot']:1d}  {raw['name']:<20s}  →  {len(candidates)} candidates (ambiguous)")
            ambiguous = True
            chosen = _pick(f"Select player for slot {raw['slot']} ({raw['name']}):", candidates)
            if chosen:
                save_slots.append({"slot": raw["slot"], "player_id": chosen["player_id"], "name": chosen["name"]})
            else:
                save_slots.append({"slot": raw["slot"], "player_id": None, "name": raw["name"]})

    # Confirm
    print()
    nulls = sum(1 for s in save_slots if s["player_id"] is None)
    if nulls:
        print(f"  Note: {nulls} slot(s) will be saved as placeholder (not in slate / skipped).")
    answer = input(f"Save {team} lineup ({len(save_slots)} slots)? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        print("Aborted.")
        sys.exit(0)

    cfg = read_config()
    fingerprint = compute_file_fingerprint(Path(cfg.paths.dk_slate))
    record = upsert_twitter_lineup(team, str(uuid.uuid4()), save_slots, fingerprint, locked=True)
    print(f"\nSaved {record['team']} — {len(record['slots'])} slots, locked=True")

    # Bake into projections CSV so slot_confirmed shows immediately without a pipeline run
    proj_path = Path(cfg.paths.projections) if cfg.paths.projections else None
    if proj_path and proj_path.exists():
        import pandas as _pd
        df = _pd.read_csv(proj_path)
        changed = False
        for s in record["slots"]:
            pid, slot = s.get("player_id"), s.get("slot")
            if pid is None:
                continue
            mask = df["player_id"] == pid
            if mask.any():
                df.loc[mask, "lineup_slot"] = slot
                df.loc[mask, "slot_confirmed"] = True
                changed = True
        if changed:
            df.to_csv(proj_path, index=False)
            print("Baked slot data into projections CSV.")


if __name__ == "__main__":
    main()
