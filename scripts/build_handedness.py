"""
Build data/handedness.csv from all DKSalaries.csv files in the archive directory.

Queries the MLB Stats API to resolve batter handedness (bats) and pitcher
throwing hand (throws) for every unique player/team combination found across
all archived slates.

Re-running is safe: existing entries are preserved and only new players are
added, so the file accumulates coverage over time.

Output: data/handedness.csv
  name   — normalised player name (lowercase ASCII, no punctuation)
  team   — DK team abbreviation (NYY, LAD, …)
  bats   — L / R / S (switch hitter)
  throws — L / R

Lookup contract (used by _build_player_pool):
  1. Match on (name, team) — handles same-name players on different teams.
  2. Fall back to name-only when exactly one entry exists for that name.
  3. Multiple entries with no team match → NaN (warning logged at build time).

Usage
-----
    python scripts/build_handedness.py
    python scripts/build_handedness.py --seasons 2025 2026
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
ARCHIVE_DIR    = PROJECT_ROOT / "archive"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "handedness.csv"
NAME_MAP_PATH  = PROJECT_ROOT / "data" / "name_map.json"

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def _normalise(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _load_name_aliases() -> dict[str, str]:
    """
    Load data/name_map.json and return {norm_dk_name: api_raw_name}.

    name_map.json format: {api_name: dk_name}  (e.g. "Enrique Hernandez" → "Kike Hernandez")
    We invert it so callers can look up "kike hernandez" → "Enrique Hernandez" (the
    name to use when querying the MLB Stats API).
    """
    import json
    if not NAME_MAP_PATH.exists():
        return {}
    with open(NAME_MAP_PATH) as f:
        name_map: dict[str, str] = json.load(f)
    return {_normalise(dk): api for api, dk in name_map.items()}


def _collect_archive_players() -> pd.DataFrame:
    """
    Scan all archive/*/DKSalaries.csv files.

    Returns a DataFrame of unique (norm_name, name, team) rows — one row per
    distinct normalised-name + DK-team combination seen across all slates.
    """
    records = []
    for csv_path in sorted(ARCHIVE_DIR.glob("*/DKSalaries.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=["Name", "TeamAbbrev"])
        except Exception as exc:
            print(f"  Warning: could not read {csv_path}: {exc}")
            continue
        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            team = str(row["TeamAbbrev"]).strip()
            if name and team:
                records.append({"name": name, "team": team})

    if not records:
        print("No DKSalaries.csv files found in archive — nothing to do.")
        sys.exit(0)

    df_all = pd.DataFrame(records)
    df_all["norm_name"] = df_all["name"].apply(_normalise)
    return (
        df_all.drop_duplicates(subset=["norm_name", "team"])
        .reset_index(drop=True)
    )


def _fetch_mlb_season(season: int) -> dict[str, dict]:
    """
    Fetch all players for one season from the MLB Stats API.

    Returns {norm_name: {"bats": str, "throws": str, "full_name": str}}.
    """
    url = f"{MLB_API_BASE}/sports/1/players"
    try:
        resp = requests.get(url, params={"season": season, "sportId": 1}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Error fetching season {season}: {exc}")
        return {}

    players = resp.json().get("people", [])
    print(f"  {season}: {len(players)} players returned")

    result: dict[str, dict] = {}
    for p in players:
        full_name  = p.get("fullName", "")
        bat_code   = p.get("batSide",   {}).get("code", "")
        throw_code = p.get("pitchHand", {}).get("code", "")
        if full_name and bat_code and throw_code:
            result[_normalise(full_name)] = {
                "bats":      bat_code,
                "throws":    throw_code,
                "full_name": full_name,
            }
    return result


def _fetch_mlb_players(seasons: list[int], name_aliases: dict[str, str]) -> dict[str, dict]:
    """
    Merge player dicts across seasons; later seasons override earlier ones.

    After building the API-keyed dict, adds DK-name aliases so that lookups by
    normalised DK name (e.g. "kike hernandez") resolve to the correct entry even
    when the API stores the player under a different name ("enrique hernandez").
    """
    combined: dict[str, dict] = {}
    for season in seasons:
        combined.update(_fetch_mlb_season(season))

    for norm_dk, api_raw in name_aliases.items():
        api_norm = _normalise(api_raw)
        if api_norm in combined and norm_dk not in combined:
            combined[norm_dk] = combined[api_norm]

    return combined


def _search_unmatched(
    unmatched: list[tuple[str, str, str]],  # (norm_name, raw_name, team)
    name_aliases: dict[str, str],           # {norm_dk_name: api_raw_name}
    delay: float = 0.15,
) -> dict[str, dict]:
    """
    Second-pass lookup for players not found in the season roster endpoint.

    Uses the MLB Stats API people/search endpoint, which covers IL players,
    minor leaguers, and anyone not on an active 40-man roster at season fetch time.

    For players with a name_map alias (e.g. "kike hernandez" → "Enrique Hernandez"),
    the API name is used as the search query and match target so the lookup succeeds
    even when DK and the MLB API use different names.

    Returns {norm_dk_name: {"bats", "throws", "full_name"}}.
    """
    found: dict[str, dict] = {}
    n = len(unmatched)
    for i, (norm_name, raw_name, _team) in enumerate(unmatched, 1):
        api_raw  = name_aliases.get(norm_name, raw_name)
        api_norm = _normalise(api_raw)
        print(f"  [{i}/{n}] searching: {api_raw}", end="\r", flush=True)
        try:
            resp = requests.get(
                f"{MLB_API_BASE}/people/search",
                params={"names": api_raw, "sportId": 1},
                timeout=15,
            )
            resp.raise_for_status()
            for p in resp.json().get("people", []):
                if _normalise(p.get("fullName", "")) == api_norm:
                    bat   = p.get("batSide",   {}).get("code", "")
                    throw = p.get("pitchHand", {}).get("code", "")
                    if bat and throw:
                        found[norm_name] = {
                            "bats":      bat,
                            "throws":    throw,
                            "full_name": p.get("fullName", raw_name),
                        }
                    break
        except requests.RequestException:
            pass
        time.sleep(delay)
    print()  # clear the \r line
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/handedness.csv from archived DK slates + MLB Stats API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2025, 2026], metavar="YEAR",
        help="MLB seasons to fetch from the Stats API (default: 2025 2026)",
    )
    args = parser.parse_args()

    # ── Collect archive players ───────────────────────────────────────────────
    print("Scanning archive DKSalaries.csv files…")
    archive_df = _collect_archive_players()
    print(f"  {len(archive_df)} unique (player, team) combinations across "
          f"{len(archive_df['norm_name'].unique())} distinct names")

    # Flag name collisions upfront so the operator knows what needs team disambiguation
    name_counts = archive_df["norm_name"].value_counts()
    collisions  = name_counts[name_counts > 1]
    if not collisions.empty:
        print(f"\n  Name collisions (team column required for disambiguation):")
        for norm_name in collisions.index:
            rows = archive_df[archive_df["norm_name"] == norm_name]
            teams = ", ".join(rows["team"].tolist())
            example = rows.iloc[0]["name"]
            print(f"    {example!r} → teams: {teams}")

    # ── Load existing handedness.csv (re-runs are additive) ──────────────────
    existing: set[tuple[str, str]] = set()
    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH, dtype=str)
        existing = set(zip(existing_df["name"], existing_df["team"]))
        print(f"\nExisting {OUTPUT_PATH.name}: {len(existing)} entries — will skip already-resolved players")

    # Only query for players not yet in the file
    new_df = archive_df[
        ~archive_df.apply(lambda r: (r["norm_name"], r["team"]) in existing, axis=1)
    ]
    if new_df.empty:
        print("All archive players already in handedness.csv — nothing to add.")
        return
    print(f"  {len(new_df)} new (player, team) pairs to resolve")

    # ── Fetch from MLB Stats API ──────────────────────────────────────────────
    name_aliases = _load_name_aliases()
    if name_aliases:
        print(f"\nLoaded {len(name_aliases)} name alias(es) from name_map.json")

    print(f"\nFetching player handedness from MLB Stats API (seasons: {args.seasons})…")
    mlb = _fetch_mlb_players(args.seasons, name_aliases)
    print(f"  {len(mlb)} players resolved from API")

    # ── Match and build new rows ──────────────────────────────────────────────
    new_rows = []
    still_unmatched: list[tuple[str, str, str]] = []  # (norm_name, raw_name, team)

    for _, row in new_df.iterrows():
        norm = row["norm_name"]
        info = mlb.get(norm)
        if info:
            new_rows.append({
                "name":   norm,
                "team":   row["team"],
                "bats":   info["bats"],
                "throws": info["throws"],
            })
        else:
            still_unmatched.append((norm, row["name"], row["team"]))

    # ── Second pass: per-name search for IL / minors players ──────────────────
    if still_unmatched:
        print(f"\nSecond pass: searching {len(still_unmatched)} unmatched players individually…")
        found2 = _search_unmatched(still_unmatched, name_aliases)
        print(f"  Resolved {len(found2)} additional players via people/search")

        final_unmatched = []
        for norm_name, raw_name, team in still_unmatched:
            info = found2.get(norm_name)
            if info:
                new_rows.append({
                    "name":   norm_name,
                    "team":   team,
                    "bats":   info["bats"],
                    "throws": info["throws"],
                })
            else:
                final_unmatched.append(f"{raw_name} ({team})")
    else:
        final_unmatched = []

    # ── Merge with existing and write ─────────────────────────────────────────
    if new_rows:
        new_frame = pd.DataFrame(new_rows)
        if OUTPUT_PATH.exists():
            combined = pd.concat(
                [pd.read_csv(OUTPUT_PATH, dtype=str), new_frame],
                ignore_index=True,
            )
        else:
            combined = new_frame
        combined = combined.sort_values(["name", "team"]).reset_index(drop=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"\nWrote {len(combined)} total rows ({len(new_rows)} added) → {OUTPUT_PATH}")
    else:
        print("\nNo new rows to write.")

    # ── Report still-unmatched ────────────────────────────────────────────────
    if final_unmatched:
        print(f"\nStill unmatched after both passes ({len(final_unmatched)}):")
        for u in sorted(final_unmatched):
            print(f"  {u}")
        print(
            "\nFor each, add a row manually to data/handedness.csv:\n"
            "  name (normalised), team (DK abbrev), bats (L/R/S), throws (L/R)"
        )
    else:
        print("All new players matched successfully.")


if __name__ == "__main__":
    main()
