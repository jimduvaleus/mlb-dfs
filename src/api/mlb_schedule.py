"""
MLB schedule fetch & doubleheader detection.

Fetches the day's schedule from the public MLB Stats API and flags any team
playing more than one game that day. Used to gate the Twitter/RotoWire/DFF
"confirmed lineup" auto-lock feature — none of those feeds carry per-game
time data, so a confirmed lineup for a doubleheader team can't be trusted as
belonging to the slate's actual game without a human checking it first.

State file: data/processed/mlb_schedule.json
{
    "date": "2026-06-24",
    "fetched_at": 1750000000.0,
    "games": [
        {"away": "CHC", "home": "NYM", "game_number": 1, "double_header": "S", "game_date": "2026-06-24T17:10:00Z"},
        ...
    ],
    "doubleheader_teams": ["CHC", "NYM"]
}

The cache is keyed to a single date and is replaced wholesale once that date
no longer matches the requested date (mirrors the reset behavior of
projection_metadata.json in projections_meta.py).
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

from .twitter_lineups import TEAM_NAME_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = PROJECT_ROOT / "data" / "processed" / "mlb_schedule.json"

_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

_EMPTY_SCHEDULE: dict = {"date": None, "fetched_at": None, "games": [], "doubleheader_teams": []}


def _team_abbr(name: str) -> Optional[str]:
    """Map a full MLB team name (as returned by the Stats API) to our abbreviation."""
    return TEAM_NAME_MAP.get(name.strip().lower())


def fetch_schedule(date_str: str) -> dict:
    """Fetch and parse the MLB schedule for *date_str* (YYYY-MM-DD).

    Raises requests.RequestException on HTTP failure — callers should treat
    that as "schedule unknown," not "no doubleheader."
    """
    resp = requests.get(_SCHEDULE_URL, params={"sportId": 1, "date": date_str}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games: list[dict] = []
    team_counts: dict[str, int] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            away_name = teams.get("away", {}).get("team", {}).get("name", "")
            home_name = teams.get("home", {}).get("team", {}).get("name", "")
            away = _team_abbr(away_name)
            home = _team_abbr(home_name)
            games.append({
                "away": away,
                "home": home,
                "game_number": g.get("gameNumber"),
                "double_header": g.get("doubleHeader"),
                "game_date": g.get("gameDate"),
            })
            for team in (away, home):
                if team:
                    team_counts[team] = team_counts.get(team, 0) + 1

    doubleheader_teams = sorted(team for team, count in team_counts.items() if count >= 2)
    return {
        "date": date_str,
        "fetched_at": time.time(),
        "games": games,
        "doubleheader_teams": doubleheader_teams,
    }


def load_cached_schedule() -> dict:
    if not SCHEDULE_PATH.exists():
        return dict(_EMPTY_SCHEDULE)
    try:
        with SCHEDULE_PATH.open() as f:
            return json.load(f)
    except Exception:
        return dict(_EMPTY_SCHEDULE)


def save_schedule(data: dict) -> None:
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCHEDULE_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def get_doubleheader_teams_cached(date_str: str) -> tuple[set[str], bool]:
    """Return (doubleheader_teams, is_fresh) for *date_str*.

    Refetches whenever the cache belongs to a different date. Fails open —
    (empty set, False) — on any fetch error, so an MLB API outage doesn't
    block the app's normal auto-lock behavior; it only loses the doubleheader
    check, which callers should surface via the is_fresh flag rather than
    silently treating as "confirmed no doubleheader."
    """
    cached = load_cached_schedule()
    if cached.get("date") == date_str:
        return set(cached.get("doubleheader_teams", [])), True
    try:
        fresh = fetch_schedule(date_str)
    except Exception:
        return set(), False
    save_schedule(fresh)
    return set(fresh["doubleheader_teams"]), True
