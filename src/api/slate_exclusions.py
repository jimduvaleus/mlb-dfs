"""Persist and retrieve game/team exclusion state for the current slate."""
import hashlib
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "slate_exclusions.json"

_EMPTY = {"slate_id": "", "excluded_teams": [], "excluded_games": []}


def compute_slate_id(games: list[str]) -> str:
    """Return a 16-char hex ID derived from the sorted game list."""
    joined = "|".join(sorted(games))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def read_exclusions() -> dict:
    """Load persisted exclusions, or return an empty structure."""
    if not EXCLUSIONS_PATH.exists():
        return dict(_EMPTY)
    with open(EXCLUSIONS_PATH) as f:
        return json.load(f)


def write_exclusions(slate_id: str, excluded_teams: list[str], excluded_games: list[str]) -> None:
    """Write exclusions to disk, creating the file if necessary."""
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXCLUSIONS_PATH, "w") as f:
        json.dump(
            {"slate_id": slate_id, "excluded_teams": excluded_teams, "excluded_games": excluded_games},
            f,
            indent=2,
        )


def get_slate_games_with_status(games: list[str]) -> tuple[str, list[dict]]:
    """
    Given the unique game strings from the current slate, compute the slate_id,
    load persisted exclusions (resetting if the slate has changed), and return
    (slate_id, list_of_game_status_dicts).

    Each game dict has keys: game, away, home, excluded, teams (list of {team, excluded}).
    """
    slate_id = compute_slate_id(games)
    stored = read_exclusions()

    if stored["slate_id"] != slate_id:
        excluded_teams: list[str] = []
        excluded_games: list[str] = []
    else:
        excluded_teams = stored.get("excluded_teams", [])
        excluded_games = stored.get("excluded_games", [])

    excluded_teams_set = set(excluded_teams)
    excluded_games_set = set(excluded_games)

    result = []
    for game in sorted(set(games)):
        parts = game.split("@")
        away = parts[0] if len(parts) == 2 else game
        home = parts[1] if len(parts) == 2 else ""
        game_excluded = game in excluded_games_set
        result.append(
            {
                "game": game,
                "away": away,
                "home": home,
                "excluded": game_excluded,
                "teams": [
                    {"team": away, "excluded": game_excluded or away in excluded_teams_set},
                    {"team": home, "excluded": game_excluded or home in excluded_teams_set},
                ],
            }
        )
    return slate_id, result
