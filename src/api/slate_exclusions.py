"""Persist and retrieve game/team/player exclusion state for the current slate."""
import hashlib
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "slate_exclusions.json"

_EMPTY = {"slate_id": "", "excluded_teams": [], "excluded_games": [], "excluded_player_ids": []}


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


def write_exclusions(
    slate_id: str,
    excluded_teams: list[str],
    excluded_games: list[str],
    excluded_player_ids: list[int] | None = None,
) -> None:
    """Write exclusions to disk, creating the file if necessary."""
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXCLUSIONS_PATH, "w") as f:
        json.dump(
            {
                "slate_id": slate_id,
                "excluded_teams": excluded_teams,
                "excluded_games": excluded_games,
                "excluded_player_ids": excluded_player_ids or [],
            },
            f,
            indent=2,
        )


def prune_player_exclusions(
    excluded_player_ids: list[int],
    excluded_teams: set[str],
    excluded_games: set[str],
    players: list[dict],
) -> list[int]:
    """Remove player IDs whose team/game is already excluded at team/game level."""
    covered = {p["player_id"] for p in players if p["team"] in excluded_teams or p["game"] in excluded_games}
    return [pid for pid in excluded_player_ids if pid not in covered]


def get_slate_games_with_status(games: list[str]) -> tuple[str, list[dict], list[int]]:
    """
    Given the unique game strings from the current slate, compute the slate_id,
    load persisted exclusions (resetting if the slate has changed), and return
    (slate_id, list_of_game_status_dicts, excluded_player_ids).

    Each game dict has keys: game, away, home, excluded, teams (list of {team, excluded}).
    """
    slate_id = compute_slate_id(games)
    stored = read_exclusions()

    if stored["slate_id"] != slate_id:
        excluded_teams: list[str] = []
        excluded_games: list[str] = []
        excluded_player_ids: list[int] = []
    else:
        excluded_teams = stored.get("excluded_teams", [])
        excluded_games = stored.get("excluded_games", [])
        excluded_player_ids = stored.get("excluded_player_ids", [])

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
    return slate_id, result, excluded_player_ids


def get_slate_players_with_status(players_df, slate_id: str) -> list[dict]:
    """
    Given a DataFrame of all slate players (with columns player_id, name, position,
    team, salary, game) and the current slate_id, return a list of player dicts for
    players NOT covered by team/game exclusions, annotated with individual exclusion status.

    Returns: list of {player_id, name, position, team, salary, excluded}
    """
    stored = read_exclusions()

    if stored.get("slate_id") != slate_id:
        excluded_player_ids_set: set[int] = set()
        excluded_teams_set: set[str] = set()
        excluded_games_set: set[str] = set()
    else:
        excluded_player_ids_set = set(stored.get("excluded_player_ids", []))
        excluded_teams_set = set(stored.get("excluded_teams", []))
        excluded_games_set = set(stored.get("excluded_games", []))

    result = []
    for _, row in players_df.iterrows():
        team = str(row["team"])
        game = str(row.get("game", ""))
        if team in excluded_teams_set or game in excluded_games_set:
            continue
        result.append({
            "player_id": int(row["player_id"]),
            "name": str(row.get("name", row["player_id"])),
            "position": str(row["position"]),
            "team": team,
            "salary": int(row["salary"]),
            "excluded": int(row["player_id"]) in excluded_player_ids_set,
        })
    return result
