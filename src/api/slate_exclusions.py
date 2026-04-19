"""Persist and retrieve game/team/player exclusion state for the current slate.

Exclusions are stored per (slate_id, file_fingerprint) key so that:
- each platform's slate keeps its own independent exclusions
- changing the underlying CSV file (detected via mtime+size fingerprint) resets
  exclusions for that slate automatically
"""
import hashlib
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "slate_exclusions.json"

_EMPTY_ENTRY = {"excluded_teams": [], "excluded_games": [], "excluded_player_ids": []}


def compute_slate_id(games: list[str]) -> str:
    """Return a 16-char hex ID derived from the sorted game list."""
    joined = "|".join(sorted(games))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def compute_file_fingerprint(path: Path | None) -> str:
    """Return a lightweight fingerprint (mtime_ns:size) for a file.

    Returns an empty string if the path is None or does not exist.
    """
    if path is None or not path.exists():
        return ""
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _entry_key(slate_id: str, file_fingerprint: str) -> str:
    return f"{slate_id}:{file_fingerprint}"


def _read_all() -> dict:
    """Load the full exclusions store from disk.

    Returns a plain dict of ``{entry_key: {excluded_teams, excluded_games,
    excluded_player_ids}}``.  Handles the legacy flat-dict format written by
    older versions by discarding it (returns empty dict) so the caller gets
    a clean slate.
    """
    if not EXCLUSIONS_PATH.exists():
        return {}
    with open(EXCLUSIONS_PATH) as f:
        data = json.load(f)
    # Legacy format had a top-level "slate_id" string key — discard it.
    if "slate_id" in data:
        return {}
    return data


def _lookup_entry(slate_id: str, file_fingerprint: str) -> dict:
    """Return the stored exclusion entry for this key, or an empty entry."""
    all_data = _read_all()
    key = _entry_key(slate_id, file_fingerprint)
    return all_data.get(key, dict(_EMPTY_ENTRY))


# ---------------------------------------------------------------------------
# Public read/write helpers
# ---------------------------------------------------------------------------

def read_exclusions(slate_id: str, file_fingerprint: str) -> dict:
    """Load persisted exclusions for a specific slate/file combination.

    Returns a dict with keys ``excluded_teams``, ``excluded_games``,
    ``excluded_player_ids`` (all lists).  Returns empty lists when no data
    exists for this combination (new slate, new file, or first run).
    """
    return _lookup_entry(slate_id, file_fingerprint)


def write_exclusions(
    slate_id: str,
    file_fingerprint: str,
    excluded_teams: list[str],
    excluded_games: list[str],
    excluded_player_ids: list[int] | None = None,
) -> None:
    """Persist exclusions for this slate/file combination."""
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_data = _read_all()
    key = _entry_key(slate_id, file_fingerprint)
    all_data[key] = {
        "excluded_teams": excluded_teams,
        "excluded_games": excluded_games,
        "excluded_player_ids": excluded_player_ids or [],
    }
    with open(EXCLUSIONS_PATH, "w") as f:
        json.dump(all_data, f, indent=2)


def prune_player_exclusions(
    excluded_player_ids: list[int],
    excluded_teams: set[str],
    excluded_games: set[str],
    players: list[dict],
) -> list[int]:
    """Remove player IDs whose team/game is already excluded at team/game level."""
    covered = {p["player_id"] for p in players if p["team"] in excluded_teams or p["game"] in excluded_games}
    return [pid for pid in excluded_player_ids if pid not in covered]


def get_slate_games_with_status(
    games: list[str],
    file_fingerprint: str = "",
) -> tuple[str, list[dict], list[int]]:
    """
    Given the unique game strings from the current slate, compute the slate_id,
    load persisted exclusions (scoped to this slate + file fingerprint), and
    return (slate_id, list_of_game_status_dicts, excluded_player_ids).

    Each game dict has keys: game, away, home, excluded, teams (list of {team, excluded}).
    """
    slate_id = compute_slate_id(games)
    stored = read_exclusions(slate_id, file_fingerprint)

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


def get_slate_players_with_status(
    players_df,
    slate_id: str,
    file_fingerprint: str = "",
) -> list[dict]:
    """
    Given a DataFrame of all slate players (with columns player_id, name, position,
    team, salary, game) and the current slate_id, return a list of player dicts for
    players NOT covered by team/game exclusions, annotated with individual exclusion status.

    Returns: list of {player_id, name, position, team, salary, excluded}
    """
    stored = read_exclusions(slate_id, file_fingerprint)

    excluded_player_ids_set: set[int] = set(stored.get("excluded_player_ids", []))
    excluded_teams_set: set[str] = set(stored.get("excluded_teams", []))
    excluded_games_set: set[str] = set(stored.get("excluded_games", []))

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
