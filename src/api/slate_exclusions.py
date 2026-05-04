"""Persist and retrieve game/team/player exclusion state for the current slate.

Exclusions are stored per (slate_id, file_fingerprint) key so that:
- each platform's slate keeps its own independent exclusions
- changing the underlying CSV file (detected via mtime+size fingerprint) resets
  exclusions for that slate automatically

Each entity (game, team, player) has an exclusion scope:
  "both"       — excluded from candidate generation AND field lineup simulation
  "candidates" — excluded from candidate generation only; still appears in the
                 sim matrix and may appear in opponent field lineups
  "none"       — not excluded (included everywhere)

The "both" scope maps to the legacy `excluded_*` fields for backward compatibility.
"""
import hashlib
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "slate_exclusions.json"

_EMPTY_ENTRY = {
    # "both" scope — existing field names (backward-compat)
    "excluded_teams": [],
    "excluded_games": [],
    "excluded_player_ids": [],
    # "candidates" scope only — new
    "candidate_excluded_teams": [],
    "candidate_excluded_games": [],
    "candidate_excluded_player_ids": [],
    "game_ppd_pcts": {},
}


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
    excluded_player_ids, ...}}``.  Handles the legacy flat-dict format written by
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
    ``excluded_player_ids``, ``candidate_excluded_teams``,
    ``candidate_excluded_games``, ``candidate_excluded_player_ids``
    (all lists).  Returns empty lists when no data exists for this
    combination (new slate, new file, or first run).
    """
    return _lookup_entry(slate_id, file_fingerprint)


def write_exclusions(
    slate_id: str,
    file_fingerprint: str,
    excluded_teams: list[str],
    excluded_games: list[str],
    excluded_player_ids: list[int] | None = None,
    candidate_excluded_teams: list[str] | None = None,
    candidate_excluded_games: list[str] | None = None,
    candidate_excluded_player_ids: list[int] | None = None,
    game_ppd_pcts: dict[str, float] | None = None,
) -> None:
    """Persist exclusions for this slate/file combination."""
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_data = _read_all()
    key = _entry_key(slate_id, file_fingerprint)
    all_data[key] = {
        "excluded_teams": excluded_teams,
        "excluded_games": excluded_games,
        "excluded_player_ids": excluded_player_ids or [],
        "candidate_excluded_teams": candidate_excluded_teams or [],
        "candidate_excluded_games": candidate_excluded_games or [],
        "candidate_excluded_player_ids": candidate_excluded_player_ids or [],
        "game_ppd_pcts": game_ppd_pcts or {},
    }
    with open(EXCLUSIONS_PATH, "w") as f:
        json.dump(all_data, f, indent=2)


def prune_player_exclusions(
    excluded_player_ids: list[int],
    excluded_teams: set[str],
    excluded_games: set[str],
    players: list[dict],
    candidate_excluded_player_ids: list[int] | None = None,
    candidate_excluded_teams: set[str] | None = None,
    candidate_excluded_games: set[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Remove player IDs whose team/game is already excluded at team/game level.

    A player in ``excluded_player_ids`` is redundant when their team/game is
    already in ``excluded_teams`` / ``excluded_games`` ("both" scope covers them).

    A player in ``candidate_excluded_player_ids`` is redundant when their
    team/game is in ``candidate_excluded_*`` OR ``excluded_*`` (either broader
    or same-scope coverage removes the need for individual tracking).

    Returns (pruned_excluded_ids, pruned_candidate_excluded_ids).
    """
    # "both"-scoped pruning: any team/game in excluded_* covers the player
    both_covered = {
        p["player_id"] for p in players
        if p["team"] in excluded_teams or p["game"] in excluded_games
    }
    pruned_both = [pid for pid in excluded_player_ids if pid not in both_covered]

    # "candidates"-scoped pruning: covered by candidate_excluded_* OR excluded_*
    cand_excl_teams = (candidate_excluded_teams or set()) | excluded_teams
    cand_excl_games = (candidate_excluded_games or set()) | excluded_games
    cand_covered = {
        p["player_id"] for p in players
        if p["team"] in cand_excl_teams or p["game"] in cand_excl_games
    }
    pruned_cand = [
        pid for pid in (candidate_excluded_player_ids or [])
        if pid not in cand_covered
    ]

    return pruned_both, pruned_cand


def _effective_game_scope(
    game: str,
    team_away: str,
    team_home: str,
    excluded_games_set: set[str],
    candidate_excluded_games_set: set[str],
    excluded_teams_set: set[str],
    candidate_excluded_teams_set: set[str],
) -> str:
    """Return the effective scope for a game.

    A game's scope is "both" if it's in excluded_games, "candidates" if it's
    in candidate_excluded_games, otherwise the strongest scope of its two teams.
    """
    if game in excluded_games_set:
        return "both"
    if game in candidate_excluded_games_set:
        return "candidates"
    # Check if both teams are at the same scope (game-level promotion)
    scopes = []
    for team in (team_away, team_home):
        if team in excluded_teams_set:
            scopes.append("both")
        elif team in candidate_excluded_teams_set:
            scopes.append("candidates")
        else:
            scopes.append("none")
    if scopes[0] == scopes[1] and scopes[0] != "none":
        return scopes[0]
    return "none"


def _effective_team_scope(
    team: str,
    game_scope: str,
    excluded_teams_set: set[str],
    candidate_excluded_teams_set: set[str],
) -> str:
    """Return the effective scope for a team given its game's scope."""
    if game_scope == "both":
        return "both"
    if team in excluded_teams_set:
        return "both"
    if game_scope == "candidates":
        return "candidates"
    if team in candidate_excluded_teams_set:
        return "candidates"
    return "none"


def get_slate_games_with_status(
    game_times: dict[str, str],
    file_fingerprint: str = "",
) -> tuple[str, list[dict], list[int]]:
    """
    Given a mapping of game_id → ISO start time string from the current slate,
    compute the slate_id, load persisted exclusions (scoped to this slate + file
    fingerprint), and return (slate_id, list_of_game_status_dicts, excluded_player_ids).

    Each game dict has keys: game, away, home, excluded, exclusion_scope,
    game_start_time, teams.
    Games are returned sorted by start time (ascending), then alphabetically.
    """
    games = list(game_times.keys())
    slate_id = compute_slate_id(games)
    stored = read_exclusions(slate_id, file_fingerprint)

    excluded_teams = stored.get("excluded_teams", [])
    excluded_games = stored.get("excluded_games", [])
    excluded_player_ids = stored.get("excluded_player_ids", [])
    candidate_excluded_teams = stored.get("candidate_excluded_teams", [])
    candidate_excluded_games = stored.get("candidate_excluded_games", [])

    excluded_teams_set = set(excluded_teams)
    excluded_games_set = set(excluded_games)
    candidate_excluded_teams_set = set(candidate_excluded_teams)
    candidate_excluded_games_set = set(candidate_excluded_games)

    game_ppd_pcts = stored.get("game_ppd_pcts", {})

    result = []
    for game in sorted(game_times, key=lambda g: (game_times[g] or "", g)):
        parts = game.split("@")
        away = parts[0] if len(parts) == 2 else game
        home = parts[1] if len(parts) == 2 else ""

        game_scope = _effective_game_scope(
            game, away, home,
            excluded_games_set, candidate_excluded_games_set,
            excluded_teams_set, candidate_excluded_teams_set,
        )
        away_scope = _effective_team_scope(away, game_scope, excluded_teams_set, candidate_excluded_teams_set)
        home_scope = _effective_team_scope(home, game_scope, excluded_teams_set, candidate_excluded_teams_set)

        ppd_pct = game_ppd_pcts.get(game)
        result.append(
            {
                "game": game,
                "away": away,
                "home": home,
                "excluded": game_scope != "none",
                "exclusion_scope": game_scope,
                "ppd_pct": ppd_pct,
                "game_start_time": game_times[game] or None,
                "teams": [
                    {"team": away, "excluded": away_scope != "none", "exclusion_scope": away_scope},
                    {"team": home, "excluded": home_scope != "none", "exclusion_scope": home_scope},
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
    team, salary, game) and the current slate_id, return a list of player dicts
    annotated with individual exclusion scope.

    Players whose team/game is "both"-excluded are hidden (they're covered by the
    game/team exclusion and don't need individual player controls).
    Players from "candidates"-scoped teams/games are shown (they may have a
    different individual scope).

    Returns: list of {player_id, name, position, team, salary, excluded,
                       exclusion_scope}
    """
    stored = read_exclusions(slate_id, file_fingerprint)

    excluded_player_ids_set: set[int] = set(stored.get("excluded_player_ids", []))
    excluded_teams_set: set[str] = set(stored.get("excluded_teams", []))
    excluded_games_set: set[str] = set(stored.get("excluded_games", []))
    candidate_excluded_player_ids_set: set[int] = set(stored.get("candidate_excluded_player_ids", []))
    candidate_excluded_teams_set: set[str] = set(stored.get("candidate_excluded_teams", []))
    candidate_excluded_games_set: set[str] = set(stored.get("candidate_excluded_games", []))

    result = []
    for _, row in players_df.iterrows():
        team = str(row["team"])
        game = str(row.get("game", ""))
        pid = int(row["player_id"])

        # Hide players whose team/game is "both"-excluded (covered at higher level)
        if team in excluded_teams_set or game in excluded_games_set:
            continue

        # Determine inherited scope from team/game
        if team in candidate_excluded_teams_set or game in candidate_excluded_games_set:
            inherited_scope = "candidates"
        else:
            inherited_scope = "none"

        # Individual player scope
        if pid in excluded_player_ids_set:
            individual_scope = "both"
        elif pid in candidate_excluded_player_ids_set:
            individual_scope = "candidates"
        else:
            individual_scope = "none"

        # Effective scope = strongest of inherited and individual
        scope_rank = {"none": 0, "candidates": 1, "both": 2}
        if scope_rank[individual_scope] >= scope_rank[inherited_scope]:
            effective_scope = individual_scope
        else:
            effective_scope = inherited_scope

        result.append({
            "player_id": pid,
            "name": str(row.get("name", row["player_id"])),
            "position": str(row["position"]),
            "team": team,
            "salary": int(row["salary"]),
            "excluded": effective_scope != "none",
            "exclusion_scope": effective_scope,
            "individual_scope": individual_scope,
        })
    return result
