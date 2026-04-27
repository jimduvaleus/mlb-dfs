import json
import re
import time
import difflib
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twitter_lineups.json"

TEAM_NAME_MAP: dict[str, str] = {
    "dodgers": "LAD", "los angeles dodgers": "LAD",
    "yankees": "NYY", "new york yankees": "NYY",
    "red sox": "BOS", "boston red sox": "BOS",
    "cubs": "CHC", "chicago cubs": "CHC",
    "white sox": "CWS", "chicago white sox": "CWS",
    "mets": "NYM", "new york mets": "NYM",
    "giants": "SF", "san francisco giants": "SF",
    "astros": "HOU", "houston astros": "HOU",
    "braves": "ATL", "atlanta braves": "ATL",
    "phillies": "PHI", "philadelphia phillies": "PHI",
    "padres": "SD", "san diego padres": "SD",
    "mariners": "SEA", "seattle mariners": "SEA",
    "rangers": "TEX", "texas rangers": "TEX",
    "tigers": "DET", "detroit tigers": "DET",
    "twins": "MIN", "minnesota twins": "MIN",
    "guardians": "CLE", "cleveland guardians": "CLE",
    "royals": "KC", "kansas city royals": "KC",
    "orioles": "BAL", "baltimore orioles": "BAL",
    "blue jays": "TOR", "toronto blue jays": "TOR",
    "rays": "TB", "tampa bay rays": "TB",
    "angels": "LAA", "los angeles angels": "LAA",
    "athletics": "OAK", "oakland athletics": "OAK",
    "reds": "CIN", "cincinnati reds": "CIN",
    "brewers": "MIL", "milwaukee brewers": "MIL",
    "cardinals": "STL", "st. louis cardinals": "STL",
    "pirates": "PIT", "pittsburgh pirates": "PIT",
    "rockies": "COL", "colorado rockies": "COL",
    "diamondbacks": "ARI", "arizona diamondbacks": "ARI",
    "marlins": "MIA", "miami marlins": "MIA",
    "nationals": "WSH", "washington nationals": "WSH",
}

_PITCHER_POSITIONS = {"SP", "RP", "P"}
_BATTER_POSITIONS = {"DH", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "C"}

_SLOT_LINE_RE = re.compile(
    r"^([A-Z]\.\s+\S+(?:\s+\S+)*?)\s+(DH|1B|2B|3B|SS|LF|CF|RF|C|SP|RP|P)\s*$"
)


def parse_notification_body(body: str) -> tuple[Optional[str], list[dict]]:
    """Parse an Underdog MLB notification body into (team_abbrev, slots).

    Returns (None, []) if the team name cannot be resolved.
    slots is a list of {"slot": int, "name": str, "position": str} for the 9 batters.

    Handles notification bodies that may contain Twitter metadata before the
    actual lineup content (e.g. account name, handle, timestamp lines).
    """
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln]

    if not lines:
        return None, []

    # Scan all lines for the team header line — it matches "Updated? TeamName Date"
    # Skip lines that look like Twitter metadata (@handle, ·, time stamps, "Underdog MLB" account name)
    _SKIP_RE = re.compile(r"^(@\w+|·|\d+[mhd]|Underdog\s+MLB)$", re.IGNORECASE)
    team_abbrev: Optional[str] = None
    header_idx: int = -1

    for i, line in enumerate(lines):
        if _SKIP_RE.match(line):
            continue
        # Strip "Updated " prefix, then trailing date
        candidate = re.sub(r"^Updated\s+", "", line, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+\d+/\d+(?:/\d+)?\s*$", "", candidate).strip().lower()
        if candidate in TEAM_NAME_MAP:
            team_abbrev = TEAM_NAME_MAP[candidate]
            header_idx = i
            break

    if team_abbrev is None or header_idx == -1:
        return None, []

    slots: list[dict] = []
    for line in lines[header_idx + 1:]:
        m = _SLOT_LINE_RE.match(line)
        if not m:
            continue
        name_part, pos = m.group(1).strip(), m.group(2)
        if pos in _PITCHER_POSITIONS:
            continue
        slot_num = len(slots) + 1
        if slot_num > 9:
            break
        slots.append({"slot": slot_num, "name": name_part, "position": pos})

    return team_abbrev, slots


def match_player_name(abbreviated: str, candidates: list[dict]) -> list[dict]:
    """Match an abbreviated player name like 'F. Freeman' against a candidate pool.

    Each candidate dict must have at least: player_id, name, team, position, salary.
    Returns a list of matching candidates (with match_confidence added), ordered best-first.
    """
    # Parse abbreviated: expect "X. LastName" or "X. First Last"
    m = re.match(r"^([A-Z])\.\s+(.+)$", abbreviated.strip())
    if not m:
        return []

    initial = m.group(1).upper()
    # The last token of the abbreviated name is taken as the last name hint
    tokens = m.group(2).strip().split()
    last_hint = tokens[-1].lower()

    results: list[dict] = []

    # First pass: exact last-name match
    for c in candidates:
        name_tokens = c["name"].split()
        c_last = name_tokens[-1].lower()
        if c_last == last_hint:
            c_first_initial = name_tokens[0][0].upper() if name_tokens else ""
            confidence = "exact" if c_first_initial == initial else "fuzzy"
            results.append({**c, "match_confidence": confidence})

    # Sort exact before fuzzy within last-name matches
    results.sort(key=lambda x: 0 if x["match_confidence"] == "exact" else 1)
    if results:
        return results

    # Second pass: difflib fuzzy match on full last name
    def _abbrev(name: str) -> str:
        parts = name.split()
        return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name

    scored: list[tuple[float, dict]] = []
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, abbreviated.lower(), _abbrev(c["name"]).lower()).ratio()
        if ratio >= 0.6:
            scored.append((ratio, c))

    scored.sort(key=lambda x: -x[0])
    for _, c in scored[:3]:
        results.append({**c, "match_confidence": "fuzzy"})

    return results


def load_twitter_lineups() -> list[dict]:
    try:
        return json.loads(_DATA_PATH.read_text())
    except Exception:
        return []


def save_twitter_lineups(lineups: list[dict]) -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(json.dumps(lineups, indent=2))


def upsert_twitter_lineup(team: str, notification_id: str, slots: list[dict]) -> dict:
    """Save or replace the confirmed lineup for a team."""
    lineups = load_twitter_lineups()
    lineups = [l for l in lineups if l.get("team") != team]
    record = {
        "team": team,
        "notification_id": notification_id,
        "confirmed_at": time.time(),
        "slots": slots,
    }
    lineups.append(record)
    save_twitter_lineups(lineups)
    return record


def delete_twitter_lineup(team: str) -> bool:
    lineups = load_twitter_lineups()
    filtered = [l for l in lineups if l.get("team") != team]
    if len(filtered) == len(lineups):
        return False
    save_twitter_lineups(filtered)
    return True


def get_twitter_overrides() -> dict[int, dict]:
    """Return {player_id: {"slot": int}} for all confirmed twitter lineups."""
    overrides: dict[int, dict] = {}
    for lineup in load_twitter_lineups():
        for slot_entry in lineup.get("slots", []):
            pid = slot_entry.get("player_id")
            slot = slot_entry.get("slot")
            if pid is not None and slot is not None:
                overrides[int(pid)] = {"slot": int(slot)}
    return overrides


def get_confirmed_team_lineups() -> dict[str, dict[int, int]]:
    """Return {team: {player_id: slot}} for all teams with a confirmed Twitter lineup.

    Used to determine which batters are authoritative starters (in the map) and
    which are scratched (in the team but not in the map).
    """
    result: dict[str, dict[int, int]] = {}
    for lineup in load_twitter_lineups():
        team = lineup.get("team")
        if not team:
            continue
        pid_to_slot: dict[int, int] = {}
        for slot_entry in lineup.get("slots", []):
            pid = slot_entry.get("player_id")
            slot = slot_entry.get("slot")
            if pid is not None and slot is not None:
                pid_to_slot[int(pid)] = int(slot)
        if pid_to_slot:
            result[team] = pid_to_slot
    return result
