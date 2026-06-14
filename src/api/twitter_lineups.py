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
    "athletics": "ATH", "oakland athletics": "ATH", "las vegas athletics": "ATH", "sacramento athletics": "ATH",
    "reds": "CIN", "cincinnati reds": "CIN",
    "brewers": "MIL", "milwaukee brewers": "MIL",
    "cardinals": "STL", "st. louis cardinals": "STL",
    "pirates": "PIT", "pittsburgh pirates": "PIT",
    "rockies": "COL", "colorado rockies": "COL",
    "diamondbacks": "ARI", "arizona diamondbacks": "ARI", "d-backs": "ARI", "dbacks": "ARI",
    "marlins": "MIA", "miami marlins": "MIA",
    "nationals": "WSH", "washington nationals": "WSH",
}

_PITCHER_POSITIONS = {"SP", "RP", "P"}
_BATTER_POSITIONS = {"DH", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "C"}

_SLOT_LINE_RE = re.compile(
    r"^((?:[A-Z]\.\s+)?\S+(?:\s+\S+)*?)\s+(DH|1B|2B|3B|SS|LF|CF|RF|C|SP|RP|P)\s*$"
)
_NOTIF_META_RE = re.compile(r"^(@\w+|·|\d+[mhd]|Underdog\s+MLB)$", re.IGNORECASE)


def looks_like_lineup(body: str) -> bool:
    """Return True if the notification body plausibly contains an Underdog lineup.

    Checks for a recognized team name using the same scan as parse_notification_body.
    """
    return extract_lineup_team(body) is not None


def extract_lineup_team(body: str) -> Optional[str]:
    """Return the team abbreviation if the body contains a recognized team header, else None."""
    for line in body.splitlines():
        line = line.strip()
        if not line or _NOTIF_META_RE.match(line):
            continue
        candidate = re.sub(r"^Updated\s+", "", line, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+\d+/\d+(?:/\d+)?\s*$", "", candidate).strip().lower()
        if candidate in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[candidate]
    return None


def parse_notification_body(body: str) -> tuple[Optional[str], list[dict], bool]:
    """Parse an Underdog MLB notification body into (team_abbrev, slots, is_updated).

    Returns (None, [], False) if the team name cannot be resolved.
    slots is a list of {"slot": int, "name": str, "position": str} for the 9 batters.
    is_updated is True when the team header line started with "Updated".

    Handles notification bodies that may contain Twitter metadata before the
    actual lineup content (e.g. account name, handle, timestamp lines).
    """
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln]

    if not lines:
        return None, [], False

    # Scan all lines for the team header line — it matches "Updated? TeamName Date"
    # Skip lines that look like Twitter metadata (@handle, ·, time stamps, "Underdog MLB" account name)
    team_abbrev: Optional[str] = None
    header_idx: int = -1
    is_updated: bool = False

    for i, line in enumerate(lines):
        if _NOTIF_META_RE.match(line):
            continue
        # Detect "Updated" prefix before stripping
        if re.match(r"^Updated\s+", line, flags=re.IGNORECASE):
            is_updated = True
        # Strip "Updated " prefix, then trailing date
        candidate = re.sub(r"^Updated\s+", "", line, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+\d+/\d+(?:/\d+)?\s*$", "", candidate).strip().lower()
        if candidate in TEAM_NAME_MAP:
            team_abbrev = TEAM_NAME_MAP[candidate]
            header_idx = i
            break

    if team_abbrev is None or header_idx == -1:
        return None, [], False

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

    return team_abbrev, slots, is_updated


def _strip_accents(s: str) -> str:
    """Decompose accented characters and drop the combining marks."""
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def match_player_name(abbreviated: str, candidates: list[dict]) -> list[dict]:
    """Match an abbreviated player name like 'F. Freeman' or bare 'Freeman' against a candidate pool.

    Each candidate dict must have at least: player_id, name, team, position, salary.
    Returns a list of matching candidates (with match_confidence added), ordered best-first.
    """
    abbreviated = abbreviated.strip()
    m = re.match(r"^([A-Z])\.\s+(.+)$", abbreviated)
    if m:
        initial: Optional[str] = m.group(1).upper()
        tokens = m.group(2).strip().split()
    else:
        # Last-name-only format: "Schmitt", "De La Cruz", etc.
        initial = None
        tokens = abbreviated.split()

    if not tokens:
        return []

    # Normalize accents so e.g. "Narváez" == "Narvaez"
    last_hint_norm = _strip_accents(tokens[-1]).lower()

    results: list[dict] = []

    # First pass: exact last-name match (accent-normalized, case-insensitive)
    for c in candidates:
        name_tokens = c["name"].split()
        c_last_norm = _strip_accents(name_tokens[-1]).lower()
        if c_last_norm == last_hint_norm:
            c_first_initial = name_tokens[0][0].upper() if name_tokens else ""
            if initial is None:
                confidence = "exact"  # no initial provided — last name match is best we can do
            else:
                confidence = "exact" if c_first_initial == initial else "fuzzy"
            results.append({**c, "match_confidence": confidence})

    results.sort(key=lambda x: 0 if x["match_confidence"] == "exact" else 1)
    if results:
        # If any exact-initial match exists, drop the cross-initial fuzzy fallbacks.
        exact_only = [r for r in results if r["match_confidence"] == "exact"]
        return exact_only if exact_only else results

    # Second pass: difflib fuzzy on last name only.
    # When an initial is known, enforce it — eliminates cross-initial false positives.
    scored: list[tuple[float, dict]] = []
    for c in candidates:
        name_tokens = c["name"].split()
        if not name_tokens:
            continue
        if initial is not None and name_tokens[0][0].upper() != initial:
            continue
        c_last_norm = _strip_accents(name_tokens[-1]).lower()
        ratio = difflib.SequenceMatcher(None, last_hint_norm, c_last_norm).ratio()
        if ratio >= 0.72:
            scored.append((ratio, c))

    scored.sort(key=lambda x: -x[0])
    for _, c in scored[:3]:
        results.append({**c, "match_confidence": "fuzzy"})

    return results


def load_twitter_lineups(slate_fingerprint: str = "") -> list[dict]:
    """Load confirmed lineups.

    If slate_fingerprint is provided and does not match the fingerprint stored with the
    lineups, the file is cleared and an empty list is returned — the slate file has changed
    since the lineups were confirmed.
    """
    try:
        raw = json.loads(_DATA_PATH.read_text())
    except Exception:
        return []

    # Support old format (plain list) as well as new format ({"slate_fingerprint": ..., "lineups": [...]})
    if isinstance(raw, list):
        lineups = raw
        stored_fp = ""
    else:
        lineups = raw.get("lineups", [])
        stored_fp = raw.get("slate_fingerprint", "")

    # Reset whenever we have a current fingerprint and the stored one doesn't match.
    # Treating stored_fp="" as a mismatch ensures lineups saved without a fingerprint
    # (old format or empty slate path at save time) are always cleared on the next
    # slate change, rather than silently surviving across slates.
    if slate_fingerprint and slate_fingerprint != stored_fp:
        save_twitter_lineups([], slate_fingerprint=slate_fingerprint)
        return []

    return lineups


def save_twitter_lineups(lineups: list[dict], slate_fingerprint: str = "") -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(json.dumps({"slate_fingerprint": slate_fingerprint, "lineups": lineups}, indent=2))


def upsert_twitter_lineup(team: str, notification_id: str, slots: list[dict], slate_fingerprint: str = "", locked: bool = True) -> dict:
    """Save or replace the confirmed lineup for a team. Defaults to locked=True."""
    lineups = load_twitter_lineups(slate_fingerprint)
    lineups = [l for l in lineups if l.get("team") != team]
    record = {
        "team": team,
        "notification_id": notification_id,
        "confirmed_at": time.time(),
        "locked": locked,
        "slots": slots,
    }
    lineups.append(record)
    save_twitter_lineups(lineups, slate_fingerprint)
    return record


def set_twitter_lineup_locked(team: str, locked: bool, slate_fingerprint: str = "") -> bool:
    """Set the locked state on a team's lineup record. Returns True if found."""
    lineups = load_twitter_lineups(slate_fingerprint)
    found = False
    for lineup in lineups:
        if lineup.get("team") == team:
            lineup["locked"] = locked
            found = True
            break
    if found:
        save_twitter_lineups(lineups, slate_fingerprint)
    return found


def delete_twitter_lineup(team: str, slate_fingerprint: str = "") -> bool:
    lineups = load_twitter_lineups(slate_fingerprint)
    filtered = [l for l in lineups if l.get("team") != team]
    if len(filtered) == len(lineups):
        return False
    save_twitter_lineups(filtered, slate_fingerprint)
    return True


def get_twitter_overrides(slate_fingerprint: str = "") -> dict[int, dict]:
    """Return {player_id: {"slot": int}} for all confirmed twitter lineups."""
    overrides: dict[int, dict] = {}
    for lineup in load_twitter_lineups(slate_fingerprint):
        for slot_entry in lineup.get("slots", []):
            pid = slot_entry.get("player_id")
            slot = slot_entry.get("slot")
            if pid is not None and slot is not None:
                overrides[int(pid)] = {"slot": int(slot)}
    return overrides


def get_confirmed_team_lineups(slate_fingerprint: str = "") -> dict[str, dict[int, int]]:
    """Return {team: {player_id: slot}} for all LOCKED teams with a confirmed Twitter lineup.

    Only locked lineups act as canonical batting orders for the pipeline.
    Placeholder slots (player_id=None) are skipped — they have no player to constrain.
    """
    result: dict[str, dict[int, int]] = {}
    for lineup in load_twitter_lineups(slate_fingerprint):
        if not lineup.get("locked", True):  # old records without the key default to locked
            continue
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
