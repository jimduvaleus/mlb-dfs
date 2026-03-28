"""
Projection metadata: per-date slate caching and fetch history.

State file: data/processed/projection_metadata.json
{
    "date": "2026-03-27",
    "slates_fetched_at": 1743123456.789,
    "slates": [
        {"slate_id": "24060", "name": "MLB $...", "is_default": true}
    ],
    "fetches": [
        {
            "timestamp_utc": 1743123456.789,
            "slate_id": "24060",
            "row_count": 145,
            "unconfirmed_count": 23,
            "projections_hash": "abc123..."
        }
    ]
}

The metadata resets entirely when the DK CSV date changes (new day).
"""

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "projection_metadata.json"

_BASE_URL = "https://www.rotowire.com/daily/mlb/api"
_SLATE_LIST_URL = f"{_BASE_URL}/slate-list.php"
_DRAFTKINGS_SITE_ID = 1

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.rotowire.com/daily/mlb/optimizer.php",
    "X-Requested-With": "XMLHttpRequest",
}

_EMPTY_META: dict = {"date": None, "slates_fetched_at": None, "slates": [], "fetches": []}


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return dict(_EMPTY_META)
    try:
        with METADATA_PATH.open() as f:
            return json.load(f)
    except Exception:
        return dict(_EMPTY_META)


def save_metadata(data: dict) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# DK date extraction (mirrors logic in fetch_rotowire_projections.py)
# ---------------------------------------------------------------------------

def _extract_dk_date(dk_path: Path) -> Optional[str]:
    """Return 'YYYY-MM-DD' parsed from the Game Info column of a DK salary CSV."""
    try:
        df = pd.read_csv(str(dk_path), usecols=["Game Info"])
        sample = df["Game Info"].dropna().iloc[0]
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", sample)
        if m:
            mo, dy, yr = m.groups()
            return f"{yr}-{mo}-{dy}"
    except Exception:
        pass
    return None


def _extract_dk_game_set(dk_path: Path) -> list[str]:
    """
    Return a sorted list of 'TEAM@TEAM_YYYY-MM-DD' identifiers from DK CSV.
    Each unique game (teams + date, ignoring time) becomes one entry.
    """
    try:
        df = pd.read_csv(str(dk_path), usecols=["Game Info"])
        games: set[str] = set()
        for raw in df["Game Info"].dropna():
            raw = str(raw)
            # "PHI@NYM 03/29/2026 01:35PM ET"
            m = re.match(r"(\w+@\w+)\s+(\d{2})/(\d{2})/(\d{4})", raw)
            if m:
                teams, mo, dy, yr = m.groups()
                games.add(f"{teams}_{yr}-{mo}-{dy}")
        return sorted(games)
    except Exception:
        return []


def _hash_game_set(games: list[str]) -> str:
    return hashlib.md5("".join(games).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Slate fetching & caching
# ---------------------------------------------------------------------------

def _fetch_slate_list_from_rw() -> list[dict]:
    """Fetch the full slate list from RotoWire (DraftKings, siteID=1)."""
    resp = requests.get(
        _SLATE_LIST_URL,
        params={"siteID": _DRAFTKINGS_SITE_ID},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    slates = data.get("slates", [])
    if isinstance(slates, str):
        return []
    return list(slates)


def _build_slate_options(slates: list[dict], target_date: Optional[str]) -> list[dict]:
    """
    Filter to Classic DraftKings slates matching target_date, and mark the default.
    Returns list of dicts: {"slate_id": str, "name": str, "is_default": bool}.
    """

    def _matches_date(s: dict) -> bool:
        return target_date is not None and target_date in (s.get("startDateOnly") or "")

    def _is_classic(s: dict) -> bool:
        return (s.get("contestType") or "").lower() == "classic"

    candidates = [s for s in slates if _matches_date(s) and _is_classic(s)]
    if not candidates:
        # Fall back to all Classic slates
        candidates = [s for s in slates if _is_classic(s)]
    if not candidates:
        candidates = slates

    # Determine the default: prefer defaultSlate flag, else first
    default_id = None
    for s in candidates:
        if s.get("defaultSlate"):
            default_id = str(s.get("slateID", ""))
            break
    if default_id is None and candidates:
        default_id = str(candidates[0].get("slateID", ""))

    return [
        {
            "slate_id": str(s.get("slateID", "")),
            "name": s.get("slateName") or f"Slate {s.get('slateID', '')}",
            "is_default": str(s.get("slateID", "")) == default_id,
        }
        for s in candidates
    ]


def get_cached_slates(dk_path: Path) -> Optional[list[dict]]:
    """
    Return cached slates for the current DK date, or None if stale / not cached.
    Caller should then fetch from RotoWire and call cache_slates().
    """
    dk_date = _extract_dk_date(dk_path)
    meta = load_metadata()
    if meta.get("date") == dk_date and meta.get("slates_fetched_at") and meta.get("slates"):
        return meta["slates"]
    return None


def fetch_and_cache_slates(dk_path: Path) -> tuple[Optional[str], list[dict]]:
    """
    Fetch slates from RotoWire, cache them keyed to the DK CSV date.
    Returns (dk_date, slate_options).
    Raises requests.RequestException on HTTP failure.
    """
    dk_date = _extract_dk_date(dk_path)
    raw_slates = _fetch_slate_list_from_rw()
    options = _build_slate_options(raw_slates, dk_date)

    meta = load_metadata()
    # Reset date-specific fields if date changed, but preserve slate_teams
    # (game sets are immutable so the cache is valid indefinitely).
    if meta.get("date") != dk_date:
        slate_teams = meta.get("slate_teams", {})
        meta = dict(_EMPTY_META)
        meta["slate_teams"] = slate_teams
    meta["date"] = dk_date
    meta["slates_fetched_at"] = time.time()
    meta["slates"] = options
    save_metadata(meta)

    return dk_date, options


# ---------------------------------------------------------------------------
# Fetch history recording
# ---------------------------------------------------------------------------

def _hash_projections(proj_path: Path) -> Optional[str]:
    """
    Compute an MD5 hash of the projections CSV content (player_id + mean, sorted).
    Returns None if the file cannot be read.
    """
    try:
        df = pd.read_csv(str(proj_path), usecols=["player_id", "mean"])
        df = df.sort_values("player_id").reset_index(drop=True)
        combined = "".join(str(r["player_id"]) + str(r["mean"]) for _, r in df.iterrows())
        return hashlib.md5(combined.encode()).hexdigest()
    except Exception:
        return None


def record_fetch_from_csv(proj_path: Path, slate_id: str, dk_path: Optional[Path] = None) -> None:
    """
    Called after a successful projection fetch. Reads the output CSV, computes
    unconfirmed_count and projections_hash, and appends to the metadata fetches list.
    Also records dk_games_hash from the DK salary file (used for freshness checks).
    """
    row_count = None
    unconfirmed_count = None
    try:
        df = pd.read_csv(str(proj_path))
        row_count = len(df)
        if "slot_confirmed" in df.columns:
            unconfirmed_count = int((~df["slot_confirmed"].astype(bool)).sum())
    except Exception:
        pass

    proj_hash = _hash_projections(proj_path)

    dk_games_hash: Optional[str] = None
    if dk_path is not None:
        games = _extract_dk_game_set(dk_path)
        if games:
            dk_games_hash = _hash_game_set(games)

    entry = {
        "timestamp_utc": time.time(),
        "slate_id": slate_id,
        "row_count": row_count,
        "unconfirmed_count": unconfirmed_count,
        "projections_hash": proj_hash,
        "dk_games_hash": dk_games_hash,
    }

    meta = load_metadata()
    fetches = meta.get("fetches", [])
    fetches.append(entry)
    meta["fetches"] = fetches
    save_metadata(meta)


# ---------------------------------------------------------------------------
# Status field derivation
# ---------------------------------------------------------------------------

def compute_freshness(dk_path: Path, proj_path: Path) -> Optional[bool]:
    """
    Check whether projections.csv is consistent with DKSalaries.csv by comparing
    player IDs directly.  Returns True if ≥90 % of projection player IDs appear in
    the DK salary file, False if not, None if either file cannot be read.

    This catches both "DK file was swapped" and "wrong RotoWire slate was fetched"
    without requiring any stored state.
    """
    try:
        proj_ids = set(pd.read_csv(str(proj_path), usecols=["player_id"])["player_id"])
        dk_ids   = set(pd.read_csv(str(dk_path),   usecols=["ID"])["ID"])
        # A complete projection has at least ~10 starters per game; fewer than 20
        # means the file is broken or nearly empty — treat as stale.
        if len(proj_ids) < 20:
            return False
        overlap = len(proj_ids & dk_ids) / len(proj_ids)
        return overlap >= 0.9
    except Exception:
        return None


def get_status_fields() -> dict:
    """
    Return extra fields for ProjectionsStatus derived from the fetch history.
    Keys: fetch_timestamp_utc, unconfirmed_count, no_changes.
    """
    meta = load_metadata()
    fetches = meta.get("fetches", [])
    if not fetches:
        return {"fetch_timestamp_utc": None, "unconfirmed_count": None, "no_changes": None}

    last = fetches[-1]
    no_changes: Optional[bool] = None
    if len(fetches) >= 2:
        prev = fetches[-2]
        h_last = last.get("projections_hash")
        h_prev = prev.get("projections_hash")
        if h_last is not None and h_prev is not None:
            no_changes = h_last == h_prev

    return {
        "fetch_timestamp_utc": last.get("timestamp_utc"),
        "unconfirmed_count": last.get("unconfirmed_count"),
        "no_changes": no_changes,
    }
