"""
Fetch MLB player projections from RotoWire and produce a projections CSV
compatible with main.py.

API flow (all discovered from the optimizer JS bundle):
  1. GET /daily/mlb/api/slate-list.php?siteID=1
       → list of slates; find the Classic DK slate for target date
  2. GET /daily/mlb/api/players.php?slateID={id}
       → 356-player array with pts, lineup.slot, rotoPos, salary, etc.
  3. Match players to DK salary file by name (exact → salary-disambiguated fuzzy)
  4. Estimate std_dev from projected pts via a position-based ratio

Output: data/processed/projections.csv
  Columns: player_id, name, mean, std_dev, lineup_slot

Usage
-----
    # Auto-detect slate from DK CSV date, write projections:
    python scripts/fetch_rotowire_projections.py

    # Override slate ID (skip auto-detection):
    python scripts/fetch_rotowire_projections.py --slate-id 24060

    # Print available slates and exit:
    python scripts/fetch_rotowire_projections.py --list-slates

    # Print raw API responses for debugging:
    python scripts/fetch_rotowire_projections.py --debug
"""

import argparse
import difflib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from src.platforms.base import Platform
from src.ingestion.factory import get_ingestor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NAME_MAP_PATH = PROJECT_ROOT / "data" / "name_map.json"
_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "projection_metadata.json"

BASE_URL = "https://www.rotowire.com/daily/mlb/api"
SLATE_LIST_URL = f"{BASE_URL}/slate-list.php"
PLAYERS_URL = f"{BASE_URL}/players.php"

DRAFTKINGS_SITE_ID = 1  # confirmed from JS bundle: siteID=1 → DraftKings
FANDUEL_SITE_ID    = 2  # TODO: confirm from RotoWire JS bundle

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.rotowire.com/daily/mlb/optimizer.php",
    "X-Requested-With": "XMLHttpRequest",
}

# std_dev estimation: no source provides this directly, so we estimate via a
# linear model fitted to empirical DK score distributions across player types.
#
# A flat σ/μ ratio systematically over-estimates variance for weak players and
# under-estimates it for strong ones.  A linear model captures the floor effect:
# variance has a baseline component that does not scale with projected output.
#
# Coefficients derived from career DK score distributions for a range of player
# types (6 batters from poor to all-time great; 5 pitchers from average to elite):
#
#   Batters:  std ≈ 4.0 + 0.40 × mean
#     σ/μ ≈ 0.97 at mean= 7,  0.84 at mean=10,  0.76 at mean=13
#     The BatterMixtureMarginal handles zero-inflation at runtime when the PCA
#     model is available; std_dev here seeds that projection.
#
#   Pitchers: std ≈ 7.2 + 0.23 × mean
#     σ/μ ≈ 0.71 at mean=15,  0.59 at mean=20,  0.52 at mean=25
#     Pitcher distributions are approximately Gaussian (no zero-inflation spike);
#     negative DK scores are possible (bad starts) and preserved in simulation.
#
# Note: fit on a small sample of historical players skewed toward greats.
# Long-term improvement: refit coefficients from historical_logs.parquet.
_BATTER_STD_INTERCEPT = 4.0
_BATTER_STD_SLOPE     = 0.40
_PITCHER_STD_INTERCEPT = 7.2
_PITCHER_STD_SLOPE     = 0.23


def _estimate_std_dev(mean: float, position: str) -> float:
    if position == "P":
        return max(_PITCHER_STD_INTERCEPT + _PITCHER_STD_SLOPE * mean, 1.0)
    return max(_BATTER_STD_INTERCEPT + _BATTER_STD_SLOPE * mean, 1.0)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_name_map(path: str | Path | None) -> dict[str, str]:
    """
    Load a JSON file mapping RotoWire names to DK canonical names.

    Example file (data/name_map.json):
        {
            "Enrique Hernandez": "Kike Hernandez"
        }

    Returns an empty dict if the file does not exist or path is None.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        mapping = json.load(f)
    log.info("Loaded %d name mapping(s) from %s", len(mapping), p)
    return mapping


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, debug: bool = False) -> dict | list:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    if debug:
        log.debug("GET %s\n%s", resp.url, resp.text[:3000])
    return resp.json()


# ---------------------------------------------------------------------------
# Name normalisation & matching
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase ASCII, strip non-alpha chars, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _match_name(
    rw_name: str,
    dk_lookup: dict[str, list[tuple[int, float]]],  # norm_name → [(player_id, salary)]
    rw_salary: float | None = None,
    cutoff: float = 0.82,
) -> int | None:
    """
    Return a DK player_id for *rw_name*, or None if no confident match.

    Uses exact normalised name first.  If there are multiple DK players with
    the same normalised name, disambiguates by comparing *rw_salary* to the
    DK salary.  Falls back to difflib fuzzy matching.
    """
    key = _normalise(rw_name)

    if key in dk_lookup:
        candidates = dk_lookup[key]
        if len(candidates) == 1:
            return candidates[0][0]
        # Multiple DK players share the same normalised name — pick closest salary
        if rw_salary is not None:
            return min(candidates, key=lambda c: abs(c[1] - rw_salary))[0]
        return candidates[0][0]

    # Fuzzy fallback
    all_keys = list(dk_lookup.keys())
    matches = difflib.get_close_matches(key, all_keys, n=1, cutoff=cutoff)
    if matches:
        candidates = dk_lookup[matches[0]]
        if rw_salary is not None and len(candidates) > 1:
            return min(candidates, key=lambda c: abs(c[1] - rw_salary))[0]
        return candidates[0][0]

    return None


# ---------------------------------------------------------------------------
# Slate detection
# ---------------------------------------------------------------------------

def _extract_date_from_dk(dk_path: str) -> str | None:
    """Return 'YYYY-MM-DD' parsed from the Game Info column of a DK salary CSV."""
    try:
        df = pd.read_csv(dk_path, usecols=["Game Info"])
        sample = df["Game Info"].dropna().iloc[0]
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", sample)
        if m:
            mo, dy, yr = m.groups()
            return f"{yr}-{mo}-{dy}"
    except Exception:
        pass
    return None


def fetch_slate_list(site_id: int = DRAFTKINGS_SITE_ID, debug: bool = False) -> list[dict]:
    data = _get(SLATE_LIST_URL, params={"siteID": site_id}, debug=debug)
    slates = data.get("slates", [])
    if isinstance(slates, str):
        return []
    return list(slates)


def find_slate(slates: list[dict], target_date: str | None) -> dict | None:
    """
    Return the best matching Classic DK slate for *target_date*.
    Priority: exact date + Classic + defaultSlate → exact date + Classic → defaultSlate.
    """
    def _matches_date(s: dict) -> bool:
        return target_date is not None and target_date in (s.get("startDateOnly") or "")

    def _is_classic(s: dict) -> bool:
        return (s.get("contestType") or "").lower() == "classic"

    # 1. Exact date + Classic + default
    for s in slates:
        if _matches_date(s) and _is_classic(s) and s.get("defaultSlate"):
            return s
    # 2. Exact date + Classic
    for s in slates:
        if _matches_date(s) and _is_classic(s):
            return s
    # 3. Default slate
    for s in slates:
        if s.get("defaultSlate"):
            return s
    # 4. First slate
    return slates[0] if slates else None


def _load_slate_teams_cache() -> dict[str, list[str]]:
    """Return the slate_teams dict from projection_metadata.json, or {} if absent."""
    try:
        with _METADATA_PATH.open() as f:
            return json.load(f).get("slate_teams", {})
    except Exception:
        return {}


def _save_slate_teams_cache(slate_id: str, teams: list[str]) -> None:
    """Persist team list for *slate_id* into projection_metadata.json."""
    meta: dict = {}
    try:
        with _METADATA_PATH.open() as f:
            meta = json.load(f)
    except Exception:
        pass
    meta.setdefault("slate_teams", {})[str(slate_id)] = sorted(teams)
    _METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _METADATA_PATH.open("w") as f:
        json.dump(meta, f, indent=2)


def _dk_teams(dk_path: str) -> set[str]:
    """Return the set of team abbreviations present in a DK salary CSV."""
    teams: set[str] = set()
    try:
        df = pd.read_csv(dk_path, usecols=["Game Info"])
        for raw in df["Game Info"].dropna():
            m = re.match(r"(\w+)@(\w+)", str(raw))
            if m:
                teams.add(m.group(1).upper())
                teams.add(m.group(2).upper())
    except Exception:
        pass
    return teams


def _slate_df_teams(slate_df: pd.DataFrame) -> set[str]:
    """Return all team abbreviations from a pre-loaded slate DataFrame.

    Works for both DK (team column + game column "ARI@PHI") and FD (team +
    opponent columns).
    """
    teams: set[str] = set()
    for col in ("team", "opponent"):
        if col in slate_df.columns:
            teams.update(
                t.upper()
                for t in slate_df[col].dropna().unique()
                if t and str(t).strip()
            )
    if "game" in slate_df.columns:
        for game in slate_df["game"].dropna().unique():
            m = re.match(r"(\w+)@(\w+)", str(game))
            if m:
                teams.add(m.group(1).upper())
                teams.add(m.group(2).upper())
    return teams


_FD_DATE_IN_PATH_RE = re.compile(r"FanDuel-MLB-(\d{4}-\d{2}-\d{2})-")


def _extract_date_from_fd_path(path: str) -> str | None:
    """Extract 'YYYY-MM-DD' from a FanDuel salary CSV filename, or None."""
    m = _FD_DATE_IN_PATH_RE.search(Path(path).name)
    return m.group(1) if m else None


def _teams_from_records(records: list[dict]) -> set[str]:
    """Extract team abbreviations from RotoWire player records.

    The API returns team as a nested object: {"team": {"abbr": "TEX", ...}}
    and opponent as: {"opponent": {"team": "PHI", ...}}.
    """
    teams: set[str] = set()
    for r in records:
        team = r.get("team")
        if isinstance(team, dict):
            abbr = team.get("abbr")
            if abbr and isinstance(abbr, str):
                teams.add(abbr.upper())
        opp = r.get("opponent")
        if isinstance(opp, dict):
            abbr = opp.get("team")
            if abbr and isinstance(abbr, str):
                teams.add(abbr.upper())
    return teams


def find_best_slate(
    slates: list[dict],
    target_date: str | None,
    slate_teams: set[str] | None = None,
    debug: bool = False,
) -> tuple[dict | None, list[dict] | None]:
    """
    Return (slate, records) where *slate* is the Classic slate that best
    matches the game set represented by *slate_teams* and *records* are the
    player records fetched for that slate during scoring (to avoid a redundant
    fetch later).  *records* is None when the team set was served from cache
    or only one candidate existed — the caller should fetch records itself.

    *slate_teams* is a set of team abbreviations extracted from the user's
    salary CSV (via ``_slate_df_teams(slate_df)``).  Works for both DK and FD.

    When multiple Classic slates share the same date (e.g. Early / Afternoon /
    Night / All), fetches player records for each uncached candidate and picks
    the one whose team set most closely matches *slate_teams*.

    Scoring (higher = better):
      primary:   fewest salary-file teams absent from the RW slate
      secondary: fewest extra RW-slate teams not in salary file
                 (avoids "All" over sub-slates)

    Falls back to defaultSlate flag if slate_teams is unavailable or no team
    data can be extracted from the player records.
    """
    def _matches_date(s: dict) -> bool:
        return target_date is not None and target_date in (s.get("startDateOnly") or "")

    def _is_classic(s: dict) -> bool:
        return (s.get("contestType") or "").lower() == "classic"

    candidates = [s for s in slates if _matches_date(s) and _is_classic(s)]
    if not candidates:
        return find_slate(slates, target_date), None
    if len(candidates) == 1:
        return candidates[0], None

    # Multiple candidates on the same date — score by game-set overlap.
    if slate_teams:
        cache = _load_slate_teams_cache()
        # scored entries: (score_primary, score_secondary, slate, fetched_records)
        scored: list[tuple] = []
        for s in candidates:
            sid = str(s["slateID"])
            fetched_records: list[dict] | None = None
            try:
                if sid in cache:
                    rw_teams = set(cache[sid])
                    log.debug("Slate %s (%s): using cached team set", sid, s.get("slateName"))
                else:
                    fetched_records = fetch_players(s["slateID"], debug=debug)
                    rw_teams = _teams_from_records(fetched_records)
                    if rw_teams:
                        _save_slate_teams_cache(sid, list(rw_teams))
                if rw_teams:
                    missing = len(slate_teams - rw_teams)
                    extra   = len(rw_teams - slate_teams)
                    scored.append((-missing, -extra, s, fetched_records))
                    log.info(
                        "Slate %s (%s): %d teams, missing=%d, extra=%d",
                        s["slateID"], s.get("slateName"),
                        len(rw_teams), missing, extra,
                    )
            except Exception as exc:
                log.debug("Could not score slate %s: %s", s.get("slateID"), exc)
        if scored:
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            return scored[0][2], scored[0][3]

    # Fallback: prefer defaultSlate among candidates, then first.
    for s in candidates:
        if s.get("defaultSlate"):
            return s, None
    return candidates[0], None


# ---------------------------------------------------------------------------
# Players fetch & parse
# ---------------------------------------------------------------------------

def fetch_players(slate_id: int | str, debug: bool = False) -> list[dict]:
    data = _get(PLAYERS_URL, params={"slateID": slate_id}, debug=debug)
    if isinstance(data, list):
        return data
    # Occasionally wrapped: {"players": [...]}
    for key in ("players", "data", "results"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _parse_slot(raw: str | None) -> int | None:
    """Convert RotoWire lineup.slot to a copula slot integer (SP → 10, '1'–'9' → int)."""
    if not raw:
        return None
    if str(raw).upper() in ("SP", "P"):
        return 10
    try:
        v = int(float(raw))
        if 1 <= v <= 9:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_players(records: list[dict]) -> pd.DataFrame:
    """
    Flatten RotoWire player records into a DataFrame with columns:
        rw_name, rw_salary, position, projected_fpts, lineup_slot, slot_confirmed
    """
    rows = []
    for r in records:
        first = r.get("firstName", "")
        last = r.get("lastName", "")
        name = f"{first} {last}".strip()
        if not name:
            continue

        pos = r.get("rotoPos") or ""
        salary = r.get("salary")
        try:
            salary = float(salary) if salary is not None else None
        except (ValueError, TypeError):
            salary = None

        pts_raw = r.get("pts")
        try:
            pts = float(pts_raw) if pts_raw not in (None, "", "null") else None
        except (ValueError, TypeError):
            pts = None

        lineup = r.get("lineup") or {}
        slot = _parse_slot(lineup.get("slot"))
        confirmed = bool(lineup.get("isConfirmed"))

        rows.append(
            {
                "rw_name": name,
                "rw_salary": salary,
                "position": pos,
                "projected_fpts": pts,
                "lineup_slot": slot,
                "slot_confirmed": confirmed,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_projections_csv(
    slate_df: pd.DataFrame,
    slate_id: int | str,
    output_path: str,
    site_id: int = DRAFTKINGS_SITE_ID,
    name_map: dict[str, str] | None = None,
    debug: bool = False,
    prefetched_records: list[dict] | None = None,
) -> pd.DataFrame:
    # --- Build player lookup from pre-loaded slate DataFrame ----------------
    # slate_df is produced by DraftKingsSlateIngestor or FanDuelSlateIngestor;
    # both expose the same standardised columns: player_id, name, salary, position.
    log.info("Building player lookup from slate (siteID=%d, %d players).", site_id, len(slate_df))
    name_lookup: dict[str, list[tuple[int, float]]] = {}
    for _, row in slate_df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        name_lookup.setdefault(key, []).append((pid, sal))

    pos_map: dict[int, str] = {}
    if "position" in slate_df.columns:
        pos_map = {int(r["player_id"]): str(r["position"]) for _, r in slate_df.iterrows()}

    # --- Fetch RotoWire players ---------------------------------------------
    if prefetched_records is not None:
        records = prefetched_records
        log.info("Using %d prefetched player records for slate %s.", len(records), slate_id)
    else:
        log.info("Fetching RotoWire players for slate %s…", slate_id)
        records = fetch_players(slate_id, debug=debug)
        if not records:
            log.error(
                "No player records returned for slateID=%s. "
                "Use --list-slates to verify the slate ID.",
                slate_id,
            )
            sys.exit(1)
        log.info("Fetched %d player records.", len(records))

    proj_df = parse_players(records)

    if debug:
        print("\n--- Parsed projections (first 5) ---")
        print(proj_df.head().to_string(index=False))

    # Warn about unconfirmed lineup slots
    unconfirmed = proj_df[
        proj_df["lineup_slot"].notna() & ~proj_df["slot_confirmed"]
    ]
    if not unconfirmed.empty:
        log.warning(
            "%d batters have projected (unconfirmed) lineup slots — "
            "slots may shift closer to game time.",
            len(unconfirmed),
        )

    # --- Match to platform player IDs --------------------------------------
    name_map = name_map or {}
    matched, unmatched = [], []
    for _, row in proj_df.iterrows():
        if row["projected_fpts"] is None:
            continue
        rw_name = name_map.get(row["rw_name"], row["rw_name"])
        if rw_name != row["rw_name"]:
            log.debug("Name map: %r → %r", row["rw_name"], rw_name)
        pid = _match_name(rw_name, name_lookup, rw_salary=row["rw_salary"])
        if pid is not None:
            matched.append(
                {
                    "player_id": pid,
                    "name": rw_name,
                    "mean": row["projected_fpts"],
                    "position": pos_map.get(pid, row["position"]),
                    "lineup_slot": row["lineup_slot"],
                    "slot_confirmed": row["slot_confirmed"],
                }
            )
        else:
            unmatched.append(row["rw_name"])

    if unmatched:
        log.warning(
            "%d RotoWire player(s) not matched to a DK ID:\n  %s",
            len(unmatched),
            "\n  ".join(unmatched[:30]),
        )

    if not matched:
        log.error("No players matched. Check name formats with --debug.")
        sys.exit(1)

    out_df = pd.DataFrame(matched)

    # --- Filter to projected starters only ----------------------------------
    # Only keep batters with a lineup slot (1-9) and starting pitchers (slot 10).
    # Bench players and relief pitchers have lineup_slot=None and must not enter
    # the simulation or optimizer — including them inflates the player pool
    # dramatically and makes optimization intractably slow.
    before_filter = len(out_df)
    out_df = out_df[out_df["lineup_slot"].notna()].copy()
    excluded = before_filter - len(out_df)
    if excluded:
        log.info(
            "Excluded %d non-starter players (no projected lineup slot).",
            excluded,
        )

    # --- Estimate std_dev ---------------------------------------------------
    out_df["std_dev"] = out_df.apply(
        lambda r: _estimate_std_dev(float(r["mean"]), str(r["position"])),
        axis=1,
    )

    # --- Write output -------------------------------------------------------
    out_cols = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
    out_df = out_df[out_cols].sort_values("mean", ascending=False).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    n_pitchers = int((out_df["lineup_slot"] == 10).sum())
    n_batters  = len(out_df) - n_pitchers
    log.info(
        "Wrote %d starter projections → %s  (pitchers=%d, batters=%d, unmatched=%d)",
        len(out_df),
        output_path,
        n_pitchers,
        n_batters,
        len(unmatched),
    )
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch RotoWire MLB projections and match to a salary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--platform",
        choices=["draftkings", "fanduel"],
        default="draftkings",
        help="Platform to fetch projections for (default: draftkings)",
    )
    parser.add_argument(
        "--dk-slate",
        default=str(PROJECT_ROOT / "data" / "raw" / "DKSalaries.csv"),
        metavar="PATH",
        help="DraftKings salary CSV (default: data/raw/DKSalaries.csv)",
    )
    parser.add_argument(
        "--fd-slate",
        default="",
        metavar="PATH",
        help="FanDuel salary CSV (default: auto-discovered in data/raw/)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "projections.csv"),
        metavar="PATH",
        help="Output CSV path (default: data/processed/projections.csv)",
    )
    parser.add_argument(
        "--slate-id",
        metavar="ID",
        default=None,
        help="RotoWire slate ID (skip auto-detection)",
    )
    parser.add_argument(
        "--list-slates",
        action="store_true",
        help="Print available slates for the selected platform and exit",
    )
    parser.add_argument(
        "--name-map",
        default=str(DEFAULT_NAME_MAP_PATH),
        metavar="PATH",
        help="JSON file mapping RotoWire names to canonical names "
             "(default: data/name_map.json; silently ignored if absent)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw API responses for debugging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Resolve platform ---------------------------------------------------
    if args.platform == "fanduel":
        platform = Platform.FANDUEL
        site_id  = FANDUEL_SITE_ID
        slate_path = args.fd_slate  # may be "" → auto-discovered by factory
    else:
        platform = Platform.DRAFTKINGS
        site_id  = DRAFTKINGS_SITE_ID
        slate_path = args.dk_slate

    # --- List slates mode ---------------------------------------------------
    if args.list_slates:
        slates = fetch_slate_list(site_id=site_id, debug=args.debug)
        if not slates:
            print(f"No slates available (siteID={site_id} returned empty list).")
            return
        print(f"\n{'ID':<8} {'Type':<10} {'Date':<12} {'Default':<8} Name")
        print("-" * 55)
        for s in slates:
            print(
                f"{s.get('slateID', '?'):<8} "
                f"{s.get('contestType', '?'):<10} "
                f"{s.get('startDateOnly', '?'):<12} "
                f"{'yes' if s.get('defaultSlate') else '':<8} "
                f"{s.get('slateName', '')}"
            )
        return

    # --- Load slate DataFrame -----------------------------------------------
    try:
        ingestor = get_ingestor(platform, slate_path)
        slate_df = ingestor.get_slate_dataframe()
    except (FileNotFoundError, ValueError) as exc:
        log.error("Could not load slate: %s", exc)
        sys.exit(1)

    # --- Extract target date ------------------------------------------------
    if platform == Platform.FANDUEL:
        fd_path = getattr(ingestor, "csv_filepath", "") or slate_path
        target_date = _extract_date_from_fd_path(fd_path)
    else:
        target_date = _extract_date_from_dk(slate_path)
    if target_date:
        log.info("Target date: %s", target_date)

    # --- Resolve slate ID ---------------------------------------------------
    slate_id = args.slate_id
    if not slate_id:
        slates = fetch_slate_list(site_id=site_id, debug=args.debug)
        if not slates:
            log.error(
                "Slate list is empty (siteID=%d returned no slates). "
                "Try --slate-id to override.",
                site_id,
            )
            sys.exit(1)
        slate_teams = _slate_df_teams(slate_df)
        slate, prefetched_records = find_best_slate(
            slates, target_date, slate_teams=slate_teams, debug=args.debug
        )
        if not slate:
            log.error("No matching slate found. Run --list-slates to see options.")
            sys.exit(1)
        slate_id = slate["slateID"]
        log.info(
            "Using slate: ID=%s  Type=%s  Date=%s  Name=%s",
            slate_id,
            slate.get("contestType"),
            slate.get("startDateOnly"),
            slate.get("slateName"),
        )
    else:
        prefetched_records = None

    # --- Build projections CSV ----------------------------------------------
    build_projections_csv(
        slate_df=slate_df,
        slate_id=slate_id,
        output_path=args.output,
        site_id=site_id,
        name_map=_load_name_map(args.name_map),
        debug=args.debug,
        prefetched_records=prefetched_records,
    )


if __name__ == "__main__":
    main()
