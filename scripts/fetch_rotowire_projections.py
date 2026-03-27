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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NAME_MAP_PATH = PROJECT_ROOT / "data" / "name_map.json"

BASE_URL = "https://www.rotowire.com/daily/mlb/api"
SLATE_LIST_URL = f"{BASE_URL}/slate-list.php"
PLAYERS_URL = f"{BASE_URL}/players.php"

DRAFTKINGS_SITE_ID = 1  # confirmed from JS bundle: siteID=1 → DraftKings

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


def fetch_slate_list(debug: bool = False) -> list[dict]:
    data = _get(SLATE_LIST_URL, params={"siteID": DRAFTKINGS_SITE_ID}, debug=debug)
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
    dk_path: str,
    slate_id: int | str,
    output_path: str,
    name_map: dict[str, str] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    # --- Load DK salary file ------------------------------------------------
    log.info("Loading DK salary file: %s", dk_path)
    dk_df = pd.read_csv(dk_path)
    dk_df.rename(columns={"ID": "player_id", "Name": "name", "Salary": "salary"}, inplace=True)

    # Build normalised-name → [(player_id, salary)] lookup
    dk_lookup: dict[str, list[tuple[int, float]]] = {}
    for _, row in dk_df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        dk_lookup.setdefault(key, []).append((pid, sal))

    dk_pos: dict[int, str] = {}
    if "Position" in dk_df.columns:
        dk_pos = {int(r["player_id"]): str(r["Position"]) for _, r in dk_df.iterrows()}

    # --- Fetch RotoWire players ---------------------------------------------
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

    # --- Match to DK player IDs ---------------------------------------------
    name_map = name_map or {}
    matched, unmatched = [], []
    for _, row in proj_df.iterrows():
        if row["projected_fpts"] is None:
            continue
        rw_name = name_map.get(row["rw_name"], row["rw_name"])
        if rw_name != row["rw_name"]:
            log.debug("Name map: %r → %r", row["rw_name"], rw_name)
        pid = _match_name(rw_name, dk_lookup, rw_salary=row["rw_salary"])
        if pid is not None:
            matched.append(
                {
                    "player_id": pid,
                    "name": rw_name,
                    "mean": row["projected_fpts"],
                    "position": dk_pos.get(pid, row["position"]),
                    "lineup_slot": row["lineup_slot"],
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
    out_cols = ["player_id", "name", "mean", "std_dev", "lineup_slot"]
    out_df = out_df[out_cols].sort_values("mean", ascending=False).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    pitchers = out_df[out_df["player_id"].map(dk_pos).eq("SP")]
    batters = out_df[out_df["player_id"].map(dk_pos).ne("SP")]
    log.info(
        "Wrote %d starter projections → %s  (pitchers=%d, batters=%d, unmatched=%d)",
        len(out_df),
        output_path,
        len(pitchers),
        len(batters),
        len(unmatched),
    )
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch RotoWire MLB projections and match to a DK salary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dk-slate",
        default=str(PROJECT_ROOT / "data" / "raw" / "DKSalaries.csv"),
        metavar="PATH",
        help="DraftKings salary CSV (default: data/raw/DKSalaries.csv)",
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
        help="Print available DraftKings slates and exit",
    )
    parser.add_argument(
        "--name-map",
        default=str(DEFAULT_NAME_MAP_PATH),
        metavar="PATH",
        help="JSON file mapping RotoWire names to DK canonical names "
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

    # --- List slates mode ---------------------------------------------------
    if args.list_slates:
        slates = fetch_slate_list(debug=args.debug)
        if not slates:
            print("No slates available (siteID=1 returned empty list).")
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

    # --- Resolve slate ID ---------------------------------------------------
    slate_id = args.slate_id
    if not slate_id:
        target_date = _extract_date_from_dk(args.dk_slate)
        if target_date:
            log.info("Target date from DK file: %s", target_date)
        slates = fetch_slate_list(debug=args.debug)
        if not slates:
            log.error(
                "Slate list is empty (siteID=1 returned no slates). "
                "Try --slate-id to override."
            )
            sys.exit(1)
        slate = find_slate(slates, target_date)
        if not slate:
            log.error("No matching slate found. Run --list-slates to see options.")
            sys.exit(1)
        slate_id = slate["slateID"]
        log.info(
            "Using slate: ID=%s  Type=%s  Date=%s  Default=%s",
            slate_id,
            slate.get("contestType"),
            slate.get("startDateOnly"),
            slate.get("defaultSlate"),
        )

    # --- Build projections CSV ----------------------------------------------
    build_projections_csv(
        dk_path=args.dk_slate,
        slate_id=slate_id,
        output_path=args.output,
        name_map=_load_name_map(args.name_map),
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
