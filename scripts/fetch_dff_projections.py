"""
Fetch MLB player projections from Daily Fantasy Fuel and produce a projections CSV
compatible with main.py.

Flow:
  1. GET /data/slates/recent/mlb/draftkings?date={YYYY-MM-DD}   [--list-slates only]
  2. Playwright headless Chromium navigates to /mlb/projections/draftkings, clicks the
     projections-links-trigger, finds slate links for the target date (e.g.
     "/mlb/projections/draftkings/2026-04-01?slate=235A9"), picks the one whose game
     count best matches the DK salary file, then navigates to it and waits for the
     fully-rendered player rows.
  3. Match players to DK salary file by name (exact → salary-disambiguated fuzzy)
  4. Estimate std_dev from projected pts via position-based linear model

Lineup slot interpretation:
  - Pitchers (data-pos == "P"): lineup_slot=10, slot_confirmed=True
    (pitcher presence in projections is taken as confirmation of starting assignment)
  - Batters: lineup_slot = data-depth_rank (if non-zero), slot_confirmed = (data-starter_flag == "1")
    Batters with depth_rank == 0 have no projected slot and are excluded, matching
    the RotoWire pipeline behaviour.

Output: data/processed/projections.csv
  Columns: player_id, name, mean, std_dev, lineup_slot, slot_confirmed

Usage
-----
    # Auto-detect slate from DK CSV date, write projections:
    python scripts/fetch_dff_projections.py

    # Override slate (skip auto-detection) using the DFF slate ID from --list-slates:
    python scripts/fetch_dff_projections.py --slate 235A9

    # Print available slates for today's DK date and exit:
    python scripts/fetch_dff_projections.py --list-slates

    # Print parsed player rows for debugging:
    python scripts/fetch_dff_projections.py --debug
"""

import argparse
import asyncio
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

BASE_URL = "https://www.dailyfantasyfuel.com"

DK_URL_SEGMENT = "draftkings"
FD_URL_SEGMENT = "fanduel"  # TODO: verify FD DFF column structure matches DK

# DraftKings defaults (unchanged for backward compatibility)
SLATE_LIST_URL  = f"{BASE_URL}/data/slates/recent/mlb/{DK_URL_SEGMENT}"
PROJECTIONS_BASE = f"{BASE_URL}/mlb/projections/{DK_URL_SEGMENT}"

_JSON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# std_dev estimation — same linear model as fetch_rotowire_projections.py
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
    """Load JSON mapping DFF names → DK canonical names. Returns {} if absent."""
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
# HTTP helpers (slate-list JSON endpoint only)
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict | None = None, debug: bool = False) -> dict | list:
    resp = requests.get(url, params=params, headers=_JSON_HEADERS, timeout=15)
    resp.raise_for_status()
    if debug:
        log.debug("GET %s\n%s", resp.url, resp.text[:3000])
    return resp.json()


# ---------------------------------------------------------------------------
# Name normalisation & matching (mirrors fetch_rotowire_projections.py)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase ASCII, strip non-alpha chars, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _match_name(
    dff_name: str,
    dk_lookup: dict[str, list[tuple[int, float]]],
    dff_salary: float | None = None,
    cutoff: float = 0.82,
) -> int | None:
    """
    Return a DK player_id for *dff_name*, or None if no confident match.
    Exact normalised name first; salary-disambiguates duplicates; fuzzy fallback.
    """
    key = _normalise(dff_name)

    if key in dk_lookup:
        candidates = dk_lookup[key]
        if len(candidates) == 1:
            return candidates[0][0]
        if dff_salary is not None:
            return min(candidates, key=lambda c: abs(c[1] - dff_salary))[0]
        return candidates[0][0]

    all_keys = list(dk_lookup.keys())
    matches = difflib.get_close_matches(key, all_keys, n=1, cutoff=cutoff)
    if matches:
        candidates = dk_lookup[matches[0]]
        if dff_salary is not None and len(candidates) > 1:
            return min(candidates, key=lambda c: abs(c[1] - dff_salary))[0]
        return candidates[0][0]

    return None


# ---------------------------------------------------------------------------
# DK salary CSV helpers (mirrors fetch_rotowire_projections.py)
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


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------

def _make_slate_link_re(url_segment: str) -> re.Pattern:
    """Regex that matches DFF slate links for a given platform URL segment."""
    return re.compile(
        rf"/mlb/projections/{re.escape(url_segment)}/(\d{{4}}-\d{{2}}-\d{{2}})\?slate=(\w+)"
    )


def _make_date_link_re(url_segment: str) -> re.Pattern:
    """Regex that matches DFF date-level links for a given platform URL segment."""
    return re.compile(
        rf"/mlb/projections/{re.escape(url_segment)}/(\d{{4}}-\d{{2}}-\d{{2}})/$"
    )


# DraftKings defaults — kept for any callers that import these directly
_SLATE_LINK_RE = _make_slate_link_re(DK_URL_SEGMENT)
_DATE_LINK_RE  = _make_date_link_re(DK_URL_SEGMENT)


def _projections_base(url_segment: str) -> str:
    return f"{BASE_URL}/mlb/projections/{url_segment}"


def _slate_list_url(url_segment: str) -> str:
    return f"{BASE_URL}/data/slates/recent/mlb/{url_segment}"


_FD_DATE_IN_PATH_RE = re.compile(r"FanDuel-MLB-(\d{4}-\d{2}-\d{2})-")


def _extract_date_from_fd_path(path: str) -> str | None:
    """Extract 'YYYY-MM-DD' from a FanDuel salary CSV filename, or None."""
    m = _FD_DATE_IN_PATH_RE.search(Path(path).name)
    return m.group(1) if m else None


def _slate_df_teams(slate_df: pd.DataFrame) -> set[str]:
    """Return all team abbreviations from a pre-loaded slate DataFrame."""
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


def _parse_game_count(text: str) -> int:
    """Extract the leading integer from text like '10 Games' or '2 Games · Turbo'."""
    m = re.match(r"(\d+)\s+games?", text.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else 0


async def _get_trigger_links(page, url_segment: str = DK_URL_SEGMENT) -> list[dict]:
    """
    Click .projections-links-trigger and return all non-showdown slate/date
    links for *url_segment* as a list of dicts.
    """
    await page.click(".projections-links-trigger")
    await page.wait_for_timeout(600)

    raw = await page.evaluate(
        f"""() => Array.from(document.querySelectorAll(
                "a[href*='/mlb/projections/{url_segment}']"
           )).map(a => ({{href: a.getAttribute("href"), text: a.innerText.trim()}}))"""
    )

    slate_link_re = _make_slate_link_re(url_segment)
    date_link_re  = _make_date_link_re(url_segment)

    results = []
    for item in raw:
        href = item["href"] or ""
        text = item["text"]
        m_slate = slate_link_re.search(href)
        m_date  = date_link_re.search(href)
        if m_slate:
            results.append({
                "href": href,
                "text": text,
                "date": m_slate.group(1),
                "slate_id": m_slate.group(2),
                "game_count": _parse_game_count(text.split("\n")[0]),
            })
        elif m_date:
            results.append({
                "href": href,
                "text": text,
                "date": m_date.group(1),
                "slate_id": None,
                "game_count": None,
            })
    return results


async def _navigate_to_slate(
    page,
    target_date: str | None,
    game_count: int,
    url_segment: str = DK_URL_SEGMENT,
) -> None:
    """
    Use the projections-links-trigger UI to navigate to the best-matching slate
    for *target_date*.  Clicks the date link first if slate links for *target_date*
    are not yet visible, then picks by game_count closest to *game_count*.
    """
    links = await _get_trigger_links(page, url_segment=url_segment)

    slate_links = [l for l in links if l["slate_id"] and l["date"] == target_date]

    if not slate_links:
        # Slate links not yet visible for this date — click the date link to load it
        date_link = next((l for l in links if l["date"] == target_date and l["slate_id"] is None), None)
        if date_link:
            log.info("Clicking date link for %s", target_date)
            await page.click(f"a[href='{date_link['href']}']")
            await page.wait_for_selector("tr.projections-listing", timeout=20000)
            links = await _get_trigger_links(page, url_segment=url_segment)
            slate_links = [l for l in links if l["slate_id"] and l["date"] == target_date]
        else:
            log.warning("No date link found for %s; using whatever is loaded.", target_date)
            return

    if not slate_links:
        log.warning("No slate links found for %s after navigation.", target_date)
        return

    # Pick slate with game_count closest to requested count; break ties by most games
    best = min(
        slate_links,
        key=lambda l: (abs((l["game_count"] or 0) - game_count), -(l["game_count"] or 0)),
    )
    label = best["text"].split("\n")[0]
    log.info("Selecting DFF slate: %s  (%s)", best["href"], label)

    await page.goto(f"{BASE_URL}{best['href']}", wait_until="domcontentloaded")
    await page.wait_for_selector("tr.projections-listing", timeout=20000)


async def _fetch_rows_playwright(
    target_date: str | None,
    game_count: int,
    slate_override: str | None = None,
    url_segment: str = DK_URL_SEGMENT,
    debug: bool = False,
) -> list[dict]:
    """
    Launch headless Chromium, navigate to the correct DFF slate, and return all
    <tr class="projections-listing"> rows as attribute dicts.

    *url_segment* selects the platform — "draftkings" or "fanduel".
    If *slate_override* is given (a DFF slate ID like '235A9'), navigation goes
    directly to /mlb/projections/{url_segment}/{target_date}?slate={slate_override}.

    # TODO: verify FD DFF column structure matches DK when url_segment="fanduel"
    #       (data-pos, data-salary, data-ppg_proj, data-depth_rank, data-starter_flag)
    """
    from playwright.async_api import async_playwright

    proj_base = _projections_base(url_segment)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        if slate_override and target_date:
            nav_url = f"{proj_base}/{target_date}?slate={slate_override}"
            log.info("Playwright: navigating directly to %s", nav_url)
            await page.goto(nav_url, wait_until="domcontentloaded")
            await page.wait_for_selector("tr.projections-listing", timeout=20000)
        else:
            log.info("Playwright: loading %s", proj_base)
            await page.goto(proj_base, wait_until="domcontentloaded")
            await page.wait_for_selector("tr.projections-listing", timeout=20000)

            if target_date:
                loaded_date: str | None = await page.evaluate(
                    "() => window.url_start_date || null"
                )
                if loaded_date != target_date:
                    log.info(
                        "Default page shows %s; navigating to %s via UI.",
                        loaded_date, target_date,
                    )
                    await _navigate_to_slate(page, target_date, game_count, url_segment)
                else:
                    # Correct date already loaded — still pick the best sub-slate
                    await _navigate_to_slate(page, target_date, game_count, url_segment)

        # Verify final loaded date
        final_date: str | None = await page.evaluate("() => window.url_start_date || null")
        if final_date and target_date and final_date != target_date:
            log.warning(
                "DFF loaded date %s but salary file is for %s. "
                "Projections may not align with the salary file.",
                final_date, target_date,
            )
        elif final_date:
            log.info("DFF loaded date: %s", final_date)

        # Extract all data-* attributes from every player row
        rows: list[dict] = await page.evaluate(
            """() => Array.from(document.querySelectorAll('tr.projections-listing'))
                        .map(tr => Object.fromEntries(
                            Array.from(tr.attributes).map(a => [a.name, a.value])
                        ))"""
        )

        if debug:
            log.debug("Playwright: extracted %d player rows", len(rows))
            if rows:
                log.debug("Sample row keys: %s", list(rows[0].keys()))

        await browser.close()

    return rows


def fetch_player_rows(
    target_date: str | None,
    game_count: int,
    slate_override: str | None = None,
    url_segment: str = DK_URL_SEGMENT,
    debug: bool = False,
) -> list[dict]:
    """Synchronous entry point for Playwright-based player data fetch."""
    return asyncio.run(
        _fetch_rows_playwright(target_date, game_count, slate_override, url_segment, debug)
    )


# ---------------------------------------------------------------------------
# Player data parse
# ---------------------------------------------------------------------------

def parse_players(rows: list[dict]) -> pd.DataFrame:
    """
    Convert DFF player attribute dicts (from Playwright) into a DataFrame with:
        dff_name, dff_salary, position, projected_fpts, lineup_slot, slot_confirmed
    """
    records = []
    for row in rows:
        name = (row.get("data-name") or "").strip()
        if not name:
            continue

        pos = (row.get("data-pos") or "").strip().upper()

        try:
            salary = float(row["data-salary"]) if row.get("data-salary") else None
        except (ValueError, TypeError):
            salary = None

        pts_raw = row.get("data-ppg_proj", "")
        try:
            val = float(pts_raw) if pts_raw not in ("", None) else None
            pts = val if val is not None and val > 0 else None
        except (ValueError, TypeError):
            pts = None

        depth_raw = row.get("data-depth_rank", "0") or "0"
        starter_flag = row.get("data-starter_flag", "0") or "0"

        if pos == "P":
            lineup_slot: int | None = 10
            slot_confirmed = True
        else:
            try:
                depth = int(float(depth_raw))
                lineup_slot = depth if depth >= 1 else None
            except (ValueError, TypeError):
                lineup_slot = None
            slot_confirmed = starter_flag == "1"

        records.append(
            {
                "dff_name": name,
                "dff_salary": salary,
                "position": pos,
                "projected_fpts": pts,
                "lineup_slot": lineup_slot,
                "slot_confirmed": slot_confirmed,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Slate listing (--list-slates mode, uses Playwright)
# ---------------------------------------------------------------------------

async def _list_slates_playwright(
    target_date: str | None,
    url_segment: str = DK_URL_SEGMENT,
) -> list[dict]:
    """Navigate the DFF page and collect all slate links for *target_date*."""
    from playwright.async_api import async_playwright

    proj_base = _projections_base(url_segment)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(proj_base, wait_until="domcontentloaded")
        await page.wait_for_selector("tr.projections-listing", timeout=20000)

        links = await _get_trigger_links(page, url_segment=url_segment)

        # If no slate links for target_date, click the date link
        slate_links = [l for l in links if l["slate_id"] and l["date"] == target_date]
        if not slate_links and target_date:
            date_link = next((l for l in links if l["date"] == target_date and l["slate_id"] is None), None)
            if date_link:
                await page.click(f"a[href='{date_link['href']}']")
                await page.wait_for_selector("tr.projections-listing", timeout=20000)
                links = await _get_trigger_links(page, url_segment=url_segment)
                slate_links = [l for l in links if l["slate_id"] and l["date"] == target_date]

        await browser.close()

    return slate_links


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_projections_csv(
    slate_df: pd.DataFrame,
    target_date: str | None,
    output_path: str,
    url_segment: str = DK_URL_SEGMENT,
    name_map: dict[str, str] | None = None,
    slate_override: str | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    # --- Build player lookup from pre-loaded slate DataFrame ----------------
    # slate_df is produced by DraftKingsSlateIngestor or FanDuelSlateIngestor;
    # both expose: player_id (int), name, salary, position.
    log.info("Building player lookup from slate (%d players).", len(slate_df))
    name_lookup: dict[str, list[tuple[int, float]]] = {}
    for _, row in slate_df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        name_lookup.setdefault(key, []).append((pid, sal))

    pos_map: dict[int, str] = {}
    if "position" in slate_df.columns:
        pos_map = {int(r["player_id"]): str(r["position"]) for _, r in slate_df.iterrows()}

    # Estimate game count from unique teams / 2
    game_count = len(_slate_df_teams(slate_df)) // 2

    # --- Fetch DFF player rows via Playwright --------------------------------
    raw_rows = fetch_player_rows(
        target_date=target_date,
        game_count=game_count,
        slate_override=slate_override,
        url_segment=url_segment,
        debug=debug,
    )

    proj_df = parse_players(raw_rows)

    if proj_df.empty:
        log.error("No player rows extracted. Try --debug to inspect the page.")
        sys.exit(1)

    log.info("Parsed %d player rows from DFF.", len(proj_df))

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

    # --- Match to platform player IDs ----------------------------------------
    name_map = name_map or {}
    matched, unmatched = [], []
    for _, row in proj_df.iterrows():
        if pd.isna(row["projected_fpts"]):
            continue
        dff_name = name_map.get(row["dff_name"], row["dff_name"])
        if dff_name != row["dff_name"]:
            log.debug("Name map: %r → %r", row["dff_name"], dff_name)
        pid = _match_name(dff_name, name_lookup, dff_salary=row["dff_salary"])
        if pid is not None:
            matched.append(
                {
                    "player_id": pid,
                    "name": dff_name,
                    "mean": row["projected_fpts"],
                    "position": pos_map.get(pid, row["position"]),
                    "lineup_slot": row["lineup_slot"],
                    "slot_confirmed": row["slot_confirmed"],
                }
            )
        else:
            unmatched.append(row["dff_name"])

    if unmatched:
        log.warning(
            "%d DFF player(s) not matched to a DK ID:\n  %s",
            len(unmatched),
            "\n  ".join(unmatched[:30]),
        )

    if not matched:
        log.error("No players matched. Check name formats with --debug.")
        sys.exit(1)

    out_df = pd.DataFrame(matched)

    # --- Filter to projected starters only -----------------------------------
    before_filter = len(out_df)
    out_df = out_df[out_df["lineup_slot"].notna()].copy()
    excluded = before_filter - len(out_df)
    if excluded:
        log.info("Excluded %d non-starter players (no projected lineup slot).", excluded)

    # --- Estimate std_dev ----------------------------------------------------
    out_df["std_dev"] = out_df.apply(
        lambda r: _estimate_std_dev(float(r["mean"]), str(r["position"])),
        axis=1,
    )

    # --- Write output --------------------------------------------------------
    out_cols = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
    out_df = out_df[out_cols].sort_values("mean", ascending=False).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    n_pitchers = int((out_df["lineup_slot"] == 10).sum())
    n_batters  = len(out_df) - n_pitchers
    log.info(
        "Wrote %d starter projections → %s  (pitchers=%d, batters=%d, unmatched=%d)",
        len(out_df), output_path, n_pitchers, n_batters, len(unmatched),
    )
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Daily Fantasy Fuel MLB projections and match to a salary file.",
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
        "--slate",
        metavar="ID",
        default=None,
        help="DFF slate ID to use (from --list-slates), e.g. '235A9'. "
             "Skips auto-detection and navigates directly to that slate.",
    )
    parser.add_argument(
        "--list-slates",
        action="store_true",
        help="Print available DFF slates for the salary file's date and exit",
    )
    parser.add_argument(
        "--name-map",
        default=str(DEFAULT_NAME_MAP_PATH),
        metavar="PATH",
        help="JSON file mapping DFF names to canonical names "
             "(default: data/name_map.json; silently ignored if absent)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print API responses and parsed rows for debugging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Resolve platform ---------------------------------------------------
    if args.platform == "fanduel":
        platform    = Platform.FANDUEL
        url_segment = FD_URL_SEGMENT
        slate_path  = args.fd_slate  # may be "" → auto-discovered by factory
    else:
        platform    = Platform.DRAFTKINGS
        url_segment = DK_URL_SEGMENT
        slate_path  = args.dk_slate

    # --- Load slate DataFrame -----------------------------------------------
    try:
        ingestor  = get_ingestor(platform, slate_path)
        slate_df  = ingestor.get_slate_dataframe()
    except (FileNotFoundError, ValueError) as exc:
        log.error("Could not load slate: %s", exc)
        sys.exit(1)

    # --- Extract target date ------------------------------------------------
    if platform == Platform.FANDUEL:
        fd_path     = getattr(ingestor, "csv_filepath", "") or slate_path
        target_date = _extract_date_from_fd_path(fd_path)
    else:
        target_date = _extract_date_from_dk(slate_path)

    if target_date:
        log.info("Target date: %s", target_date)
    else:
        log.warning("Could not extract date from salary file.")

    # --- List slates mode ---------------------------------------------------
    if args.list_slates:
        slate_links = asyncio.run(
            _list_slates_playwright(target_date, url_segment=url_segment)
        )
        if not slate_links:
            print("No slates found for date %s." % (target_date or "today"))
            return
        print(f"\n{'Slate ID':<12} {'Games':<7} Description")
        print("-" * 50)
        for s in sorted(slate_links, key=lambda l: -(l["game_count"] or 0)):
            label = s["text"].replace("\n", "  ").strip()
            print(f"{s['slate_id']:<12} {s['game_count'] or '?':<7} {label}")
        return

    # --- Build projections CSV -----------------------------------------------
    build_projections_csv(
        slate_df=slate_df,
        target_date=target_date,
        output_path=args.output,
        url_segment=url_segment,
        name_map=_load_name_map(args.name_map),
        slate_override=args.slate,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
