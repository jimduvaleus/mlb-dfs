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
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

# DFF team abbreviation → DK canonical abbreviation.
# Add entries here whenever DFF uses a non-standard abbreviation.
_DFF_TEAM_MAP: dict[str, str] = {
    "OAK": "ATH",   # Athletics (DFF keeps legacy Oakland abbr; DK uses ATH)
    "WAS": "WSH",   # Washington Nationals
}


def _normalise_dff_team(raw: str) -> str:
    """Map a DFF team abbreviation to the DK canonical abbreviation."""
    upper = raw.strip().upper()
    return _DFF_TEAM_MAP.get(upper, upper)


# std_dev estimation — same linear model as fetch_rotowire_projections.py.
# See that file for derivation and calibration notes.
_DK_BATTER_STD_INTERCEPT  = 4.0
_DK_BATTER_STD_SLOPE      = 0.40
_DK_PITCHER_STD_INTERCEPT = 7.2
_DK_PITCHER_STD_SLOPE     = 0.23

_FD_BATTER_STD_INTERCEPT  = 6.0   # 4.0 × 1.5 (FD/DK batter scoring ratio)
_FD_BATTER_STD_SLOPE      = 0.40
_FD_PITCHER_STD_INTERCEPT = 12.6  # 7.2 × 1.75 (FD/DK pitcher scoring ratio)
_FD_PITCHER_STD_SLOPE     = 0.23

# Keep module-level aliases for backward compatibility (DK defaults).
_BATTER_STD_INTERCEPT  = _DK_BATTER_STD_INTERCEPT
_BATTER_STD_SLOPE      = _DK_BATTER_STD_SLOPE
_PITCHER_STD_INTERCEPT = _DK_PITCHER_STD_INTERCEPT
_PITCHER_STD_SLOPE     = _DK_PITCHER_STD_SLOPE


def _estimate_std_dev(mean: float, position: str, platform: str = "draftkings") -> float:
    if platform == "fanduel":
        if position == "P":
            return max(_FD_PITCHER_STD_INTERCEPT + _FD_PITCHER_STD_SLOPE * mean, 1.0)
        return max(_FD_BATTER_STD_INTERCEPT + _FD_BATTER_STD_SLOPE * mean, 1.0)
    if position == "P":
        return max(_DK_PITCHER_STD_INTERCEPT + _DK_PITCHER_STD_SLOPE * mean, 1.0)
    return max(_DK_BATTER_STD_INTERCEPT + _DK_BATTER_STD_SLOPE * mean, 1.0)


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
    dk_lookup: dict[str, list[tuple[int, float, str]]],  # norm_name → [(player_id, salary, team)]
    dff_salary: float | None = None,
    dff_team: str | None = None,
    cutoff: float = 0.82,
) -> int | None:
    """
    Return a DK player_id for *dff_name*, or None if no confident match.

    Disambiguation priority:
      1. Exact name + same team → immediate win.
      2. Exact name + team mismatch (single candidate) → fuzzy-search within
         *dff_team* to catch suffix differences (e.g. "Luis Garcia" vs "Luis Garcia Jr.").
      3. Multiple exact-name candidates → prefer same team, then closest salary.
      4. Fuzzy fallback → prefer same-team result, then closest salary.
    """
    key = _normalise(dff_name)
    all_keys = list(dk_lookup.keys())

    def _pick_from(candidates: list[tuple[int, float, str]]) -> int:
        if len(candidates) == 1:
            return candidates[0][0]
        if dff_team:
            team_matches = [c for c in candidates if c[2].upper() == dff_team.upper()]
            if len(team_matches) == 1:
                return team_matches[0][0]
            if team_matches:
                if dff_salary is not None:
                    return min(team_matches, key=lambda c: abs(c[1] - dff_salary))[0]
                return team_matches[0][0]
        if dff_salary is not None:
            return min(candidates, key=lambda c: abs(c[1] - dff_salary))[0]
        return candidates[0][0]

    if key in dk_lookup:
        candidates = dk_lookup[key]
        if len(candidates) == 1 and dff_team and candidates[0][2]:
            if candidates[0][2].upper() != dff_team.upper():
                fuzzy = difflib.get_close_matches(key, all_keys, n=5, cutoff=cutoff)
                for fkey in fuzzy:
                    if fkey == key:
                        continue
                    for c in dk_lookup[fkey]:
                        if c[2].upper() == dff_team.upper():
                            return c[0]
        return _pick_from(candidates)

    matches = difflib.get_close_matches(key, all_keys, n=3, cutoff=cutoff)
    if not matches:
        return None
    if dff_team:
        for fkey in matches:
            team_matches = [c for c in dk_lookup[fkey] if c[2].upper() == dff_team.upper()]
            if team_matches:
                return _pick_from(team_matches)
        # Fuzzy name match but no team confirmation — reject to avoid cross-slate false positives
        return None
    return _pick_from(dk_lookup[matches[0]])


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


def _playwright_proxy() -> dict | None:
    """Return a Playwright proxy dict from env vars, or None if not set."""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(var)
        if val:
            return {"server": val}
    return None


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


def _select_slate_candidates(
    slate_links: list[dict],
    game_count: int,
) -> list[dict]:
    """Return slate links sorted by preference (best first).

    Prefers slates where game_count >= dk_game_count (potential superset of DK
    games), picking the smallest such superset.  Falls back to closest available
    when no superset exists.  Ties broken by most games.
    """
    supersets = [l for l in slate_links if (l["game_count"] or 0) >= game_count]
    pool = supersets if supersets else slate_links
    return sorted(pool, key=lambda l: (
        abs((l["game_count"] or 0) - game_count),
        -(l["game_count"] or 0),
    ))


async def _navigate_to_slate(
    page,
    target_date: str | None,
    game_count: int,
    slate_teams: set[str] | None = None,
    url_segment: str = DK_URL_SEGMENT,
) -> None:
    """
    Use the projections-links-trigger UI to navigate to the best-matching slate
    for *target_date*.

    Selection strategy:
    1. Prefer slates whose game_count >= dk_game_count (potential superset).
       Among those, pick the smallest.  Fall back to closest if none qualify.
    2. When multiple slates tie on game_count and slate_teams is provided,
       navigate to each in turn, score team overlap, and keep the best.
       This handles days with two same-size slates (e.g. two 6-game slates).
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

    ordered = _select_slate_candidates(slate_links, game_count)
    top_gc = ordered[0]["game_count"] or 0
    tied = [c for c in ordered if (c["game_count"] or 0) == top_gc]

    if len(tied) == 1 or not slate_teams:
        best = tied[0]
        label = best["text"].split("\n")[0]
        log.info("Selecting DFF slate: %s  (%s)", best["href"], label)
        await page.goto(f"{BASE_URL}{best['href']}", wait_until="domcontentloaded")
        await page.wait_for_selector("tr.projections-listing", timeout=20000)
        return

    # Multiple slates with the same game count — pick by team overlap with the DK slate.
    best = tied[0]
    best_overlap = -1
    current_href: str | None = None

    for candidate in tied:
        if current_href != candidate["href"]:
            await page.goto(f"{BASE_URL}{candidate['href']}", wait_until="domcontentloaded")
            await page.wait_for_selector("tr.projections-listing", timeout=20000)
            current_href = candidate["href"]

        raw_teams: list[str] = await page.evaluate(
            """() => Array.from(document.querySelectorAll('tr.projections-listing'))
                        .map(tr => tr.getAttribute('data-team') || '')
                        .filter(t => t !== '')"""
        )
        dff_teams_on_page = {_normalise_dff_team(t) for t in raw_teams}
        overlap = len(dff_teams_on_page & slate_teams)
        label = candidate["text"].split("\n")[0]
        log.info(
            "Slate candidate %s (%s): %d/%d DK teams found.",
            candidate["href"], label, overlap, len(slate_teams),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best = candidate

    if current_href != best["href"]:
        await page.goto(f"{BASE_URL}{best['href']}", wait_until="domcontentloaded")
        await page.wait_for_selector("tr.projections-listing", timeout=20000)

    label = best["text"].split("\n")[0]
    log.info("Selected DFF slate: %s  (%s)  [team overlap %d/%d]", best["href"], label, best_overlap, len(slate_teams))


async def _fetch_rows_playwright(
    target_date: str | None,
    game_count: int,
    slate_override: str | None = None,
    url_segment: str = DK_URL_SEGMENT,
    debug: bool = False,
    slate_teams: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Launch headless Chromium, navigate to the correct DFF slate, and return
    (player_rows, team_rows) where each element is a list of attribute dicts.

    player_rows — all <tr class="projections-listing"> rows (one per player).
    team_rows   — all <tr class="team-listing"> rows (one per team, contains
                  TM PTS implied totals).  Empty list if DFF changes the markup.

    *url_segment* selects the platform — "draftkings" or "fanduel".
    If *slate_override* is given (a DFF slate ID like '235A9'), navigation goes
    directly to /mlb/projections/{url_segment}/{target_date}?slate={slate_override}.

    # TODO: verify FD DFF column structure matches DK when url_segment="fanduel"
    #       (data-pos, data-salary, data-ppg_proj, data-depth_rank, data-starter_flag)
    """
    from playwright.async_api import async_playwright

    proj_base = _projections_base(url_segment)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, proxy=_playwright_proxy())
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
                    await _navigate_to_slate(page, target_date, game_count, slate_teams, url_segment)
                else:
                    # Correct date already loaded — still pick the best sub-slate
                    await _navigate_to_slate(page, target_date, game_count, slate_teams, url_segment)

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
        elif rows:
            # Always log the full attribute key set at INFO level so we can
            # identify TM PTS and other useful fields without --debug.
            log.info("Playwright: player row attribute keys: %s", sorted(rows[0].keys()))

        # TM PTS (data-proj_score) and game O/U (data-ou) are embedded directly on
        # each player row — no separate team rows needed.
        team_rows: list[dict] = []

        await browser.close()

    return rows, team_rows


def fetch_player_rows(
    target_date: str | None,
    game_count: int,
    slate_override: str | None = None,
    url_segment: str = DK_URL_SEGMENT,
    debug: bool = False,
    slate_teams: set[str] | None = None,
) -> list[dict]:
    """Synchronous entry point for Playwright-based player data fetch (player rows only)."""
    player_rows, _ = asyncio.run(
        _fetch_rows_playwright(target_date, game_count, slate_override, url_segment, debug, slate_teams)
    )
    return player_rows


def fetch_player_and_team_rows(
    target_date: str | None,
    game_count: int,
    slate_override: str | None = None,
    url_segment: str = DK_URL_SEGMENT,
    debug: bool = False,
    slate_teams: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Synchronous entry point returning (player_rows, team_rows) from one browser session."""
    return asyncio.run(
        _fetch_rows_playwright(target_date, game_count, slate_override, url_segment, debug, slate_teams)
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

        dff_team = _normalise_dff_team(row.get("data-team") or "")

        def _fval(key: str) -> float | None:
            raw = row.get(key, "")
            try:
                v = float(raw) if raw not in ("", None) else None
                return v if v is not None and v > 0 else None
            except (ValueError, TypeError):
                return None

        records.append(
            {
                "dff_name": name,
                "dff_salary": salary,
                "dff_team": dff_team,
                "position": pos,
                "projected_fpts": pts,
                "lineup_slot": lineup_slot,
                "slot_confirmed": slot_confirmed,
                "proj_score": _fval("data-proj_score"),  # TM PTS (own team for batters, opp for pitchers)
                "game_ou": _fval("data-ou"),              # game over/under total
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
        browser = await pw.chromium.launch(headless=True, proxy=_playwright_proxy())
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
# Archive helpers
# ---------------------------------------------------------------------------

def _date_to_archive_dir(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'MMDDYYYY' for archive directory naming."""
    yr, mo, dy = date_str.split("-")
    return f"{mo}{dy}{yr}"


def _archive_dff_slate(
    target_date: str | None,
    proj_df: pd.DataFrame,
) -> None:
    """Mirror projections and team totals into archive/MMDDYYYY/ for later evaluation.

    proj_df must contain proj_score and game_ou columns (added by parse_players).
    Team totals are derived from batter rows where proj_score == own team's implied total.
    """
    if not target_date:
        return
    try:
        archive_dir = PROJECT_ROOT / "archive" / _date_to_archive_dir(target_date)
        archive_dir.mkdir(parents=True, exist_ok=True)
        _dk_src = PROJECT_ROOT / "data" / "raw" / "DKSalaries.csv"
        _dk_dst = archive_dir / "DKSalaries.csv"
        if _dk_src.exists() and not _dk_dst.exists():
            import shutil as _shutil
            _shutil.copy2(_dk_src, _dk_dst)

        # Archive the full player projection data including proj_score / game_ou.
        archive_cols = ["player_id", "name", "mean", "std_dev", "lineup_slot",
                        "slot_confirmed", "proj_score", "game_ou"]
        proj_df[[c for c in archive_cols if c in proj_df.columns]].to_csv(
            archive_dir / "dff_projections.csv", index=False
        )
        log.info("Archived DFF projections → %s", archive_dir / "dff_projections.csv")

        # Derive team totals from batter rows: for non-pitchers, proj_score is the
        # batter's own team's implied run total.
        if "proj_score" in proj_df.columns and "lineup_slot" in proj_df.columns:
            batters = proj_df[proj_df["lineup_slot"] != 10].copy()
            if not batters.empty and "team" in batters.columns:
                totals = (
                    batters.groupby("team")["proj_score"]
                    .first()
                    .dropna()
                    .reset_index()
                    .rename(columns={"proj_score": "implied_total"})
                )
                if not totals.empty:
                    totals.to_csv(archive_dir / "dff_team_totals.csv", index=False)
                    log.info(
                        "Archived DFF team totals (%d teams) → %s",
                        len(totals), archive_dir / "dff_team_totals.csv",
                    )
    except Exception as exc:
        log.warning("Could not archive DFF projections: %s", exc)


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
    platform: str = "draftkings",
) -> pd.DataFrame:
    # --- Build player lookup from pre-loaded slate DataFrame ----------------
    # slate_df is produced by DraftKingsSlateIngestor or FanDuelSlateIngestor;
    # both expose: player_id (int), name, salary, position.
    log.info("Building player lookup from slate (%d players).", len(slate_df))
    name_lookup: dict[str, list[tuple[int, float, str]]] = {}
    for _, row in slate_df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        team = str(row["team"]).upper() if "team" in slate_df.columns else ""
        name_lookup.setdefault(key, []).append((pid, sal, team))

    pos_map: dict[int, str] = {}
    if "position" in slate_df.columns:
        pos_map = {int(r["player_id"]): str(r["position"]) for _, r in slate_df.iterrows()}

    # Estimate game count from unique teams / 2; keep slate_teams for slate matching
    slate_teams = _slate_df_teams(slate_df)
    game_count = len(slate_teams) // 2

    # --- Fetch DFF player rows via Playwright --------------------------------
    raw_rows, team_rows = fetch_player_and_team_rows(
        target_date=target_date,
        game_count=game_count,
        slate_override=slate_override,
        url_segment=url_segment,
        debug=debug,
        slate_teams=slate_teams,
    )

    proj_df = parse_players(raw_rows)

    if proj_df.empty:
        log.error("No player rows extracted. Try --debug to inspect the page.")
        sys.exit(1)

    log.info("Parsed %d player rows from DFF.", len(proj_df))

    # Discard players whose team is not in the DK slate — guards against the DFF
    # script landing on a wrong/larger slate (e.g. main vs turbo mismatch).
    slate_teams = _slate_df_teams(slate_df)
    if slate_teams:
        before_team_filter = len(proj_df)
        proj_df = proj_df[
            proj_df["dff_team"].str.upper().isin(slate_teams) | (proj_df["dff_team"] == "")
        ].copy()
        discarded = before_team_filter - len(proj_df)
        if discarded:
            log.warning(
                "Discarded %d DFF player row(s) from teams not in the DK slate — "
                "possible slate mismatch. Run with --debug to investigate.",
                discarded,
            )

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
        pid = _match_name(dff_name, name_lookup, dff_salary=row["dff_salary"], dff_team=row.get("dff_team") or None)
        if pid is not None:
            matched.append(
                {
                    "player_id": pid,
                    "name": dff_name,
                    "mean": row["projected_fpts"],
                    "position": pos_map.get(pid, row["position"]),
                    "team": row.get("dff_team", ""),
                    "lineup_slot": row["lineup_slot"],
                    "slot_confirmed": row["slot_confirmed"],
                    "proj_score": row.get("proj_score"),
                    "game_ou": row.get("game_ou"),
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
        lambda r: _estimate_std_dev(float(r["mean"]), str(r["position"]), platform),
        axis=1,
    )

    # --- Write output --------------------------------------------------------
    archive_df = out_df.copy()  # preserve full columns (team, proj_score, game_ou) for archive
    out_cols = ["player_id", "name", "team", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
    out_df = out_df[[c for c in out_cols if c in out_df.columns]].sort_values("mean", ascending=False).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    n_pitchers = int((out_df["lineup_slot"] == 10).sum())
    n_batters  = len(out_df) - n_pitchers
    log.info(
        "Wrote %d starter projections → %s  (pitchers=%d, batters=%d, unmatched=%d)",
        len(out_df), output_path, n_pitchers, n_batters, len(unmatched),
    )

    # --- Auto-archive for historical ownership evaluation --------------------
    _archive_dff_slate(target_date, archive_df)

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
        "--team",
        default=None,
        metavar="ABBREV",
        help="Filter output to a single team abbreviation (e.g. NYY). "
             "Other teams are excluded from the written CSV.",
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
    out_df = build_projections_csv(
        slate_df=slate_df,
        target_date=target_date,
        output_path=args.output,
        url_segment=url_segment,
        name_map=_load_name_map(args.name_map),
        slate_override=args.slate,
        debug=args.debug,
        platform=args.platform,
    )

    # --- Apply team filter (--team flag) ------------------------------------
    if args.team and out_df is not None and "team" in out_df.columns:
        filtered = out_df[out_df["team"].str.upper() == args.team.strip().upper()].copy()
        filtered = filtered.drop(columns=["team"], errors="ignore")
        filtered.to_csv(args.output, index=False)
        log.info("--team %s: wrote %d rows to %s", args.team, len(filtered), args.output)


if __name__ == "__main__":
    main()
