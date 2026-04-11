"""
Fetch MLB player projections from CrazyNinjaOdds market odds and produce a projections CSV
compatible with the main pipeline.

Flow:
  1. Load DK salary CSV to extract slate games (team pairs + times) and player lookup
  2. Playwright navigates to crazyninjaodds.com/site/browse/games.aspx
  3. Games are matched to the slate by league (MLB), date, team pair, and game time (±30 min)
  4. For each matched game, navigate to the game detail page and:
     a. Set Devig Method to Liquidity-Weighted Additive/Shin
     b. For each relevant market, select All players and parse the Fair Odds table
        Batter markets: Player Runs, Player RBIs, Player Singles, Player Doubles,
          Player Triples, Player Home Runs, Player Stolen Bases, Player Walks
        Pitcher markets: Player Outs Recorded, Player Pitching Strikeouts,
          Player Record A Win, Player Hits Allowed, Player Walks Allowed,
          Player Earned Runs Allowed
  5. Convert O/U fair odds to E[X]:
     - Batters: Geometric NB (r=1) closed-form solver — corrects for the heavy
       overdispersion in per-game batter counting stats vs. the Poisson assumption
     - Pitchers: Poisson (scipy brentq solver) — appropriate for outs/strikeouts
  6. Apply DK scoring formula to compute mean; derive std_dev from NB/Poisson variance
  7. Match player names to DK player IDs (normalise + fuzzy match)
  8. Write projections CSV (player_id, name, mean, std_dev)
     lineup_slot is NOT set here — provided by RotoWire during server merge

Table column layout (verified):
  [0] Bet Name  — e.g. "Corbin Carroll Over 0.5" (player + Over/Under + line in one cell)
  [1] Best      — best available market odds (ignored)
  [2] Fair Odds — American odds after Additive/Shin devig, e.g. "-308" or "+276"

Over/Under rows are collected in pairs per (player, line) and the two implied
probabilities are normalized so they sum to 1 before being used as P(Over).

If the table layout changes, adjust _BET_NAME_COL and _FAIR_ODDS_COL below.

Output: data/processed/projections_mo.csv  (or --output path)
  Columns: player_id, name, mean, std_dev

Usage
-----
    python scripts/fetch_market_odds_projections.py
    python scripts/fetch_market_odds_projections.py --debug
    python scripts/fetch_market_odds_projections.py --dk-slate data/raw/DKSalaries.csv
"""

import argparse
import asyncio
import difflib
import json
import logging
import math
import re
import sys
import unicodedata
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson as scipy_poisson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NAME_MAP_PATH = PROJECT_ROOT / "data" / "name_map.json"
GAME_ID_CACHE_PATH    = PROJECT_ROOT / "data" / "processed" / "market_odds_game_ids.json"

GAMES_URL = "https://crazyninjaodds.com/site/browse/games.aspx"
GAME_URL  = "https://crazyninjaodds.com/site/browse/game.aspx?game_id={}"

# Dropdown value for Liquidity-Weighted - Additive/Shin devig method
DEVIG_VALUE = "2"

BATTER_MARKETS = [
    "Player Runs",
    "Player RBIs",
    "Player Singles",
    "Player Doubles",
    "Player Triples",
    "Player Home Runs",
    "Player Stolen Bases",
    "Player Walks",
]

PITCHER_MARKETS = [
    "Player Outs Recorded",
    "Player Pitching Strikeouts",
    "Player Record A Win",
    "Player Hits Allowed",
    "Player Walks Allowed",
    "Player Earned Runs Allowed",
]

# Internal market key names
MK_RUNS    = "runs"
MK_RBIS    = "rbis"
MK_SINGLES = "singles"
MK_DOUBLES = "doubles"
MK_TRIPLES = "triples"
MK_HR      = "home_runs"
MK_SB      = "stolen_bases"
MK_WALKS   = "walks"
MK_OUTS    = "outs_recorded"
MK_K       = "strikeouts"
MK_WIN     = "win"
MK_HA      = "hits_allowed"
MK_BBA     = "walks_allowed"
MK_ER      = "earned_runs_allowed"

MARKET_KEY_MAP: dict[str, str] = {
    "Player Runs":                MK_RUNS,
    "Player RBIs":                MK_RBIS,
    "Player Singles":             MK_SINGLES,
    "Player Doubles":             MK_DOUBLES,
    "Player Triples":             MK_TRIPLES,
    "Player Home Runs":           MK_HR,
    "Player Stolen Bases":        MK_SB,
    "Player Walks":               MK_WALKS,
    "Player Outs Recorded":       MK_OUTS,
    "Player Pitching Strikeouts": MK_K,
    "Player Record A Win":        MK_WIN,
    "Player Hits Allowed":        MK_HA,
    "Player Walks Allowed":       MK_BBA,
    "Player Earned Runs Allowed": MK_ER,
}

# HBP heuristics (league-average constants)
HBP_PER_WALK            = 0.11   # batter: E[HBP] = E[BB] * this
HBP_PER_INNING_PITCHER  = 0.039  # pitcher: E[HBP] = E[IP] * this

# Playwright selector IDs (verified from page inspection)
SEL_DEVIG  = (
    "#ContentPlaceHolderMain_ContentPlaceHolderRight_"
    "WebUserControl_FilterDevigMethod_DropDownListDevigMethod"
)
SEL_MARKET = (
    "#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListMarket"
)
SEL_PLAYER = (
    "#ContentPlaceHolderMain_ContentPlaceHolderRight_DropDownListSubMarket"
)

# Table column indices — adjust if site layout changes (use --debug to inspect)
_BET_NAME_COL  = 0   # "Player Name Over/Under Line" combined cell
_FAIR_ODDS_COL = 2   # American odds after devig (e.g. "-308", "+276")

# MLB team abbreviation → full name
TEAM_FULL_NAME: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",          # Sacramento/Las Vegas Athletics (DK abbr since relocation)
    "OAK": "Oakland Athletics",  # Legacy name kept for fallback matching
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD":  "San Diego Padres",
    "SF":  "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB":  "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

FULL_NAME_TO_ABBR: dict[str, str] = {v: k for k, v in TEAM_FULL_NAME.items()}

# Nickname → abbreviation: last word of each team name (e.g. "Athletics" → "ATH").
# Used as a fallback when the odds site uses a shortened city-less team name.
# Single-word entries (no city) take priority over multi-word entries so that
# "Athletics" (ATH) wins over "Oakland Athletics" (OAK) for the key "athletics".
_NICKNAME_TO_ABBR: dict[str, str] = {}
for _abbr, _full in TEAM_FULL_NAME.items():
    _words = _full.split()
    _nickname = _words[-1].lower()
    if _nickname not in _NICKNAME_TO_ABBR or len(_words) == 1:
        _NICKNAME_TO_ABBR[_nickname] = _abbr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name normalisation & matching (mirrors fetch_dff_projections.py)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _match_name(
    source_name: str,
    dk_lookup: dict[str, list[tuple[int, float]]],
    cutoff: float = 0.82,
) -> int | None:
    key = _normalise(source_name)
    if key in dk_lookup:
        candidates = dk_lookup[key]
        if len(candidates) == 1:
            return candidates[0][0]
        return candidates[0][0]
    all_keys = list(dk_lookup.keys())
    matches = difflib.get_close_matches(key, all_keys, n=1, cutoff=cutoff)
    if matches:
        return dk_lookup[matches[0]][0]
    return None


def _load_name_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DK salary CSV helpers
# ---------------------------------------------------------------------------

def _extract_date_from_dk(dk_path: str) -> str | None:
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


def _extract_game_info_from_dk(dk_path: str) -> dict[tuple[str, str], dtime]:
    """
    Parse DK Game Info column → {(away_abbr, home_abbr): game_time_ET}.
    Example Game Info: "ARI@PHI 04/11/2026 01:05PM ET"
    """
    games: dict[tuple[str, str], dtime] = {}
    try:
        df = pd.read_csv(dk_path, usecols=["Game Info"])
        for raw in df["Game Info"].dropna().unique():
            m = re.match(
                r"(\w+)@(\w+)\s+\d{2}/\d{2}/\d{4}\s+(\d{1,2}):(\d{2})(AM|PM)",
                str(raw).strip(),
            )
            if m:
                away, home, h_str, mn_str, ampm = m.groups()
                h, mn = int(h_str), int(mn_str)
                if ampm == "PM" and h != 12:
                    h += 12
                elif ampm == "AM" and h == 12:
                    h = 0
                games[(away.upper(), home.upper())] = dtime(h, mn)
    except Exception as e:
        log.warning("Could not extract game info from DK file: %s", e)
    return games


# ---------------------------------------------------------------------------
# Game ID cache
# ---------------------------------------------------------------------------

def _game_id_cache_key(slate_date: str, slate_games: dict[tuple[str, str], dtime]) -> str:
    """Stable cache key: date + sorted game matchups."""
    games_str = "|".join(sorted(f"{a}@{h}" for a, h in slate_games.keys()))
    return f"{slate_date}|{games_str}"


def _load_game_id_cache(
    cache_key: str,
) -> dict[tuple[str, str], int] | None:
    """
    Return cached game IDs if the cache key matches the current slate, else None.
    Stored as {away}@{home} string keys; converted back to tuple keys on load.
    """
    if not GAME_ID_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(GAME_ID_CACHE_PATH.read_text())
        if data.get("cache_key") != cache_key:
            return None
        raw: dict[str, int] = data["game_ids"]
        result: dict[tuple[str, str], int] = {}
        for matchup, gid in raw.items():
            away, home = matchup.split("@", 1)
            result[(away, home)] = gid
        log.info("Using cached game IDs for slate %s", cache_key)
        return result
    except Exception as e:
        log.debug("Could not read game ID cache: %s", e)
        return None


def _save_game_id_cache(
    cache_key: str,
    game_id_map: dict[tuple[str, str], int],
) -> None:
    """Persist game IDs keyed by the current slate fingerprint."""
    try:
        GAME_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cache_key": cache_key,
            "game_ids": {f"{a}@{h}": gid for (a, h), gid in game_id_map.items()},
        }
        GAME_ID_CACHE_PATH.write_text(json.dumps(data, indent=2))
        log.debug("Saved game ID cache: %s", GAME_ID_CACHE_PATH)
    except Exception as e:
        log.warning("Could not save game ID cache: %s", e)


# ---------------------------------------------------------------------------
# Poisson O/U → E[X] conversion
# ---------------------------------------------------------------------------

def _poisson_lambda(line: float, p_over: float, tol: float = 1e-9) -> float | None:
    """
    Find Poisson λ such that P(X > line) = p_over.
    For a half-point line (e.g. 1.5), P(X > 1.5) = P(X >= 2) = 1 - Poisson.cdf(1, λ).
    Uses scipy.optimize.brentq.
    """
    p_over = max(tol, min(1.0 - tol, p_over))
    k = math.floor(line)   # P(X > line) = 1 - P(X <= k) for half-point lines

    def objective(lam: float) -> float:
        return (1.0 - float(scipy_poisson.cdf(k, lam))) - p_over

    lo, hi = 1e-6, 60.0
    try:
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            return None
        return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return None


def _fit_lambda(lines_probs: list[tuple[float, float]]) -> float | None:
    """
    Given multiple (line, p_over) tuples, return a weighted-average λ estimate.
    Weight = |p_over - 0.5| so sharper / more informative lines count more.
    Used for pitcher stats, which are well-approximated by Poisson.
    """
    estimates, weights = [], []
    for line, p_over in lines_probs:
        lam = _poisson_lambda(line, p_over)
        if lam is not None:
            estimates.append(lam)
            weights.append(max(abs(p_over - 0.5), 0.01))
    if not estimates:
        return None
    total_w = sum(weights)
    return sum(e * w for e, w in zip(estimates, weights)) / total_w


# ---------------------------------------------------------------------------
# Geometric NB O/U → E[X] conversion (for batter stats)
# ---------------------------------------------------------------------------

def _nb_mean_geometric(line: float, p_over: float, tol: float = 1e-9) -> float | None:
    """
    Find the mean μ of a Geometric / NB(r=1) distribution such that P(X > line) = p_over.

    Batter per-game counting stats (singles, HR, runs, etc.) are far more
    overdispersed than Poisson assumes.  The geometric distribution (NB r=1)
    is better motivated: it models a "first-success" process where the number
    of events in a game is 0 or 1 most of the time, with a long right tail.

    Closed-form solution:
        P(X > line) = P(X >= k+1) = (μ/(1+μ))^(k+1)  where k = floor(line)
        → q = p_over^(1/(k+1)),  μ = q / (1 - q)
    """
    p_over = max(tol, min(1.0 - tol, p_over))
    k = math.floor(line)
    try:
        q = p_over ** (1.0 / (k + 1))
        if q >= 1.0 - tol:
            return None
        return q / (1.0 - q)
    except (ValueError, ZeroDivisionError):
        return None


def _fit_nb_mean(lines_probs: list[tuple[float, float]]) -> float | None:
    """
    Given multiple (line, p_over) tuples, return a weighted-average geometric-NB
    mean estimate.  Weight = |p_over - 0.5| so sharper lines count more.
    Used for batter stats in place of _fit_lambda.
    """
    estimates, weights = [], []
    for line, p_over in lines_probs:
        mu = _nb_mean_geometric(line, p_over)
        if mu is not None:
            estimates.append(mu)
            weights.append(max(abs(p_over - 0.5), 0.01))
    if not estimates:
        return None
    total_w = sum(weights)
    return sum(e * w for e, w in zip(estimates, weights)) / total_w


# ---------------------------------------------------------------------------
# Fair odds → probability
# ---------------------------------------------------------------------------

def _parse_fair_odds(odds_str: str) -> float | None:
    """
    Convert a Fair Odds cell value to an implied probability in [0, 1].
    Handles: American odds (+110, -130), decimal odds (2.10), direct probability.
    """
    try:
        s = odds_str.strip().replace(",", "").replace(" ", "")
        if not s:
            return None
        if s.startswith("+") or (s.startswith("-") and not s[1:].startswith("0.")):
            # American odds
            odds = float(s)
            if odds < 0:
                return abs(odds) / (abs(odds) + 100.0)
            else:
                return 100.0 / (odds + 100.0)
        val = float(s)
        if 0.0 < val <= 1.0:
            return val        # Already a probability
        if 1.0 < val < 100.0:
            return 1.0 / val  # Decimal odds
        if val >= 100.0:
            return 100.0 / (val + 100.0)  # Positive American odds without sign
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Bet Name parsing
# ---------------------------------------------------------------------------

_BET_NAME_RE = re.compile(
    r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s*$", re.IGNORECASE
)
_BET_NAME_YES_NO_RE = re.compile(
    r"^(.+?)\s+(Yes|No)\s*$", re.IGNORECASE
)


def _parse_bet_name(bet_name: str) -> tuple[str, str, float] | None:
    """
    Parse a Bet Name cell into (player_name, outcome, line).
    Handles two formats:
      - "Corbin Carroll Over 0.5"  → ('Corbin Carroll', 'over', 0.5)
      - "Brandon Pfaadt Yes"       → ('Brandon Pfaadt', 'over', 0.5)
      - "Brandon Pfaadt No"        → ('Brandon Pfaadt', 'under', 0.5)
    The Yes/No form is used by binary markets (e.g. Player Record A Win).
    'Yes' maps to 'over' and 'No' maps to 'under' with an implicit line of 0.5.
    """
    m = _BET_NAME_RE.match(bet_name.strip())
    if m:
        return m.group(1).strip(), m.group(2).lower(), float(m.group(3))
    m2 = _BET_NAME_YES_NO_RE.match(bet_name.strip())
    if m2:
        outcome = "over" if m2.group(2).lower() == "yes" else "under"
        return m2.group(1).strip(), outcome, 0.5
    return None


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def _validate_table_columns(rows: list[list[str]], market_name: str) -> None:
    """
    Sanity-check that _BET_NAME_COL and _FAIR_ODDS_COL point at the right data.
    Raises RuntimeError with a diagnostic message if validation fails.

    Checks (on a sample of up to 5 data rows):
      _BET_NAME_COL  — parseable by _parse_bet_name ("Name Over/Under Line")
      _FAIR_ODDS_COL — parseable by _parse_fair_odds (American odds e.g. "-308")
    """
    min_cols = _FAIR_ODDS_COL + 1
    data_rows = [r for r in rows if len(r) >= min_cols and any(c.strip() for c in r)]

    if not data_rows:
        raise RuntimeError(
            f"Column validation ({market_name!r}): table loaded but no rows with "
            f">= {min_cols} cells found. The table structure may have changed."
        )

    sample = data_rows[:5]
    bet_name_bad: list[str] = []
    odds_bad: list[str] = []

    for row in sample:
        if _parse_bet_name(row[_BET_NAME_COL]) is None:
            bet_name_bad.append(repr(row[_BET_NAME_COL]))
        odds_cell = row[_FAIR_ODDS_COL]
        # Rows with ⚠️ are intentionally skipped by the parser — don't count as bad.
        if "\u26a0" not in odds_cell and _parse_fair_odds(odds_cell) is None:
            odds_bad.append(repr(odds_cell))

    threshold = max(1, len(sample) // 2 + 1)
    issues: list[str] = []
    if len(bet_name_bad) >= threshold:
        issues.append(
            f"  _BET_NAME_COL={_BET_NAME_COL}: expected 'Name Over/Under Line', "
            f"found {bet_name_bad}"
        )
    if len(odds_bad) >= threshold:
        issues.append(
            f"  _FAIR_ODDS_COL={_FAIR_ODDS_COL}: expected American odds (e.g. '-308'), "
            f"found {odds_bad}"
        )

    if not issues:
        return

    row_dump = "\n".join(
        "  row {}: {}".format(
            i,
            "  ".join(f"[{j}]={repr(c)}" for j, c in enumerate(row[:min_cols + 2]))
        )
        for i, row in enumerate(data_rows[:3])
    )
    raise RuntimeError(
        f"Table column layout mismatch in market {market_name!r}:\n"
        + "\n".join(issues)
        + f"\n\nFirst {min(3, len(data_rows))} data rows:\n{row_dump}\n"
        + "Adjust _BET_NAME_COL and _FAIR_ODDS_COL constants at the top of this script."
    )


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------

async def _wait_for_ajax(page, timeout: int = 15_000) -> None:
    """Wait for ASP.NET AJAX postback to complete after a dropdown change."""
    # Give the onChange handler 150 ms to fire and disable the market dropdown.
    await page.wait_for_timeout(150)
    # Then wait for the market dropdown to be re-enabled (AJAX complete).
    try:
        await page.wait_for_function(
            f"""() => {{
                const el = document.querySelector('{SEL_MARKET}');
                return el && !el.disabled;
            }}""",
            timeout=timeout,
        )
    except Exception:
        # Fallback: fixed delay if function-wait fails
        await page.wait_for_timeout(2500)


async def _fetch_game_odds(
    page,
    game_id: int,
    markets: list[str],
    debug: bool = False,
    validate_columns: bool = False,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """
    Navigate to a game page, set the devig method, and scrape all markets.

    If validate_columns=True, the column layout is sanity-checked on the first
    non-empty table result and a RuntimeError is raised if it looks wrong.

    Returns:
        {player_name: {market_key: [(line, p_over), ...]}}
    """
    url = GAME_URL.format(game_id)
    log.info("Fetching game odds: %s", url)
    await page.goto(url, wait_until="domcontentloaded")

    # Wait for the initial AJAX load (timer fires automatically on page load)
    await page.wait_for_timeout(2000)
    try:
        await page.wait_for_selector('[id*="GridView1"] tr', timeout=20_000)
    except Exception:
        log.warning("Game page table did not load for game_id=%d", game_id)
        return {}

    # Set devig method: Liquidity-Weighted - Additive/Shin
    try:
        await page.select_option(SEL_DEVIG, value=DEVIG_VALUE)
        await _wait_for_ajax(page)
    except Exception as e:
        log.warning("Could not set devig method for game %d: %s", game_id, e)

    market_data: dict[str, dict[str, list[tuple[float, float]]]] = {}
    _validated = False

    for market_name in markets:
        market_key = MARKET_KEY_MAP.get(market_name)
        if not market_key:
            continue

        # Select market by display text
        try:
            await page.select_option(SEL_MARKET, label=market_name, timeout=5000)
        except Exception as e:
            log.debug("Market '%s' not found for game %d: %s", market_name, game_id, e)
            continue
        await _wait_for_ajax(page)

        # Select All players (value="0")
        try:
            await page.select_option(SEL_PLAYER, value="0")
            await _wait_for_ajax(page)
        except Exception as e:
            log.debug("Could not select All players for '%s': %s", market_name, e)

        # Parse the GridView1 table
        rows: list[list[str]] = await page.evaluate(
            """() => {
                const table = document.querySelector('[id*="GridView1"]');
                if (!table) return [];
                const out = [];
                for (const tr of table.querySelectorAll('tr')) {
                    const cells = Array.from(tr.querySelectorAll('td'))
                                       .map(c => c.innerText.trim());
                    if (cells.length > 0) out.push(cells);
                }
                return out;
            }"""
        )

        if debug and rows:
            log.debug("Market '%s' raw rows (first 4): %s", market_name, rows[:4])

        # Validate column layout once on the first non-empty table result
        if validate_columns and not _validated and rows:
            _validate_table_columns(rows, market_name)  # raises RuntimeError on bad layout
            _validated = True

        # Collect Over and Under odds separately, keyed by (player, line).
        # Both sides are needed so we can normalize implied probabilities to sum to 1.
        # Rows whose Fair Odds cell contains ⚠️ are skipped — the site uses this icon
        # to flag odds devigged from a one-way line with estimated juice, making them
        # less reliable.
        raw: dict[tuple[str, float], dict[str, float]] = {}
        for row in rows:
            if len(row) <= _FAIR_ODDS_COL:
                continue
            parsed = _parse_bet_name(row[_BET_NAME_COL])
            if parsed is None:
                continue
            player_name, outcome, line = parsed
            odds_cell = row[_FAIR_ODDS_COL]
            if "\u26a0" in odds_cell:   # ⚠️ — estimated juice, skip
                log.debug("Skipping flagged odds row: %s %s", player_name, odds_cell)
                continue
            prob = _parse_fair_odds(odds_cell.strip())
            if prob is None:
                continue
            raw.setdefault((player_name, line), {})[outcome] = prob

        # Normalize Over/Under pair → P(Over), then store
        for (player_name, line), sides in raw.items():
            p_over_raw = sides.get("over")
            if p_over_raw is None:
                continue   # can't compute without the Over side
            p_under_raw = sides.get("under")
            if p_under_raw is not None:
                # Re-normalize so the two implied probs sum to 1
                total = p_over_raw + p_under_raw
                p_over = p_over_raw / total if total > 0 else p_over_raw
            else:
                p_over = p_over_raw
            market_data.setdefault(player_name, {}).setdefault(market_key, []).append(
                (line, p_over)
            )

    if debug:
        for mkt in markets:
            mk = MARKET_KEY_MAP.get(mkt)
            count = sum(1 for p in market_data.values() if mk in p) if mk else 0
            status = f"{count} players" if count else "NO DATA"
            log.debug("  %-40s %s", mkt, status)
    log.info(
        "game_id=%d: collected market data for %d players", game_id, len(market_data)
    )
    return market_data


async def _find_game_ids(
    page,
    slate_games: dict[tuple[str, str], dtime],
    slate_date: str,
    debug: bool = False,
) -> dict[tuple[str, str], int]:
    """
    Navigate to games.aspx and return game_ids matching slate games.
    Matches by: league=MLB, date=slate_date, team pair, game time ±30 min.
    """
    log.info("Loading games list: %s", GAMES_URL)
    await page.goto(GAMES_URL, wait_until="domcontentloaded")
    # ASP.NET timer fires with interval:1 to populate the GridView
    await page.wait_for_timeout(3000)
    try:
        await page.wait_for_selector('[id*="GridView1"] tr', timeout=20_000)
    except Exception:
        log.warning("Games list table did not load within timeout.")
        return {}

    # Extract rows: game_id from href, all cell text
    rows: list[dict] = await page.evaluate(
        """() => {
            const out = [];
            for (const a of document.querySelectorAll('a[href*="game.aspx?game_id="]')) {
                const href = a.getAttribute('href') || '';
                const m = href.match(/game_id=(\\d+)/);
                if (!m) continue;
                const game_id = parseInt(m[1], 10);
                const tr = a.closest('tr');
                if (!tr) continue;
                const cells = Array.from(tr.querySelectorAll('td'))
                                   .map(c => c.innerText.trim());
                out.push({ game_id, cells });
            }
            return out;
        }"""
    )

    if debug:
        log.debug("Games list rows (first 5): %s", rows[:5])

    slate_date_obj = datetime.strptime(slate_date, "%Y-%m-%d").date()
    today = datetime.now().date()

    game_id_map: dict[tuple[str, str], int] = {}

    for row_data in rows:
        cells: list[str] = row_data["cells"]
        game_id: int = row_data["game_id"]
        row_text = " ".join(cells)

        # Filter to MLB only
        if "MLB" not in row_text.upper():
            continue

        # Find the date cell and parse it
        resolved_date = None
        game_time: dtime | None = None

        for cell in cells:
            # "Today at 1:05 PM" or "Tomorrow at 7:05 PM" or "Apr 11 at 1:05 PM"
            m_today = re.match(
                r"Today\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM)", cell, re.IGNORECASE
            )
            m_other = re.match(
                r"(\w+\s+\d+)\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM)", cell, re.IGNORECASE
            )
            if m_today:
                resolved_date = today
                h, mn, ampm = int(m_today.group(1)), int(m_today.group(2)), m_today.group(3)
            elif m_other:
                try:
                    resolved_date = datetime.strptime(
                        f"{m_other.group(1)} {today.year}", "%b %d %Y"
                    ).date()
                except ValueError:
                    continue
                h, mn, ampm = int(m_other.group(2)), int(m_other.group(3)), m_other.group(4)
            else:
                continue

            if ampm.upper() == "PM" and h != 12:
                h += 12
            elif ampm.upper() == "AM" and h == 12:
                h = 0
            game_time = dtime(h, mn)
            break

        if resolved_date != slate_date_obj or game_time is None:
            continue

        # Identify team abbreviations from cell text.
        # First pass: exact full-name match (e.g. "Arizona Diamondbacks").
        # Second pass: nickname match (last word, e.g. "Athletics") for teams
        # whose city name changed or is omitted by the site.
        found_abbrs: list[str] = []
        for cell in cells:
            cell_lower = cell.lower()
            for full_name, abbr in FULL_NAME_TO_ABBR.items():
                if full_name.lower() in cell_lower and abbr not in found_abbrs:
                    found_abbrs.append(abbr)
            if len(found_abbrs) >= 2:
                break

        if len(found_abbrs) < 2:
            # Fallback: match individual words in each cell against team nicknames
            for cell in cells:
                for word in re.split(r"[\s/|]+", cell):
                    abbr = _NICKNAME_TO_ABBR.get(word.lower().rstrip("s"))  # try singular too
                    if abbr is None:
                        abbr = _NICKNAME_TO_ABBR.get(word.lower())
                    if abbr and abbr not in found_abbrs:
                        found_abbrs.append(abbr)
                if len(found_abbrs) >= 2:
                    break

        if len(found_abbrs) < 2:
            continue

        game_away, game_home = found_abbrs[0], found_abbrs[1]

        # Match to slate game by team pair + time tolerance ±30 min
        for (slate_away, slate_home), slate_time in slate_games.items():
            if {game_away, game_home} == {slate_away, slate_home}:
                delta = abs(
                    (game_time.hour * 60 + game_time.minute)
                    - (slate_time.hour * 60 + slate_time.minute)
                )
                if delta <= 30:
                    game_id_map[(slate_away, slate_home)] = game_id
                    log.info(
                        "Matched %s@%s → game_id=%d", slate_away, slate_home, game_id
                    )
                    break

    for key in slate_games:
        if key not in game_id_map:
            log.warning("No CrazyNinjaOdds game found for %s@%s", *key)

    return game_id_map


# ---------------------------------------------------------------------------
# Projection calculation
# ---------------------------------------------------------------------------

def _compute_batter_projection(
    player_markets: dict[str, list[tuple[float, float]]]
) -> tuple[float, float] | tuple[None, str]:
    """
    Compute (mean_dk, std_dev_dk) for a batter from market mean estimates.

    Uses the Geometric NB (r=1) model to back-solve E[X] from O/U fair odds.
    This corrects the systematic underestimation Poisson produces for batter
    counting stats, which are heavily overdispersed: most games a player gets
    0 of a given stat, but the right tail is long.

    Runs and RBIs are fetched as dedicated markets rather than derived as a
    residual from the combined Hits+Runs+RBIs market.

    Returns (None, reason) when key market data is absent for this player —
    the caller should use a RotoWire fallback projection in that case.
    Returns (mean, std_dev) on success.
    """
    def get_mu(key: str) -> float:
        lp = player_markets.get(key, [])
        if not lp:
            return 0.0
        mu = _fit_nb_mean(lp)
        return mu if mu is not None else 0.0

    e_s   = get_mu(MK_SINGLES)
    e_d   = get_mu(MK_DOUBLES)
    e_t   = get_mu(MK_TRIPLES)
    e_hr  = get_mu(MK_HR)
    e_sb  = get_mu(MK_SB)
    e_bb  = get_mu(MK_WALKS)
    e_r   = get_mu(MK_RUNS)
    e_rbi = get_mu(MK_RBIS)

    # Require at least one hit-type market and both Runs and RBIs.
    # A zero here means the market wasn't available for this player — fall back
    # to RotoWire rather than silently underestimating their projection.
    if e_s == 0 and e_d == 0 and e_t == 0 and e_hr == 0:
        return None, "no hit market data (Singles/Doubles/Triples/HR all missing)"
    missing = [name for name, val in [("Runs", e_r), ("RBIs", e_rbi)] if val == 0]
    if missing:
        return None, f"{' and '.join(missing)} market{'s' if len(missing) > 1 else ''} unavailable"

    e_hbp = e_bb * HBP_PER_WALK

    mean = (
        e_s * 3 + e_d * 5 + e_t * 8 + e_hr * 10
        + e_r * 2 + e_rbi * 2 + e_bb * 2 + e_sb * 5 + e_hbp * 2
    )
    # Var[aX] = a² * μ(1+μ) for Geometric NB(r=1).
    # The μ(1+μ) term (vs. Poisson's μ) captures the overdispersion inherent
    # in batter game-log distributions and gives a more realistic std_dev.
    var = (
        e_s   * (1 + e_s)   * 9
        + e_d   * (1 + e_d)   * 25
        + e_t   * (1 + e_t)   * 64
        + e_hr  * (1 + e_hr)  * 100
        + e_r   * (1 + e_r)   * 4
        + e_rbi * (1 + e_rbi) * 4
        + e_bb  * (1 + e_bb)  * 4
        + e_sb  * (1 + e_sb)  * 25
        + e_hbp * (1 + e_hbp) * 4
    )
    return mean, max(math.sqrt(var), 1.0)


def _compute_pitcher_projection(
    player_markets: dict[str, list[tuple[float, float]]]
) -> tuple[float, float] | None:
    """
    Compute (mean_dk, std_dev_dk) for a pitcher from market lambda estimates.
    Returns None if there is insufficient data.
    """
    def get_lam(key: str) -> float:
        lp = player_markets.get(key, [])
        if not lp:
            return 0.0
        lam = _fit_lambda(lp)
        return lam if lam is not None else 0.0

    e_outs = get_lam(MK_OUTS)
    e_k    = get_lam(MK_K)
    e_win  = get_lam(MK_WIN)
    e_ha   = get_lam(MK_HA)
    e_bba  = get_lam(MK_BBA)
    e_er   = get_lam(MK_ER)

    if e_outs == 0 and e_k == 0:
        return None

    e_ip  = e_outs / 3.0
    e_hbp = e_ip * HBP_PER_INNING_PITCHER

    mean = (
        e_outs * 0.75 + e_k * 2.0 + e_win * 4.0
        + e_er * (-2.0) + e_ha * (-0.6) + e_bba * (-0.6) + e_hbp * (-0.6)
    )
    var = (
        e_outs * 0.5625 + e_k * 4.0 + e_win * 16.0
        + e_er * 4.0 + e_ha * 0.36 + e_bba * 0.36 + e_hbp * 0.36
    )
    return mean, max(math.sqrt(var), 1.0)


# ---------------------------------------------------------------------------
# Playwright orchestration
# ---------------------------------------------------------------------------

async def _run_playwright(
    dk_path: str,
    name_map: dict[str, str],
    debug: bool = False,
    games_filter: set[tuple[str, str]] | None = None,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """
    Launch headless Chromium, find slate games on CrazyNinjaOdds, and scrape
    all relevant markets. Returns raw market data keyed by player name.

    games_filter: if provided, only fetch these (away, home) matchups.
    """
    from playwright.async_api import async_playwright

    slate_date = _extract_date_from_dk(dk_path)
    if not slate_date:
        log.error("Could not extract date from DK file.")
        return {}
    log.info("Slate date: %s", slate_date)

    slate_games = _extract_game_info_from_dk(dk_path)
    if not slate_games:
        log.error("Could not extract games from DK file.")
        return {}

    if games_filter:
        unknown = games_filter - slate_games.keys()
        if unknown:
            log.warning(
                "Games not in DK slate, ignoring: %s",
                ", ".join(f"{a}@{h}" for a, h in sorted(unknown)),
            )
        slate_games = {k: v for k, v in slate_games.items() if k in games_filter}
        if not slate_games:
            log.error("No requested games found in DK slate.")
            return {}

    log.info("Fetching for games: %s", [f"{a}@{h}" for a, h in slate_games.keys()])

    all_markets = BATTER_MARKETS + PITCHER_MARKETS
    all_market_data: dict[str, dict[str, list[tuple[float, float]]]] = {}

    cache_key   = _game_id_cache_key(slate_date, slate_games)
    game_id_map = _load_game_id_cache(cache_key)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        if game_id_map is None:
            game_id_map = await _find_game_ids(page, slate_games, slate_date, debug=debug)
            if not game_id_map:
                log.error("No slate games matched on CrazyNinjaOdds.")
                await browser.close()
                return {}
            _save_game_id_cache(cache_key, game_id_map)
        else:
            log.info("Skipping games.aspx — using %d cached game IDs.", len(game_id_map))

        first_game = True
        for (away, home), game_id in game_id_map.items():
            log.info("Fetching odds for %s@%s (game_id=%d)", away, home, game_id)
            try:
                game_data = await _fetch_game_odds(
                    page, game_id, all_markets, debug=debug,
                    validate_columns=first_game,
                )
                first_game = False
                for player_name, markets in game_data.items():
                    mapped = name_map.get(player_name, player_name)
                    entry = all_market_data.setdefault(mapped, {})
                    for mk, lp in markets.items():
                        entry.setdefault(mk, []).extend(lp)
            except Exception as e:
                log.warning(
                    "Error fetching game %d (%s@%s): %s", game_id, away, home, e
                )

        await browser.close()

    return all_market_data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_projections_csv(
    dk_path: str,
    output_path: str,
    name_map: dict[str, str] | None = None,
    debug: bool = False,
    games_filter: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Fetch market odds and produce projections CSV.
    Output columns: player_id, name, mean, std_dev  (no lineup_slot).

    games_filter: if provided, only fetch these (away, home) matchups.
    """
    name_map = name_map or {}

    log.info("Loading DK salary file: %s", dk_path)
    dk_df = pd.read_csv(dk_path)
    dk_df.rename(
        columns={"ID": "player_id", "Name": "name", "Salary": "salary"},
        inplace=True,
    )

    dk_lookup: dict[str, list[tuple[int, float]]] = {}
    for _, row in dk_df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        dk_lookup.setdefault(key, []).append((pid, sal))

    dk_pos: dict[int, str] = {}
    if "Position" in dk_df.columns:
        dk_pos = {int(r["player_id"]): str(r["Position"]) for _, r in dk_df.iterrows()}

    # Fetch market data
    all_market_data = asyncio.run(
        _run_playwright(dk_path, name_map, debug=debug, games_filter=games_filter)
    )

    if not all_market_data:
        log.error("No market data fetched.")
        sys.exit(1)

    log.info("Market data collected for %d players.", len(all_market_data))

    matched: list[dict] = []
    unmatched: list[str] = []
    # player_id → reason string for batters that couldn't be projected from market data
    fallback_reasons: dict[int, str] = {}

    for player_name, player_markets in all_market_data.items():
        pid = _match_name(player_name, dk_lookup)
        if pid is None:
            unmatched.append(player_name)
            continue

        position = dk_pos.get(pid, "")
        is_pitcher = any(p.upper() in {"P", "SP", "RP"} for p in position.split("/"))

        if is_pitcher:
            result = _compute_pitcher_projection(player_markets)
            if result is None:
                log.info(
                    "No projection for pitcher %s (pid=%d) — markets present: %s",
                    player_name, pid, list(player_markets.keys()) or "none",
                )
                continue
            mean, std_dev = result
        else:
            batter_result = _compute_batter_projection(player_markets)
            if batter_result[0] is None:
                reason = str(batter_result[1])
                log.info(
                    "No MO projection for %s (pid=%d) — %s; using RotoWire fallback",
                    player_name, pid, reason,
                )
                fallback_reasons[pid] = reason
                continue
            mean, std_dev = float(batter_result[0]), float(batter_result[1])  # type: ignore[arg-type]

        matched.append({
            "player_id": pid,
            "name": player_name,
            "mean": round(mean, 4),
            "std_dev": round(std_dev, 4),
        })

    if unmatched:
        log.warning(
            "%d player(s) not matched to DK IDs:\n  %s",
            len(unmatched),
            "\n  ".join(unmatched[:30]),
        )

    if not matched:
        log.error("No players matched. Run with --debug to inspect market data.")
        sys.exit(1)

    out_df = (
        pd.DataFrame(matched)
        .drop_duplicates("player_id")
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Write sidecar JSON mapping player_id → fallback reason.
    # The server reads this to annotate the merge_info callout in the UI.
    _op = Path(output_path)
    sidecar_path = _op.parent / (_op.stem + "_fallback.json")
    try:
        sidecar_path.write_text(
            json.dumps({str(pid): reason for pid, reason in fallback_reasons.items()}, indent=2)
        )
        log.debug("Wrote fallback reasons for %d players → %s", len(fallback_reasons), sidecar_path)
    except Exception as e:
        log.warning("Could not write fallback sidecar: %s", e)

    n_pitchers = int(out_df["player_id"].map(dk_pos).apply(
        lambda pos: any(p.upper() in {"P", "SP", "RP"} for p in str(pos).split("/"))
    ).sum())
    n_batters  = len(out_df) - n_pitchers
    log.info(
        "Wrote %d projections → %s  (batters=%d, pitchers=%d, unmatched=%d)",
        len(out_df), output_path, n_batters, n_pitchers, len(unmatched),
    )
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch CrazyNinjaOdds market odds and produce DK projections CSV."
        ),
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
        default=str(PROJECT_ROOT / "data" / "processed" / "projections_mo.csv"),
        metavar="PATH",
        help="Output CSV path (default: data/processed/projections_mo.csv)",
    )
    parser.add_argument(
        "--name-map",
        default=str(DEFAULT_NAME_MAP_PATH),
        metavar="PATH",
        help=(
            "JSON file mapping CrazyNinjaOdds player names to DK canonical names "
            "(default: data/name_map.json; silently ignored if absent)"
        ),
    )
    parser.add_argument(
        "--games",
        metavar="AWAY@HOME[,...]",
        default=None,
        help=(
            "Comma-separated matchups to fetch, e.g. 'ARI@PHI' or 'ARI@PHI,MIA@DET'. "
            "Defaults to all games in the DK slate. "
            "Useful for testing or refreshing a subset of games."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (prints raw table rows and per-market data counts)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    games_filter: set[tuple[str, str]] | None = None
    if args.games:
        games_filter = set()
        for token in args.games.split(","):
            token = token.strip().upper()
            if "@" not in token:
                parser.error(f"Invalid game format {token!r} — expected AWAY@HOME")
            away, home = token.split("@", 1)
            games_filter.add((away, home))

    build_projections_csv(
        dk_path=args.dk_slate,
        output_path=args.output,
        name_map=_load_name_map(args.name_map),
        debug=args.debug,
        games_filter=games_filter,
    )


if __name__ == "__main__":
    main()
