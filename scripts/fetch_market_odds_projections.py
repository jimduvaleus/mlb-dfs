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
          Player Triples, Player Home Runs, Player Stolen Bases, Player Batting Walks
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
Flagged (⚠️) Over lines > 0.5 are accepted directly into the NB fit without
normalization; flagged 0.5 lines remain a fallback-only path.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    "Player Batting Walks",
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
    "Player Batting Walks":       MK_WALKS,
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

# Scoring coefficients keyed by platform
# Batter: keys match the E[X] variable names used in _compute_batter_projection
_DK_BATTER = dict(single=3.0, double=5.0, triple=8.0, home_run=10.0,
                  run=2.0, rbi=2.0, walk=2.0, hbp=2.0, sb=5.0)
_FD_BATTER = dict(single=3.0, double=6.0, triple=9.0, home_run=12.0,
                  run=3.2, rbi=3.5, walk=3.0, hbp=3.0, sb=6.0)

# Pitcher: 'out' is per out recorded (DK: 0.75/out = 2.25/IP; FD: 1.0/out = 3.0/IP)
# 'h', 'bb', 'hbp' are 0 on FD (no penalty for allowing baserunners).
_DK_PITCHER = dict(out=0.75, k=2.0, win=4.0, qs=0.0,
                   er=-2.0, h=-0.6, bb=-0.6, hbp=-0.6)
_FD_PITCHER = dict(out=1.0,  k=3.0, win=6.0, qs=4.0,
                   er=-3.0,  h=0.0,  bb=0.0,  hbp=0.0)

# P(ER ≤ 3 | IP ≥ 6, no ER market) — historical MLB league-average fallback used
# when the Earned Runs market is unavailable for a pitcher.
_QS_ER_FALLBACK_PROB = 0.60

# Max browser pages open simultaneously while fetching game odds.
# Tune down to 1-2 if the site starts returning empty tables or HTTP 429s.
_MAX_CONCURRENT_GAMES = 4

# Playwright selector IDs (verified from page inspection)
SEL_DEVIG  = (
    "#ContentPlaceHolderMain_ContentPlaceHolderRight_"
    "WebUserControl_FilterDevigMethod_DropDownListDevigMethod"
)
SEL_DEVIG_UPDATE = (
    "#ContentPlaceHolderMain_ContentPlaceHolderRight_ButtonUpdate"
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

# Override legacy/alias site names that differ from the current DK abbreviation.
# CrazyNinjaOdds still uses "Oakland Athletics" even after relocation; DK uses ATH.
FULL_NAME_TO_ABBR["Oakland Athletics"] = "ATH"

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
    dk_lookup: dict[str, list[tuple[int, float, str]]],  # norm_name → [(player_id, salary, team)]
    source_teams: set[str] | None = None,
    cutoff: float = 0.82,
) -> int | None:
    """
    Return a DK player_id for *source_name*, or None if no confident match.

    *source_teams* is the set of team abbreviations for the game the player
    appeared in (i.e. {away, home}).  When provided, DK candidates are filtered
    to only players on those teams before any further disambiguation, which
    resolves same-name players in different games (e.g. Max Muncy LAD vs ATH).

    Falls back to difflib fuzzy matching within the same team set.
    """
    key = _normalise(source_name)
    all_keys = list(dk_lookup.keys())

    def _team_filter(candidates: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
        """Filter candidates to those on a game team; returns [] on no match (no fallback)."""
        if not source_teams:
            return candidates
        return [c for c in candidates if c[2].upper() in source_teams]

    def _fuzzy_within_teams() -> int | None:
        """Try fuzzy name match restricted to source_teams; return pid or None."""
        fuzzy = difflib.get_close_matches(key, all_keys, n=5, cutoff=cutoff)
        for fkey in fuzzy:
            if fkey == key:
                continue
            hits = _team_filter(dk_lookup[fkey])
            if hits:
                return hits[0][0]
        return None

    if key in dk_lookup:
        candidates = dk_lookup[key]
        filtered = _team_filter(candidates)

        if filtered:
            # At least one exact-name candidate is on a game team.
            return filtered[0][0] if len(filtered) == 1 else filtered[0][0]

        if source_teams:
            # Exact name match exists but no candidate is on a game team.
            # Try a fuzzy name variant that IS on a game team
            # (e.g. CrazyNinja "Luis Garcia" → DK "Luis Garcia Jr." WSH).
            hit = _fuzzy_within_teams()
            if hit is not None:
                return hit

        # No team info or fuzzy found nothing — accept the exact match as-is.
        return candidates[0][0]

    # No exact name match — fuzzy search, preferring game-team candidates.
    matches = difflib.get_close_matches(key, all_keys, n=3, cutoff=cutoff)
    if not matches:
        return None
    if source_teams:
        for fkey in matches:
            hits = _team_filter(dk_lookup[fkey])
            if hits:
                return hits[0][0]
    return dk_lookup[matches[0]][0]


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


def _extract_game_info_from_dk(dk_path: str) -> dict[tuple[str, str], dtime | None]:
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
# FanDuel slate helpers
# ---------------------------------------------------------------------------

_FD_DATE_IN_PATH_RE = re.compile(r"FanDuel-MLB-(\d{4}-\d{2}-\d{2})-")


def _extract_date_from_fd_path(path: str) -> str | None:
    """
    Extract 'YYYY-MM-DD' from a FanDuel salary CSV filename.
    FD filenames follow the pattern: FanDuel-MLB-YYYY-MM-DD-*.csv
    Returns None if the pattern is not found.
    """
    m = _FD_DATE_IN_PATH_RE.search(Path(path).name)
    return m.group(1) if m else None


def _extract_game_info_from_fd(fd_path: str) -> dict[tuple[str, str], dtime | None]:
    """
    Parse FanDuel salary CSV → {(away_abbr, home_abbr): game_time_ET_or_None}.

    FD's "Game" column contains only "AWAY@HOME" (no date/time), so game_time
    is always None.  The time-match step in _find_game_ids skips the tolerance
    check when the slate time is None (matching on team pairs only).
    """
    from src.ingestion.fd_slate import FanDuelSlateIngestor

    games: dict[tuple[str, str], dtime | None] = {}
    try:
        df = FanDuelSlateIngestor(fd_path).get_slate_dataframe()
        for raw in df["game"].dropna().unique():
            m = re.match(r"(\w+)@(\w+)", str(raw).strip())
            if m:
                away, home = m.group(1).upper(), m.group(2).upper()
                games[(away, home)] = None  # no time info in FD CSV
    except Exception as e:
        log.warning("Could not extract game info from FD file: %s", e)
    return games


def _build_fd_player_lookup(
    fd_path: str,
) -> tuple[dict[str, list[tuple[int, float, str]]], dict[int, str]]:
    """
    Build player lookup dicts from a FanDuel salary CSV via FanDuelSlateIngestor.

    Returns:
        (lookup, pos_map) matching the DK equivalents built in build_projections_csv:
        lookup  — {normalised_name: [(player_id, salary, team_abbr), ...]}
        pos_map — {player_id: position_string}
    """
    from src.ingestion.fd_slate import FanDuelSlateIngestor

    df = FanDuelSlateIngestor(fd_path).get_slate_dataframe()
    lookup: dict[str, list[tuple[int, float, str]]] = {}
    for _, row in df.iterrows():
        key = _normalise(str(row["name"]))
        pid = int(row["player_id"])
        sal = float(row["salary"])
        team = str(row["team"]).upper()
        lookup.setdefault(key, []).append((pid, sal, team))
    pos_map: dict[int, str] = {int(r["player_id"]): str(r["position"]) for _, r in df.iterrows()}
    return lookup, pos_map


# ---------------------------------------------------------------------------
# Game ID cache
# ---------------------------------------------------------------------------

def _game_id_cache_key(slate_date: str, slate_games: dict[tuple[str, str], dtime | None]) -> str:
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

def _fit_nb_mean(lines_probs: list[tuple[float, float]], tol: float = 1e-9) -> float | None:
    """
    Fit Geometric NB(r=1) mean via weighted log-linear regression across all
    available (line, p_over) pairs.

    Under NB(r=1): P(X > k) = q^(k+1)  where q = μ/(1+μ) ∈ (0,1).
    Taking logs:   log P(X > k) = (k+1) · θ  (θ = log q < 0).

    Estimate θ by weighted least squares through the origin:

        θ = Σ[w_i · (k_i+1) · log p_i] / Σ[w_i · (k_i+1)²]

    where w_i = max(|p_i − 0.5|, 0.01) (sharper lines carry more information)
    and k_i = floor(line_i).

    Higher lines contribute with weight proportional to (k+1)², so a 1.5-line
    constrains the tail of the distribution more strongly than a 0.5-line.
    This prevents the estimate from exploding when p_over on the 0.5 line is
    high but no higher line is available to anchor the tail.

    For a single line the estimator reduces to the previous closed-form:
        θ = log(p_over) / 1  →  q = p_over  →  μ = p_over / (1 − p_over).
    """
    numerator = 0.0
    denominator = 0.0
    for line, p_over in lines_probs:
        p = max(tol, min(1.0 - tol, p_over))
        k = math.floor(line)
        w = max(abs(p - 0.5), 0.01)
        x = float(k + 1)
        numerator += w * x * math.log(p)
        denominator += w * x * x
    if denominator == 0.0:
        return None
    theta = numerator / denominator
    if theta >= 0.0:
        return None  # degenerate: q must be in (0,1) so log(q) must be negative
    q = math.exp(theta)
    if q >= 1.0 - tol:
        return None
    return q / (1.0 - q)


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
    # Phase 1: wait for the market dropdown to become disabled, confirming the
    # postback has actually started.  Use a short timeout — it should happen in
    # well under a second.  If it times out (e.g. no postback was triggered),
    # fall through so Phase 2 still guards the read.
    try:
        await page.wait_for_function(
            f"""() => {{
                const el = document.querySelector('{SEL_MARKET}');
                return !el || el.disabled;
            }}""",
            timeout=2000,
        )
    except Exception:
        pass  # postback may have already completed or was not triggered

    # Phase 2: wait for the market dropdown to be re-enabled (AJAX complete,
    # table values updated with the new devig method).
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
        await page.wait_for_timeout(1200)


async def _wait_and_read_table(page, timeout: int = 15_000) -> list[list[str]]:
    """
    Wait for an ASP.NET AJAX postback to complete and return the GridView table
    rows in a single browser round-trip.

    Polls a JS function that returns null while the market dropdown is disabled
    (postback in-flight) and returns the table rows array once it is re-enabled.
    This replaces the separate _wait_for_ajax() + page.evaluate() pattern,
    saving one browser ↔ Python round-trip per market per game.

    Falls back to a fixed delay + direct evaluate() if the poll times out.
    """
    # Phase 1: wait for the market dropdown to become disabled (postback started).
    try:
        await page.wait_for_function(
            f"""() => {{
                const el = document.querySelector('{SEL_MARKET}');
                return !el || el.disabled;
            }}""",
            timeout=2000,
        )
    except Exception:
        pass  # postback may have already completed or was not triggered

    # Phase 2: once the dropdown is re-enabled, the table contains updated values.
    try:
        handle = await page.wait_for_function(
            f"""() => {{
                const el = document.querySelector('{SEL_MARKET}');
                if (!el || el.disabled) return null;
                const table = document.querySelector('[id*="GridView1"]');
                if (!table) return [];
                const out = [];
                for (const tr of table.querySelectorAll('tr')) {{
                    const cells = Array.from(tr.querySelectorAll('td'))
                                       .map(c => c.innerText.trim());
                    if (cells.length > 0) out.push(cells);
                }}
                return out;
            }}""",
            timeout=timeout,
        )
        return await handle.json_value()
    except Exception:
        # Fallback: fixed delay then direct evaluate
        await page.wait_for_timeout(1200)
        return await page.evaluate(
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

    # Wait for the ASP.NET timer to fire and populate the GridView.
    # No fixed sleep: wait_for_selector polls efficiently until the table appears.
    try:
        await page.wait_for_selector('[id*="GridView1"] tr', timeout=20_000)
    except Exception:
        log.warning(
            "Game page table did not load for game_id=%d — "
            "possible rate-limit, redirect, or site change. "
            "Check for HTTP 429/403 warnings above.",
            game_id,
        )
        return {}, set()

    # Set devig method: Liquidity-Weighted - Additive/Shin
    try:
        await page.select_option(SEL_DEVIG, value=DEVIG_VALUE)
        await page.click(SEL_DEVIG_UPDATE)
        await _wait_for_ajax(page)
    except Exception as e:
        log.warning("Could not set devig method for game %d: %s", game_id, e)

    market_data: dict[str, dict[str, list[tuple[float, float]]]] = {}
    seen_players: set[str] = set()
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

        # Select All players (value="0"), then combined-wait + table read in one
        # round-trip via _wait_and_read_table (saves a separate evaluate() call).
        try:
            await page.select_option(SEL_PLAYER, value="0")
        except Exception as e:
            log.debug("Could not select All players for '%s': %s", market_name, e)
        rows: list[list[str]] = await _wait_and_read_table(page)

        if debug and rows:
            log.debug("Market '%s' raw rows (first 4): %s", market_name, rows[:4])

        # Validate column layout once on the first non-empty table result
        if validate_columns and not _validated and rows:
            _validate_table_columns(rows, market_name)  # raises RuntimeError on bad layout
            _validated = True

        # Collect Over and Under odds separately, keyed by (player, line).
        # Both sides are needed so we can normalize implied probabilities to sum to 1.
        # Rows whose Fair Odds cell contains ⚠️ were devigged from a one-way line.
        # We save ⚠️-flagged 0.5-line odds in a separate dict and use them as a
        # fallback for any player who has no valid (non-⚠️) 0.5 data for this market.
        # The one-way devigged odds are treated as fair: P(Over) = implied_prob(Over).
        # Flagged 0.5 lines are stored in flagged_half for fallback.
        # Flagged Over lines > 0.5 are accepted directly into raw — the one-way
        # devig introduces some bias, but these lines still anchor the tail of the
        # NB fit and prevent the model from exploding on a single high-p 0.5 line.
        raw: dict[tuple[str, float], dict[str, float]] = {}
        flagged_half: dict[str, dict[str, float]] = {}  # player → {outcome: prob}
        for row in rows:
            if len(row) <= _FAIR_ODDS_COL:
                continue
            parsed = _parse_bet_name(row[_BET_NAME_COL])
            if parsed is None:
                continue
            player_name, outcome, line = parsed
            seen_players.add(player_name)  # track before ⚠️ skip
            odds_cell = row[_FAIR_ODDS_COL]
            if "\u26a0" in odds_cell:   # ⚠️ — one-way devigged odds
                odds_str = odds_cell.split("\u26a0")[0].strip()
                prob = _parse_fair_odds(odds_str)
                if prob is not None:
                    if line == 0.5:
                        # Store for flagged-0.5 fallback (existing logic)
                        flagged_half.setdefault(player_name, {})[outcome] = prob
                    elif outcome == "over":
                        # Accept flagged Over lines > 0.5 into raw so the NB
                        # regression can use them as tail constraints.  No Under
                        # side is available so p_over is used without normalization.
                        raw.setdefault((player_name, line), {})["over"] = prob
                        log.debug(
                            "Accepting flagged over %.1f odds for %s (p_over=%.4f)",
                            line, player_name, prob,
                        )
                    else:
                        log.debug("Skipping flagged odds row: %s %s", player_name, odds_cell)
                else:
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

        # Flagged-0.5 fallback: for each player whose 0.5 line had no valid odds,
        # add the flagged 0.5 odds so they aren't silently projected with a missing
        # market.  Players with valid 1.5/2.5 data but flagged 0.5 also benefit —
        # the 0.5 point is the most informative line and improves the μ estimate.
        for player_name, sides in flagged_half.items():
            if (player_name, 0.5) in raw:
                continue  # valid 0.5 already present — don't mix in flagged data
            p_over_raw = sides.get("over")
            if p_over_raw is None:
                continue  # no Over side available
            p_under_raw = sides.get("under")
            if p_under_raw is not None:
                total = p_over_raw + p_under_raw
                p_over = p_over_raw / total if total > 0 else p_over_raw
            else:
                p_over = p_over_raw
            market_data.setdefault(player_name, {}).setdefault(market_key, []).append(
                (0.5, p_over)
            )
            log.debug(
                "game_id=%d: %s — using flagged 0.5 odds for %s (p_over=%.4f)",
                game_id, market_name, player_name, p_over,
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
    return market_data, seen_players


async def _find_game_ids(
    page,
    slate_games: dict[tuple[str, str], dtime | None],
    slate_date: str,
    debug: bool = False,
) -> dict[tuple[str, str], int]:
    """
    Navigate to games.aspx and return game_ids matching slate games.
    Matches by: league=MLB, date=slate_date, team pair, game time ±30 min.
    When slate_time is None (e.g. FD CSV has no time info), the time check
    is skipped and any game on the correct date with matching teams is accepted.
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

        # Match to slate game by team pair + optional time tolerance ±30 min.
        # When slate_time is None (e.g. FD CSV has no game time), match on
        # team pairs only — any CrazyNinjaOdds game on the correct date with
        # the same two teams is accepted.
        for (slate_away, slate_home), slate_time in slate_games.items():
            if {game_away, game_home} == {slate_away, slate_home}:
                if slate_time is not None:
                    delta = abs(
                        (game_time.hour * 60 + game_time.minute)
                        - (slate_time.hour * 60 + slate_time.minute)
                    )
                    if delta > 30:
                        continue
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

def _compute_qs_probability(e_outs: float, e_er: float) -> float:
    """
    Estimate P(Quality Start) from Poisson market-odds estimates.

    A Quality Start is IP ≥ 6.0 AND ER ≤ 3.  Both dimensions are modelled as
    independent Poisson random variables using the λ values back-solved from
    the Outs Recorded and Earned Runs Allowed market odds.

    P(QS) ≈ P(Outs ≥ 18 | λ_outs) × P(ER ≤ 3 | λ_er)

    The independence assumption slightly overestimates P(QS) (high-ER games
    tend to be shorter), but the ER market already captures that signal
    through p_er3.  When ER market data is absent (e_er == 0), we fall back
    to the historical league-average P(ER ≤ 3 | IP ≥ 6) ≈ 0.60.
    """
    if e_outs <= 0:
        return 0.0
    p_ip6 = 1.0 - float(scipy_poisson.cdf(17, e_outs))       # P(Outs ≥ 18)
    if e_er > 0:
        p_er3 = float(scipy_poisson.cdf(3, e_er))             # P(ER ≤ 3)
    else:
        p_er3 = _QS_ER_FALLBACK_PROB
    return p_ip6 * p_er3


def _compute_batter_projection(
    player_markets: dict[str, list[tuple[float, float]]],
    platform: str = "draftkings",
) -> tuple[float, float] | tuple[None, str]:
    """
    Compute (mean, std_dev) for a batter from market mean estimates.

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
    c = _FD_BATTER if platform == "fanduel" else _DK_BATTER

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

    # Require at least one hit-type market and all three rate markets (Runs, RBIs,
    # Batting Walks).  A zero here means the market wasn't available for this
    # player — fall back to RotoWire rather than silently underestimating their
    # projection.  Triples and Stolen Bases are legitimately absent for many
    # players and are not required.
    if e_s == 0 and e_d == 0 and e_t == 0 and e_hr == 0:
        return None, "no hit market data (Singles/Doubles/Triples/HR all missing)"
    missing = [
        name for name, val in [
            ("Runs", e_r), ("RBIs", e_rbi), ("Batting Walks", e_bb)
        ] if val == 0
    ]
    if missing:
        return None, f"{' and '.join(missing)} market{'s' if len(missing) > 1 else ''} unavailable"

    e_hbp = e_bb * HBP_PER_WALK

    mean = (
        e_s * c["single"] + e_d * c["double"] + e_t * c["triple"] + e_hr * c["home_run"]
        + e_r * c["run"] + e_rbi * c["rbi"] + e_bb * c["walk"] + e_sb * c["sb"]
        + e_hbp * c["hbp"]
    )
    # Var[aX] = a² * μ(1+μ) for Geometric NB(r=1).
    # The μ(1+μ) term (vs. Poisson's μ) captures the overdispersion inherent
    # in batter game-log distributions and gives a more realistic std_dev.
    var = (
        e_s   * (1 + e_s)   * c["single"]   ** 2
        + e_d   * (1 + e_d)   * c["double"]   ** 2
        + e_t   * (1 + e_t)   * c["triple"]   ** 2
        + e_hr  * (1 + e_hr)  * c["home_run"] ** 2
        + e_r   * (1 + e_r)   * c["run"]      ** 2
        + e_rbi * (1 + e_rbi) * c["rbi"]      ** 2
        + e_bb  * (1 + e_bb)  * c["walk"]     ** 2
        + e_sb  * (1 + e_sb)  * c["sb"]       ** 2
        + e_hbp * (1 + e_hbp) * c["hbp"]      ** 2
    )
    return mean, max(math.sqrt(var), 1.0)


def _compute_pitcher_projection(
    player_markets: dict[str, list[tuple[float, float]]],
    platform: str = "draftkings",
) -> tuple[float, float] | None:
    """
    Compute (mean, std_dev) for a pitcher from market lambda estimates.
    Returns None if there is insufficient data.

    For FanDuel, a Quality Start bonus (4 pts) is included.  P(QS) is
    estimated from the Outs Recorded and Earned Runs Allowed Poisson lambdas
    via _compute_qs_probability.  When ER market data is absent, a
    league-average fallback of P(ER ≤ 3 | IP ≥ 6) ≈ 0.60 is used.
    """
    c = _FD_PITCHER if platform == "fanduel" else _DK_PITCHER

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

    # QS: only meaningful for FD (qs coef = 0 on DK); treat QS indicator as
    # Bernoulli(p_qs) — E[QS] = p_qs, Var[QS] = p_qs*(1-p_qs).
    p_qs = _compute_qs_probability(e_outs, e_er) if platform == "fanduel" else 0.0

    mean = (
        e_outs * c["out"] + e_k * c["k"] + e_win * c["win"] + p_qs * c["qs"]
        + e_er  * c["er"]
        + e_ha  * c["h"] + e_bba * c["bb"] + e_hbp * c["hbp"]
    )
    var = (
        e_outs * c["out"] ** 2
        + e_k   * c["k"]   ** 2
        + e_win * c["win"] ** 2
        + e_er  * c["er"]  ** 2
        + e_ha  * c["h"]   ** 2
        + e_bba * c["bb"]  ** 2
        + e_hbp * c["hbp"] ** 2
        + p_qs * (1.0 - p_qs) * c["qs"] ** 2  # Bernoulli QS variance
    )
    return mean, max(math.sqrt(var), 1.0)


# ---------------------------------------------------------------------------
# Playwright orchestration
# ---------------------------------------------------------------------------

def _install_rate_limit_handler(page, label: str) -> None:
    """
    Install a response listener on a Playwright page that logs HTTP status
    codes indicative of rate-limiting or IP blocks from crazyninjaodds.com.
    Call once per page immediately after creation.
    """
    async def _on_response(response) -> None:
        if "crazyninjaodds" not in response.url:
            return
        status = response.status
        if status == 429:
            log.warning(
                "RATE LIMIT (HTTP 429) [%s] — %s. "
                "Reduce _MAX_CONCURRENT_GAMES (currently %d) or add delays.",
                label, response.url, _MAX_CONCURRENT_GAMES,
            )
        elif status == 403:
            log.warning(
                "HTTP 403 Forbidden [%s] — %s. Possible IP block or rate limit.",
                label, response.url,
            )
        elif status == 503:
            log.warning(
                "HTTP 503 Service Unavailable [%s] — %s. "
                "Possible rate limit or server overload.",
                label, response.url,
            )
        elif status >= 400:
            log.debug("HTTP %d [%s] — %s", status, label, response.url)

    page.on("response", _on_response)


async def _run_playwright(
    slate_path: str,
    name_map: dict[str, str],
    debug: bool = False,
    games_filter: set[tuple[str, str]] | None = None,
    platform: str = "draftkings",
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    set[tuple[str, str, str]],
]:
    """
    Launch headless Chromium, find slate games on CrazyNinjaOdds, and scrape
    all relevant markets. Games are fetched in parallel (up to
    _MAX_CONCURRENT_GAMES simultaneous pages).

    Returns (all_market_data, all_seen_players) where both are keyed/tagged by
    (player_name, away_team, home_team) so that same-name players in different
    games (e.g. Max Muncy on LAD and ATH) are kept as separate entries and
    matched to the correct DK player_id at projection time.

    games_filter: if provided, only fetch these (away, home) matchups.
    platform: 'draftkings' or 'fanduel' — controls date/game extraction path.
    """
    from playwright.async_api import async_playwright

    if platform == "fanduel":
        slate_date = _extract_date_from_fd_path(slate_path)
        if not slate_date:
            log.error("Could not extract date from FD filename: %s", Path(slate_path).name)
            return {}, set()
    else:
        slate_date = _extract_date_from_dk(slate_path)
        if not slate_date:
            log.error("Could not extract date from DK file.")
            return {}, set()
    log.info("Slate date: %s", slate_date)

    if platform == "fanduel":
        slate_games = _extract_game_info_from_fd(slate_path)
    else:
        slate_games = _extract_game_info_from_dk(slate_path)
    if not slate_games:
        log.error("Could not extract games from %s file.", platform.upper())
        return {}, set()

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
            return {}, set()

    log.info("Fetching for games: %s", [f"{a}@{h}" for a, h in slate_games.keys()])

    all_markets = BATTER_MARKETS + PITCHER_MARKETS
    all_market_data: dict[str, dict[str, list[tuple[float, float]]]] = {}
    all_seen_players: set[str] = set()

    cache_key   = _game_id_cache_key(slate_date, slate_games)
    game_id_map = _load_game_id_cache(cache_key)

    # Discard a partial cache — if any slate game is missing a game ID the
    # cache was built before a previous match failure was resolved.
    if game_id_map is not None:
        missing = [k for k in slate_games if k not in game_id_map]
        if missing:
            log.info(
                "Cached game IDs are incomplete (%d missing); re-scraping.",
                len(missing),
            )
            game_id_map = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # Phase 1: game ID discovery uses a single dedicated page.
        if game_id_map is None:
            lookup_page = await browser.new_page()
            _install_rate_limit_handler(lookup_page, "games.aspx")
            game_id_map = await _find_game_ids(
                lookup_page, slate_games, slate_date, debug=debug
            )
            await lookup_page.close()
            if not game_id_map:
                log.error("No slate games matched on CrazyNinjaOdds.")
                await browser.close()
                return {}, set()
            _save_game_id_cache(cache_key, game_id_map)
        else:
            log.info("Skipping games.aspx — using %d cached game IDs.", len(game_id_map))

        # Phase 2: fetch all games in parallel, capped by _MAX_CONCURRENT_GAMES.
        game_entries = list(game_id_map.items())
        log.info(
            "Fetching %d game(s) with up to %d concurrent pages.",
            len(game_entries), _MAX_CONCURRENT_GAMES,
        )
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GAMES)

        async def _fetch_one(
            away: str, home: str, game_id: int, validate: bool
        ) -> tuple[dict, set]:
            async with semaphore:
                page = await browser.new_page()
                _install_rate_limit_handler(page, f"{away}@{home}")
                log.info("Fetching odds for %s@%s (game_id=%d)", away, home, game_id)
                try:
                    return await _fetch_game_odds(
                        page, game_id, all_markets, debug=debug,
                        validate_columns=validate,
                    )
                except Exception as e:
                    log.warning(
                        "Error fetching game %d (%s@%s): %s", game_id, away, home, e
                    )
                    return {}, set()
                finally:
                    await page.close()

        tasks = [
            _fetch_one(away, home, game_id, validate=(i == 0))
            for i, ((away, home), game_id) in enumerate(game_entries)
        ]
        results = await asyncio.gather(*tasks)

        await browser.close()

    # Merge results, keeping market data separate per (player_name, away, home) so
    # that same-name players in different games (e.g. Max Muncy on both LAD and ATH)
    # remain distinct entries and can each be matched to the correct DK player_id.
    all_market_data: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = {}
    all_seen_players: set[tuple[str, str, str]] = set()
    consecutive_empty = 0
    for i, ((away, home), _) in enumerate(game_entries):
        game_data, seen_players = results[i]
        if not game_data:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                log.warning(
                    "≥2 consecutive games returned no market data (%s@%s and prior) — "
                    "possible rate-limiting. Check for HTTP 429/403 warnings above. "
                    "Consider reducing _MAX_CONCURRENT_GAMES (currently %d).",
                    away, home, _MAX_CONCURRENT_GAMES,
                )
        else:
            consecutive_empty = 0
        for player_name, markets in game_data.items():
            mapped = name_map.get(player_name, player_name)
            game_key = (mapped, away, home)
            entry = all_market_data.setdefault(game_key, {})
            for mk, lp in markets.items():
                entry.setdefault(mk, []).extend(lp)
        for player_name in seen_players:
            mapped = name_map.get(player_name, player_name)
            all_seen_players.add((mapped, away, home))

    return all_market_data, all_seen_players


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_projections_csv(
    dk_path: str,
    output_path: str,
    name_map: dict[str, str] | None = None,
    debug: bool = False,
    games_filter: set[tuple[str, str]] | None = None,
    rw_player_ids: set[int] | None = None,
    platform: str = "draftkings",
    fd_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch market odds and produce projections CSV.
    Output columns: player_id, name, mean, std_dev  (no lineup_slot).

    rw_player_ids: if provided, only compute projections for players whose
        player_id appears in this set (i.e. players with a RotoWire lineup
        slot).  Players not in the set are silently skipped.

    games_filter: if provided, only fetch these (away, home) matchups.

    platform: 'draftkings' (default) or 'fanduel'.
    fd_path: path to FanDuel salary CSV (required when platform='fanduel').
    """
    name_map = name_map or {}

    # Build player lookup (name → [(player_id, salary, team)]) and position map.
    if platform == "fanduel":
        if not fd_path:
            log.error("--fd-slate is required when --platform fanduel")
            sys.exit(1)
        log.info("Loading FD salary file: %s", fd_path)
        dk_lookup, dk_pos = _build_fd_player_lookup(fd_path)
        slate_path = fd_path
    else:
        log.info("Loading DK salary file: %s", dk_path)
        dk_df = pd.read_csv(dk_path)
        dk_df.rename(
            columns={"ID": "player_id", "Name": "name", "Salary": "salary"},
            inplace=True,
        )
        dk_lookup: dict[str, list[tuple[int, float, str]]] = {}
        for _, row in dk_df.iterrows():
            key = _normalise(str(row["name"]))
            pid = int(row["player_id"])
            sal = float(row["salary"])
            team = str(row.get("TeamAbbrev", "")).upper()
            dk_lookup.setdefault(key, []).append((pid, sal, team))
        dk_pos: dict[int, str] = {}
        if "Position" in dk_df.columns:
            dk_pos = {int(r["player_id"]): str(r["Position"]) for _, r in dk_df.iterrows()}
        slate_path = dk_path

    # Fetch market data
    all_market_data, all_seen_players = asyncio.run(
        _run_playwright(
            slate_path, name_map, debug=debug,
            games_filter=games_filter, platform=platform,
        )
    )

    if not all_market_data:
        log.error("No market data fetched.")
        sys.exit(1)

    log.info("Market data collected for %d player-game entries.", len(all_market_data))

    matched: list[dict] = []
    unmatched: list[str] = []
    # player_id → reason string for batters that couldn't be projected from market data
    fallback_reasons: dict[int, str] = {}

    for (player_name, away, home), player_markets in all_market_data.items():
        source_teams = {away, home}
        pid = _match_name(player_name, dk_lookup, source_teams=source_teams)
        if pid is None:
            unmatched.append(player_name)
            continue

        if rw_player_ids is not None and pid not in rw_player_ids:
            log.debug("Skipping %s (pid=%d) — not in RotoWire lineup pool", player_name, pid)
            continue

        position = dk_pos.get(pid, "")
        is_pitcher = any(p.upper() in {"P", "SP", "RP"} for p in position.split("/"))

        if is_pitcher:
            result = _compute_pitcher_projection(player_markets, platform=platform)
            if result is None:
                log.info(
                    "No projection for pitcher %s (pid=%d) — markets present: %s",
                    player_name, pid, list(player_markets.keys()) or "none",
                )
                continue
            mean, std_dev = result
        else:
            batter_result = _compute_batter_projection(player_markets, platform=platform)
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

    # Players who appeared in odds tables but had every line ⚠️-flagged never entered
    # all_market_data, so the loop above silently skipped them.  Detect and warn here
    # so the log is explicit and the UI can show a reason for the RotoWire fallback.
    all_flagged_keys = all_seen_players - set(all_market_data.keys())
    for (player_name, away, home) in sorted(all_flagged_keys):
        source_teams = {away, home}
        pid = _match_name(player_name, dk_lookup, source_teams=source_teams)
        if pid is None:
            continue  # not on this DK slate — ignore
        if rw_player_ids is not None and pid not in rw_player_ids:
            continue  # not in RotoWire lineup pool — skip fallback reason too
        position = dk_pos.get(pid, "")
        is_pitcher = any(p.upper() in {"P", "SP", "RP"} for p in position.split("/"))
        if is_pitcher:
            log.debug(
                "No MO projection for pitcher %s (pid=%d) — all market odds lines flagged (⚠️)",
                player_name, pid,
            )
            continue
        reason = "all market odds lines flagged (\u26a0\ufe0f)"
        log.info(
            "No MO projection for %s (pid=%d) — %s; using RotoWire fallback",
            player_name, pid, reason,
        )
        fallback_reasons[pid] = reason

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
        "--platform",
        choices=["draftkings", "fanduel"],
        default="draftkings",
        help="DFS platform (default: draftkings)",
    )
    parser.add_argument(
        "--dk-slate",
        default=str(PROJECT_ROOT / "data" / "raw" / "DKSalaries.csv"),
        metavar="PATH",
        help="DraftKings salary CSV (default: data/raw/DKSalaries.csv)",
    )
    parser.add_argument(
        "--fd-slate",
        default=None,
        metavar="PATH",
        help=(
            "FanDuel salary CSV (required when --platform fanduel). "
            "Filename must follow the FanDuel-MLB-YYYY-MM-DD-*.csv convention "
            "so the slate date can be extracted from it."
        ),
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
        "--rw-output",
        metavar="PATH",
        default=None,
        help=(
            "Path to the RotoWire projections CSV produced by fetch_rotowire_projections.py. "
            "When provided, only players with a RotoWire lineup slot are projected; "
            "all other players in the market odds data are skipped."
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

    rw_player_ids: set[int] | None = None
    if args.rw_output:
        try:
            rw_df = pd.read_csv(args.rw_output)
            rw_player_ids = set(int(x) for x in rw_df["player_id"].dropna())
            log.info("RotoWire player pool: %d players with lineup slots", len(rw_player_ids))
        except Exception as exc:
            log.warning("Could not load RotoWire output %s: %s — projecting all players", args.rw_output, exc)

    build_projections_csv(
        dk_path=args.dk_slate,
        output_path=args.output,
        name_map=_load_name_map(args.name_map),
        debug=args.debug,
        games_filter=games_filter,
        rw_player_ids=rw_player_ids,
        platform=args.platform,
        fd_path=args.fd_slate,
    )


if __name__ == "__main__":
    main()
