"""
FastAPI server for the MLB DFS optimizer web UI.

Endpoints:
  GET  /api/config              — return current config.yaml
  POST /api/config              — save config.yaml
  GET  /api/projections/status  — projections file metadata
  POST /api/projections/fetch   — stream fetch_rotowire_projections.py output
  GET  /api/run/stream          — SSE: run pipeline, stream progress events
  GET  /api/run/status          — current run state
  GET  /api/portfolio           — last completed portfolio
  GET  /*                       — serve React SPA (ui/dist/)
"""
import asyncio
import json
import os
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config_io import read_config, write_config
from .models import AppConfig, ExclusionsUpdate, GameStatus, ParsedSlot, PlayerExclusionStatus, PlayerExclusionsUpdate, PlayerMatch, PortfolioResult, ProjectionsStatus, SlateGamesResponse, SlateListResponse, SlateOption, SlatePlayersResponse, TwitterLineupParseRequest, TwitterLineupParseResponse, TwitterLineupRecord, TwitterLineupSaveRequest, TwitterLineupSlot
from .twitter_lineups import (
    delete_twitter_lineup,
    get_twitter_overrides,
    load_twitter_lineups,
    match_player_name,
    parse_notification_body,
    upsert_twitter_lineup,
)
from .projections_meta import (
    compute_freshness,
    fetch_and_cache_slates,
    get_cached_slates,
    get_status_fields,
    record_fetch_from_csv,
)
from .slate_exclusions import compute_file_fingerprint, get_slate_games_with_status, get_slate_players_with_status, prune_player_exclusions, read_exclusions, write_exclusions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIST = PROJECT_ROOT / "ui" / "dist"

app = FastAPI(title="MLB DFS Optimizer")

# ---------------------------------------------------------------------------
# Run state (single-user local tool — no locking needed beyond a flag)
# ---------------------------------------------------------------------------

_state: dict = {
    "status": "idle",      # idle | running | complete | stopped | error
    "portfolio": None,     # PortfolioResult dict or None
    "error": None,
    "_runner_last": None,  # Last PipelineRunner instance (for upload file writing)
}

_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# X/Twitter notification store
# ---------------------------------------------------------------------------

_notifications: deque = deque(maxlen=25)
_notifications_lock = threading.Lock()

_CHROME_APPS = {'chrome', 'chromium', 'google-chrome', 'chromium-browser', 'Google Chrome'}
_TWITTER_RE = re.compile(
    r'(@\w+|liked your|replied to|retweeted|mentioned you|'
    r'Direct Message|new post|followed you|quote tweeted|X\.com|twitter\.com)',
    re.IGNORECASE,
)
_NOTIFY_HEADER_RE = re.compile(r'member=Notify\b')
_STRING_START_RE = re.compile(r'^\s+string\s+"(.*)$')


def _maybe_commit_notification(str_args: list[str]) -> None:
    if len(str_args) < 4:
        return
    app_name, _icon, summary, body = str_args[0], str_args[1], str_args[2], str_args[3]
    if app_name in _CHROME_APPS and _TWITTER_RE.search(summary + ' ' + body):
        notif = {
            'id': str(uuid.uuid4()),
            'summary': summary,
            'body': body,
            'app_name': app_name,
            'captured_at': time.time(),
        }
        with _notifications_lock:
            _notifications.append(notif)


def _dbus_monitor_loop() -> None:
    while True:
        try:
            proc = subprocess.Popen(
                [
                    'dbus-monitor', '--session', '--monitor',
                    "type='method_call',interface='org.freedesktop.Notifications',member='Notify'",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            in_notify = False
            str_args: list[str] = []
            in_multiline: bool = False
            multiline_buf: list[str] = []

            for line in proc.stdout:
                line_s = line.rstrip('\n')

                # Accumulate multi-line string values
                if in_multiline:
                    if line_s.endswith('"'):
                        multiline_buf.append(line_s[:-1])
                        val = '\n'.join(multiline_buf)
                        in_multiline = False
                        if len(str_args) < 4:
                            str_args.append(val)
                            if len(str_args) == 4:
                                _maybe_commit_notification(str_args)
                                str_args = []
                                in_notify = False
                    else:
                        multiline_buf.append(line_s)
                    continue

                if not in_notify:
                    if _NOTIFY_HEADER_RE.search(line_s):
                        in_notify = True
                        str_args = []
                    continue

                m = _STRING_START_RE.match(line_s)
                if m and len(str_args) < 4:
                    content = m.group(1)
                    if content.endswith('"'):
                        # Single-line string
                        str_args.append(content[:-1])
                        if len(str_args) == 4:
                            _maybe_commit_notification(str_args)
                            str_args = []
                            in_notify = False
                    else:
                        # Start of multi-line string
                        in_multiline = True
                        multiline_buf = [content]
        except Exception:
            time.sleep(5)


def _portfolio_csv_path(platform_val: str | None = None) -> Path:
    cfg = read_config()
    if platform_val is None:
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    output_dir = cfg.paths.output_dir or "outputs"
    base = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    return base / f"portfolio_{platform_val}.csv"


def _portfolio_entries_path(platform_val: str | None = None) -> Path:
    cfg = read_config()
    if platform_val is None:
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    output_dir = cfg.paths.output_dir or "outputs"
    base = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    return base / f"portfolio_entries_{platform_val}.json"


def _load_portfolio_from_csv(platform_val: str) -> list[dict] | None:
    """Load a persisted portfolio from a platform-specific CSV. Returns None if unavailable."""
    import pandas as pd

    try:
        path = _portfolio_csv_path(platform_val)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        portfolio: list[dict] = []
        for lineup_idx, group in df.groupby("lineup", sort=True):
            first = group.iloc[0]
            players = [
                {
                    "player_id": int(row["player_id"]),
                    "name": str(row["name"]),
                    "position": str(row["position"]),
                    "team": str(row["team"]),
                    "salary": int(row["salary"]),
                    "mean": float(row["mean"]) if "mean" in row and pd.notna(row["mean"]) else None,
                    "slot": int(row["slot"]) if "slot" in row and pd.notna(row["slot"]) else None,
                    "slot_confirmed": bool(row["slot_confirmed"]) if "slot_confirmed" in row and pd.notna(row["slot_confirmed"]) else False,
                }
                for _, row in group.iterrows()
            ]
            portfolio.append({
                "lineup_index": int(lineup_idx),
                "p_hit_target": float(first["p_hit_target"]),
                "lineup_salary": int(first["lineup_salary"]),
                "players": players,
            })
        if not portfolio:
            return None
        meta_path = _portfolio_entries_path(platform_val)
        if meta_path.exists():
            with open(meta_path) as mf:
                entry_meta = json.load(mf)
            for lr in portfolio:
                info = entry_meta.get(str(lr["lineup_index"]))
                if info:
                    lr.update(info)
        return portfolio
    except Exception:
        return None  # corrupt or missing CSV — return nothing


@app.on_event("startup")
def _load_persisted_portfolio() -> None:
    """Restore the last portfolio from the current-platform CSV so the UI shows it after restart."""
    try:
        cfg = read_config()
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
        portfolio = _load_portfolio_from_csv(platform_val)
        if portfolio:
            _state["portfolio"] = portfolio
            _state["status"] = "complete"
    except Exception:
        pass  # corrupt or missing config/CSV — start fresh


@app.on_event("startup")
def _start_dbus_monitor() -> None:
    threading.Thread(target=_dbus_monitor_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config() -> AppConfig:
    return read_config()


@app.post("/api/config")
def post_config(cfg: AppConfig) -> AppConfig:
    write_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Notification endpoints
# ---------------------------------------------------------------------------

@app.get("/api/notifications")
def get_notifications():
    with _notifications_lock:
        items = list(_notifications)
    items.sort(key=lambda n: n['captured_at'], reverse=True)
    return items


@app.delete("/api/notifications/{notification_id}")
def delete_notification(notification_id: str):
    with _notifications_lock:
        before = len(_notifications)
        keep = [n for n in _notifications if n['id'] != notification_id]
        _notifications.clear()
        _notifications.extend(keep)
    if len(keep) == before:
        raise HTTPException(404, detail="Not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Twitter lineup endpoints
# ---------------------------------------------------------------------------

@app.post("/api/twitter-lineups/parse")
def parse_twitter_lineup(req: TwitterLineupParseRequest) -> TwitterLineupParseResponse:
    team, raw_slots = parse_notification_body(req.body)
    if team is None:
        return TwitterLineupParseResponse(
            team=None,
            notification_id=req.notification_id,
            slots=[],
            team_in_slate=False,
            warning="Team name not recognized",
        )

    # Load all slate hitters for that team (include excluded players — exclusion ≠ slot confirmation)
    slate_df = _load_slate_df()
    team_hitters: list[dict] = []
    team_in_slate = False
    if slate_df is not None:
        rows = slate_df[
            (slate_df["team"] == team) & (slate_df["position"] != "P")
        ]
        team_in_slate = not rows.empty
        team_hitters = [
            {
                "player_id": int(r["player_id"]),
                "name": str(r["name"]),
                "team": str(r["team"]),
                "position": str(r["position"]),
                "salary": int(r["salary"]),
            }
            for _, r in rows.iterrows()
        ]

    warning: str | None = None
    if not team_in_slate:
        warning = f"{team} not found on the current slate"

    parsed_slots: list[ParsedSlot] = []
    for raw in raw_slots:
        candidate_dicts = match_player_name(raw["name"], team_hitters)
        matches = [
            PlayerMatch(
                player_id=c["player_id"],
                name=c["name"],
                team=c["team"],
                position=c["position"],
                salary=c["salary"],
                match_confidence=c["match_confidence"],
            )
            for c in candidate_dicts
        ]
        parsed_slots.append(ParsedSlot(
            slot=raw["slot"],
            raw_name=raw["name"],
            position=raw["position"],
            matches=matches,
        ))

    return TwitterLineupParseResponse(
        team=team,
        notification_id=req.notification_id,
        slots=parsed_slots,
        team_in_slate=team_in_slate,
        warning=warning,
    )


@app.get("/api/twitter-lineups")
def get_twitter_lineups() -> list[TwitterLineupRecord]:
    return [TwitterLineupRecord(**l) for l in load_twitter_lineups()]


@app.post("/api/twitter-lineups")
def save_twitter_lineup(req: TwitterLineupSaveRequest) -> TwitterLineupRecord:
    slots = [s.model_dump() for s in req.slots]
    record = upsert_twitter_lineup(req.team, req.notification_id, slots)
    return TwitterLineupRecord(**record)


@app.delete("/api/twitter-lineups/{team}")
def remove_twitter_lineup(team: str):
    found = delete_twitter_lineup(team)
    if not found:
        raise HTTPException(404, detail="No confirmed lineup for that team")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Slate game/team exclusion endpoints
# ---------------------------------------------------------------------------

def _resolve_proj_path(cfg: AppConfig) -> Path:
    """Return the platform-appropriate projections CSV path."""
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    if platform_val == "fanduel":
        path_str = (getattr(cfg.paths, "fd_projections", None) or "data/processed/projections_fd.csv")
    else:
        path_str = cfg.paths.projections or "data/processed/projections_dk.csv"
    return PROJECT_ROOT / path_str if not Path(path_str).is_absolute() else Path(path_str)


def _resolve_fd_slate_path(cfg: AppConfig) -> str:
    """Return the fd_slate path, auto-discovering a newer file if the configured one is missing.

    If the configured path is absent or points to a non-existent file, scans
    data/raw/ for the most-recent FanDuel-MLB-*.csv and updates config.yaml so
    the new path persists for subsequent calls.  Returns an empty string if no
    FD CSV is found at all.
    """
    from src.ingestion.factory import find_fd_slate
    raw = getattr(cfg.paths, "fd_slate", "") or ""
    if raw:
        p = PROJECT_ROOT / raw if not Path(raw).is_absolute() else Path(raw)
        if p.exists():
            return raw  # configured path is valid — nothing to do
    # Either empty or stale — try auto-discovery.
    discovered = find_fd_slate(str(PROJECT_ROOT / "data/raw"))
    if not discovered:
        return ""
    try:
        rel = str(Path(discovered).relative_to(PROJECT_ROOT))
    except ValueError:
        rel = discovered
    if rel != raw:
        cfg.paths.fd_slate = rel
        write_config(cfg)
    return rel


def _get_slate_file_path() -> Path | None:
    """Return the resolved Path of the current slate CSV, or None if not configured/found."""
    cfg = read_config()
    from src.platforms.base import Platform
    platform = cfg.platform if hasattr(cfg, 'platform') else Platform.DRAFTKINGS
    if platform == Platform.FANDUEL:
        slate_path = _resolve_fd_slate_path(cfg)
    else:
        slate_path = cfg.paths.dk_slate
    if not slate_path:
        return None
    p = PROJECT_ROOT / slate_path if not Path(slate_path).is_absolute() else Path(slate_path)
    return p if p.exists() else None


def _load_slate_df():
    """Parse the configured slate CSV and return a DataFrame (or None)."""
    p = _get_slate_file_path()
    if p is None:
        return None
    cfg = read_config()
    from src.platforms.base import Platform
    from src.ingestion.factory import get_ingestor
    platform = cfg.platform if hasattr(cfg, 'platform') else Platform.DRAFTKINGS
    return get_ingestor(platform, str(p)).get_slate_dataframe()


def _load_slate_games() -> dict[str, str]:
    """Parse the configured slate CSV and return {game_id: ISO_start_time} for unique games."""
    df = _load_slate_df()
    if df is None:
        return {}
    result: dict[str, str] = {}
    has_time = "game_start_time" in df.columns
    for _, row in df.drop_duplicates("game").iterrows():
        game = str(row["game"])
        if not game:
            continue
        result[game] = str(row["game_start_time"]) if has_time and row["game_start_time"] else ""
    return result


@app.get("/api/slate/games")
def get_slate_games() -> SlateGamesResponse:
    game_times = _load_slate_games()
    if not game_times:
        return SlateGamesResponse(slate_id="", games=[], excluded_player_ids=[])
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    slate_id, game_dicts, excluded_player_ids = get_slate_games_with_status(game_times, fingerprint)
    return SlateGamesResponse(
        slate_id=slate_id,
        games=[GameStatus(**g) for g in game_dicts],
        excluded_player_ids=excluded_player_ids,
    )


@app.post("/api/slate/exclusions")
def post_slate_exclusions(update: ExclusionsUpdate) -> SlateGamesResponse:
    fingerprint = compute_file_fingerprint(_get_slate_file_path())

    # Load existing player exclusions so we can prune them
    stored = read_exclusions(update.slate_id, fingerprint)
    existing_player_ids: list[int] = stored.get("excluded_player_ids", [])

    # Prune player exclusions now covered by team/game exclusions
    df = _load_slate_df()
    if df is not None and existing_player_ids:
        all_players = [
            {"player_id": int(r["player_id"]), "team": str(r["team"]), "game": str(r.get("game", ""))}
            for _, r in df.iterrows()
        ]
        existing_player_ids = prune_player_exclusions(
            existing_player_ids,
            set(update.excluded_teams),
            set(update.excluded_games),
            all_players,
        )

    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=update.excluded_teams,
        excluded_games=update.excluded_games,
        excluded_player_ids=existing_player_ids,
    )
    game_times = _load_slate_games()
    if not game_times:
        return SlateGamesResponse(slate_id=update.slate_id, games=[], excluded_player_ids=[])
    slate_id, game_dicts, excluded_player_ids = get_slate_games_with_status(game_times, fingerprint)
    return SlateGamesResponse(
        slate_id=slate_id,
        games=[GameStatus(**g) for g in game_dicts],
        excluded_player_ids=excluded_player_ids,
    )


@app.get("/api/slate/players")
def get_slate_players() -> SlatePlayersResponse:
    df = _load_slate_df()
    if df is None or df.empty:
        return SlatePlayersResponse(slate_id="", players=[])
    games = [g for g in df["game"].dropna().unique().tolist() if g]
    if not games:
        return SlatePlayersResponse(slate_id="", players=[])
    from .slate_exclusions import compute_slate_id
    slate_id = compute_slate_id(games)
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    player_dicts = get_slate_players_with_status(df, slate_id, fingerprint)
    return SlatePlayersResponse(
        slate_id=slate_id,
        players=[PlayerExclusionStatus(**p) for p in player_dicts],
    )


@app.post("/api/slate/player-exclusions")
def post_player_exclusions(update: PlayerExclusionsUpdate) -> SlatePlayersResponse:
    fingerprint = compute_file_fingerprint(_get_slate_file_path())

    stored = read_exclusions(update.slate_id, fingerprint)
    excluded_teams: list[str] = stored.get("excluded_teams", [])
    excluded_games: list[str] = stored.get("excluded_games", [])

    df = _load_slate_df()
    pruned_ids = update.excluded_player_ids
    if df is not None and pruned_ids:
        all_players = [
            {"player_id": int(r["player_id"]), "team": str(r["team"]), "game": str(r.get("game", ""))}
            for _, r in df.iterrows()
        ]
        pruned_ids = prune_player_exclusions(pruned_ids, set(excluded_teams), set(excluded_games), all_players)

    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=excluded_teams,
        excluded_games=excluded_games,
        excluded_player_ids=pruned_ids,
    )

    if df is None or df.empty:
        return SlatePlayersResponse(slate_id=update.slate_id, players=[])
    player_dicts = get_slate_players_with_status(df, update.slate_id, fingerprint)
    return SlatePlayersResponse(
        slate_id=update.slate_id,
        players=[PlayerExclusionStatus(**p) for p in player_dicts],
    )


# ---------------------------------------------------------------------------
# Projections endpoints
# ---------------------------------------------------------------------------

@app.get("/api/projections/status")
def projections_status() -> ProjectionsStatus:
    cfg = read_config()
    p = _resolve_proj_path(cfg)
    extra = get_status_fields()
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"

    # Resolve platform-appropriate slate path for freshness check
    slate_path: Path | None = None
    if platform_val == "fanduel":
        fd_raw = _resolve_fd_slate_path(cfg)
        if fd_raw:
            _fp = PROJECT_ROOT / fd_raw if not Path(fd_raw).is_absolute() else Path(fd_raw)
            if _fp.exists():
                slate_path = _fp
    else:
        dk_raw = cfg.paths.dk_slate
        if dk_raw:
            _dp = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
            if _dp.exists():
                slate_path = _dp

    if not p.exists():
        return ProjectionsStatus(exists=False, path=str(p.relative_to(PROJECT_ROOT)), **extra)

    stat = p.stat()
    age = time.time() - stat.st_mtime

    row_count = None
    try:
        import pandas as pd
        df = pd.read_csv(p)
        row_count = len(df)
    except Exception:
        pass

    is_fresh = (
        compute_freshness(slate_path, p, platform=platform_val)
        if slate_path is not None
        else None
    )

    status = ProjectionsStatus(
        exists=True,
        path=str(p.relative_to(PROJECT_ROOT)),
        last_modified=stat.st_mtime,
        age_seconds=age,
        row_count=row_count,
        is_fresh=is_fresh,
        **extra,
    )
    if status.unconfirmed_count is not None:
        n_twitter = len(get_twitter_overrides())
        status.unconfirmed_count = max(0, status.unconfirmed_count - n_twitter)
    return status


@app.get("/api/projections/unconfirmed")
def projections_unconfirmed():
    cfg = read_config()
    p = _resolve_proj_path(cfg)
    if not p.exists():
        return {"player_ids": []}
    try:
        import pandas as pd
        df = pd.read_csv(p)
        if "slot_confirmed" not in df.columns or "player_id" not in df.columns:
            return {"player_ids": []}
        unconfirmed = df[~df["slot_confirmed"].astype(bool)]["player_id"].tolist()
        twitter_confirmed = set(get_twitter_overrides().keys())
        unconfirmed = [int(x) for x in unconfirmed if int(x) not in twitter_confirmed]
        return {"player_ids": unconfirmed}
    except Exception:
        return {"player_ids": []}


@app.get("/api/projections/players")
def projections_players():
    import pandas as pd
    cfg = read_config()
    proj_path = _resolve_proj_path(cfg)
    if not proj_path.exists():
        return []
    slate_df = _load_slate_df()
    if slate_df is None or slate_df.empty:
        return []
    try:
        proj_df = pd.read_csv(proj_path)
        if proj_df.empty:
            return []
        for df in (slate_df, proj_df):
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        slate_sub = slate_df[["player_id", "name", "position", "team", "salary"]]
        proj_sub  = proj_df[["player_id", "mean", "lineup_slot", "slot_confirmed"]]
        merged = slate_sub.merge(proj_sub, on="player_id", how="inner")
        result = []
        for _, row in merged.iterrows():
            slot_raw = row["lineup_slot"]
            result.append({
                "player_id":      int(row["player_id"]),
                "name":           str(row["name"]),
                "position":       str(row["position"]),
                "team":           str(row["team"]),
                "salary":         int(row["salary"]),
                "slot":           int(slot_raw) if pd.notna(slot_raw) else None,
                "slot_confirmed": bool(row["slot_confirmed"]),
                "mean":           float(row["mean"]),
            })
        return result
    except Exception:
        return []


@app.get("/api/projections/slates")
async def projections_slates() -> SlateListResponse:
    cfg = read_config()
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    site_id = 2 if platform_val == "fanduel" else 1  # 1=DK, 2=FD

    # Resolve platform-appropriate slate path
    slate_raw: str = ""
    if platform_val == "fanduel":
        slate_raw = _resolve_fd_slate_path(cfg)
    else:
        slate_raw = cfg.paths.dk_slate

    if not slate_raw:
        return SlateListResponse(date=None, slates=[])
    slate_path = PROJECT_ROOT / slate_raw if not Path(slate_raw).is_absolute() else Path(slate_raw)
    if not slate_path.exists():
        return SlateListResponse(date=None, slates=[])

    cached = get_cached_slates(slate_path, site_id=site_id, platform=platform_val)
    if cached is not None:
        return SlateListResponse(
            date=None,
            slates=[SlateOption(**s) for s in cached],
        )

    try:
        slate_date, options = fetch_and_cache_slates(
            slate_path, site_id=site_id, platform=platform_val
        )
    except Exception:
        return SlateListResponse(date=None, slates=[])

    return SlateListResponse(
        date=slate_date,
        slates=[SlateOption(**s) for s in options],
    )


@app.get("/api/projections/fetch")
async def projections_fetch(request: Request):
    import re as _re
    import pandas as pd

    if _state["status"] == "running":
        raise HTTPException(409, "A pipeline run is already in progress")

    cfg = read_config()
    proj_path = _resolve_proj_path(cfg)
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"

    # Resolve DK path (used for partial-fetch game filtering — DK only for now)
    dk_raw = cfg.paths.dk_slate
    dk_path: Path | None = None
    if dk_raw:
        _dp = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
        if _dp.exists():
            dk_path = _dp

    # Resolve the canonical slate path for metadata recording
    if platform_val == "fanduel":
        fd_raw = _resolve_fd_slate_path(cfg)
        _slate_path_for_meta: Path | None = None
        if fd_raw:
            _fp2 = PROJECT_ROOT / fd_raw if not Path(fd_raw).is_absolute() else Path(fd_raw)
            if _fp2.exists():
                _slate_path_for_meta = _fp2
    else:
        _slate_path_for_meta = dk_path

    # Build platform-specific args appended to every script subprocess call.
    # For FD: --platform fanduel [--fd-slate PATH]
    # For DK: --platform draftkings  (default DKSalaries.csv path used by scripts)
    _platform_args: list[str] = ["--platform", platform_val]
    if platform_val == "fanduel":
        fd_raw2 = fd_raw  # already resolved above
        if fd_raw2:
            _platform_args += ["--fd-slate", str(
                PROJECT_ROOT / fd_raw2 if not Path(fd_raw2).is_absolute() else Path(fd_raw2)
            )]
    elif dk_path is not None:
        _platform_args += ["--dk-slate", str(dk_path)]

    # --- Parse games filter from query param ---------------------------------
    # exclude_games=ARI@PHI,NYM@STL means "only fetch the other games and merge
    # the result into the existing projections.csv" (partial fetch mode).
    exclude_games_raw = request.query_params.get("exclude_games", "").strip()
    excluded_pairs: set[tuple[str, str]] = set()
    if exclude_games_raw:
        for token in exclude_games_raw.split(","):
            token = token.strip().upper()
            if "@" in token:
                away, home = token.split("@", 1)
                excluded_pairs.add((away, home))

    # Resolve which games are actually in this slate (from DK CSV) and which
    # to include in the fetch.  Only used when excluded_pairs is non-empty.
    included_pairs: set[tuple[str, str]] = set()
    included_teams: set[str] = set()
    included_pids:  set[int] = set()
    id_to_team: dict[int, str] = {}

    if excluded_pairs and dk_path is not None:
        try:
            dk_gi = pd.read_csv(dk_path, usecols=["ID", "TeamAbbrev", "Game Info"])
            id_to_team = {int(r["ID"]): str(r["TeamAbbrev"]) for _, r in dk_gi.iterrows()}
            all_pairs: set[tuple[str, str]] = set()
            for gi in dk_gi["Game Info"].dropna().unique():
                m = _re.match(r"(\w+)@(\w+)\s", str(gi).strip())
                if m:
                    all_pairs.add((m.group(1).upper(), m.group(2).upper()))
            included_pairs = all_pairs - excluded_pairs
            included_teams = {t for a, h in included_pairs for t in (a, h)}
            included_pids  = {
                int(r["ID"]) for _, r in dk_gi.iterrows()
                if str(r["TeamAbbrev"]) in included_teams
            }
        except Exception:
            pass  # fall back to full fetch if DK CSV parse fails

    is_partial = bool(excluded_pairs) and proj_path.exists()

    async def _stream():
        source = (cfg.paths.projections_source or "rotowire").strip().lower()
        is_dff_preferred    = source == "dailyfantasyfuel"
        is_market_preferred = source == "market_odds"
        if is_market_preferred:
            preferred_label = "Market Odds (CrazyNinjaOdds)"
            fallback_label  = "RotoWire"
        elif is_dff_preferred:
            preferred_label = "Daily Fantasy Fuel"
            fallback_label  = "RotoWire"
        else:
            preferred_label = "RotoWire"
            fallback_label  = "Daily Fantasy Fuel"

        python     = PROJECT_ROOT / "venv" / "bin" / "python"
        dff_script = PROJECT_ROOT / "scripts" / "fetch_dff_projections.py"
        rw_script  = PROJECT_ROOT / "scripts" / "fetch_rotowire_projections.py"
        mo_script  = PROJECT_ROOT / "scripts" / "fetch_market_odds_projections.py"
        dff_out    = proj_path.parent / "projections_dff.csv"
        rw_out     = proj_path.parent / "projections_rw.csv"
        mo_out     = proj_path.parent / "projections_mo.csv"
        mo_sidecar   = proj_path.parent / "projections_mo_fallback.json"
        mo_caps_path = proj_path.parent / "projections_mo_caps.json"

        def _log(msg: str) -> str:
            return f"data: {json.dumps({'type': 'log', 'line': msg, 'timestamp': int(time.time() * 1000)})}\n\n"

        # Clean up any stale temp files left by a prior incomplete fetch.
        for _p in (dff_out, rw_out, mo_out, mo_sidecar, mo_caps_path):
            try:
                _p.unlink(missing_ok=True)
            except Exception:
                pass

        if is_partial:
            n_excl = len(excluded_pairs)
            n_incl = len(included_pairs) if included_pairs else "?"
            yield _log(
                f"--- Partial fetch: {n_incl} game(s) included, {n_excl} excluded — "
                f"will merge into existing projections ---"
            )

        # Helper: read fallback reasons written by the MO fetch script sidecar.
        def _read_mo_sidecar() -> dict[int, str]:
            if not mo_sidecar.exists():
                return {}
            try:
                raw = json.loads(mo_sidecar.read_text())
                return {int(k): v for k, v in raw.items()}
            except Exception:
                return {}

        # Helper: read hard-cap warnings written by the MO fetch script.
        def _read_mo_caps() -> dict[int, list[str]]:
            if not mo_caps_path.exists():
                return {}
            try:
                raw = json.loads(mo_caps_path.read_text())
                return {int(k): v for k, v in raw.items()}
            except Exception:
                return {}

        # Helper: build the id_to_team map if not already built (non-partial paths).
        def _ensure_id_to_team() -> dict[int, str]:
            if id_to_team:
                return id_to_team
            if dk_path is not None:
                try:
                    dk_df2 = pd.read_csv(dk_path, usecols=["ID", "TeamAbbrev"])
                    return {int(r["ID"]): str(r["TeamAbbrev"]) for _, r in dk_df2.iterrows()}
                except Exception:
                    pass
            return {}

        # Helper: return teams whose total batter projection mean is below threshold.
        def _low_team_projections(
            merged_df: "pd.DataFrame",
            itm: dict[int, str],
            threshold: float = 40.0,
        ) -> list[dict]:
            if merged_df.empty or "mean" not in merged_df.columns:
                return []
            if "lineup_slot" in merged_df.columns:
                batters = merged_df[merged_df["lineup_slot"] != 10].copy()
            else:
                batters = merged_df.copy()
            if batters.empty:
                return []
            batters["_team"] = batters["player_id"].apply(lambda pid: itm.get(int(pid), ""))
            sums = batters.groupby("_team")["mean"].sum()
            low = [
                {"team": team, "total": round(float(total), 2)}
                for team, total in sums.items()
                if team and total < threshold
            ]
            return sorted(low, key=lambda x: x["total"])

        # Helper: write final merged_df to proj_path, handling partial merge.
        def _write_proj(merged_df: "pd.DataFrame") -> None:
            out_cols = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
            result = merged_df[[c for c in out_cols if c in merged_df.columns]]
            if is_partial and proj_path.exists():
                existing = pd.read_csv(proj_path)
                # Purge ALL players from the fetched games before merging in the new
                # result.  Using only result["player_id"] would leave stale rows for
                # players who dropped out of the lineup since the last fetch (e.g. a
                # player whose slot is now unconfirmed and absent from the new result).
                purge_ids = included_pids or (
                    set(result["player_id"].tolist()) if "player_id" in result.columns else set()
                )
                other = existing[~existing["player_id"].isin(purge_ids)] if not existing.empty else pd.DataFrame()
                out_cols2 = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
                other = other[[c for c in out_cols2 if c in other.columns]]
                result = pd.concat([other, result], ignore_index=True)
            result = result.sort_values("mean", ascending=False).reset_index(drop=True)
            result.to_csv(proj_path, index=False)

        # ---- Market Odds path -----------------------------------------------
        if is_market_preferred:
            returncode = 0
            proj_written = False
            result_event: str | None = None

            # The entire fetch is one try/finally so that cleanup and metadata
            # update run reliably even when the client disconnects mid-stream.
            # GeneratorExit (thrown by aclose() on disconnect) is not caught by
            # "except Exception", but finally always runs — that's the guarantee.
            # No yields occur inside the merge step, so once it starts it
            # completes atomically before the generator can be suspended again.
            try:
                # Step 1: RotoWire is always required (player pool + lineup slots)
                yield _log("--- Fetching RotoWire projections (player pool) ---")
                proc = await asyncio.create_subprocess_exec(
                    str(python), str(rw_script), "--output", str(rw_out),
                    *_platform_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                )
                assert proc.stdout is not None
                async for line in proc.stdout:
                    if await request.is_disconnected():
                        proc.kill()
                        break
                    yield _log(line.decode().rstrip())
                rw_rc = await proc.wait()

                if rw_rc != 0:
                    returncode = rw_rc
                else:
                    # Step 2: Market odds — pass --games filter when doing a partial fetch
                    yield _log("--- Fetching Market Odds (CrazyNinjaOdds) ---")
                    mo_cmd = [str(python), str(mo_script), "--output", str(mo_out),
                              "--rw-output", str(rw_out), *_platform_args]
                    if included_pairs:
                        games_arg = ",".join(f"{a}@{h}" for a, h in included_pairs)
                        mo_cmd += ["--games", games_arg]
                    proc2 = await asyncio.create_subprocess_exec(
                        *mo_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=str(PROJECT_ROOT),
                    )
                    assert proc2.stdout is not None
                    async for line in proc2.stdout:
                        if await request.is_disconnected():
                            proc2.kill()
                            break
                        yield _log(line.decode().rstrip())
                    mo_rc = await proc2.wait()

                    # Step 3: Merge — no yields from here until after finally
                    rw_df = pd.read_csv(rw_out) if rw_out.exists() else pd.DataFrame()
                    mo_df = (
                        pd.read_csv(mo_out)
                        if (mo_out.exists() and mo_rc == 0)
                        else pd.DataFrame()
                    )
                    # Normalize player_id to int64 — pandas 3 may infer str dtype
                    for _df in (rw_df, mo_df):
                        if not _df.empty and "player_id" in _df.columns:
                            _df["player_id"] = pd.to_numeric(_df["player_id"], errors="coerce").astype("Int64")
                    # When partial: restrict the RW pool to only the included games
                    if is_partial and included_pids and not rw_df.empty:
                        rw_df = rw_df[rw_df["player_id"].isin(included_pids)]

                    if rw_df.empty:
                        returncode = 1
                        result_event = _log("Error: RotoWire produced no usable data.")
                    else:
                        pool = rw_df.copy()
                        fallback_players: list[dict] = []
                        mo_sidecar_reasons = _read_mo_sidecar()
                        _itm = _ensure_id_to_team()

                        if not mo_df.empty and {"player_id", "mean", "std_dev"}.issubset(mo_df.columns):
                            pref_lookup = (
                                mo_df.drop_duplicates("player_id")
                                .set_index("player_id")[["mean", "std_dev"]]
                            )
                            pool = pool.merge(
                                pref_lookup.rename(columns={"mean": "_pm", "std_dev": "_ps"}),
                                on="player_id",
                                how="left",
                            )
                            has_pref = pool["_pm"].notna()
                            pool.loc[has_pref, "mean"]    = pool.loc[has_pref, "_pm"]
                            pool.loc[has_pref, "std_dev"] = pool.loc[has_pref, "_ps"]
                            pool = pool.drop(columns=["_pm", "_ps"])
                            # Scale RotoWire fallback projections for hitters down to 80%
                            # (RotoWire tends to be more optimistic than market odds for batters)
                            is_hitter_fallback = ~has_pref & (pool.get("lineup_slot", 0) != 10)
                            pool.loc[is_hitter_fallback, "mean"]    *= 0.8
                            pool.loc[is_hitter_fallback, "std_dev"] *= 0.8
                            fallback_rows = pool.loc[~has_pref] if "name" in pool.columns else pd.DataFrame()

                            if not fallback_rows.empty:
                                for _, row in fallback_rows.iterrows():
                                    pid_val = int(row["player_id"]) if "player_id" in fallback_rows.columns else 0
                                    is_pitcher = bool(row.get("lineup_slot") == 10) if "lineup_slot" in fallback_rows.columns else False
                                    fallback_players.append({
                                        "name": row["name"],
                                        "team": _itm.get(pid_val, ""),
                                        "reason": mo_sidecar_reasons.get(pid_val, ""),
                                        "player_id": pid_val,
                                        "is_pitcher": is_pitcher,
                                    })
                        else:
                            result_event = _log(
                                "Warning: Market odds unavailable; using RotoWire projections for all players."
                            )

                        _write_proj(pool)
                        proj_written = True

                        # Build capped_players list from the caps sidecar + pool.
                        mo_cap_data = _read_mo_caps()
                        capped_players: list[dict] = []
                        if mo_cap_data and "name" in pool.columns:
                            pid_to_name = (
                                pool[["player_id", "name"]]
                                .drop_duplicates("player_id")
                                .set_index("player_id")["name"]
                                .to_dict()
                            )
                            for cap_pid, cap_mkts in mo_cap_data.items():
                                if cap_pid in pid_to_name:
                                    capped_players.append({
                                        "name": pid_to_name[cap_pid],
                                        "team": _itm.get(cap_pid, ""),
                                        "markets": cap_mkts,
                                    })

                        low_team_projs = _low_team_projections(pool, _itm)

                        if fallback_players or capped_players or low_team_projs:
                            result_event = f"data: {json.dumps({'type': 'merge_info', 'secondary_source': 'RotoWire', 'count': len(fallback_players), 'players': fallback_players, 'capped_players': capped_players, 'low_team_projections': low_team_projs, 'timestamp': int(time.time() * 1000)})}\n\n"
                        elif result_event is None:
                            result_event = _log("All player projections sourced from Market Odds (CrazyNinjaOdds).")

            except Exception as exc:
                returncode = 1
                result_event = _log(f"Warning: merge error — {exc}")
            finally:
                for p in (rw_out, mo_out, mo_sidecar, mo_caps_path):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                if proj_written:
                    try:
                        record_fetch_from_csv(proj_path, "auto", _slate_path_for_meta)
                    except Exception:
                        pass

            if result_event:
                yield result_event
            yield f"data: {json.dumps({'type': 'done', 'returncode': returncode, 'timestamp': int(time.time() * 1000)})}\n\n"
            return

        # ---- Run preferred + fallback scripts (live-streamed) ---------------
        returncode = 0
        proj_written = False
        result_event2: str | None = None

        try:
            yield _log(f"--- Fetching {preferred_label} projections ---")
            preferred_script = dff_script if is_dff_preferred else rw_script
            preferred_out    = dff_out    if is_dff_preferred else rw_out

            proc = await asyncio.create_subprocess_exec(
                str(python), str(preferred_script), "--output", str(preferred_out),
                *_platform_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            assert proc.stdout is not None
            async for line in proc.stdout:
                if await request.is_disconnected():
                    proc.kill()
                    break
                yield _log(line.decode().rstrip())
            preferred_rc = await proc.wait()

            if preferred_rc != 0:
                returncode = preferred_rc
            else:
                yield _log(f"--- Fetching {fallback_label} projections ---")
                fallback_script = rw_script  if is_dff_preferred else dff_script
                fallback_out    = rw_out     if is_dff_preferred else dff_out

                proc2 = await asyncio.create_subprocess_exec(
                    str(python), str(fallback_script), "--output", str(fallback_out),
                    *_platform_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                )
                assert proc2.stdout is not None
                async for line in proc2.stdout:
                    if await request.is_disconnected():
                        proc2.kill()
                        break
                    yield _log(line.decode().rstrip())
                await proc2.wait()

                # ---- Merge both outputs into final projections.csv (no yields) ---
                dff_df = pd.read_csv(dff_out) if dff_out.exists() else pd.DataFrame()
                rw_df  = pd.read_csv(rw_out)  if rw_out.exists()  else pd.DataFrame()
                # Normalize player_id to int64 — pandas 3 may infer str dtype
                for _df in (dff_df, rw_df):
                    if not _df.empty and "player_id" in _df.columns:
                        _df["player_id"] = pd.to_numeric(_df["player_id"], errors="coerce").astype("Int64")

                # RW pool  — all starters already filtered by the fetch script
                # DFF pool — confirmed batters only (slot_confirmed=True) + all pitchers
                rw_pool = rw_df.copy()

                if not dff_df.empty and {"lineup_slot", "slot_confirmed"}.issubset(dff_df.columns):
                    dff_pitchers = dff_df[dff_df["lineup_slot"] == 10]
                    dff_batters  = dff_df[(dff_df["lineup_slot"] != 10) & dff_df["slot_confirmed"].astype(bool)]
                    dff_pool     = pd.concat([dff_batters, dff_pitchers], ignore_index=True)
                else:
                    dff_pool = dff_df.copy()

                if rw_pool.empty and dff_pool.empty:
                    returncode = 1
                    result_event2 = _log("Error: both projection sources produced no usable data.")
                else:
                    # Union: start with RW starters, append any DFF-pool players not in RW
                    rw_ids    = set(rw_pool["player_id"].tolist()) if not rw_pool.empty else set()
                    dff_extra = dff_pool[~dff_pool["player_id"].isin(rw_ids)] if not dff_pool.empty else pd.DataFrame()
                    pool      = pd.concat([rw_pool, dff_extra], ignore_index=True)

                    # Apply preferred source's mean/std_dev
                    pref_proj_df = dff_df if is_dff_preferred else rw_df

                    _itm2 = _ensure_id_to_team()
                    fallback_players2: list[dict] = []
                    if not pref_proj_df.empty and {"player_id", "mean", "std_dev"}.issubset(pref_proj_df.columns):
                        pref_lookup = (
                            pref_proj_df.drop_duplicates("player_id")
                            .set_index("player_id")[["mean", "std_dev"]]
                        )
                        pool = pool.merge(
                            pref_lookup.rename(columns={"mean": "_pm", "std_dev": "_ps"}),
                            on="player_id",
                            how="left",
                        )
                        has_pref = pool["_pm"].notna()
                        pool.loc[has_pref, "mean"]    = pool.loc[has_pref, "_pm"]
                        pool.loc[has_pref, "std_dev"] = pool.loc[has_pref, "_ps"]
                        pool = pool.drop(columns=["_pm", "_ps"])
                        fallback_rows2 = pool.loc[~has_pref] if "name" in pool.columns else pd.DataFrame()

                        if not fallback_rows2.empty:
                            for _, row in fallback_rows2.iterrows():
                                pid_val = int(row["player_id"]) if "player_id" in fallback_rows2.columns else 0
                                is_pitcher2 = bool(row.get("lineup_slot") == 10) if "lineup_slot" in fallback_rows2.columns else False
                                fallback_players2.append({
                                    "name": row["name"],
                                    "team": _itm2.get(pid_val, ""),
                                    "reason": "",
                                    "player_id": pid_val,
                                    "is_pitcher": is_pitcher2,
                                })

                    # When partial: keep only the included games' players from this fetch
                    if is_partial and included_pids:
                        pool = pool[pool["player_id"].isin(included_pids)]
                        fallback_players2 = [
                            p for p in fallback_players2
                            if _itm2.get(int(p.get("player_id", 0)), "") in included_teams
                        ]

                    _write_proj(pool)
                    proj_written = True

                    low_team_projs2 = _low_team_projections(pool, _itm2)

                    if fallback_players2 or low_team_projs2:
                        result_event2 = f"data: {json.dumps({'type': 'merge_info', 'secondary_source': fallback_label, 'count': len(fallback_players2), 'players': fallback_players2, 'low_team_projections': low_team_projs2, 'timestamp': int(time.time() * 1000)})}\n\n"
                    else:
                        result_event2 = _log(f"All player projections sourced from {preferred_label}.")

        except Exception as exc:
            returncode = 1
            result_event2 = _log(f"Warning: merge error — {exc}")
        finally:
            for p in (dff_out, rw_out):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if proj_written:
                try:
                    record_fetch_from_csv(proj_path, "auto", _slate_path_for_meta)
                except Exception:
                    pass

        if result_event2:
            yield result_event2
        yield f"data: {json.dumps({'type': 'done', 'returncode': returncode, 'timestamp': int(time.time() * 1000)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Pipeline run endpoints
# ---------------------------------------------------------------------------

@app.get("/api/run/status")
def run_status():
    return {"status": _state["status"], "error": _state.get("error")}


@app.post("/api/run/stop")
def stop_run():
    if _state["status"] != "running":
        raise HTTPException(400, "No run in progress")
    _stop_event.set()
    return {"ok": True}


@app.post("/api/run/write_upload")
def write_upload():
    runner = _state.get("_runner_last")
    if runner is None or not hasattr(runner, "_raw_portfolio"):
        raise HTTPException(400, "No portfolio available for upload")
    try:
        paths = runner.write_upload_files()
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"paths": paths}


@app.get("/api/run/stream")
async def run_stream(request: Request):
    if _state["status"] in ("running", "replacing"):
        raise HTTPException(409, "A run is already in progress")

    _state["status"] = "running"
    _state["portfolio"] = None
    _state["error"] = None
    _stop_event.clear()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_cb(stage: str, data: dict) -> None:
        payload = {"stage": stage, "timestamp": int(time.time() * 1000), **data}
        asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

    def run_pipeline() -> None:
        try:
            from .pipeline import PipelineRunner
            runner = PipelineRunner(
                str(PROJECT_ROOT / "config.yaml"),
                progress_cb,
                stop_check=_stop_event.is_set,
            )
            portfolio = runner.run()
            _state["portfolio"] = portfolio
            _state["_runner_last"] = runner
            _state["status"] = "stopped" if _stop_event.is_set() else "complete"
        except Exception as exc:
            _state["status"] = "error"
            _state["error"] = str(exc)
            err_payload = {"stage": "error", "message": str(exc), "timestamp": int(time.time() * 1000)}
            asyncio.run_coroutine_threadsafe(queue.put(err_payload), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    async def _sse_generator():
        asyncio.get_event_loop().run_in_executor(None, run_pipeline)
        while True:
            if await request.is_disconnected():
                break
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(_sse_generator(), media_type="text/event-stream")


@app.post("/api/portfolio/replace/{lineup_index}")
async def replace_lineup_endpoint(lineup_index: int):
    if _state["status"] in ("running", "replacing"):
        raise HTTPException(409, "Cannot replace lineup while a run is in progress")
    runner = _state.get("_runner_last")
    if runner is None or not hasattr(runner, "_raw_portfolio"):
        raise HTTPException(400, "No portfolio available")
    if not hasattr(runner, "_sim_results"):
        raise HTTPException(400, "Simulation results not available — please re-run the portfolio")
    if lineup_index < 1 or lineup_index > len(runner._raw_portfolio):
        raise HTTPException(400, f"Invalid lineup index: {lineup_index}")
    _state["status"] = "replacing"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, runner.replace_lineup, lineup_index)
        _state["portfolio"] = result
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        _state["status"] = "complete"


@app.get("/api/portfolio")
def get_portfolio(platform: str | None = None):
    if platform is not None:
        # Platform explicitly requested — load directly from the platform-specific CSV.
        portfolio = _load_portfolio_from_csv(platform)
        if portfolio is None:
            raise HTTPException(404, f"No portfolio available for platform '{platform}'")
        return portfolio
    # No platform param — fall back to the in-memory portfolio from the last run.
    if _state["portfolio"] is None:
        raise HTTPException(404, "No portfolio available")
    return _state["portfolio"]


# ---------------------------------------------------------------------------
# Static files (React SPA)
# ---------------------------------------------------------------------------

if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")
    app.mount("/team-logos", StaticFiles(directory=str(UI_DIST / "team-logos")), name="team-logos")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes (already handled above)
        index = UI_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(404, "UI not built — run: cd ui && npm run build")
else:
    @app.get("/")
    async def ui_not_built():
        return {"message": "UI not built. Run: cd ui && npm run build"}
