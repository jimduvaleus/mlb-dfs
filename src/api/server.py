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
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config_io import read_config, write_config
from .models import AppConfig, DoubleheaderStatusResponse, ExclusionsUpdate, GameStatus, ParsedSlot, PlayerExclusionStatus, PlayerExclusionsUpdate, PlayerMatch, PlayerProjectionOverridesResponse, PlayerProjectionOverridesUpdate, PortfolioResult, ProjectionsStatus, SlateGamesResponse, SlateListResponse, SlateOption, SlatePlayersResponse, TeamOwnershipReductionsResponse, TeamOwnershipReductionsUpdate, TwitterLineupParseRequest, TwitterLineupParseResponse, TwitterLineupRecord, TwitterLineupSaveRequest, TwitterLineupSlot
from .mlb_schedule import get_doubleheader_teams_cached
from .twitter_lineups import (
    delete_twitter_lineup,
    get_confirmed_team_lineups,
    get_twitter_overrides,
    load_twitter_lineups,
    extract_lineup_team,
    extract_lineup_header_date,
    looks_like_lineup,
    match_player_name,
    parse_notification_body,
    set_twitter_lineup_locked,
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

import logging as _logging
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="MLB DFS Optimizer")

# ---------------------------------------------------------------------------
# Run state (single-user local tool — no locking needed beyond a flag)
# ---------------------------------------------------------------------------

_state: dict = {
    "status": "idle",      # idle | running | complete | stopped | error
    "portfolio": None,     # PortfolioResult dict or None
    "optimal_lineups": None,  # list[LineupResult] or None
    "error": None,
    "_runner_last": None,  # Last PipelineRunner instance (for upload file writing)
}

_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# X/Twitter notification store
# ---------------------------------------------------------------------------

_notifications: deque = deque(maxlen=25)
_notifications_lock = threading.Lock()
_NOTIFICATIONS_PATH = PROJECT_ROOT / "data" / "notifications.json"


def _save_notifications() -> None:
    """Persist current notifications to disk. Must be called while holding _notifications_lock."""
    try:
        _NOTIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _NOTIFICATIONS_PATH.write_text(json.dumps(list(_notifications), indent=2))
    except Exception:
        pass


def _load_notifications() -> None:
    """Load persisted notifications into the deque on startup."""
    try:
        raw = json.loads(_NOTIFICATIONS_PATH.read_text())
        if isinstance(raw, list):
            with _notifications_lock:
                _notifications.clear()
                for n in raw:
                    _notifications.append(n)
    except Exception:
        pass


_CHROME_APPS = {'chrome', 'chromium', 'google-chrome', 'chromium-browser', 'Google Chrome'}
_TWITTER_RE = re.compile(
    r'(@\w+|liked your|replied to|retweeted|mentioned you|'
    r'Direct Message|new post|followed you|quote tweeted|X\.com|twitter\.com)',
    re.IGNORECASE,
)
_NOTIFY_HEADER_RE = re.compile(r'member=Notify\b')
_STRING_START_RE = re.compile(r'^\s+string\s+"(.*)$')
_SCRATCH_RE = re.compile(r'scratch(?:ed)?', re.IGNORECASE)

# Append-only forensic log of every dbus Notify call seen, regardless of the
# app/content filter below. The live _notifications deque only retains undismissed
# items, so it can't answer "did anything arrive around 7:12?" after the fact — this can.
# Cleared whenever a new slate is detected — only useful for diagnosing the slate in progress.
_RAW_NOTIF_LOG_PATH = PROJECT_ROOT / "data" / "notification_log.jsonl"
_RAW_NOTIF_LOG_FP_PATH = PROJECT_ROOT / "data" / "notification_log.fingerprint"
_raw_notif_log_lock = threading.Lock()


def _clear_notification_log_if_new_slate(fp: str) -> None:
    if not fp:
        return
    try:
        stored = _RAW_NOTIF_LOG_FP_PATH.read_text().strip() if _RAW_NOTIF_LOG_FP_PATH.exists() else ""
    except Exception:
        stored = ""
    if fp == stored:
        return
    try:
        with _raw_notif_log_lock:
            _RAW_NOTIF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _RAW_NOTIF_LOG_PATH.write_text("")
            _RAW_NOTIF_LOG_FP_PATH.write_text(fp)
    except Exception:
        pass


def _log_raw_notification(app_name: str, summary: str, body: str, committed: bool) -> None:
    entry = {
        "logged_at": time.time(),
        "app_name": app_name,
        "summary": summary,
        "body": body,
        "committed": committed,
    }
    try:
        with _raw_notif_log_lock:
            _RAW_NOTIF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _RAW_NOTIF_LOG_PATH.open("a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _maybe_commit_notification(str_args: list[str]) -> None:
    if len(str_args) < 4:
        return
    app_name, _icon, summary, body = str_args[0], str_args[1], str_args[2], str_args[3]
    committed = app_name in _CHROME_APPS and bool(_TWITTER_RE.search(summary + ' ' + body))
    _log_raw_notification(app_name, summary, body, committed)
    if committed:
        notif = {
            'id': str(uuid.uuid4()),
            'summary': summary,
            'body': body,
            'app_name': app_name,
            'captured_at': time.time(),
        }
        with _notifications_lock:
            _notifications.append(notif)
            _save_notifications()

        # Scratch alerts are time-critical from 10 min before slate lock onward —
        # email the full notification text in addition to the in-app entry above.
        if _SCRATCH_RE.search(summary + ' ' + body) and _slate_first_pitch_started():
            from .email_notify import send_notification_email
            full_text = f"{summary}\n\n{body}" if body else summary
            threading.Thread(
                target=send_notification_email,
                args=(f"Scratch alert: {summary}", full_text),
                daemon=True,
            ).start()


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


def _upload_order_players(players: list[dict]) -> list[dict]:
    """Reorder serialized players into DK upload column order so persisted
    portfolios written before the ordering change display consistently with
    the upload_*.csv files. No-op when assignment doesn't apply (FD rosters)."""
    import pandas as pd
    from .dk_entries import assign_players_to_slots
    try:
        sub = pd.DataFrame({
            "player_id": [p["player_id"] for p in players],
            "name": [p.get("name", "") for p in players],
            "position": [str(p["position"]).split("/")[0] for p in players],
            "eligible_positions": [str(p["position"]).split("/") for p in players],
        })
        ordered = assign_players_to_slots([p["player_id"] for p in players], sub)
        by_id = {p["player_id"]: p for p in players}
        return [by_id[pid] for pid in ordered]
    except Exception:
        return players


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
                "mean_ev": float(first["mean_ev"]) if "mean_ev" in first and pd.notna(first["mean_ev"]) else None,
                "players": _upload_order_players(players),
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


def _optimal_lineups_path(platform_val: str | None = None) -> Path:
    cfg = read_config()
    if platform_val is None:
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    output_dir = cfg.paths.output_dir or "outputs"
    base = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    return base / f"optimal_lineups_{platform_val}.json"


def _load_optimal_lineups_from_json(platform_val: str) -> list[dict] | None:
    """Load persisted optimal lineups, returning None if missing or slate fingerprint is stale."""
    try:
        path = _optimal_lineups_path(platform_val)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        stored_fp = data.get("slate_fingerprint", "")
        if stored_fp:
            current_fp = compute_file_fingerprint(_get_slate_file_path())
            if current_fp != stored_fp:
                return None  # slate changed — discard
        lineups = data.get("lineups")
        if not lineups:
            return None
        for lr in lineups:
            if isinstance(lr.get("players"), list):
                lr["players"] = _upload_order_players(lr["players"])
        return lineups
    except Exception:
        return None


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
def _load_persisted_optimal_lineups() -> None:
    """Restore the last optimal lineups from JSON (if slate fingerprint still matches)."""
    try:
        cfg = read_config()
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
        lineups = _load_optimal_lineups_from_json(platform_val)
        if lineups:
            _state["optimal_lineups"] = lineups
    except Exception:
        pass


@app.on_event("startup")
def _reset_stale_twitter_lineups() -> None:
    """Invalidate any locked lineups that belong to a different slate.

    Runs eagerly on startup so a DKSalaries.csv swap takes effect immediately
    rather than waiting for the first API request.
    """
    try:
        fp = _slate_fingerprint()
        if fp:
            load_twitter_lineups(fp)  # triggers clear-and-save if fingerprint changed
            _clear_notification_log_if_new_slate(fp)
    except Exception:
        pass


@app.on_event("startup")
def _start_dbus_monitor() -> None:
    _load_notifications()
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
    # A notification belongs to the current slate if it was captured after the slate
    # file was last modified.  Notifications captured before the slate was replaced
    # contain lineup assignments for a different day and must not auto-confirm.
    slate_path = _get_slate_file_path()
    slate_mtime: float = slate_path.stat().st_mtime if slate_path else 0.0
    from datetime import date as _date
    today = _date.today()

    def _is_current_slate(n: dict) -> bool:
        if n.get('captured_at', 0) >= slate_mtime:
            return True
        # Fallback: if the lineup header explicitly carries today's date, treat as
        # current-slate even if the notification arrived just before the slate was placed.
        hd = extract_lineup_header_date(n.get('body', ''))
        return hd is not None and hd == (today.month, today.day)

    # Teams playing in the currently-loaded slate, so out-of-slate lineup notifications
    # can be excluded from the count returned to the client instead of being counted and
    # then dismissed a moment later once the Slate tab mounts and re-derives this set.
    slate_teams = {t for game in _load_slate_games() for t in game.split('@')}

    result = []
    for n in items:
        lineup_team = extract_lineup_team(n.get('body', ''))
        result.append({
            **n,
            'could_be_lineup': looks_like_lineup(n.get('body', '')),
            'lineup_team': lineup_team,
            'is_current_slate': _is_current_slate(n),
            'lineup_team_in_slate': not lineup_team or not slate_teams or lineup_team in slate_teams,
        })
    return result


@app.delete("/api/notifications/{notification_id}")
def delete_notification(notification_id: str):
    with _notifications_lock:
        before = len(_notifications)
        keep = [n for n in _notifications if n['id'] != notification_id]
        _notifications.clear()
        _notifications.extend(keep)
        if len(keep) < before:
            _save_notifications()
    if len(keep) == before:
        raise HTTPException(404, detail="Not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Twitter lineup endpoints
# ---------------------------------------------------------------------------

def _slate_fingerprint() -> str:
    """Return the current slate file fingerprint, or '' if no slate is configured."""
    return compute_file_fingerprint(_get_slate_file_path())


@app.post("/api/twitter-lineups/parse")
def parse_twitter_lineup(req: TwitterLineupParseRequest) -> TwitterLineupParseResponse:
    team, raw_slots, is_updated = parse_notification_body(req.body)
    if team is None:
        return TwitterLineupParseResponse(
            team=None,
            notification_id=req.notification_id,
            slots=[],
            team_in_slate=False,
            warning="Team name not recognized",
            is_updated=False,
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
        is_updated=is_updated,
    )


@app.get("/api/twitter-lineups")
def get_twitter_lineups() -> list[TwitterLineupRecord]:
    doubleheader_teams, _ = get_doubleheader_teams_cached(date.today().isoformat())
    records = []
    for l in load_twitter_lineups(_slate_fingerprint()):
        l = {**l, "needs_game_confirmation": l.get("team") in doubleheader_teams}
        records.append(TwitterLineupRecord(**l))
    return records


def _emit_lineup_diff_notification(
    team: str,
    summary: str,
    old_slots: list[dict],
    new_slots: list[TwitterLineupSlot],
) -> None:
    """Post an In/Out notification for the batter diff between two slot lists."""
    old_ids = {s["player_id"] for s in old_slots if s.get("player_id") is not None}
    new_ids = {s.player_id for s in new_slots if s.player_id is not None}
    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    if not (added_ids or removed_ids):
        return

    name_map: dict[int, str] = {}
    slate_df = _load_slate_df()
    if slate_df is not None:
        for pid in added_ids | removed_ids:
            row = slate_df[slate_df["player_id"] == pid]
            if not row.empty:
                name_map[pid] = str(row.iloc[0]["name"])
    for s in new_slots:
        if s.player_id is not None and s.player_id not in name_map:
            name_map[s.player_id] = s.name
    for s in old_slots:
        pid = s.get("player_id")
        if pid is not None and pid not in name_map:
            name_map[pid] = s.get("name", str(pid))

    parts: list[str] = []
    if added_ids:
        parts.append("In: " + ", ".join(name_map.get(p, str(p)) for p in sorted(added_ids)))
    if removed_ids:
        parts.append("Out: " + ", ".join(name_map.get(p, str(p)) for p in sorted(removed_ids)))

    diff_notif = {
        "id": str(uuid.uuid4()),
        "summary": summary,
        "body": " | ".join(parts),
        "app_name": "system",
        "captured_at": time.time(),
    }
    with _notifications_lock:
        _notifications.append(diff_notif)
        _save_notifications()

    from .email_notify import send_notification_email
    threading.Thread(
        target=send_notification_email, args=(summary, diff_notif["body"]), daemon=True
    ).start()


_LINEUP_DIFF_WINDOW = timedelta(minutes=10)


def _slate_first_pitch_started() -> bool:
    """True from 10 minutes before the slate's earliest game start onward.

    The 10-minute lead-in covers lineups posted just ahead of lock, which are
    just as noteworthy a diff against the best guess as ones posted after the
    first game has actually begun.
    """
    starts = [t for t in _load_slate_games().values() if t]
    if not starts:
        return False
    try:
        earliest = min(datetime.fromisoformat(t) for t in starts)
    except ValueError:
        return False
    return datetime.now() >= earliest - _LINEUP_DIFF_WINDOW


def _best_guess_lineup_slots(team: str) -> list[dict] | None:
    """Return the current best-guess (pre-confirmation) starting batters for a team.

    Sourced from the projections CSV's `lineup_slot` column, which holds the
    most recent RW/DFF-fetched lineup — this hasn't been overwritten yet when
    a Twitter/Underdog notification is first being saved for the team, so it
    reflects what the pipeline was using right up until this confirmation.
    Returns None if there's no projections file or no slotted batters.
    """
    import pandas as pd
    cfg = read_config()
    proj_path = _resolve_proj_path(cfg)
    if not proj_path.exists():
        return None
    try:
        proj_df = pd.read_csv(proj_path)
    except Exception:
        return None
    if "lineup_slot" not in proj_df.columns or "player_id" not in proj_df.columns:
        return None

    slate_df = _load_slate_df()
    if slate_df is None:
        return None
    team_pids = set(
        slate_df.loc[(slate_df["team"] == team) & (slate_df["position"] != "P"), "player_id"]
    )
    if not team_pids:
        return None

    starters = proj_df[
        proj_df["player_id"].isin(team_pids)
        & proj_df["lineup_slot"].notna()
        & proj_df["lineup_slot"].between(1, 9)
    ]
    if starters.empty:
        return None
    return [
        {"player_id": int(r["player_id"]), "name": str(r.get("name", ""))}
        for _, r in starters.iterrows()
    ]


@app.post("/api/twitter-lineups")
def save_twitter_lineup(req: TwitterLineupSaveRequest) -> TwitterLineupRecord:
    fp = _slate_fingerprint()

    existing_lineups = load_twitter_lineups(fp)
    existing = next((l for l in existing_lineups if l.get("team") == req.team), None)
    if existing and existing.get("locked"):
        # Updating an already-locked lineup — diff against what was locked in.
        _emit_lineup_diff_notification(
            req.team, f"{req.team} lineup update", existing.get("slots", []), req.slots
        )
    elif _slate_first_pitch_started():
        # First official confirmation for this team, arriving after the slate's
        # first game began — diff against the best-guess lineup it's replacing.
        best_guess = _best_guess_lineup_slots(req.team)
        if best_guess is not None:
            _emit_lineup_diff_notification(
                req.team, f"{req.team} lineup confirmed", best_guess, req.slots
            )

    # Teams playing a doubleheader today can't be auto-trusted: Twitter/RotoWire/DFF
    # carry no game-time data, so a confirmed lineup might belong to the wrong game.
    # Save it, but veto the lock and flag it for manual confirmation.
    doubleheader_teams, _ = get_doubleheader_teams_cached(date.today().isoformat())
    needs_game_confirmation = req.team in doubleheader_teams
    locked = False if needs_game_confirmation else req.locked

    slots = [s.model_dump() for s in req.slots]
    record = upsert_twitter_lineup(req.team, req.notification_id, slots, fp, locked=locked)
    record["needs_game_confirmation"] = needs_game_confirmation
    return TwitterLineupRecord(**record)


@app.delete("/api/twitter-lineups/{team}")
def remove_twitter_lineup(team: str):
    fp = _slate_fingerprint()
    lineups = load_twitter_lineups(fp)
    lineup = next((l for l in lineups if l.get("team") == team), None)
    if lineup is None:
        raise HTTPException(404, detail="No confirmed lineup for that team")
    delete_twitter_lineup(team, fp)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Lineup lock / refresh endpoints
# ---------------------------------------------------------------------------

@app.post("/api/lineups/{team}/lock")
def lock_lineup(team: str):
    found = set_twitter_lineup_locked(team, True, _slate_fingerprint())
    if not found:
        raise HTTPException(404, detail=f"No confirmed lineup for {team}")
    return {"ok": True, "team": team, "locked": True}


@app.delete("/api/lineups/{team}/lock")
def unlock_lineup(team: str):
    found = set_twitter_lineup_locked(team, False, _slate_fingerprint())
    if not found:
        raise HTTPException(404, detail=f"No confirmed lineup for {team}")
    return {"ok": True, "team": team, "locked": False}


@app.post("/api/lineups/{team}/refresh")
async def refresh_lineup(team: str):
    """Fetch a fresh confirmed lineup for an unlocked team from RotoWire (DFF fallback)."""
    import asyncio
    import tempfile
    import os
    import pandas as pd

    fp = _slate_fingerprint()
    lineups = load_twitter_lineups(fp)
    record = next((l for l in lineups if l.get("team") == team), None)
    if record is not None and record.get("locked", False):
        raise HTTPException(409, detail=f"{team} lineup is locked — unlock before refreshing")

    cfg = read_config()
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    python = PROJECT_ROOT / "venv" / "bin" / "python"
    rw_script = PROJECT_ROOT / "scripts" / "fetch_rotowire_projections.py"

    platform_args: list[str] = ["--platform", platform_val]
    slate_path = _get_slate_file_path()
    if slate_path:
        if platform_val == "fanduel":
            platform_args += ["--fd-slate", str(slate_path)]
        else:
            platform_args += ["--dk-slate", str(slate_path)]

    async def _run_fetch(script: Path, extra_args: list[str]) -> str | None:
        """Run a fetch script with --team and --output to a temp file. Returns temp path or None."""
        fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            proc = await asyncio.create_subprocess_exec(
                str(python), str(script),
                *platform_args,
                "--team", team,
                "--output", tmp_path,
                *extra_args,
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=90)
            return tmp_path
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return None

    def _extract_confirmed_slots(tmp_path: str) -> list[dict] | None:
        """Read temp CSV and return confirmed batter slots for the team, or None if none found."""
        try:
            df = pd.read_csv(tmp_path)
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if df.empty or "lineup_slot" not in df.columns or "slot_confirmed" not in df.columns:
            return None
        confirmed = df[
            df["slot_confirmed"].astype(str).str.lower().isin(["true", "1"]) &
            df["lineup_slot"].notna() &
            (df["lineup_slot"].astype(float) >= 1) &
            (df["lineup_slot"].astype(float) <= 9)
        ]
        if confirmed.empty:
            return None
        slots: list[dict] = []
        for _, row in confirmed.iterrows():
            pid = int(row["player_id"]) if "player_id" in row and not pd.isna(row["player_id"]) else None
            slots.append({
                "slot": int(float(row["lineup_slot"])),
                "player_id": pid,
                "name": str(row.get("name", "")),
            })
        slots.sort(key=lambda s: s["slot"])
        return slots if slots else None

    # Try RotoWire first
    tmp = await _run_fetch(rw_script, [])
    confirmed_slots = _extract_confirmed_slots(tmp) if tmp else None

    # DFF fallback
    if confirmed_slots is None:
        dff_script = PROJECT_ROOT / "scripts" / "fetch_dff_projections.py"
        if dff_script.exists():
            tmp = await _run_fetch(dff_script, [])
            confirmed_slots = _extract_confirmed_slots(tmp) if tmp else None

    if not confirmed_slots:
        raise HTTPException(422, detail=f"No confirmed lineup data found for {team} in RotoWire or DFF")

    new_record = upsert_twitter_lineup(
        team=team,
        notification_id="rotowire-refresh",
        slots=confirmed_slots,
        slate_fingerprint=fp,
        locked=True,
    )
    _bake_twitter_lineup_to_projections(new_record)
    return TwitterLineupRecord(**new_record)


def _bake_twitter_lineup_to_projections(lineup: dict) -> None:
    """Write confirmed slot data from a Twitter lineup into projections.csv."""
    import pandas as pd
    try:
        cfg = read_config()
        proj_path = _resolve_proj_path(cfg)
        if not proj_path.exists():
            return
        df = pd.read_csv(proj_path)
        if "player_id" not in df.columns:
            return
        changed = False
        for slot_entry in lineup.get("slots", []):
            pid = slot_entry.get("player_id")
            slot = slot_entry.get("slot")
            if pid is None or slot is None:
                continue
            mask = df["player_id"] == pid
            if mask.any():
                df.loc[mask, "lineup_slot"] = slot
                df.loc[mask, "slot_confirmed"] = True
                changed = True
        if changed:
            df.to_csv(proj_path, index=False)
    except Exception:
        pass


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


def _get_archive_dir(create: bool = False) -> Path | None:
    """Return archive/MMDDYYYY path derived from the slate CSV Game Info date.

    When create=True, creates the directory and copies DKSalaries.csv into it
    (needed by evaluate_ownership dry-run).  When create=False (default), just
    returns the path without touching the filesystem.  Returns None when the
    slate is missing or the date cannot be parsed.
    """
    import re as _re
    import pandas as _pd
    slate_path = _get_slate_file_path()
    if slate_path is None:
        return None
    try:
        gi_df = _pd.read_csv(slate_path, usecols=["Game Info"])
        m = _re.search(r'(\d{2})/(\d{2})/(\d{4})', str(gi_df["Game Info"].dropna().iloc[0]))
        if not m:
            return None
        mo, dy, yr = m.groups()
        d = PROJECT_ROOT / "archive" / f"{mo}{dy}{yr}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
            dk_archive = d / "DKSalaries.csv"
            if not dk_archive.exists():
                import shutil as _shutil
                _shutil.copy2(str(slate_path), str(dk_archive))
        return d
    except Exception:
        return None


def _archive_contest_artifacts(archive_dir: Path, player_fpts_by_id: dict, zip_name: str) -> None:
    """Snapshot pipeline-run artifacts into the contest archive dir, once per
    "Analyze Contest" click. Best-effort: each artifact's failure is logged
    and skipped so the endpoint's primary by-name response is never blocked
    by an archival problem (missing file, disk full, permissions, etc.).
    """
    import shutil as _shutil
    log = _logging.getLogger(__name__)
    out_dir = _output_dir_path()
    cfg = read_config()
    from src.platforms.base import Platform
    platform = cfg.platform if hasattr(cfg, 'platform') else Platform.DRAFTKINGS

    try:
        src = out_dir / "candidate_pool_debug.csv"
        if src.exists():
            _shutil.copy2(str(src), str(archive_dir / "candidate_pool_debug.csv"))
        else:
            log.info("Analyze Contest: no candidate_pool_debug.csv to archive.")
    except Exception as exc:
        log.warning("Analyze Contest: failed to archive candidate pool dump: %s", exc)

    try:
        sweep_name = f"portfolio_sweep_{platform.value}.json"
        src = out_dir / sweep_name
        if src.exists():
            _shutil.copy2(str(src), str(archive_dir / sweep_name))
        else:
            log.info("Analyze Contest: no %s to archive.", sweep_name)
    except Exception as exc:
        log.warning("Analyze Contest: failed to archive portfolio sweep: %s", exc)

    try:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "zip_source": zip_name,
            "player_fpts": {str(pid): fpts for pid, fpts in player_fpts_by_id.items()},
        }
        with open(archive_dir / "contest_player_fpts.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        log.warning("Analyze Contest: failed to write contest_player_fpts.json: %s", exc)


@app.get("/api/contest/analyze")
def get_contest_analysis():
    """Return a player-name → FPTS map parsed from the contest standings zip in the archive dir."""
    import csv as _csv
    import io as _io
    import zipfile as _zipfile

    archive_dir = _get_archive_dir()
    if archive_dir is None:
        raise HTTPException(status_code=404, detail="Cannot resolve archive directory from slate")
    if not archive_dir.exists():
        raise HTTPException(status_code=404, detail=f"Archive directory does not exist: {archive_dir.name}")

    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    if not zips:
        raise HTTPException(status_code=404, detail=f"No contest standings zip found in {archive_dir.name}")

    zip_path = zips[0]
    try:
        with _zipfile.ZipFile(zip_path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            content = zf.read(csv_name).decode("utf-8-sig")
        reader = _csv.reader(_io.StringIO(content))
        rows = list(reader)
        if not rows:
            raise HTTPException(status_code=500, detail="Contest standings CSV is empty")
        header = rows[0]
        player_col   = header.index("Player")
        fpts_col     = header.index("FPTS")
        drafted_col  = header.index("%Drafted")

        # name -> distinct (fpts, drafted) pairs seen for that name in the contest export.
        # DK's contest export keys this sidebar table by name only, so when two real
        # players share a name (e.g. two different "Max Muncy"s), their rows collide here.
        name_pairs: dict[str, list[tuple[float, float]]] = {}
        for row in rows[1:]:
            if len(row) > player_col and row[player_col].strip():
                name = row[player_col].strip()
                raw  = row[fpts_col].strip() if len(row) > fpts_col else ""
                if not raw:
                    continue
                try:
                    fpts = float(raw)
                except ValueError:
                    continue
                drafted_raw = row[drafted_col].strip().rstrip("%") if len(row) > drafted_col else ""
                try:
                    drafted = float(drafted_raw) if drafted_raw else 0.0
                except ValueError:
                    drafted = 0.0
                pairs = name_pairs.setdefault(name, [])
                if (fpts, drafted) not in pairs:
                    pairs.append((fpts, drafted))

        # name -> candidate players from the slate pool, used to tell a genuine name
        # collision (two distinct player_ids) apart from duplicate rows for one player.
        candidates_by_name: dict[str, list[dict]] = {}
        slate_df = _load_slate_df()
        if slate_df is not None and not slate_df.empty:
            for _, r in slate_df.drop_duplicates("player_id").iterrows():
                candidates_by_name.setdefault(str(r["name"]).strip(), []).append({
                    "player_id": int(r["player_id"]),
                    "team": str(r["team"]),
                    "salary": int(r["salary"]),
                })

        player_fpts: dict[str, float] = {}
        collisions: list[dict] = []
        for name, pairs in name_pairs.items():
            slate_candidates = candidates_by_name.get(name, [])
            distinct_fpts = {p[0] for p in pairs}
            # Multiple rows for a name only matter if they actually disagree on FPTS —
            # if both real players happened to score the same, there's nothing to resolve.
            if len(distinct_fpts) > 1 and len(slate_candidates) > 1:
                # Genuine collision: the contest file has no team/ID for this table, so we
                # can only suggest a pairing — rank both lists by their best ownership proxy
                # (salary for slate candidates, %Drafted for contest rows) and zip them.
                # The user picks the actual mapping in the UI; this is just the default.
                sorted_pairs = sorted(pairs, key=lambda p: -p[1])
                sorted_candidates = sorted(slate_candidates, key=lambda c: -c["salary"])
                entries = [
                    {
                        **cand,
                        "fpts": sorted_pairs[i][0] if i < len(sorted_pairs) else None,
                        "drafted": sorted_pairs[i][1] if i < len(sorted_pairs) else None,
                        "suggested": i == 0,
                    }
                    for i, cand in enumerate(sorted_candidates)
                ]
                collisions.append({"name": name, "candidates": entries})
                player_fpts[name] = entries[0]["fpts"] if entries[0]["fpts"] is not None else max(pairs, key=lambda p: p[1])[0]
            else:
                # No collision (or nothing to disambiguate against) — keep the
                # highest-%Drafted value, same as before.
                player_fpts[name] = max(pairs, key=lambda p: p[1])[0]

        # Resolved player_id -> FPTS map (unambiguous names + heuristic-guessed
        # collisions) — an additional, durable artifact for retrospective
        # candidate-pool analysis; the by-name response below is unchanged.
        player_fpts_by_id: dict[int, float] = {}
        collided_names = {c["name"] for c in collisions}
        for name in name_pairs:
            if name in collided_names:
                continue
            for cand in candidates_by_name.get(name, []):
                player_fpts_by_id[cand["player_id"]] = player_fpts[name]
        for collision in collisions:
            for cand in collision["candidates"]:
                if cand["fpts"] is not None:
                    player_fpts_by_id[cand["player_id"]] = cand["fpts"]

        try:
            _archive_contest_artifacts(archive_dir, player_fpts_by_id, zip_path.name)
        except Exception as exc:
            _logging.getLogger(__name__).warning("Analyze Contest: artifact archiving failed: %s", exc)

        return {"player_fpts": player_fpts, "collisions": collisions}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse contest zip: {exc}")


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


@app.get("/api/schedule/doubleheaders")
def get_schedule_doubleheaders() -> DoubleheaderStatusResponse:
    """Doubleheader teams for today, per the real MLB schedule (not the slate file).

    Used to veto auto-lock for Twitter/RotoWire/DFF confirmed lineups, since
    those feeds carry no game-time data and can't tell which of a team's
    games a confirmed lineup belongs to.
    """
    today = date.today().isoformat()
    teams, is_fresh = get_doubleheader_teams_cached(today)
    return DoubleheaderStatusResponse(date=today, doubleheader_teams=sorted(teams), is_fresh=is_fresh)


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

    # Split scope dicts into separate "both" and "candidates" lists
    both_games = [g for g, s in update.game_scopes.items() if s == "both"]
    cand_games = [g for g, s in update.game_scopes.items() if s == "candidates"]
    both_teams = [t for t, s in update.team_scopes.items() if s == "both"]
    cand_teams = [t for t, s in update.team_scopes.items() if s == "candidates"]

    # Load existing player exclusions so we can apply scoped pruning
    stored = read_exclusions(update.slate_id, fingerprint)
    existing_both_pids: list[int] = stored.get("excluded_player_ids", [])
    existing_cand_pids: list[int] = stored.get("candidate_excluded_player_ids", [])

    df = _load_slate_df()
    if df is not None:
        all_players = [
            {"player_id": int(r["player_id"]), "team": str(r["team"]), "game": str(r.get("game", ""))}
            for _, r in df.iterrows()
        ]
        existing_both_pids, existing_cand_pids = prune_player_exclusions(
            existing_both_pids,
            set(both_teams),
            set(both_games),
            all_players,
            candidate_excluded_player_ids=existing_cand_pids,
            candidate_excluded_teams=set(cand_teams),
            candidate_excluded_games=set(cand_games),
        )

    stored_reductions = stored.get("team_ownership_reductions", {})
    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=both_teams,
        excluded_games=both_games,
        excluded_player_ids=existing_both_pids,
        candidate_excluded_teams=cand_teams,
        candidate_excluded_games=cand_games,
        candidate_excluded_player_ids=existing_cand_pids,
        game_ppd_pcts=update.game_ppd_pcts,
        team_ownership_reductions=stored_reductions,
        player_projection_overrides={int(k): v for k, v in stored.get("player_projection_overrides", {}).items()},
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
    cand_excluded_teams: list[str] = stored.get("candidate_excluded_teams", [])
    cand_excluded_games: list[str] = stored.get("candidate_excluded_games", [])
    game_ppd_pcts: dict[str, float] = stored.get("game_ppd_pcts", {})

    # Split scope dict into "both" and "candidates" lists
    both_pids = [int(pid) for pid, s in update.player_scopes.items() if s == "both"]
    cand_pids = [int(pid) for pid, s in update.player_scopes.items() if s == "candidates"]

    df = _load_slate_df()
    if df is not None:
        all_players = [
            {"player_id": int(r["player_id"]), "team": str(r["team"]), "game": str(r.get("game", ""))}
            for _, r in df.iterrows()
        ]
        both_pids, cand_pids = prune_player_exclusions(
            both_pids,
            set(excluded_teams),
            set(excluded_games),
            all_players,
            candidate_excluded_player_ids=cand_pids,
            candidate_excluded_teams=set(cand_excluded_teams),
            candidate_excluded_games=set(cand_excluded_games),
        )

    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=excluded_teams,
        excluded_games=excluded_games,
        excluded_player_ids=both_pids,
        candidate_excluded_teams=cand_excluded_teams,
        candidate_excluded_games=cand_excluded_games,
        candidate_excluded_player_ids=cand_pids,
        game_ppd_pcts=game_ppd_pcts,
        player_projection_overrides={int(k): v for k, v in stored.get("player_projection_overrides", {}).items()},
    )

    if df is None or df.empty:
        return SlatePlayersResponse(slate_id=update.slate_id, players=[])
    player_dicts = get_slate_players_with_status(df, update.slate_id, fingerprint)
    return SlatePlayersResponse(
        slate_id=update.slate_id,
        players=[PlayerExclusionStatus(**p) for p in player_dicts],
    )


@app.get("/api/slate/ownership-reductions")
def get_team_ownership_reductions() -> TeamOwnershipReductionsResponse:
    """Return the current per-team ownership reductions for the active slate."""
    from .slate_exclusions import compute_slate_id
    game_times = _load_slate_games()
    if not game_times:
        return TeamOwnershipReductionsResponse(slate_id="", team_ownership_reductions={})
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    slate_id = compute_slate_id(list(game_times.keys()))
    stored = read_exclusions(slate_id, fingerprint)
    return TeamOwnershipReductionsResponse(
        slate_id=slate_id,
        team_ownership_reductions=stored.get("team_ownership_reductions", {}),
    )


@app.post("/api/slate/ownership-reductions")
def post_team_ownership_reductions(update: TeamOwnershipReductionsUpdate) -> TeamOwnershipReductionsResponse:
    """Update per-team ownership reductions without touching any other exclusion state."""
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    stored = read_exclusions(update.slate_id, fingerprint)
    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=stored.get("excluded_teams", []),
        excluded_games=stored.get("excluded_games", []),
        excluded_player_ids=stored.get("excluded_player_ids"),
        candidate_excluded_teams=stored.get("candidate_excluded_teams"),
        candidate_excluded_games=stored.get("candidate_excluded_games"),
        candidate_excluded_player_ids=stored.get("candidate_excluded_player_ids"),
        game_ppd_pcts=stored.get("game_ppd_pcts"),
        team_ownership_reductions=update.team_ownership_reductions,
        player_projection_overrides={int(k): v for k, v in stored.get("player_projection_overrides", {}).items()},
    )
    try:
        import json as _json
        from datetime import datetime as _dt
        _arc = _get_archive_dir(create=True)
        if _arc is not None:
            (_arc / "ownership_settings.json").write_text(
                _json.dumps({
                    "team_ownership_reductions": update.team_ownership_reductions,
                    "saved_at": _dt.utcnow().isoformat(timespec="seconds"),
                }, indent=2)
            )
    except Exception:
        pass
    return TeamOwnershipReductionsResponse(
        slate_id=update.slate_id,
        team_ownership_reductions=update.team_ownership_reductions,
    )


@app.get("/api/slate/projection-overrides")
def get_player_projection_overrides() -> PlayerProjectionOverridesResponse:
    """Return current per-player projection overrides for the active slate."""
    from .slate_exclusions import compute_slate_id
    game_times = _load_slate_games()
    if not game_times:
        return PlayerProjectionOverridesResponse(slate_id="", player_projection_overrides={})
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    slate_id = compute_slate_id(list(game_times.keys()))
    stored = read_exclusions(slate_id, fingerprint)
    raw = stored.get("player_projection_overrides", {}) or {}
    return PlayerProjectionOverridesResponse(
        slate_id=slate_id,
        player_projection_overrides={int(k): v for k, v in raw.items()},
    )


@app.post("/api/slate/projection-overrides")
def post_player_projection_overrides(update: PlayerProjectionOverridesUpdate) -> PlayerProjectionOverridesResponse:
    """Update per-player projection overrides without touching any other exclusion state."""
    fingerprint = compute_file_fingerprint(_get_slate_file_path())
    stored = read_exclusions(update.slate_id, fingerprint)
    write_exclusions(
        slate_id=update.slate_id,
        file_fingerprint=fingerprint,
        excluded_teams=stored.get("excluded_teams", []),
        excluded_games=stored.get("excluded_games", []),
        excluded_player_ids=stored.get("excluded_player_ids"),
        candidate_excluded_teams=stored.get("candidate_excluded_teams"),
        candidate_excluded_games=stored.get("candidate_excluded_games"),
        candidate_excluded_player_ids=stored.get("candidate_excluded_player_ids"),
        game_ppd_pcts=stored.get("game_ppd_pcts"),
        team_ownership_reductions=stored.get("team_ownership_reductions"),
        player_projection_overrides=update.player_projection_overrides,
    )
    return PlayerProjectionOverridesResponse(
        slate_id=update.slate_id,
        player_projection_overrides=update.player_projection_overrides,
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
    live_unconfirmed_count = None
    try:
        import pandas as pd
        from .twitter_lineups import get_confirmed_team_lineups
        df = pd.read_csv(p)
        row_count = len(df)
        if {"slot_confirmed", "player_id"}.issubset(df.columns):
            unconf = df[~df["slot_confirmed"].astype(bool)].copy()
            confirmed = get_confirmed_team_lineups(_slate_fingerprint())
            if confirmed:
                twitter_pids = {pid for pid_to_slot in confirmed.values() for pid in pid_to_slot}
                unconf = unconf[~unconf["player_id"].isin(twitter_pids)]
                if "team" in unconf.columns and "position" in unconf.columns:
                    for team, pid_to_slot in confirmed.items():
                        scratched = (
                            (unconf["position"] != "P")
                            & (unconf["team"] == team)
                            & ~unconf["player_id"].isin(pid_to_slot)
                        )
                        unconf = unconf[~scratched]
            live_unconfirmed_count = len(unconf)
    except Exception:
        pass

    is_fresh = (
        compute_freshness(slate_path, p, platform=platform_val)
        if slate_path is not None
        else None
    )

    # Override stored unconfirmed_count with the live computation if available
    if live_unconfirmed_count is not None:
        extra["unconfirmed_count"] = live_unconfirmed_count

    return ProjectionsStatus(
        exists=True,
        path=str(p.relative_to(PROJECT_ROOT)),
        last_modified=stat.st_mtime,
        age_seconds=age,
        row_count=row_count,
        is_fresh=is_fresh,
        **extra,
    )


@app.get("/api/projections/merge_info")
def projections_merge_info():
    """Return the persisted merge_info state from the last projection fetch."""
    cfg = read_config()
    p   = _resolve_proj_path(cfg)
    state_path = p.parent / (p.stem + "_merge_info.json")
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


@app.get("/api/projections/unconfirmed")
def projections_unconfirmed():
    cfg = read_config()
    p = _resolve_proj_path(cfg)
    if not p.exists():
        return {"player_ids": []}
    try:
        import pandas as pd
        from .twitter_lineups import get_confirmed_team_lineups
        df = pd.read_csv(p)
        if "slot_confirmed" not in df.columns or "player_id" not in df.columns:
            return {"player_ids": []}
        unconfirmed = df[~df["slot_confirmed"].astype(bool)].copy()

        confirmed = get_confirmed_team_lineups(_slate_fingerprint())
        if confirmed:
            # Players in Twitter-confirmed lineups are confirmed regardless of CSV value
            twitter_pids = {pid for pid_to_slot in confirmed.values() for pid in pid_to_slot}
            unconfirmed = unconfirmed[~unconfirmed["player_id"].isin(twitter_pids)]
            # Scratched batters (in a confirmed-lineup team but not in the lineup)
            # are not in the active pool — exclude them from the unconfirmed list too
            if "team" in unconfirmed.columns and "position" in unconfirmed.columns:
                for team, pid_to_slot in confirmed.items():
                    scratched_mask = (
                        (unconfirmed["position"] != "P")
                        & (unconfirmed["team"] == team)
                        & ~unconfirmed["player_id"].isin(pid_to_slot)
                    )
                    unconfirmed = unconfirmed[~scratched_mask]

        return {"player_ids": [int(x) for x in unconfirmed["player_id"].tolist()]}
    except Exception:
        return {"player_ids": []}


@app.get("/api/projections/players")
def projections_players():
    import pandas as pd
    from .twitter_lineups import get_confirmed_team_lineups
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
        slate_cols = ["player_id", "name", "position", "team", "opponent", "salary", "game"]
        for _opt in ("game_start_time", "eligible_positions", "avg_pts"):
            if _opt in slate_df.columns:
                slate_cols.append(_opt)
        slate_sub = slate_df[slate_cols]
        proj_sub  = proj_df[["player_id", "mean", "lineup_slot", "slot_confirmed"]]
        merged = slate_sub.merge(proj_sub, on="player_id", how="inner")

        # Twitter confirmed lineups are authoritative: drop scratched batters and
        # update slot/slot_confirmed for the players who are actually starting.
        confirmed = get_confirmed_team_lineups(_slate_fingerprint())
        if confirmed:
            batter_mask = merged["position"] != "P"
            for team, pid_to_slot in confirmed.items():
                scratched = batter_mask & (merged["team"] == team) & ~merged["player_id"].isin(pid_to_slot)
                merged = merged[~scratched]
            merged = merged.copy()
            for pid_to_slot in confirmed.values():
                for pid, slot in pid_to_slot.items():
                    mask = merged["player_id"] == pid
                    if mask.any():
                        merged.loc[mask, "lineup_slot"] = slot
                        merged.loc[mask, "slot_confirmed"] = True

        # Players in "both"-excluded games are zeroed out and not factored into
        # the ownership softmax at all — their projections are irrelevant to the slate.
        both_excl_pids: set = set()
        _team_ownership_reductions: dict = {}
        _proj_overrides: dict = {}
        if "game" in slate_df.columns:
            try:
                from .slate_exclusions import compute_slate_id
                _games = [str(g) for g in slate_df["game"].dropna().unique()]
                _sid = compute_slate_id(_games)
                _fp = compute_file_fingerprint(_get_slate_file_path())
                _excl = read_exclusions(_sid, _fp)
                _both_games = set(_excl.get("excluded_games", []))
                if _both_games:
                    both_excl_pids = set(
                        slate_df.loc[slate_df["game"].isin(_both_games), "player_id"].dropna().astype(int)
                    )
                _team_ownership_reductions = _excl.get("team_ownership_reductions", {}) or {}
                _raw_overrides = _excl.get("player_projection_overrides", {}) or {}
                _proj_overrides = {int(k): v for k, v in _raw_overrides.items()}
                if _proj_overrides:
                    merged = merged.copy()
                    for _pid, _val in _proj_overrides.items():
                        merged.loc[merged["player_id"] == _pid, "mean"] = _val
            except Exception:
                pass

        # Compute heuristic ownership — Model D if team totals available, else C.
        try:
            from src.optimization.ownership import (
                apply_ownership_calibration,
                compute_heuristic_ownership,
                load_ownership_calibrator,
            )
            from .pipeline import PipelineRunner
            slate_path = _get_slate_file_path()
            team_totals = PipelineRunner._load_team_totals(str(slate_path) if slate_path else "")
            excl_mask = merged["player_id"].isin(both_excl_pids) if both_excl_pids else pd.Series(False, index=merged.index)
            non_excl = merged[~excl_mask]
            if non_excl.empty:
                ow_pct = [0.0] * len(merged)
            else:
                hr_odds = PipelineRunner._load_hr_fair_odds(str(slate_path) if slate_path else "")
                if hr_odds and "name" in non_excl.columns:
                    import unicodedata as _ud, re as _re
                    def _norm_hr(n: str) -> str:
                        nfkd = _ud.normalize("NFKD", n)
                        return _re.sub(r"[^a-z ]", "", nfkd.encode("ascii", "ignore").decode("ascii").lower()).strip()
                    non_excl = non_excl.copy()
                    non_excl["hr_prob"] = non_excl["name"].apply(lambda n: hr_odds.get(_norm_hr(str(n))))
                ow_sub = compute_heuristic_ownership(
                    non_excl, team_totals,
                    team_ownership_reductions=_team_ownership_reductions or None,
                )
                # Apply the same isotonic calibration (W_resid) the live pipeline
                # uses, so the UI shows the same magnitude-corrected ownership the
                # optimizer actually runs on. Loader returns None when the
                # artifact is missing or stale against current model constants.
                _calibrator = load_ownership_calibrator()
                if _calibrator is not None:
                    ow_sub = apply_ownership_calibration(
                        ow_sub, non_excl["position"].values, _calibrator
                    )
                ow_pct = [0.0] * len(merged)
                sub_i = 0
                for i, is_excl in enumerate(excl_mask):
                    if not is_excl:
                        ow_pct[i] = round(float(ow_sub[sub_i]) * 100, 1)
                        sub_i += 1
        except Exception:
            ow_pct = [None] * len(merged)

        result = []
        for i, (_, row) in enumerate(merged.iterrows()):
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
                "ownership_pct":  ow_pct[i],
                "is_overridden":  int(row["player_id"]) in _proj_overrides,
            })

        # Archive ownership projections whenever team reductions are active so
        # post-contest evaluation can reconstruct what the model projected.
        if _team_ownership_reductions:
            try:
                _arc = _get_archive_dir(create=True)
                if _arc is not None:
                    _proj_rows = [
                        {
                            "player_id": r["player_id"],
                            "name": r["name"],
                            "team": r["team"],
                            "game": str(merged.iloc[i].get("game", "")),
                            "position": r["position"],
                            "ownership_pct": r["ownership_pct"],
                        }
                        for i, r in enumerate(result)
                    ]
                    pd.DataFrame(_proj_rows).to_csv(
                        _arc / "ownership_projections.csv", index=False
                    )
            except Exception:
                pass

        return result
    except Exception:
        return []


@app.get("/api/projections/ownership_sync")
async def projections_ownership_sync():
    """
    Run evaluate_ownership.py --dry-run on today's archive and compare its
    E_production ownership against the live compute_heuristic_ownership output.

    Only meaningful when projections_source == "market_odds" (returns
    status="unavailable" otherwise).  The dry-run writes ownership_projections.csv
    to the archive dir; we then match players by ID and compute Spearman ρ.
    """
    import re as _re
    import pandas as pd
    from scipy.stats import spearmanr

    cfg = read_config()
    source = (cfg.paths.projections_source or "rotowire").strip().lower()
    if source != "market_odds":
        return {"status": "unavailable", "reason": "not_market_odds"}

    dk_raw = cfg.paths.dk_slate
    if not dk_raw:
        return {"status": "unavailable", "reason": "no_slate"}
    dk_path = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
    if not dk_path.exists():
        return {"status": "unavailable", "reason": "slate_missing"}

    # Derive archive directory from slate game date (e.g. "05/09/2026" → "05092026")
    try:
        gi_df = pd.read_csv(dk_path, usecols=["Game Info"])
        m = _re.search(r'(\d{2})/(\d{2})/(\d{4})', str(gi_df["Game Info"].dropna().iloc[0]))
        if not m:
            return {"status": "unavailable", "reason": "no_date_in_slate"}
        mo, dy, yr = m.groups()
        archive_dir = PROJECT_ROOT / "archive" / f"{mo}{dy}{yr}"
        if not archive_dir.exists():
            return {"status": "unavailable", "reason": "no_archive_dir"}
    except Exception:
        return {"status": "unavailable", "reason": "date_parse_error"}

    # Run the dry-run subprocess — writes ownership_projections.csv to archive_dir
    python = PROJECT_ROOT / "venv" / "bin" / "python"
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_ownership.py"
    try:
        proc = await asyncio.create_subprocess_exec(
            str(python), str(eval_script), "--dry-run", str(archive_dir),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
        )
        await asyncio.wait_for(proc.wait(), timeout=60)
    except asyncio.TimeoutError:
        return {"status": "error", "reason": "dry_run_timeout"}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    proj_csv = archive_dir / "ownership_projections.csv"
    if not proj_csv.exists():
        return {"status": "error", "reason": "no_dry_run_output"}

    try:
        dry_df = pd.read_csv(proj_csv)
        if "pred_E_production" not in dry_df.columns or "player_id" not in dry_df.columns:
            return {"status": "error", "reason": "missing_columns_in_output"}
        dry_df = dry_df[["player_id", "pred_E_production"]].dropna()
        dry_df["player_id"] = dry_df["player_id"].astype(int)

        # Live ownership comes from the same function the pipeline uses at runtime
        live_rows = projections_players()
        if not live_rows:
            return {"status": "unavailable", "reason": "no_live_projections"}
        live_df = pd.DataFrame(live_rows)[["player_id", "ownership_pct"]].dropna()
        live_df["player_id"] = live_df["player_id"].astype(int)
        live_df["ownership_frac"] = live_df["ownership_pct"] / 100.0

        merged = dry_df.merge(live_df[["player_id", "ownership_frac"]], on="player_id", how="inner")

        # Players whose whole game is excluded from the live slate (e.g. a postponed
        # game) are correctly zeroed out by projections_players(), but the dry-run
        # archive predates that exclusion and still assigns them nonzero
        # pred_E_production. Comparing those rows is a spurious mismatch, not a real
        # model disagreement, so drop them from the sync check.
        both_excl_pids: set = set()
        slate_df = _load_slate_df()
        if slate_df is not None and "game" in slate_df.columns:
            try:
                from .slate_exclusions import compute_slate_id
                games = [str(g) for g in slate_df["game"].dropna().unique()]
                sid = compute_slate_id(games)
                fp = compute_file_fingerprint(_get_slate_file_path())
                excl = read_exclusions(sid, fp)
                both_games = set(excl.get("excluded_games", []))
                if both_games:
                    slate_df["player_id"] = pd.to_numeric(slate_df["player_id"], errors="coerce")
                    both_excl_pids = set(
                        slate_df.loc[slate_df["game"].isin(both_games), "player_id"].dropna().astype(int)
                    )
            except Exception:
                pass
        if both_excl_pids:
            merged = merged[~merged["player_id"].isin(both_excl_pids)]

        if len(merged) < 5:
            return {"status": "unavailable", "reason": "too_few_matched_players"}

        r, _ = spearmanr(merged["pred_E_production"], merged["ownership_frac"])
        max_diff = float((merged["pred_E_production"] - merged["ownership_frac"]).abs().max())

        return {
            "status": "synced" if float(r) >= 0.95 else "out_of_sync",
            "spearman_r": round(float(r), 3),
            "max_diff": round(max_diff, 4),
            "n_checked": len(merged),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


@app.get("/api/projections/team_totals")
def projections_team_totals():
    """Return implied run totals and their source.

    Response: {"totals": {team: float, ...}, "source": "fantasylabs"|"cno"|"dff"|null}
    """
    try:
        from .pipeline import PipelineRunner
        slate_path = _get_slate_file_path()
        slate_str = str(slate_path) if slate_path else ""
        totals = PipelineRunner._load_team_totals(slate_str)
        source = PipelineRunner._get_team_totals_source(slate_str)
        return {"totals": totals or {}, "source": source}
    except Exception:
        return {"totals": {}, "source": None}


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

    # Auto-skip games that are "both"-excluded in stored exclusions.
    # Market Odds is the only source with per-game filtering (--games arg),
    # but adding these to excluded_pairs is a no-op for other sources.
    if dk_path is not None:
        try:
            from .slate_exclusions import compute_slate_id, compute_file_fingerprint as _cfp, read_exclusions as _re_excl
            _dk_gi_auto = pd.read_csv(dk_path, usecols=["Game Info"])
            _auto_games: list[str] = []
            for _gi in _dk_gi_auto["Game Info"].dropna().unique():
                _m = _re.match(r"(\w+)@(\w+)\s", str(_gi).strip())
                if _m:
                    _auto_games.append(f"{_m.group(1).upper()}@{_m.group(2).upper()}")
            if _auto_games:
                _auto_slate_id = compute_slate_id(_auto_games)
                _auto_fp = _cfp(dk_path)
                _auto_stored = _re_excl(_auto_slate_id, _auto_fp)
                for _g in _auto_stored.get("excluded_games", []):
                    if "@" in _g:
                        _a, _h = _g.split("@", 1)
                        excluded_pairs.add((_a.upper(), _h.upper()))
        except Exception:
            pass  # non-fatal; fall back to query-param-only exclusions

    # Resolve which games are actually in this slate (from DK CSV) and which
    # to include in the fetch.  Only used when excluded_pairs is non-empty.
    included_pairs: set[tuple[str, str]] = set()
    included_teams: set[str] = set()
    included_pids:  set[int] = set()
    id_to_team: dict[int, str] = {}
    team_to_game: dict[str, str] = {}

    if excluded_pairs and dk_path is not None:
        try:
            dk_gi = pd.read_csv(dk_path, usecols=["ID", "TeamAbbrev", "Game Info"])
            id_to_team = {int(r["ID"]): str(r["TeamAbbrev"]) for _, r in dk_gi.iterrows()}
            for _, r in dk_gi.iterrows():
                team = str(r["TeamAbbrev"])
                gi   = str(r.get("Game Info", ""))
                _gm  = _re.match(r"(\w+)@(\w+)\s", gi.strip())
                if _gm and team not in team_to_game:
                    team_to_game[team] = f"{_gm.group(1).upper()}@{_gm.group(2).upper()}"
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
        rw_seen_ids_path = proj_path.parent / "projections_rw_seen_ids.json"
        mo_out     = proj_path.parent / "projections_mo.csv"
        mo_sidecar               = proj_path.parent / "projections_mo_fallback.json"
        mo_caps_path             = proj_path.parent / "projections_mo_caps.json"
        mo_missing_opt_path      = proj_path.parent / "projections_mo_missing_opt.json"
        mo_team_warn_path        = proj_path.parent / "projections_mo_team_warnings.json"
        mo_pitcher_partial_path  = proj_path.parent / "projections_mo_pitcher_partial.json"
        # Persistent across partial fetches; overwritten only on a full fetch.
        merge_info_state_path = proj_path.parent / (proj_path.stem + "_merge_info.json")

        def _log(msg: str) -> str:
            return f"data: {json.dumps({'type': 'log', 'line': msg, 'timestamp': int(time.time() * 1000)})}\n\n"

        # Clean up any stale temp files left by a prior incomplete fetch.
        for _p in (dff_out, rw_out, rw_seen_ids_path, mo_out, mo_sidecar, mo_caps_path, mo_missing_opt_path, mo_team_warn_path, mo_pitcher_partial_path):
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

        def _read_mo_pitcher_partial() -> dict[int, dict]:
            if not mo_pitcher_partial_path.exists():
                return {}
            try:
                raw = json.loads(mo_pitcher_partial_path.read_text())
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

        def _read_mo_missing_opt() -> dict[int, list[str]]:
            if not mo_missing_opt_path.exists():
                return {}
            try:
                raw = json.loads(mo_missing_opt_path.read_text())
                return {int(k): v for k, v in raw.items()}
            except Exception:
                return {}

        def _read_mo_team_warnings() -> list[dict]:
            if not mo_team_warn_path.exists():
                return []
            try:
                return json.loads(mo_team_warn_path.read_text())
            except Exception:
                return []

        def _copy_proj_to_mo_archive() -> None:
            """Overwrite the archive market_odds_projections.csv with the final
            merged projections.csv so fallback players are included."""
            if dk_path is None or not proj_path.exists():
                return
            try:
                import shutil as _sh
                _gi = pd.read_csv(dk_path, usecols=["Game Info"])
                _m = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(_gi["Game Info"].dropna().iloc[0]))
                if not _m:
                    return
                _mo, _dy, _yr = _m.groups()
                _dest = PROJECT_ROOT / "archive" / f"{_mo}{_dy}{_yr}" / "market_odds_projections.csv"
                if _dest.exists():
                    _sh.copy2(proj_path, _dest)
            except Exception:
                pass

        def _load_merge_info_state() -> dict:
            if not merge_info_state_path.exists():
                return {}
            try:
                return json.loads(merge_info_state_path.read_text())
            except Exception:
                return {}

        def _save_merge_info_state(state: dict) -> None:
            try:
                merge_info_state_path.write_text(json.dumps(state, indent=2))
            except Exception:
                pass

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

        def _ensure_team_to_game() -> dict[str, str]:
            if team_to_game:
                return team_to_game
            if dk_path is not None:
                try:
                    dk_df3 = pd.read_csv(dk_path, usecols=["TeamAbbrev", "Game Info"])
                    t2g: dict[str, str] = {}
                    for _, r in dk_df3.iterrows():
                        team = str(r["TeamAbbrev"])
                        gi   = str(r.get("Game Info", ""))
                        _gm  = _re.match(r"(\w+)@(\w+)\s", gi.strip())
                        if _gm and team not in t2g:
                            t2g[team] = f"{_gm.group(1).upper()}@{_gm.group(2).upper()}"
                    return t2g
                except Exception:
                    pass
            return {}

        def _whole_team_fallbacks(
            fallback_players: list[dict],
            t2g: dict[str, str],
            total_per_team: dict[str, int] | None = None,
        ) -> list[dict]:
            """Return teams where every batter in the slate fell back to the secondary source."""
            counts: dict[str, int] = {}
            for p in fallback_players:
                if not p.get("is_pitcher") and p.get("team"):
                    counts[p["team"]] = counts.get(p["team"], 0) + 1
            result = [
                {"team": team, "game": t2g.get(team, ""), "count": count}
                for team, count in counts.items()
                if total_per_team and count == total_per_team.get(team)
            ]
            return sorted(result, key=lambda x: -x["count"])

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

        def _confirmed_not_in_batter_ids(pool_df: "pd.DataFrame", itm: dict[int, str]) -> set[int]:
            """Batter PIDs whose team has announced its official lineup but who are absent from it.
            Detected by: team has ≥1 slot_confirmed=True batter, but this batter is not confirmed."""
            if "slot_confirmed" not in pool_df.columns or "lineup_slot" not in pool_df.columns:
                return set()
            batters = pool_df[pool_df["lineup_slot"] != 10].copy()
            if batters.empty:
                return set()
            batters["_t"] = batters["player_id"].astype(int).map(lambda p: itm.get(p, ""))
            conf_teams: set[str] = set(
                batters.loc[batters["slot_confirmed"].astype(bool), "_t"].tolist()
            ) - {""}
            if not conf_teams:
                return set()
            unconf = batters.loc[
                ~batters["slot_confirmed"].astype(bool) & batters["_t"].isin(conf_teams)
            ]
            return set(unconf["player_id"].astype(int).tolist())

        # Helper: write final merged_df to proj_path, handling partial merge.
        def _write_proj(merged_df: "pd.DataFrame") -> "str | None":
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
            # Filter out players no longer on the current slate (e.g. from games that
            # were postponed and removed from the DK/FD CSV after projections were fetched).
            warn_msg: "str | None" = None
            if _slate_path_for_meta is not None and _slate_path_for_meta.exists():
                try:
                    if platform_val == "fanduel":
                        _id_col = pd.read_csv(_slate_path_for_meta, usecols=["Id"])
                        _current_pids: set = set(
                            pd.to_numeric(
                                _id_col["Id"].astype(str).str.split("-").str[-1], errors="coerce"
                            ).dropna().astype(int)
                        )
                    else:
                        _id_col = pd.read_csv(_slate_path_for_meta, usecols=["ID"])
                        _current_pids = set(pd.to_numeric(_id_col["ID"], errors="coerce").dropna().astype(int))
                    _before = len(result)
                    if "player_id" in result.columns:
                        result = result[
                            pd.to_numeric(result["player_id"], errors="coerce").isin(_current_pids)
                        ].copy()
                    _n_dropped = _before - len(result)
                    if _n_dropped:
                        warn_msg = (
                            f"Removed {_n_dropped} stale projection row(s) for player(s) "
                            f"no longer on the current slate — likely from a postponed game."
                        )
                except Exception:
                    pass
            result = result.sort_values("mean", ascending=False).reset_index(drop=True)
            result.to_csv(proj_path, index=False)
            return warn_msg

        # Helper: inject confirmed Twitter lineup players missing from pool.
        # The projection sources only output their own "confirmed starter" pool;
        # a player confirmed via Underdog lineup notification may have a projection
        # in the raw source data but be absent from the filtered output.  This
        # ensures they're always included so _apply_twitter_overrides can slot them.
        # When the market-odds output projected the player (late adds are passed
        # to the MO fetch via --include-player-ids), that projection is used.
        # Falls back to the salary / 600.0 heuristic only when no source had
        # them at all; heuristic injections are surfaced in merge_info.
        def _inject_twitter_confirmed(
            pool: "pd.DataFrame",
            mo_df: "pd.DataFrame | None" = None,
            mo_reasons: "dict[int, str] | None" = None,
        ) -> "tuple[pd.DataFrame, list[dict]]":
            from .twitter_lineups import get_confirmed_team_lineups
            confirmed = get_confirmed_team_lineups(_slate_fingerprint())
            if not confirmed:
                return pool, []
            slate_file = _slate_path_for_meta
            if slate_file is None or not slate_file.exists():
                return pool, []
            pid_to_slot: dict[int, int] = {
                pid: slot
                for pid_to_slot in confirmed.values()
                for pid, slot in pid_to_slot.items()
            }
            pool_pids = (
                {int(p) for p in pool["player_id"].tolist()}
                if not pool.empty and "player_id" in pool.columns
                else set()
            )
            missing_pids = set(pid_to_slot.keys()) - pool_pids
            if is_partial and included_pids:
                missing_pids &= {int(p) for p in included_pids}
            if not missing_pids:
                return pool, []
            try:
                if platform_val == "fanduel":
                    sl = pd.read_csv(slate_file, usecols=["Id", "Nickname", "Salary"])
                    sl["_pid"] = pd.to_numeric(
                        sl["Id"].astype(str).str.split("-").str[-1], errors="coerce"
                    ).astype("Int64")
                    sl = sl[sl["_pid"].isin(missing_pids)]
                    name_col = "Nickname"
                else:
                    sl = pd.read_csv(slate_file, usecols=["ID", "Name", "Salary"])
                    sl["_pid"] = pd.to_numeric(sl["ID"], errors="coerce").astype("Int64")
                    sl = sl[sl["_pid"].isin(missing_pids)]
                    name_col = "Name"
            except Exception:
                return pool, []
            if sl.empty:
                return pool, []
            mo_lookup: dict[int, tuple[float, float]] = {}
            if mo_df is not None and not mo_df.empty and {"player_id", "mean", "std_dev"}.issubset(mo_df.columns):
                mo_lookup = {
                    int(r.player_id): (float(r.mean), float(r.std_dev))
                    for r in mo_df.drop_duplicates("player_id").itertuples(index=False)
                }
            _itm_inj = _ensure_id_to_team()
            rows = []
            injected: list[dict] = []
            for _, row in sl.iterrows():
                pid = int(row["_pid"])
                salary = float(row["Salary"]) if pd.notna(row.get("Salary")) else 3000.0
                if pid in mo_lookup:
                    proj_mean, proj_std = mo_lookup[pid]
                    source = "market_odds"
                else:
                    proj_mean = round(salary / 600.0, 2)
                    proj_std = round(proj_mean * 0.85, 2)
                    source = "heuristic"
                rows.append({
                    "player_id": pid,
                    "name": str(row[name_col]),
                    "mean": proj_mean,
                    "std_dev": proj_std,
                    "lineup_slot": pid_to_slot.get(pid, 1),
                    "slot_confirmed": True,
                })
                injected.append({
                    "player_id": pid,
                    "name": str(row[name_col]),
                    "team": _itm_inj.get(pid, ""),
                    "salary": salary,
                    "mean": proj_mean,
                    "source": source,
                    "reason": (mo_reasons or {}).get(pid, ""),
                })
            if not rows:
                return pool, []
            new_df = pd.DataFrame(rows)
            new_df["player_id"] = new_df["player_id"].astype("Int64")
            return pd.concat([pool, new_df], ignore_index=True), injected

        # Clear stale merge-info state at the start of every full (non-partial)
        # fetch so that server restarts don't resurface banners from a prior slate.
        # Partial fetches preserve the file so the purge+accumulate logic works.
        if not is_partial:
            try:
                merge_info_state_path.unlink(missing_ok=True)
            except Exception:
                pass

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
                # Fire DFF archive fetch in the background: runs concurrently with
                # RW+MO, archives to archive/MMDDYYYY/ but does NOT write to processed/.
                _dff_tmp = proj_path.parent / ".dff_archive_tmp.csv"

                async def _archive_dff_silent() -> None:
                    try:
                        _proc = await asyncio.create_subprocess_exec(
                            str(python), str(dff_script),
                            "--output", str(_dff_tmp),
                            *_platform_args,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                            cwd=str(PROJECT_ROOT),
                        )
                        await _proc.wait()
                    except Exception:
                        pass
                    finally:
                        _dff_tmp.unlink(missing_ok=True)

                asyncio.create_task(_archive_dff_silent())

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
                    # Late adds confirmed via Twitter lineups may be absent from
                    # the RotoWire pool; pass them explicitly so they can receive
                    # market projections instead of the salary heuristic.
                    try:
                        from .twitter_lineups import get_confirmed_team_lineups as _gctl
                        _confirmed_pids = sorted({
                            int(pid)
                            for _m in _gctl(_slate_fingerprint()).values()
                            for pid in _m
                        })
                        if _confirmed_pids:
                            mo_cmd += [
                                "--include-player-ids",
                                ",".join(str(p) for p in _confirmed_pids),
                            ]
                    except Exception:
                        pass
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
                        mo_sidecar_reasons  = _read_mo_sidecar()
                        mo_pitcher_partials = _read_mo_pitcher_partial()
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
                            # Scale RotoWire fallback projections for hitters down to 90%
                            # (RotoWire tends to be more optimistic than market odds for batters)
                            is_hitter_fallback = ~has_pref & (pool.get("lineup_slot", 0) != 10)
                            pool.loc[is_hitter_fallback, "mean"]    *= 0.9
                            pool.loc[is_hitter_fallback, "std_dev"] *= 0.9
                            # For pitchers missing only the wins market, use partial MO
                            # projection + 1.5 pts (win bonus estimate) instead of RW fallback.
                            is_pitcher_fallback = ~has_pref & (pool.get("lineup_slot", 0) == 10)
                            if is_pitcher_fallback.any() and mo_pitcher_partials:
                                for idx in pool[is_pitcher_fallback].index:
                                    pid = int(pool.at[idx, "player_id"])
                                    if pid in mo_pitcher_partials:
                                        pool.at[idx, "mean"]    = mo_pitcher_partials[pid]["mean"] + 1.5
                                        pool.at[idx, "std_dev"] = mo_pitcher_partials[pid]["std_dev"]
                            fallback_rows = pool.loc[~has_pref] if "name" in pool.columns else pd.DataFrame()

                            if not fallback_rows.empty:
                                for _, row in fallback_rows.iterrows():
                                    pid_val = int(row["player_id"]) if "player_id" in fallback_rows.columns else 0
                                    is_pitcher = bool(row.get("lineup_slot") == 10) if "lineup_slot" in fallback_rows.columns else False
                                    entry: dict = {
                                        "name": row["name"],
                                        "team": _itm.get(pid_val, ""),
                                        "reason": mo_sidecar_reasons.get(pid_val, ""),
                                        "player_id": pid_val,
                                        "is_pitcher": is_pitcher,
                                    }
                                    if is_pitcher and pid_val in mo_pitcher_partials:
                                        entry["partial_mean"]    = mo_pitcher_partials[pid_val]["mean"]
                                        entry["partial_std_dev"] = mo_pitcher_partials[pid_val]["std_dev"]
                                    fallback_players.append(entry)
                        else:
                            result_event = _log(
                                "Warning: Market odds unavailable; using RotoWire projections for all players."
                            )
                            # RotoWire is systematically optimistic for batters vs
                            # market-implied values, so scale down even when MO
                            # failed entirely (e.g. Walks market absent for all batters).
                            is_hitter = pool.get("lineup_slot", 0) != 10
                            pool.loc[is_hitter, "mean"]    *= 0.9
                            pool.loc[is_hitter, "std_dev"] *= 0.9

                        pool, _injected = _inject_twitter_confirmed(
                            pool, mo_df=mo_df, mo_reasons=mo_sidecar_reasons
                        )
                        heuristic_players: list[dict] = [
                            e for e in _injected if e["source"] == "heuristic"
                        ]
                        for e in _injected:
                            if e["source"] == "market_odds":
                                _logging.getLogger(__name__).info(
                                    "Late add %s (%s) projected from market odds.",
                                    e["name"], e["team"],
                                )
                        _stale_warn = _write_proj(pool)
                        if _stale_warn:
                            yield _log(f"Warning: {_stale_warn}")
                        proj_written = True
                        _copy_proj_to_mo_archive()

                        _conf_not_in = _confirmed_not_in_batter_ids(pool, _itm)
                        if _conf_not_in:
                            fallback_players = [
                                p for p in fallback_players
                                if p.get("is_pitcher") or p.get("player_id") not in _conf_not_in
                            ]

                        # Build capped_players, missing_opt_players, and team_name_warnings from sidecars.
                        mo_cap_data = _read_mo_caps()
                        mo_missing_opt_data = _read_mo_missing_opt()
                        team_name_warnings: list[dict] = _read_mo_team_warnings()
                        capped_players: list[dict] = []
                        missing_opt_players: list[dict] = []
                        if "name" in pool.columns:
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
                            for opt_pid, opt_mkts in mo_missing_opt_data.items():
                                if opt_pid in pid_to_name and opt_pid not in _conf_not_in:
                                    missing_opt_players.append({
                                        "name": pid_to_name[opt_pid],
                                        "team": _itm.get(opt_pid, ""),
                                        "markets": opt_mkts,
                                    })

                        low_team_projs = _low_team_projections(pool, _itm)

                        # Merge with persisted state for partial fetches: purge
                        # entries for teams we just re-fetched, keep the rest.
                        if is_partial and included_teams:
                            _prev = _load_merge_info_state()
                            def _purge(lst: list[dict]) -> list[dict]:
                                return [x for x in lst if x.get("team", "") not in included_teams]
                            def _purge_warnings(lst: list[dict]) -> list[dict]:
                                return [x for x in lst if x.get("game", "") not in {
                                    f"{a}@{h}" for a, h in included_pairs
                                }]
                            fallback_players    = _purge(_prev.get("players", []))         + fallback_players
                            capped_players      = _purge(_prev.get("capped_players", []))  + capped_players
                            missing_opt_players = _purge(_prev.get("missing_opt_players", [])) + missing_opt_players
                            heuristic_players   = _purge(_prev.get("heuristic_players", [])) + heuristic_players
                            team_name_warnings  = _purge_warnings(_prev.get("team_name_warnings", [])) + team_name_warnings

                        _total_batters: dict[str, int] = {}
                        if "lineup_slot" in pool.columns:
                            for _pid in pool.loc[pool["lineup_slot"] != 10, "player_id"].astype(int):
                                _t = _itm.get(_pid, "")
                                if _t:
                                    _total_batters[_t] = _total_batters.get(_t, 0) + 1
                        fallback_teams = _whole_team_fallbacks(fallback_players, _ensure_team_to_game(), _total_batters)

                        _save_merge_info_state({
                            "players":             fallback_players,
                            "capped_players":      capped_players,
                            "missing_opt_players": missing_opt_players,
                            "heuristic_players":   heuristic_players,
                            "fallback_teams":      fallback_teams,
                            "team_name_warnings":  team_name_warnings,
                            "secondary_source":    "RotoWire",
                        })

                        if fallback_players or capped_players or low_team_projs or missing_opt_players or team_name_warnings or heuristic_players:
                            result_event = f"data: {json.dumps({'type': 'merge_info', 'secondary_source': 'RotoWire', 'count': len(fallback_players), 'players': fallback_players, 'capped_players': capped_players, 'low_team_projections': low_team_projs, 'fallback_teams': fallback_teams, 'missing_opt_players': missing_opt_players, 'heuristic_players': heuristic_players, 'team_name_warnings': team_name_warnings, 'included_teams': sorted(included_teams), 'timestamp': int(time.time() * 1000)})}\n\n"
                        elif result_event is None:
                            result_event = _log("All player projections sourced from Market Odds (CrazyNinjaOdds).")

            except Exception as exc:
                returncode = 1
                result_event = _log(f"Warning: merge error — {exc}")
            finally:
                for p in (rw_out, mo_out, mo_sidecar, mo_caps_path, mo_missing_opt_path, mo_team_warn_path, mo_pitcher_partial_path):
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
                    # Union: start with RW starters, then append DFF players that RW has
                    # no knowledge of at all.  Players RW matched (even those excluded from
                    # the starter output because they lost their lineup slot — e.g. a
                    # scratched batter) must not re-enter via DFF fallback.
                    rw_ids: set[int] = set(rw_pool["player_id"].tolist()) if not rw_pool.empty else set()

                    rw_seen_ids: set[int] = set()
                    try:
                        if rw_seen_ids_path.exists():
                            rw_seen_ids = set(json.loads(rw_seen_ids_path.read_text()))
                    except Exception:
                        pass
                    rw_footprint = rw_ids | rw_seen_ids

                    # Also block DFF pitchers for teams where RW already has a
                    # confirmed starting pitcher.  rw_footprint handles players
                    # that RW matched and excluded; this handles players that RW
                    # never mentioned at all (e.g. a same-team alternative SP).
                    _itm_early = _ensure_id_to_team()
                    if not rw_pool.empty and "lineup_slot" in rw_pool.columns:
                        _rw_pitcher_teams: set[str] = {
                            _itm_early.get(int(pid), "")
                            for pid in rw_pool.loc[rw_pool["lineup_slot"] == 10, "player_id"].astype(int)
                        } - {""}
                    else:
                        _rw_pitcher_teams = set()

                    if not dff_pool.empty:
                        _dff_is_pitcher = (
                            (dff_pool["lineup_slot"] == 10)
                            if "lineup_slot" in dff_pool.columns
                            else pd.Series(False, index=dff_pool.index)
                        )
                        _dff_team = dff_pool["player_id"].astype(int).map(
                            lambda pid: _itm_early.get(pid, "")
                        )
                        _dff_blocked = _dff_is_pitcher & _dff_team.isin(_rw_pitcher_teams)
                        dff_extra = dff_pool[
                            ~dff_pool["player_id"].isin(rw_footprint) & ~_dff_blocked
                        ]
                        if _dff_blocked.any():
                            _logging.getLogger(__name__).info(
                                "Blocked %d DFF pitcher(s) for teams with RW confirmed starter: %s",
                                int(_dff_blocked.sum()),
                                sorted(set(_dff_team[_dff_blocked].tolist())),
                            )
                    else:
                        dff_extra = pd.DataFrame()
                    pool = pd.concat([rw_pool, dff_extra], ignore_index=True)

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

                    pool, _injected2 = _inject_twitter_confirmed(pool)
                    heuristic_players2: list[dict] = [
                        e for e in _injected2 if e["source"] == "heuristic"
                    ]
                    _stale_warn = _write_proj(pool)
                    if _stale_warn:
                        yield _log(f"Warning: {_stale_warn}")
                    proj_written = True

                    _conf_not_in2 = _confirmed_not_in_batter_ids(pool, _itm2)
                    if _conf_not_in2:
                        fallback_players2 = [
                            p for p in fallback_players2
                            if p.get("is_pitcher") or p.get("player_id") not in _conf_not_in2
                        ]

                    low_team_projs2 = _low_team_projections(pool, _itm2)
                    _total_batters2: dict[str, int] = {}
                    if "lineup_slot" in pool.columns:
                        for _pid2 in pool.loc[pool["lineup_slot"] != 10, "player_id"].astype(int):
                            _t2 = _itm2.get(_pid2, "")
                            if _t2:
                                _total_batters2[_t2] = _total_batters2.get(_t2, 0) + 1
                    fallback_teams2 = _whole_team_fallbacks(fallback_players2, _ensure_team_to_game(), _total_batters2)

                    # Merge with persisted state for partial fetches.
                    if is_partial and included_teams:
                        _prev2 = _load_merge_info_state()
                        def _purge2(lst: list[dict]) -> list[dict]:
                            return [x for x in lst if x.get("team", "") not in included_teams]
                        fallback_players2 = _purge2(_prev2.get("players", [])) + fallback_players2
                        heuristic_players2 = _purge2(_prev2.get("heuristic_players", [])) + heuristic_players2

                    _save_merge_info_state({
                        "players":             fallback_players2,
                        "capped_players":      [],
                        "missing_opt_players": [],
                        "heuristic_players":   heuristic_players2,
                        "fallback_teams":      fallback_teams2,
                        "team_name_warnings":  [],
                        "secondary_source":    fallback_label,
                    })

                    if fallback_players2 or low_team_projs2 or heuristic_players2:
                        result_event2 = f"data: {json.dumps({'type': 'merge_info', 'secondary_source': fallback_label, 'count': len(fallback_players2), 'players': fallback_players2, 'low_team_projections': low_team_projs2, 'fallback_teams': fallback_teams2, 'heuristic_players': heuristic_players2, 'included_teams': sorted(included_teams), 'timestamp': int(time.time() * 1000)})}\n\n"
                    else:
                        result_event2 = _log(f"All player projections sourced from {preferred_label}.")

        except Exception as exc:
            returncode = 1
            result_event2 = _log(f"Warning: merge error — {exc}")
        finally:
            for p in (dff_out, rw_out, rw_seen_ids_path):
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


@app.get("/api/run/cache_status")
def run_cache_status():
    """Return lineup cache availability for the current slate."""
    from .lineup_cache import get_cache_status
    cfg = read_config()
    slate_path = cfg.paths.dk_slate or ""
    abs_slate = (PROJECT_ROOT / slate_path) if slate_path else None
    is_gpp = True
    status = get_cache_status(abs_slate or "")

    n_batter_teams = 0
    try:
        slate_df = _load_slate_df()
        if slate_df is not None:
            all_batter_teams = set(slate_df[slate_df["position"] != "P"]["team"])
            game_times = _load_slate_games()
            if game_times and abs_slate:
                from .slate_exclusions import compute_file_fingerprint, compute_slate_id, read_exclusions as _re
                fp = compute_file_fingerprint(abs_slate)
                sid = compute_slate_id(list(game_times.keys()))
                stored = _re(sid, fp)
                excluded_games = set(stored.get("excluded_games", []))
                if excluded_games:
                    excl_teams = set(
                        slate_df[
                            slate_df["game"].isin(excluded_games) &
                            (slate_df["position"] != "P")
                        ]["team"]
                    )
                    all_batter_teams -= excl_teams
            n_batter_teams = len(all_batter_teams)
    except Exception:
        pass

    return {
        **status,
        "is_gpp": is_gpp,
        "n_configured_candidates": cfg.gpp.n_candidates,
        "n_configured_field_k": cfg.gpp.n_field_samples,
        "n_batter_teams": n_batter_teams,
    }


@app.get("/api/run/stream")
async def run_stream(
    request: Request,
    use_candidates: bool = False,
    use_field: bool = False,
    seed_optimal: bool = False,
):
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
            # Release large arrays from the previous run before allocating new
            # ones.  The old runner's robust_payout (~400 MB), field_sorted
            # (~600 MB), and sim_results can otherwise co-exist in memory with
            # the new run's equivalents, causing significant memory pressure.
            _old = _state.get("_runner_last")
            if _old is not None:
                for _attr in (
                    "_gpp_robust_payout",
                    "_gpp_candidates",
                    "_gpp_coverage_threshold",
                    "_sim_results",
                    "_raw_portfolio",
                ):
                    try:
                        setattr(_old, _attr, None)
                    except Exception:
                        pass
            del _old

            from .pipeline import PipelineRunner
            import yaml as _yaml
            _cfg_path = str(PROJECT_ROOT / "config.yaml")
            if seed_optimal:
                with open(_cfg_path) as _f:
                    _cfg = _yaml.safe_load(_f) or {}
                _cfg.setdefault("gpp", {})["seed_optimal_lineups"] = True
                import tempfile, os as _os
                _tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False,
                    dir=str(PROJECT_ROOT),
                )
                _yaml.dump(_cfg, _tmp)
                _tmp.close()
                _cfg_path = _tmp.name
            runner = PipelineRunner(
                _cfg_path,
                progress_cb,
                stop_check=_stop_event.is_set,
                use_cached_candidates=use_candidates,
                use_cached_field=use_field,
            )
            portfolio = runner.run()
            _state["portfolio"] = portfolio
            _state["_runner_last"] = runner
            # Re-sync the optimal-lineups cache with what this run just wrote to disk —
            # it's otherwise never refreshed after the first load, so a later run (e.g.
            # one that changed slate/player exclusions) would leave GET /api/portfolio/optimal
            # serving a stale pre-exclusion snapshot indefinitely.
            _state["optimal_lineups"] = _load_optimal_lineups_from_json(runner._platform.value)
            _state["status"] = "stopped" if _stop_event.is_set() else "complete"
        except Exception as exc:
            _state["status"] = "error"
            _state["error"] = str(exc)
            err_payload = {"stage": "error", "message": str(exc), "timestamp": int(time.time() * 1000)}
            asyncio.run_coroutine_threadsafe(queue.put(err_payload), loop)
        finally:
            if seed_optimal and _cfg_path != str(PROJECT_ROOT / "config.yaml"):
                import os as _os
                try:
                    _os.unlink(_cfg_path)
                except OSError:
                    pass
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


@app.post("/api/portfolio/activate_risk")
async def activate_risk_endpoint(body: dict):
    """Switch the active det_ev portfolio to a different risk level."""
    if _state["status"] in ("running", "replacing"):
        raise HTTPException(409, "Cannot switch active portfolio while a run is in progress")
    runner = _state.get("_runner_last")
    if runner is None or not getattr(runner, "_sweep_portfolios_raw", None):
        raise HTTPException(400, "No det_ev sweep portfolios available — please run the pipeline first.")
    try:
        risk = float(body.get("risk", 1))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, runner.activate_sweep_risk, risk)
        _state["portfolio"] = result
        return {"lineups": result}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/portfolio/replace/{lineup_index}")
async def replace_lineup_endpoint(lineup_index: int):
    if _state["status"] in ("running", "replacing"):
        raise HTTPException(409, "Cannot replace lineup while a run is in progress")
    runner = _state.get("_runner_last")
    if runner is None or getattr(runner, "_raw_portfolio", None) is None:
        raise HTTPException(
            400,
            "Simulation data unavailable — please re-run the portfolio to enable lineup replacement.",
        )
    if getattr(runner, "_sim_results", None) is None:
        raise HTTPException(
            400,
            "Simulation results unavailable — please re-run the portfolio to enable lineup replacement.",
        )
    if lineup_index < 1 or lineup_index > len(runner._raw_portfolio):
        raise HTTPException(400, f"Invalid lineup index: {lineup_index}")
    _state["status"] = "replacing"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, runner.replace_lineup, lineup_index)
        _state["portfolio"] = result
        return result
    except KeyError as exc:
        raise HTTPException(500, f"Lineup replacement failed: player {exc} not found in simulation data — try re-running the portfolio")
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        _state["status"] = "complete"


@app.get("/api/portfolio/sweep")
def get_portfolio_sweep():
    """Return the persisted det_ev sweep portfolios if they match the current slate."""
    try:
        cfg = read_config()
        platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
        output_dir = cfg.paths.output_dir or "outputs"
        base = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        sweep_path = base / f"portfolio_sweep_{platform_val}.json"

        if not sweep_path.exists():
            return {"sweep": [], "active_risk": 1}

        with open(sweep_path) as f:
            data = json.load(f)

        # Validate against the current slate fingerprint.
        from .slate_exclusions import compute_file_fingerprint
        if platform_val == "draftkings":
            slate_rel = cfg.paths.dk_slate or ""
        else:
            slate_rel = cfg.paths.fd_slate or ""
        slate_abs = (PROJECT_ROOT / slate_rel) if slate_rel and not Path(slate_rel).is_absolute() else Path(slate_rel)
        current_fp = compute_file_fingerprint(slate_abs) if slate_rel else ""
        if data.get("slate_fingerprint") != current_fp:
            return {"sweep": [], "active_risk": 1}

        return {"sweep": data.get("sweep", []), "active_risk": data.get("active_risk", 1)}
    except Exception:
        return {"sweep": [], "active_risk": 1}


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


@app.get("/api/portfolio/optimal")
def get_optimal_lineups(platform: str | None = None):
    """Return persisted optimal lineups if they match the current slate fingerprint."""
    try:
        cfg = read_config()
        platform_val = platform or (cfg.platform.value if hasattr(cfg, "platform") else "draftkings")
    except Exception:
        platform_val = platform or "draftkings"

    # If a run just completed (in-memory), check in-memory first.
    if _state["optimal_lineups"] is not None:
        # Validate against current slate fingerprint before serving in-memory copy.
        current_fp = compute_file_fingerprint(_get_slate_file_path())
        path = _optimal_lineups_path(platform_val)
        if path.exists():
            try:
                with open(path) as f:
                    stored_fp = json.load(f).get("slate_fingerprint", "")
                if stored_fp and stored_fp != current_fp:
                    _state["optimal_lineups"] = None
                    raise HTTPException(404, "Optimal lineups invalidated by slate change")
            except HTTPException:
                raise
            except Exception:
                pass
        return _state["optimal_lineups"]

    # Fall through to disk.
    lineups = _load_optimal_lineups_from_json(platform_val)
    if lineups is None:
        raise HTTPException(404, "No optimal lineups available")
    _state["optimal_lineups"] = lineups
    return lineups


# ---------------------------------------------------------------------------
# Late swap
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel


class LateSwapRunRequest(_BaseModel):
    entry_marks: dict[str, list[int]] = {}
    bulk_marked_player_ids: list[int] = []
    bulk_marked_teams: list[str] = []


class LateSwapOverrideRequest(_BaseModel):
    entry_id: str
    slot_index: int
    player_id: int


_late_swap_lock = threading.Lock()


def _now_eastern():
    """Naive Eastern-time now — slate start times are naive Eastern ISO strings."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


def _output_dir_path() -> Path:
    cfg = read_config()
    output_dir = cfg.paths.output_dir or "outputs"
    return PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)


def _late_swap_context(apply_saved: bool = True) -> dict:
    """Parse entries + slate, build pools/lookup/states, apply persisted swaps.

    apply_saved=True (the default) carries forward previously committed
    swaps onto the freshly parsed states — including into run_swap, so a
    re-run extends prior swaps rather than recomputing from the untouched
    original entry file. Lock status is then recomputed from each slot's
    current occupant (the swap-in, if any, else the original) via
    recompute_locks: a slot only locks once whoever is actually rostered
    there now has their game start, so a swapped-in player whose game
    hasn't started keeps the slot open even though the player they
    replaced has already started. A still-open slot remains fully
    re-computable on every run.

    Returns {"status": "ok", ...} or a terminal {"status": <reason>}.
    """
    import pandas as pd
    from . import late_swap
    from .dk_entries import parse_entry_file
    from .slate_exclusions import compute_slate_id

    cfg = read_config()
    platform_val = cfg.platform.value if hasattr(cfg, "platform") else "draftkings"
    if platform_val != "draftkings":
        return {"status": "unsupported_platform"}
    slate_path = _get_slate_file_path()
    if slate_path is None:
        return {"status": "no_slate"}
    # Entries come from outputs/ (the upload_*.csv files written at portfolio
    # completion reflect what was actually submitted to DK), not data/raw.
    entry_files = late_swap.scan_swap_entry_files(str(_output_dir_path()))
    if not entry_files:
        return {"status": "no_entries"}
    slate_df = _load_slate_df()
    if slate_df is None or slate_df.empty:
        return {"status": "no_slate"}

    proj_path = _resolve_proj_path(cfg)
    proj_df = pd.read_csv(proj_path) if proj_path.exists() else None

    fingerprint = compute_file_fingerprint(slate_path)
    slate_id = compute_slate_id(sorted({str(g) for g in slate_df["game"] if g}))
    exclusions = read_exclusions(slate_id, fingerprint)
    confirmed_lineups = get_confirmed_team_lineups(fingerprint)

    lookup_df, candidates_df = late_swap.build_swap_pools(
        slate_df, proj_df, exclusions, confirmed_team_lineups=confirmed_lineups,
    )
    lookup = late_swap.build_player_lookup(lookup_df)
    raw_salaries = late_swap.load_raw_salaries(slate_path)
    now = _now_eastern()

    all_file_entries = [(p, parse_entry_file(p)) for p in entry_files]
    states = late_swap.build_entry_states(all_file_entries, lookup, raw_salaries, now)

    output_dir = str(_output_dir_path())
    saved = late_swap.load_state(output_dir, fingerprint)
    if saved and apply_saved:
        late_swap.apply_saved_state(states, saved)
        late_swap.recompute_locks(states, lookup, now)

    return {
        "status": "ok",
        "states": states,
        "lookup": lookup,
        "candidates_df": candidates_df,
        "slate_df": slate_df,
        "now": now,
        "fingerprint": fingerprint,
        "saved": saved,
        "output_dir": output_dir,
        "all_file_entries": all_file_entries,
    }


def _late_swap_state_response(ctx: dict) -> dict:
    from . import late_swap

    if ctx["status"] != "ok":
        return {
            "status": ctx["status"], "now": None, "files": [], "entries": [],
            "bulk_marked_player_ids": [], "bulk_marked_teams": [], "teams": [],
            "last_run_at": None, "written_files": [],
        }
    saved = ctx.get("saved") or {}
    file_counts: dict[str, int] = {}
    for path, records in ctx["all_file_entries"]:
        file_counts[path.name] = len(records)
    return {
        "status": "ok",
        "now": ctx["now"].isoformat(),
        "files": [{"file_name": n, "n_entries": c} for n, c in file_counts.items()],
        "entries": [
            late_swap.entry_to_dict(e, ctx["lookup"], ctx["now"]) for e in ctx["states"]
        ],
        "bulk_marked_player_ids": saved.get("bulk_marked_player_ids", []),
        "bulk_marked_teams": saved.get("bulk_marked_teams", []),
        "teams": sorted({str(t) for t in ctx["slate_df"]["team"] if t}),
        "last_run_at": saved.get("run_at"),
        "written_files": saved.get("written_files", []),
    }


@app.get("/api/late-swap/state")
def get_late_swap_state():
    return _late_swap_state_response(_late_swap_context())


@app.post("/api/late-swap/run")
def run_late_swap(req: LateSwapRunRequest):
    from . import late_swap

    if not _late_swap_lock.acquire(blocking=False):
        raise HTTPException(409, "A late swap run is already in progress")
    try:
        ctx = _late_swap_context()
        if ctx["status"] != "ok":
            raise HTTPException(400, f"Late swap unavailable: {ctx['status']}")
        bulk_pids = set(req.bulk_marked_player_ids)
        bulk_teams = set(req.bulk_marked_teams)
        entry_marks = {k: set(v) for k, v in req.entry_marks.items()}
        late_swap.run_swap(
            ctx["states"], entry_marks, bulk_pids, bulk_teams,
            ctx["candidates_df"], ctx["lookup"], late_swap.HeuristicScorer(),
            ctx["now"],
        )
        written = late_swap.write_swap_files(ctx["states"], ctx["output_dir"], ctx["lookup"])
        late_swap.save_state(
            ctx["output_dir"], ctx["fingerprint"], _now_eastern().isoformat(),
            bulk_pids, bulk_teams, ctx["states"], written,
        )
        ctx["saved"] = late_swap.load_state(ctx["output_dir"], ctx["fingerprint"])
        return _late_swap_state_response(ctx)
    finally:
        _late_swap_lock.release()


@app.get("/api/late-swap/candidates")
def get_late_swap_candidates(entry_id: str, slot_index: int):
    from . import late_swap

    ctx = _late_swap_context()
    if ctx["status"] != "ok":
        raise HTTPException(400, f"Late swap unavailable: {ctx['status']}")
    entry = next((e for e in ctx["states"] if e.entry_id == entry_id), None)
    if entry is None:
        raise HTTPException(404, f"Entry {entry_id} not found")
    if not (0 <= slot_index < len(entry.slots)):
        raise HTTPException(404, f"Invalid slot index {slot_index}")
    saved = ctx.get("saved") or {}
    cands = late_swap.candidates_for_slot(
        entry, slot_index, ctx["candidates_df"], ctx["lookup"],
        set(saved.get("bulk_marked_player_ids", [])),
        set(saved.get("bulk_marked_teams", [])),
        ctx["now"],
    )
    return {
        "candidates": [
            {
                "player_id": c["player_id"], "name": c["name"], "team": c["team"],
                "position": c["position"], "eligible_positions": c["eligible_positions"],
                "salary": c["salary"], "mean": c["mean"], "score": c["score"],
                "newly_confirmed": bool(c.get("newly_confirmed", False)),
            }
            for c in cands
        ],
        "max_salary": late_swap.slot_max_salary(entry, slot_index, ctx["lookup"]),
    }


@app.post("/api/late-swap/override")
def override_late_swap(req: LateSwapOverrideRequest):
    from . import late_swap

    if not _late_swap_lock.acquire(blocking=False):
        raise HTTPException(409, "A late swap run is already in progress")
    try:
        ctx = _late_swap_context()
        if ctx["status"] != "ok":
            raise HTTPException(400, f"Late swap unavailable: {ctx['status']}")
        entry = next((e for e in ctx["states"] if e.entry_id == req.entry_id), None)
        if entry is None:
            raise HTTPException(404, f"Entry {req.entry_id} not found")
        saved = ctx.get("saved") or {}
        bulk_pids = set(saved.get("bulk_marked_player_ids", []))
        bulk_teams = set(saved.get("bulk_marked_teams", []))
        err = late_swap.apply_override(
            entry, req.slot_index, req.player_id,
            ctx["candidates_df"], ctx["lookup"], bulk_pids, bulk_teams, ctx["now"],
        )
        if err:
            raise HTTPException(422, err)
        written = late_swap.write_swap_files(ctx["states"], ctx["output_dir"], ctx["lookup"])
        # Stamp the write time so the UI banner reflects the latest file update.
        updated_at = _now_eastern().isoformat()
        late_swap.save_state(
            ctx["output_dir"], ctx["fingerprint"], updated_at,
            bulk_pids, bulk_teams, ctx["states"], written,
        )
        return {
            "entry": late_swap.entry_to_dict(entry, ctx["lookup"], ctx["now"]),
            "written_files": written,
            "last_run_at": updated_at,
        }
    finally:
        _late_swap_lock.release()


@app.post("/api/late-swap/reset")
def reset_late_swap():
    from . import late_swap

    output_dir = str(_output_dir_path())
    late_swap.clear_state(output_dir)
    late_swap.delete_swap_files(output_dir)
    return _late_swap_state_response(_late_swap_context())


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
