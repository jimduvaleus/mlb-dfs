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
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config_io import read_config, write_config
from .models import AppConfig, ExclusionsUpdate, GameStatus, PlayerExclusionStatus, PlayerExclusionsUpdate, PortfolioResult, ProjectionsStatus, SlateGamesResponse, SlateListResponse, SlateOption, SlatePlayersResponse
from .projections_meta import (
    compute_freshness,
    fetch_and_cache_slates,
    get_cached_slates,
    get_status_fields,
    record_fetch_from_csv,
)
from .slate_exclusions import get_slate_games_with_status, get_slate_players_with_status, prune_player_exclusions, read_exclusions, write_exclusions

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


def _portfolio_csv_path() -> Path:
    cfg = read_config()
    output_dir = cfg.paths.output_dir or "outputs"
    base = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    return base / "portfolio.csv"


@app.on_event("startup")
def _load_persisted_portfolio() -> None:
    """Restore the last portfolio from portfolio.csv so the UI shows it after restart."""
    import pandas as pd

    try:
        path = _portfolio_csv_path()
        if not path.exists():
            return
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
                }
                for _, row in group.iterrows()
            ]
            portfolio.append({
                "lineup_index": int(lineup_idx),
                "p_hit_target": float(first["p_hit_target"]),
                "lineup_salary": int(first["lineup_salary"]),
                "players": players,
            })
        if portfolio:
            _state["portfolio"] = portfolio
            _state["status"] = "complete"
    except Exception:
        pass  # corrupt or missing CSV — start fresh


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
# Slate game/team exclusion endpoints
# ---------------------------------------------------------------------------

def _load_slate_df():
    """Parse the configured DK slate CSV and return a DataFrame (or None)."""
    cfg = read_config()
    dk_path = cfg.paths.dk_slate
    if not dk_path:
        return None
    p = PROJECT_ROOT / dk_path if not Path(dk_path).is_absolute() else Path(dk_path)
    if not p.exists():
        return None
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    return DraftKingsSlateIngestor(str(p)).get_slate_dataframe()


def _load_slate_games() -> list[str]:
    """Parse the configured DK slate CSV and return unique game strings."""
    df = _load_slate_df()
    if df is None:
        return []
    return [g for g in df["game"].dropna().unique().tolist() if g]


@app.get("/api/slate/games")
def get_slate_games() -> SlateGamesResponse:
    games = _load_slate_games()
    if not games:
        return SlateGamesResponse(slate_id="", games=[], excluded_player_ids=[])
    slate_id, game_dicts, excluded_player_ids = get_slate_games_with_status(games)
    return SlateGamesResponse(
        slate_id=slate_id,
        games=[GameStatus(**g) for g in game_dicts],
        excluded_player_ids=excluded_player_ids,
    )


@app.post("/api/slate/exclusions")
def post_slate_exclusions(update: ExclusionsUpdate) -> SlateGamesResponse:
    # Load existing player exclusions so we can prune them
    stored = read_exclusions()
    existing_player_ids: list[int] = []
    if stored.get("slate_id") == update.slate_id:
        existing_player_ids = stored.get("excluded_player_ids", [])

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
        excluded_teams=update.excluded_teams,
        excluded_games=update.excluded_games,
        excluded_player_ids=existing_player_ids,
    )
    games = _load_slate_games()
    if not games:
        return SlateGamesResponse(slate_id=update.slate_id, games=[], excluded_player_ids=[])
    slate_id, game_dicts, excluded_player_ids = get_slate_games_with_status(games)
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
    player_dicts = get_slate_players_with_status(df, slate_id)
    return SlatePlayersResponse(
        slate_id=slate_id,
        players=[PlayerExclusionStatus(**p) for p in player_dicts],
    )


@app.post("/api/slate/player-exclusions")
def post_player_exclusions(update: PlayerExclusionsUpdate) -> SlatePlayersResponse:
    stored = read_exclusions()
    excluded_teams: list[str] = []
    excluded_games: list[str] = []
    if stored.get("slate_id") == update.slate_id:
        excluded_teams = stored.get("excluded_teams", [])
        excluded_games = stored.get("excluded_games", [])

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
        excluded_teams=excluded_teams,
        excluded_games=excluded_games,
        excluded_player_ids=pruned_ids,
    )

    if df is None or df.empty:
        return SlatePlayersResponse(slate_id=update.slate_id, players=[])
    player_dicts = get_slate_players_with_status(df, update.slate_id)
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
    proj_path = cfg.paths.projections
    extra = get_status_fields()

    # Resolve DK path for freshness check
    dk_raw = cfg.paths.dk_slate
    dk_path: Path | None = None
    if dk_raw:
        dk_path = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
        if not dk_path.exists():
            dk_path = None

    if not proj_path:
        return ProjectionsStatus(exists=False, **extra)

    p = PROJECT_ROOT / proj_path if not Path(proj_path).is_absolute() else Path(proj_path)
    if not p.exists():
        return ProjectionsStatus(exists=False, path=str(proj_path), **extra)

    stat = p.stat()
    age = time.time() - stat.st_mtime

    row_count = None
    try:
        import pandas as pd
        df = pd.read_csv(p)
        row_count = len(df)
    except Exception:
        pass

    is_fresh = compute_freshness(dk_path, p) if dk_path is not None else None

    return ProjectionsStatus(
        exists=True,
        path=str(proj_path),
        last_modified=stat.st_mtime,
        age_seconds=age,
        row_count=row_count,
        is_fresh=is_fresh,
        **extra,
    )


@app.get("/api/projections/slates")
async def projections_slates() -> SlateListResponse:
    cfg = read_config()
    dk_raw = cfg.paths.dk_slate
    if not dk_raw:
        return SlateListResponse(date=None, slates=[])
    dk_path = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
    if not dk_path.exists():
        return SlateListResponse(date=None, slates=[])

    cached = get_cached_slates(dk_path)
    if cached is not None:
        return SlateListResponse(
            date=None,
            slates=[SlateOption(**s) for s in cached],
        )

    try:
        dk_date, options = fetch_and_cache_slates(dk_path)
    except Exception:
        return SlateListResponse(date=None, slates=[])

    return SlateListResponse(
        date=dk_date,
        slates=[SlateOption(**s) for s in options],
    )


@app.get("/api/projections/fetch")
async def projections_fetch(request: Request):
    if _state["status"] == "running":
        raise HTTPException(409, "A pipeline run is already in progress")

    cfg = read_config()
    proj_path_str = cfg.paths.projections or "data/processed/projections.csv"
    proj_path = PROJECT_ROOT / proj_path_str if not Path(proj_path_str).is_absolute() else Path(proj_path_str)

    dk_raw = cfg.paths.dk_slate
    dk_path: Path | None = None
    if dk_raw:
        _dp = PROJECT_ROOT / dk_raw if not Path(dk_raw).is_absolute() else Path(dk_raw)
        if _dp.exists():
            dk_path = _dp

    async def _stream():
        script = PROJECT_ROOT / "scripts" / "fetch_rotowire_projections.py"
        python = PROJECT_ROOT / "venv" / "bin" / "python"
        cmd = [str(python), str(script)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            if await request.is_disconnected():
                proc.kill()
                break
            payload = {"type": "log", "line": line.decode().rstrip(), "timestamp": int(time.time() * 1000)}
            yield f"data: {json.dumps(payload)}\n\n"
        returncode = await proc.wait()
        if returncode == 0:
            try:
                record_fetch_from_csv(proj_path, "auto", dk_path)
            except Exception:
                pass
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
    if _state["status"] == "running":
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


@app.get("/api/portfolio")
def get_portfolio():
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
