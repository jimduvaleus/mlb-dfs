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
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config_io import read_config, write_config
from .models import AppConfig, ExclusionsUpdate, GameStatus, PortfolioResult, ProjectionsStatus, SlateGamesResponse, SlateListResponse, SlateOption
from .projections_meta import (
    fetch_and_cache_slates,
    get_cached_slates,
    get_status_fields,
    record_fetch_from_csv,
)
from .slate_exclusions import get_slate_games_with_status, write_exclusions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIST = PROJECT_ROOT / "ui" / "dist"

app = FastAPI(title="MLB DFS Optimizer")

# ---------------------------------------------------------------------------
# Run state (single-user local tool — no locking needed beyond a flag)
# ---------------------------------------------------------------------------

_state: dict = {
    "status": "idle",      # idle | running | complete | error
    "portfolio": None,     # PortfolioResult dict or None
    "error": None,
}


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

def _load_slate_games() -> list[str]:
    """Parse the configured DK slate CSV and return unique game strings."""
    cfg = read_config()
    dk_path = cfg.paths.dk_slate
    if not dk_path:
        return []
    p = PROJECT_ROOT / dk_path if not Path(dk_path).is_absolute() else Path(dk_path)
    if not p.exists():
        return []
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    df = DraftKingsSlateIngestor(str(p)).get_slate_dataframe()
    return [g for g in df["game"].dropna().unique().tolist() if g]


@app.get("/api/slate/games")
def get_slate_games() -> SlateGamesResponse:
    games = _load_slate_games()
    if not games:
        return SlateGamesResponse(slate_id="", games=[])
    slate_id, game_dicts = get_slate_games_with_status(games)
    return SlateGamesResponse(
        slate_id=slate_id,
        games=[GameStatus(**g) for g in game_dicts],
    )


@app.post("/api/slate/exclusions")
def post_slate_exclusions(update: ExclusionsUpdate) -> SlateGamesResponse:
    write_exclusions(
        slate_id=update.slate_id,
        excluded_teams=update.excluded_teams,
        excluded_games=update.excluded_games,
    )
    games = _load_slate_games()
    if not games:
        return SlateGamesResponse(slate_id=update.slate_id, games=[])
    slate_id, game_dicts = get_slate_games_with_status(games)
    return SlateGamesResponse(
        slate_id=slate_id,
        games=[GameStatus(**g) for g in game_dicts],
    )


# ---------------------------------------------------------------------------
# Projections endpoints
# ---------------------------------------------------------------------------

@app.get("/api/projections/status")
def projections_status() -> ProjectionsStatus:
    cfg = read_config()
    proj_path = cfg.paths.projections
    extra = get_status_fields()

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

    return ProjectionsStatus(
        exists=True,
        path=str(proj_path),
        last_modified=stat.st_mtime,
        age_seconds=age,
        row_count=row_count,
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
async def projections_fetch(request: Request, slate_id: str | None = None):
    if _state["status"] == "running":
        raise HTTPException(409, "A pipeline run is already in progress")

    cfg = read_config()
    proj_path_str = cfg.paths.projections or "data/processed/projections.csv"
    proj_path = PROJECT_ROOT / proj_path_str if not Path(proj_path_str).is_absolute() else Path(proj_path_str)

    async def _stream():
        script = PROJECT_ROOT / "scripts" / "fetch_rotowire_projections.py"
        python = PROJECT_ROOT / "venv" / "bin" / "python"
        cmd = [str(python), str(script)]
        if slate_id:
            cmd += ["--slate-id", slate_id]
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
                record_fetch_from_csv(proj_path, slate_id or "auto")
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


@app.get("/api/run/stream")
async def run_stream(request: Request):
    if _state["status"] == "running":
        raise HTTPException(409, "A run is already in progress")

    _state["status"] = "running"
    _state["portfolio"] = None
    _state["error"] = None

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_cb(stage: str, data: dict) -> None:
        payload = {"stage": stage, "timestamp": int(time.time() * 1000), **data}
        asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

    def run_pipeline() -> None:
        try:
            from .pipeline import PipelineRunner
            runner = PipelineRunner(str(PROJECT_ROOT / "config.yaml"), progress_cb)
            portfolio = runner.run()
            _state["portfolio"] = portfolio
            _state["status"] = "complete"
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
