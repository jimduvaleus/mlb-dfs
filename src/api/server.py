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
        if portfolio:
            meta_path = path.parent / "portfolio_entries.json"
            if meta_path.exists():
                import json
                with open(meta_path) as mf:
                    entry_meta = json.load(mf)
                for lr in portfolio:
                    info = entry_meta.get(str(lr["lineup_index"]))
                    if info:
                        lr.update(info)
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


@app.get("/api/projections/unconfirmed")
def projections_unconfirmed():
    cfg = read_config()
    proj_path_str = cfg.paths.projections
    if not proj_path_str:
        return {"player_ids": []}
    p = PROJECT_ROOT / proj_path_str if not Path(proj_path_str).is_absolute() else Path(proj_path_str)
    if not p.exists():
        return {"player_ids": []}
    try:
        import pandas as pd
        df = pd.read_csv(p)
        if "slot_confirmed" not in df.columns or "player_id" not in df.columns:
            return {"player_ids": []}
        unconfirmed = df[~df["slot_confirmed"].astype(bool)]["player_id"].tolist()
        return {"player_ids": [int(x) for x in unconfirmed]}
    except Exception:
        return {"player_ids": []}


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

        def _log(msg: str) -> str:
            return f"data: {json.dumps({'type': 'log', 'line': msg, 'timestamp': int(time.time() * 1000)})}\n\n"

        # ---- Market Odds path -----------------------------------------------
        if is_market_preferred:
            # Step 1: RotoWire is always required (player pool + lineup slots)
            yield _log("--- Fetching RotoWire projections (player pool) ---")
            proc = await asyncio.create_subprocess_exec(
                str(python), str(rw_script), "--output", str(rw_out),
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
                yield f"data: {json.dumps({'type': 'done', 'returncode': rw_rc, 'timestamp': int(time.time() * 1000)})}\n\n"
                return

            # Step 2: Market odds (non-fatal — fall back to RW entirely on failure)
            yield _log("--- Fetching Market Odds (CrazyNinjaOdds) ---")
            proc2 = await asyncio.create_subprocess_exec(
                str(python), str(mo_script), "--output", str(mo_out),
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

            # Step 3: Merge — RW is pool, market odds overlays values
            try:
                import pandas as pd

                rw_df = pd.read_csv(rw_out) if rw_out.exists() else pd.DataFrame()
                mo_df = (
                    pd.read_csv(mo_out)
                    if (mo_out.exists() and mo_rc == 0)
                    else pd.DataFrame()
                )

                if rw_df.empty:
                    yield _log("Error: RotoWire produced no usable data.")
                    yield f"data: {json.dumps({'type': 'done', 'returncode': 1, 'timestamp': int(time.time() * 1000)})}\n\n"
                    return

                pool = rw_df.copy()
                fallback_players: list[dict] = []

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
                    fallback_rows = pool.loc[~has_pref] if "name" in pool.columns else pd.DataFrame()

                    if not fallback_rows.empty:
                        id_to_team: dict = {}
                        if dk_path is not None:
                            try:
                                dk_df = pd.read_csv(dk_path, usecols=["ID", "TeamAbbrev"])
                                id_to_team = dict(zip(dk_df["ID"], dk_df["TeamAbbrev"]))
                            except Exception:
                                pass
                        for _, row in fallback_rows.iterrows():
                            fallback_players.append({
                                "name": row["name"],
                                "team": id_to_team.get(int(row["player_id"]), "") if "player_id" in fallback_rows.columns else "",
                            })
                else:
                    yield _log(
                        "Warning: Market odds unavailable; using RotoWire projections for all players."
                    )

                out_cols  = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
                merged_df = pool[[c for c in out_cols if c in pool.columns]]
                merged_df = merged_df.sort_values("mean", ascending=False).reset_index(drop=True)
                merged_df.to_csv(proj_path, index=False)

                if fallback_players:
                    yield f"data: {json.dumps({'type': 'merge_info', 'secondary_source': 'RotoWire', 'count': len(fallback_players), 'players': fallback_players, 'timestamp': int(time.time() * 1000)})}\n\n"
                else:
                    yield _log("All player projections sourced from Market Odds (CrazyNinjaOdds).")

            except Exception as exc:
                yield _log(f"Warning: merge error — {exc}")
            finally:
                for p in (rw_out, mo_out):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

            try:
                record_fetch_from_csv(proj_path, "auto", dk_path)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'done', 'returncode': 0, 'timestamp': int(time.time() * 1000)})}\n\n"
            return

        # ---- Run preferred script (live-streamed) ----------------------------
        yield _log(f"--- Fetching {preferred_label} projections ---")
        preferred_script = dff_script if is_dff_preferred else rw_script
        preferred_out    = dff_out    if is_dff_preferred else rw_out

        proc = await asyncio.create_subprocess_exec(
            str(python), str(preferred_script), "--output", str(preferred_out),
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
            yield f"data: {json.dumps({'type': 'done', 'returncode': preferred_rc, 'timestamp': int(time.time() * 1000)})}\n\n"
            return

        # ---- Run fallback script (live-streamed) -----------------------------
        yield _log(f"--- Fetching {fallback_label} projections ---")
        fallback_script = rw_script  if is_dff_preferred else dff_script
        fallback_out    = rw_out     if is_dff_preferred else dff_out

        proc2 = await asyncio.create_subprocess_exec(
            str(python), str(fallback_script), "--output", str(fallback_out),
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
        fallback_rc = await proc2.wait()

        # ---- Merge both outputs into final projections.csv ------------------
        try:
            import pandas as pd

            dff_df = pd.read_csv(dff_out) if dff_out.exists() else pd.DataFrame()
            rw_df  = pd.read_csv(rw_out)  if rw_out.exists()  else pd.DataFrame()

            # --- Step 1: build player pool (slot/roster membership) ----------
            # RW pool  — all starters already filtered by the fetch script
            # DFF pool — confirmed batters only (slot_confirmed=True) + all pitchers
            #            (DFF already excludes zero-projection pitchers in its parser)
            rw_pool = rw_df.copy()

            if not dff_df.empty and {"lineup_slot", "slot_confirmed"}.issubset(dff_df.columns):
                dff_pitchers = dff_df[dff_df["lineup_slot"] == 10]
                dff_batters  = dff_df[(dff_df["lineup_slot"] != 10) & dff_df["slot_confirmed"].astype(bool)]
                dff_pool     = pd.concat([dff_batters, dff_pitchers], ignore_index=True)
            else:
                dff_pool = dff_df.copy()

            if rw_pool.empty and dff_pool.empty:
                yield _log("Error: both projection sources produced no usable data.")
                yield f"data: {json.dumps({'type': 'done', 'returncode': 1, 'timestamp': int(time.time() * 1000)})}\n\n"
                return

            # Union: start with RW starters, append any DFF-pool players not in RW
            rw_ids    = set(rw_pool["player_id"].tolist()) if not rw_pool.empty else set()
            dff_extra = dff_pool[~dff_pool["player_id"].isin(rw_ids)] if not dff_pool.empty else pd.DataFrame()
            pool      = pd.concat([rw_pool, dff_extra], ignore_index=True)

            # --- Step 2: apply preferred source's mean/std_dev ---------------
            # The full DFF CSV (including unconfirmed batters) and the RW CSV are
            # both used as projection-value lookup tables.  Pool membership (step 1)
            # and projection-value preference are independent: a RW batter enters
            # the pool via RW but still gets DFF's fantasy-point projection if DFF
            # is preferred and has a value for them.
            pref_proj_df = dff_df if is_dff_preferred else rw_df

            fallback_players: list[dict] = []
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
                fallback_rows = pool.loc[~has_pref] if "name" in pool.columns else pd.DataFrame()

                # Build player list with team info from DK slate
                if not fallback_rows.empty:
                    id_to_team: dict = {}
                    if dk_path is not None:
                        try:
                            dk_df = pd.read_csv(dk_path, usecols=["ID", "TeamAbbrev"])
                            id_to_team = dict(zip(dk_df["ID"], dk_df["TeamAbbrev"]))
                        except Exception:
                            pass
                    for _, row in fallback_rows.iterrows():
                        fallback_players.append({
                            "name": row["name"],
                            "team": id_to_team.get(int(row["player_id"]), "") if "player_id" in fallback_rows.columns else "",
                        })


            out_cols  = ["player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"]
            merged_df = pool[[c for c in out_cols if c in pool.columns]]
            merged_df = merged_df.sort_values("mean", ascending=False).reset_index(drop=True)
            merged_df.to_csv(proj_path, index=False)

            if fallback_players:
                yield f"data: {json.dumps({'type': 'merge_info', 'secondary_source': fallback_label, 'count': len(fallback_players), 'players': fallback_players, 'timestamp': int(time.time() * 1000)})}\n\n"
            else:
                yield _log(f"All player projections sourced from {preferred_label}.")

        except Exception as exc:
            yield _log(f"Warning: merge error — {exc}")
        finally:
            for p in (dff_out, rw_out):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        try:
            record_fetch_from_csv(proj_path, "auto", dk_path)
        except Exception:
            pass
        yield f"data: {json.dumps({'type': 'done', 'returncode': 0, 'timestamp': int(time.time() * 1000)})}\n\n"

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
