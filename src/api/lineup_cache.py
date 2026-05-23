"""Disk cache for GPP candidate and field lineups.

Keyed by slate file fingerprint (mtime_ns:size), matching the mechanism in
slate_exclusions.py. Cache is stored in data/lineup_cache/ as .npz files.
Old fingerprint files are pruned automatically after each write.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.optimization.lineup import Lineup

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _PROJECT_ROOT / "data" / "lineup_cache"


def _safe_fp(fingerprint: str) -> str:
    """Convert 'mtime_ns:size' to a path-safe string."""
    return fingerprint.replace(":", "_")


def get_cache_status(slate_path: str | Path) -> dict:
    """Return cache availability for the given slate file.

    Returns
    -------
    dict with keys:
        fingerprint : str  (empty if slate_path missing)
        candidates  : int | None  (None = no cache)
        field_k     : int | None  (None = no cache)
    """
    from src.api.slate_exclusions import compute_file_fingerprint

    fp = compute_file_fingerprint(Path(slate_path) if slate_path else None)
    result: dict = {"fingerprint": fp, "candidates": None, "field_k": None}

    if not fp:
        return result

    safe = _safe_fp(fp)
    cand_path = CACHE_DIR / f"candidates_{safe}.npz"
    field_path = CACHE_DIR / f"field_{safe}.npz"

    if cand_path.exists():
        try:
            with np.load(cand_path) as npz:
                arr = npz["arr"]
                if arr.ndim == 2 and arr.shape[1] == 10:
                    result["candidates"] = int(arr.shape[0])
        except Exception:
            pass

    if field_path.exists():
        try:
            with np.load(field_path) as npz:
                keys = [k for k in npz.files if k.startswith("field_")]
                valid = all(
                    npz[k].ndim == 2 and npz[k].shape[1] == 10 for k in keys
                )
                if keys and valid:
                    result["field_k"] = len(keys)
        except Exception:
            pass

    return result


def load_candidates(fingerprint: str) -> list[Lineup] | None:
    """Load cached candidates. Returns None on miss or corruption."""
    from src.optimization.lineup import Lineup

    safe = _safe_fp(fingerprint)
    path = CACHE_DIR / f"candidates_{safe}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as npz:
            arr = npz["arr"]
            if arr.ndim != 2 or arr.shape[1] != 10:
                logger.warning("lineup_cache: candidates shape mismatch %s", arr.shape)
                return None
            candidates = [Lineup(player_ids=row.tolist()) for row in arr]
        logger.info("lineup_cache: loaded %d candidates from cache", len(candidates))
        return candidates
    except Exception as exc:
        logger.warning("lineup_cache: failed to load candidates: %s", exc)
        return None


def save_candidates(fingerprint: str, candidates: list[Lineup]) -> None:
    """Serialize candidates to disk. Overwrites existing file then prunes stale files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.array([lu.player_ids for lu in candidates], dtype=np.int64)
    safe = _safe_fp(fingerprint)
    path = CACHE_DIR / f"candidates_{safe}.npz"
    np.savez_compressed(path, arr=arr)
    logger.info("lineup_cache: saved %d candidates to %s", len(candidates), path.name)
    prune_stale_cache(fingerprint)


def load_field(fingerprint: str) -> list[np.ndarray] | None:
    """Load cached field arrays. Returns list of K arrays (n_field, 10) or None."""
    safe = _safe_fp(fingerprint)
    path = CACHE_DIR / f"field_{safe}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as npz:
            keys = sorted(k for k in npz.files if k.startswith("field_"))
            if not keys:
                return None
            arrays = []
            for k in keys:
                a = npz[k]
                if a.ndim != 2 or a.shape[1] != 10:
                    logger.warning("lineup_cache: field array %s shape mismatch %s", k, a.shape)
                    return None
                arrays.append(a.astype(np.int64))
        logger.info(
            "lineup_cache: loaded %d field samples (%d lineups each) from cache",
            len(arrays), arrays[0].shape[0] if arrays else 0,
        )
        return arrays
    except Exception as exc:
        logger.warning("lineup_cache: failed to load field: %s", exc)
        return None


def save_field(fingerprint: str, field_list: list[np.ndarray]) -> None:
    """Serialize K field arrays to disk. Overwrites existing file then prunes stale files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_fp(fingerprint)
    path = CACHE_DIR / f"field_{safe}.npz"
    arrays = {f"field_{k}": arr.astype(np.int64) for k, arr in enumerate(field_list)}
    np.savez_compressed(path, **arrays)
    total = sum(a.shape[0] for a in field_list)
    logger.info(
        "lineup_cache: saved %d field samples (%d total lineups) to %s",
        len(field_list), total, path.name,
    )
    prune_stale_cache(fingerprint)


def prune_stale_cache(current_fingerprint: str) -> None:
    """Delete cache files whose fingerprint doesn't match the current one."""
    if not CACHE_DIR.exists():
        return
    safe = _safe_fp(current_fingerprint)
    for p in CACHE_DIR.iterdir():
        if p.suffix == ".npz" and not p.stem.endswith(safe):
            try:
                p.unlink()
                logger.info("lineup_cache: pruned stale file %s", p.name)
            except Exception as exc:
                logger.warning("lineup_cache: could not prune %s: %s", p.name, exc)
