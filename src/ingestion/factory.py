"""
Ingestor factory for platform-specific slate parsing.

Auto-discovery
--------------
FanDuel exports a uniquely-named file each time
(e.g. ``FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv``).
If ``slate_path`` is empty when building a FD ingestor, ``find_fd_slate``
scans ``data/raw/`` for all matching files and returns the one whose
filename encodes the most recent date.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

from src.ingestion.dk_slate import BaseSlateIngestor, DraftKingsSlateIngestor
from src.ingestion.fd_slate import FanDuelSlateIngestor
from src.platforms.base import Platform

# Default search directory, relative to repo root.
_DEFAULT_RAW_DIR = "data/raw"

# Filename pattern for FD salary/upload CSV exports.
_FD_FILENAME_PATTERN = "FanDuel-MLB-*.csv"

# Regex to extract the YYYY-MM-DD embedded in the FD filename.
_FD_DATE_RE = re.compile(r"FanDuel-MLB-(\d{4})-(\d{2})-(\d{2})-")


def find_fd_slate(raw_dir: Optional[str] = None) -> Optional[str]:
    """
    Scan *raw_dir* for FanDuel salary CSV exports and return the path of the
    file whose filename encodes the most recent date.

    *raw_dir* defaults to ``data/raw`` (looked up at call time so that tests
    can patch ``src.ingestion.factory._DEFAULT_RAW_DIR``).

    Returns ``None`` if no matching files are found.

    File naming convention::

        FanDuel-MLB-{YYYY}-{MM}-{DD}-{slate_id}-entries-upload-template.csv

    If multiple files share the same date (different slate IDs on the same
    day), the lexicographically last path is returned, which is deterministic
    even if arbitrary.
    """
    if raw_dir is None:
        raw_dir = _DEFAULT_RAW_DIR

    pattern = os.path.join(raw_dir, _FD_FILENAME_PATTERN)
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _date_key(path: str):
        m = _FD_DATE_RE.search(os.path.basename(path))
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return (0, 0, 0)

    return max(candidates, key=_date_key)


def get_ingestor(platform: Platform, slate_path: str = "") -> BaseSlateIngestor:
    """
    Return the appropriate :class:`BaseSlateIngestor` for *platform*.

    For FanDuel, if *slate_path* is empty the factory will auto-discover the
    most recent FD CSV in ``data/raw/`` via :func:`find_fd_slate`.

    Raises
    ------
    FileNotFoundError
        If a FD slate is needed but cannot be located.
    ValueError
        If *platform* is unrecognised.
    """
    if platform == Platform.DRAFTKINGS:
        return DraftKingsSlateIngestor(slate_path)

    if platform == Platform.FANDUEL:
        path = slate_path or find_fd_slate() or ""
        if not path:
            raise FileNotFoundError(
                "No FanDuel salary CSV found in data/raw/. "
                "Download the FanDuel entries upload template and place it in data/raw/, "
                "or set paths.fd_slate in config.yaml."
            )
        return FanDuelSlateIngestor(path)

    raise ValueError(f"Unknown platform: {platform!r}")
