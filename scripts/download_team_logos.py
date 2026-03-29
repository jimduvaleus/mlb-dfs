"""
Download MLB team logos from ESPN CDN and cache them for local use.

Logos are saved to ui/public/team-logos/{ABBREV}.png
URL pattern: https://a.espncdn.com/i/teamlogos/mlb/500/{abbrev_lower}.png

Usage:
    python scripts/download_team_logos.py
    python scripts/download_team_logos.py --force   # re-download even if cached
"""

import argparse
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ui" / "public" / "team-logos"

ESPN_LOGO_URL = "https://a.espncdn.com/i/teamlogos/mlb/500/{abbrev}.png"

# All 30 MLB teams using DraftKings abbreviations
MLB_TEAMS = [
    "ATL", "ARI", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC",  "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD",  "SF",  "SEA", "STL", "TB",  "TEX", "TOR", "WSH",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


def download_logos(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = [], [], []

    for abbrev in MLB_TEAMS:
        dest = OUTPUT_DIR / f"{abbrev}.png"
        if dest.exists() and not force:
            log.info("Skipping %s (already cached)", abbrev)
            skipped.append(abbrev)
            continue

        url = ESPN_LOGO_URL.format(abbrev=abbrev.lower())
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            log.info("Downloaded %s (%d bytes)", abbrev, len(resp.content))
            ok.append(abbrev)
        except Exception as exc:
            log.error("Failed to download %s: %s", abbrev, exc)
            failed.append(abbrev)

    log.info(
        "Done. downloaded=%d  skipped=%d  failed=%d  → %s",
        len(ok), len(skipped), len(failed), OUTPUT_DIR,
    )

    if failed:
        log.error("Failed teams: %s", failed)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MLB team logos from ESPN CDN.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download logos even if already cached",
    )
    args = parser.parse_args()
    download_logos(force=args.force)


if __name__ == "__main__":
    main()
