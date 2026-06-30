"""Bootstrap real historical contest score distributions as field samples
for ContestScorer.

When ``field_source = "historical"`` is set in the gpp config, ContestScorer
replaces its model-generated opponent field with bootstrapped samples drawn
from past real DraftKings contest standings ZIPs.  This grounds the EV
calculation in the actual score distribution a lineup must beat, rather than
a distribution that is doubly bounded by the same copula/marginal model used
to generate our own lineup scores.

The model-simulated field has a tail ceiling set by historical bootstrapping
of the copula — real tail outcomes (team explosion, non-consensus stack going
off) are underweighted because the model assigns them low probability by
construction.  Grounding the *opponent field* in observed past score
distributions means that candidates are judged against a harder, more realistic
ceiling at the 99th percentile, which changes what "high EV" means: lineups
that survive now must have high-ceiling potential, not just high-mean potential.

Public interface
----------------
  load_historical_distributions(archive_root, n_slates, exclude_date)
      → list of sorted (N_i,) float64 arrays (one per historical slate)

  estimate_current_slate_ref(sim_matrix, field_ownership_vec)
      → float  (median expected lineup score for the current slate)

  build_historical_field_samples(distributions, n_field, n_sims, current_ref, rng, K)
      → list of K (n_sims, n_field) float32 C-contiguous arrays, shaped
        exactly as ContestScorer._field_sorted_list entries.
"""
import csv
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _slate_sort_key(name: str) -> tuple:
    """Chronological sort key for MMDDYYYY or MMDDYYYY<suffix> dir names."""
    m = re.match(r"^(\d{2})(\d{2})(\d{4})", name)
    if m:
        mm, dd, yyyy = m.groups()
        return (int(yyyy), int(mm), int(dd), name[8:])
    return (9999, 0, 0, name)


def parse_contest_points_from_zip(zip_path: Path) -> np.ndarray:
    """Extract a sorted float64 array of field entry scores from a DK
    contest-standings ZIP.

    Parses the per-entry table section (Rank, EntryId, EntryName,
    TimeRemaining, Points, Lineup) that precedes the player-ownership sidebar
    in the glued CSV export.  Raises ValueError if the file cannot be parsed
    or contains no Points data.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        content = zf.read(csv_name).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise ValueError(f"contest standings CSV in {zip_path.name} is empty")
    try:
        points_col = rows[0].index("Points")
    except ValueError:
        raise ValueError(f"no 'Points' column header in {zip_path.name}")
    points = []
    for row in rows[1:]:
        if len(row) > points_col and row[points_col].strip():
            try:
                points.append(float(row[points_col]))
            except ValueError:
                continue
    if not points:
        raise ValueError(f"no parseable Points values in {zip_path.name}")
    return np.sort(np.array(points, dtype=np.float64))


def load_historical_distributions(
    archive_root: Path,
    n_slates: int = 10,
    exclude_date: str | None = None,
) -> list[np.ndarray]:
    """Load sorted real field Points distributions from the N most recent
    archived contest-standings ZIPs, excluding the current slate date.

    Directories without a contest-standings-*.zip, or ZIPs that fail to parse,
    are silently skipped.  Returns an empty list if no qualifying slates are
    found — callers should fall back to simulated mode in that case.
    """
    dirs = sorted(
        (
            d for d in archive_root.iterdir()
            if d.is_dir() and d.name != exclude_date
        ),
        key=lambda d: _slate_sort_key(d.name),
    )
    selected = dirs[-n_slates:]
    distributions: list[np.ndarray] = []
    for d in selected:
        zips = sorted(d.glob("contest-standings-*.zip"))
        if not zips:
            continue
        try:
            pts = parse_contest_points_from_zip(zips[0])
        except Exception as exc:
            logger.debug("historical_field: skipping %s — %s", d.name, exc)
            continue
        if len(pts) < 100:
            logger.warning(
                "historical_field: %s has only %d entries — skipping",
                d.name, len(pts),
            )
            continue
        distributions.append(pts)
    logger.info(
        "historical_field: loaded %d distributions (n_slates=%d, exclude=%s)",
        len(distributions), n_slates, exclude_date,
    )
    return distributions


# ---------------------------------------------------------------------------
# Current-slate scoring reference
# ---------------------------------------------------------------------------

def estimate_current_slate_ref(
    sim_matrix: np.ndarray,
    field_ownership_vec: np.ndarray | None = None,
    n_sample: int = 500,
    rng_seed: int = 0,
) -> float:
    """Estimate the median expected lineup score for the current slate.

    Samples n_sample random 10-player combinations from the sim_matrix
    (weighted by field_ownership_vec when provided) and returns the median
    total score across all (n_sims × n_sample) values.  Used as the
    normalization reference when scaling historical field distributions to
    the current slate's scoring environment.

    The random sampling ignores roster-construction constraints (position
    eligibility, salary cap), which is fine here — we only need an
    order-of-magnitude estimate of the current slate's typical lineup score
    to anchor the linear scaling applied to each historical distribution.
    """
    rng = np.random.default_rng(rng_seed)
    n_players = sim_matrix.shape[1]
    if field_ownership_vec is not None and len(field_ownership_vec) == n_players:
        w = np.asarray(field_ownership_vec, dtype=float)
        total = w.sum()
        probs = w / total if total > 0 else None
    else:
        probs = None
    indices = rng.choice(n_players, size=(n_sample, 10), replace=True, p=probs)
    # sim_matrix: (n_sims, n_players); result: (n_sims, n_sample)
    sample_scores = sim_matrix[:, indices].sum(axis=2)
    return float(np.median(sample_scores))


# ---------------------------------------------------------------------------
# Field sample construction
# ---------------------------------------------------------------------------

def build_historical_field_samples(
    distributions: list[np.ndarray],
    n_field: int,
    n_sims: int,
    current_ref: float,
    rng: np.random.Generator,
    K: int = 3,
) -> list[np.ndarray]:
    """Produce K field samples shaped as ContestScorer._field_sorted_list expects.

    Each of the K samples:
      1. Picks one historical distribution uniformly at random.
      2. Scales it linearly so its median matches current_ref (aligns the
         historical scoring level to the current slate's environment — a 15-game
         slate scores higher than a 5-game slate, so the scale factor corrects
         for that difference).
      3. Bootstrap-resamples to exactly n_field entries with replacement
         (standardises the field size so the existing payout_lookup remains
         valid; also injects healthy inter-sample variance so the K draws are
         not identical even when the same historical slate is chosen twice).
      4. Tiles to shape (n_sims, n_field) float32 — the field scores are
         *fixed* across simulations because they are observed historical data,
         not modeled draws.  Each candidate's simulated score (which varies
         across sims via the copula/marginals) is then measured against this
         fixed real-world reference.

    Raises ValueError if distributions is empty.
    """
    if not distributions:
        raise ValueError(
            "build_historical_field_samples: no historical distributions loaded."
        )

    samples: list[np.ndarray] = []
    for _ in range(K):
        dist_raw = distributions[rng.integers(len(distributions))]

        hist_ref = float(np.median(dist_raw))
        scale = (current_ref / hist_ref) if hist_ref > 0 else 1.0
        dist_scaled = dist_raw * scale

        # Bootstrap resample to target field size
        resampled = np.sort(
            rng.choice(dist_scaled, size=n_field, replace=True).astype(np.float32)
        )

        # Tile to (n_sims, n_field) — same distribution in every sim row
        row = resampled.reshape(1, -1)
        field_sorted = np.ascontiguousarray(np.tile(row, (n_sims, 1)))
        samples.append(field_sorted)

    return samples
