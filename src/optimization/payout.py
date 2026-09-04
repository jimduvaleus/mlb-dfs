"""Payout structure loading utilities."""

import json
from typing import Optional
from pathlib import Path

import numpy as np


PAYOUT_STRUCTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "payout_structures"


def load_payout_structure(name: str = "dk_classic_gpp") -> dict:
    """Load a payout structure JSON file by name.

    Returns the parsed dict with keys: name, entry_fee, total_entries, payouts.
    """
    path = PAYOUT_STRUCTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def payout_table_to_array(structure: dict) -> np.ndarray:
    """Expand a payout structure into a (total_entries,) array of payouts.

    Returns an array where index i is the payout for finishing in position i+1.
    """
    total = structure["total_entries"]
    payouts = np.zeros(total, dtype=np.float64)
    for tier in structure["payouts"]:
        start = tier["start"] - 1  # 0-indexed
        end = tier["end"]          # exclusive upper bound
        payouts[start:end] = tier["amount"]
    return payouts


def scaled_payout_curve(structure: dict, n_field: int) -> tuple[np.ndarray, float]:
    """Per-rank gross payout (rank 1..n_field): the reference structure's
    payout curve sampled at each rank's percentile, renormalized so the
    paid fraction of collected fees matches the reference exactly (DK's
    ~16% rake is fixed across contest sizes).

    *** ONLY VALID NEAR THE REFERENCE SIZE. *** Because it SAMPLES discrete
    tiers, the top sampled ranks each capture a whole reference tier, so at
    small n renormalising to a small pool leaves 1st place with an absurd
    share: 84.5% of the pool at n=416 and 74.6% at n=694, against a real
    DK small-contest figure near 20%. It is also too FLAT at the other end
    for top-heavy formats (10.0% at n=9,803 vs a real 33.3% for Bat Flip).

    Real DK payout SHAPE is a property of contest DESIGN, not field size --
    measured first-place shares are 20.0% at n=352, 10.0% at n=490, 10.0% at
    n=4,458, 33.3% at n=9,803 and 10.0% at n=17,835 -- so no function of
    n alone can represent it. Use `structure_for_contest()` and the real
    tables in data/payout_structures/ for anything contest-size-dependent;
    reserve this for a single contest at or near the reference size.

    Percentile-sampling — not rank-interval scaling — avoids single-rank
    top tiers (1st, 2nd, 3rd...) overwriting each other at scaled indices,
    which previously destroyed 20-50% of the top-heavy prize mass (implied
    rake 24-29% instead of ~16% at common field sizes; see
    scripts/replay_slate.py commit 0897acf, "fix replay payout curve").

    Returns (curve, entry_fee) where curve is a (n_field,) float64 array of
    gross dollar payouts by descending rank (curve[0] = 1st place).
    """
    fee = float(structure.get("entry_fee", 4.0))
    ref = payout_table_to_array(structure)
    ref_n = len(ref)
    idx = np.minimum((np.arange(n_field) * ref_n) // n_field, ref_n - 1)
    curve = ref[idx].astype(np.float64)
    ref_pool_frac = ref.sum() / (ref_n * fee)
    if curve.sum() > 0:
        curve *= (n_field * fee * ref_pool_frac) / curve.sum()
    return curve, fee


# Real DK payout tables captured 2026-07-30 (data/payout_structures/*.json).
# Keyed by the short contest name as it appears in DK entry files and in
# portfolio_sweep_draftkings.json's `contest_name`.
CONTEST_STRUCTURES = {
    "skipper": ["dk_skipper", "dk_skipper_235", "dk_skipper_470", "dk_skipper_564"],
    "base hit": ["dk_base_hit", "dk_base_hit_392", "dk_base_hit_588", "dk_base_hit_784", "dk_base_hit_980"],
    "hot corner": ["dk_hot_corner", "dk_hot_corner_396", "dk_hot_corner_594", "dk_hot_corner_1189", "dk_hot_corner_1585"],
    "four-seamer": ["dk_four_seamer", "dk_four_seamer_2972", "dk_four_seamer_5945"],
    "rally cap": ["dk_rally_cap", "dk_rally_cap_8823", "dk_rally_cap_10294", "dk_rally_cap_11029", "dk_rally_cap_11764", "dk_rally_cap_29411"],
    "solo shot": ["dk_solo_shot", "dk_solo_shot_4756", "dk_solo_shot_7134", "dk_solo_shot_8917"],
    "moonshot": ["dk_moonshot", "dk_moonshot_4756"],
    "bat flip": ["dk_bat_flip", "dk_bat_flip_8169", "dk_bat_flip_11437"],
    "mini-max": ["dk_mini_max", "dk_mini_max_14268", "dk_mini_max_23781"],
    "knuckleball": ["dk_knuckleball", "dk_knuckleball_14268"],
    "pickoff": ["dk_pickoff", "dk_pickoff_396", "dk_pickoff_792"],
    "chin music": ["dk_chin_music", "dk_chin_music_1189", "dk_chin_music_1783", "dk_chin_music_1902", "dk_chin_music_2378"],
    "relay throw": ["dk_relay_throw", "dk_relay_throw_9803"],
    "five-tool player": ["dk_five_tool"],
    "extra inning": ["dk_extra_inning"],
}


def _contest_structure_key(contest_name: str) -> Optional[str]:
    """Match `contest_name` against CONTEST_STRUCTURES's short keys, either
    exactly (the offline backtest substrate's convention -- pre-normalized
    short display names like "Base Hit", see tests/bt_core.py::ZIP_TO_CONTEST)
    or as a substring (DK's REAL entries-file convention -- full names like
    "MLB $10K Skipper [Single Entry]". Confirmed via a live smoke test
    2026-08-08: every one of a real slate's 12 contests only matched this
    way -- the exact-match path alone never fires against real DK entries
    files, only against the offline substrate's pre-normalized names).

    Returns None if nothing matches OR if more than one registered key
    substring-matches (ambiguous) -- callers should treat that the same as
    "no table" rather than guess wrong."""
    name = str(contest_name).strip().lower()
    if name in CONTEST_STRUCTURES:
        return name
    hits = [k for k in CONTEST_STRUCTURES if k in name]
    return hits[0] if len(hits) == 1 else None


def structure_for_contest(
    contest_name: str, n_entries: Optional[int] = None,
) -> Optional[dict]:
    """The real payout structure for `contest_name`, or None when we have no
    table for it. Matches either the offline substrate's exact short names
    or DK's real full entries-file names (see _contest_structure_key).

    DK runs several size variants of the same contest (Four-Seamer appears as
    both a $15K/4,458 and a $20K/5,945 version), so pass `n_entries` — the
    contest's actual field that day, e.g. from its standings zip — to pick the
    closest captured variant. Without it the first registered variant is used.

    Prefer this over scaling a reference curve by field size: payout shape
    tracks contest design, not size (see scaled_payout_curve's warning). A
    caller with no table should skip the contest or be explicit that it is
    extrapolating.
    """
    key = _contest_structure_key(contest_name)
    if key is None:
        return None
    keys = CONTEST_STRUCTURES[key]
    if len(keys) == 1 or n_entries is None:
        return load_payout_structure(keys[0])
    cands = [load_payout_structure(k) for k in keys]
    return min(cands, key=lambda s: abs(int(s["total_entries"]) - int(n_entries)))


def nearest_payout_structure(
    contest_name: str, n_entries: Optional[float] = None,
) -> tuple[dict, bool]:
    """Like `structure_for_contest`, but never returns None: falls back to
    the closest-*size* table across ALL registered contest types when
    `contest_name` isn't one of the 15 known DK types (e.g. a live contest
    name outside the archive this codebase's tables were captured from).

    Returns `(structure, is_approximate)` — `is_approximate` is True
    whenever the fallback fired (name unmatched, or `n_entries` unusable),
    so callers that need a REAL payout curve every round (unlike
    `structure_for_contest`'s callers, which can just skip the contest) can
    surface a warning rather than silently trusting an approximation.

    `n_entries <= 0` or `None` (e.g. `implied_field_size` returning 0.0 for
    an unparseable prize pool, external_pool.py) has no size to match
    against, so it falls back to the smallest registered variant rather
    than guessing a size.
    """
    exact = structure_for_contest(contest_name, int(n_entries) if n_entries else None)
    if exact is not None:
        return exact, False
    all_structs = [
        load_payout_structure(k) for keys in CONTEST_STRUCTURES.values() for k in keys
    ]
    if not n_entries or n_entries <= 0:
        return min(all_structs, key=lambda s: int(s["total_entries"])), True
    return min(all_structs, key=lambda s: abs(int(s["total_entries"]) - int(n_entries))), True
