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
    "skipper": "dk_skipper",
    "base hit": "dk_base_hit",
    "four-seamer": "dk_four_seamer",
    "bat flip": "dk_bat_flip",
    "solo shot": "dk_solo_shot",
    "rally cap": "dk_rally_cap",
    "hot corner": "dk_hot_corner",
    "moonshot": "dk_moonshot",
    "mini-max": "dk_mini_max",
}


def structure_for_contest(contest_name: str) -> Optional[dict]:
    """The real payout structure for `contest_name`, or None when we have no
    table for it.

    Prefer this over scaling a reference curve by field size: payout shape
    tracks contest design, not size (see scaled_payout_curve's warning). A
    caller with no table should either skip the contest or be explicit that
    it is extrapolating.
    """
    key = CONTEST_STRUCTURES.get(str(contest_name).strip().lower())
    return load_payout_structure(key) if key else None
