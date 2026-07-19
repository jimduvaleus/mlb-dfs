"""Payout structure loading utilities."""

import json
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
