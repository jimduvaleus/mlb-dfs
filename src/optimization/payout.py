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
