"""Payout functions for GPP portfolio optimization.

Provides a power-law payout function used by the marginal_payout optimizer
objective. The function P(s) = max(0, s - cash_line)^beta captures the
top-heavy nature of GPP payout structures: a lineup scoring 200 is worth
far more than two lineups scoring 140.

A reference GPP payout structure (DraftKings $20 Classic) is stored in
data/payout_structures/dk_classic_gpp.json for calibration purposes.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numba import njit


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


@njit(cache=True)
def power_law_payout(scores: np.ndarray, cash_line: float, beta: float) -> np.ndarray:
    """Compute payout values using a power-law function.

    P(s) = max(0, s - cash_line)^beta

    Parameters
    ----------
    scores : ndarray
        Raw DK scores.
    cash_line : float
        Minimum score to receive any payout (typically the target).
    beta : float
        Convexity exponent.  Higher values weight top scores more heavily.
        Recommended range: 2.0-3.0 for typical GPPs.

    Returns
    -------
    ndarray
        Payout values, same shape as scores.
    """
    n = scores.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff = scores[i] - cash_line
        if diff > 0.0:
            out[i] = diff ** beta
        else:
            out[i] = 0.0
    return out


def get_cash_line_score(structure: dict, score_percentiles: np.ndarray) -> float:
    """Compute the score at the profitability break-even point.

    Finds the last contest position where ``payout > entry_fee``, maps it to
    a score percentile, and returns that score as the power-law cash line.
    With correct contest data this gives the score at roughly the 74th
    percentile (top ~26% cash), providing dense gradient signal for the
    marginal_payout objective instead of the sparse 3% signal produced by
    using the optimization target as the cash line.

    Parameters
    ----------
    structure : dict
        Loaded payout structure (from ``load_payout_structure``).
    score_percentiles : ndarray, shape (N,)
        Score values at evenly-spaced percentiles (e.g. percentiles 1..100
        of the simulated best-lineup scores).

    Returns
    -------
    float
        The minimum score needed to profit given the entry fee.
    """
    total = structure["total_entries"]
    entry_fee = structure["entry_fee"]
    payouts_arr = payout_table_to_array(structure)

    # Last 0-indexed position where payout > entry_fee
    last_cash_idx = -1
    for i in range(len(payouts_arr)):
        if payouts_arr[i] > entry_fee:
            last_cash_idx = i

    if last_cash_idx < 0:
        return float(np.median(score_percentiles))

    n_pctiles = len(score_percentiles)
    pctile_idx = int((1.0 - (last_cash_idx + 1) / total) * (n_pctiles - 1))
    pctile_idx = max(0, min(pctile_idx, n_pctiles - 1))
    return float(score_percentiles[pctile_idx])


def calibrate_beta(
    structure: dict,
    score_percentiles: np.ndarray,
    cash_line: float,
    beta_range: Tuple[float, float] = (1.5, 4.0),
    n_steps: int = 100,
) -> float:
    """Find the beta that best fits a payout structure.

    Given a mapping from score percentiles to actual payouts, finds the
    beta that minimizes the squared error between the power-law payout
    curve and the actual payout curve (both normalized to [0, 1]).

    Parameters
    ----------
    structure : dict
        Loaded payout structure (from load_payout_structure).
    score_percentiles : ndarray, shape (N,)
        Score values at evenly-spaced percentiles of the score distribution.
        E.g. percentiles 1..100 of the simulated best-lineup scores.
    cash_line : float
        The score at the cash line (minimum score that pays out).
    beta_range : tuple
        (min_beta, max_beta) to search over.
    n_steps : int
        Number of beta values to evaluate.

    Returns
    -------
    float
        The beta that best fits the payout structure.
    """
    payouts_arr = payout_table_to_array(structure)
    total = len(payouts_arr)

    # Map positions to normalized payouts (0 to 1 scale)
    max_payout = payouts_arr[0]
    if max_payout == 0:
        return 2.5  # fallback
    norm_payouts = payouts_arr / max_payout

    # Map positions to score percentiles
    # Position 1 = 100th percentile, position N = lowest paid percentile
    n_pctiles = len(score_percentiles)
    position_scores = np.zeros(total, dtype=np.float64)
    for i in range(total):
        # Position i+1 corresponds to percentile (1 - (i+1)/total) * 100
        pctile_idx = int((1.0 - (i + 1) / total) * (n_pctiles - 1))
        pctile_idx = max(0, min(pctile_idx, n_pctiles - 1))
        position_scores[i] = score_percentiles[pctile_idx]

    betas = np.linspace(beta_range[0], beta_range[1], n_steps)
    best_beta = 2.5
    best_err = float("inf")

    for b in betas:
        pred = power_law_payout(position_scores, cash_line, b)
        pred_max = pred[0] if pred[0] > 0 else 1.0
        norm_pred = pred / pred_max
        err = float(np.sum((norm_pred - norm_payouts) ** 2))
        if err < best_err:
            best_err = err
            best_beta = float(b)

    return best_beta
