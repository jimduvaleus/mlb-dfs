"""
Tests for src/optimization/payout.py.

Covers payout_table_to_array, get_cash_line_score, and calibrate_beta
using the corrected dk_classic_gpp.json (14863 total entries, $4 entry fee,
3855 paying positions with minimum payout $6).
"""
import numpy as np
import pytest

from src.optimization.payout import (
    calibrate_beta,
    get_cash_line_score,
    load_payout_structure,
    payout_table_to_array,
    power_law_payout,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def dk_structure():
    """Load the corrected dk_classic_gpp.json."""
    return load_payout_structure("dk_classic_gpp")


@pytest.fixture
def score_percentiles():
    """Synthetic score percentiles for a typical best-lineup distribution.

    100 evenly-spaced values from 80 to 220 DK pts (percentiles 1-100).
    """
    return np.linspace(80.0, 220.0, 100)


# ------------------------------------------------------------------ #
#  load_payout_structure                                               #
# ------------------------------------------------------------------ #

def test_load_payout_structure_has_correct_total_entries(dk_structure):
    assert dk_structure["total_entries"] == 14863


def test_load_payout_structure_has_correct_entry_fee(dk_structure):
    assert dk_structure["entry_fee"] == pytest.approx(4.00)


def test_load_payout_structure_has_payouts_list(dk_structure):
    assert isinstance(dk_structure["payouts"], list)
    assert len(dk_structure["payouts"]) > 0


# ------------------------------------------------------------------ #
#  payout_table_to_array                                               #
# ------------------------------------------------------------------ #

def test_payout_table_to_array_length(dk_structure):
    """Array length must equal total_entries (14863), not payout positions."""
    arr = payout_table_to_array(dk_structure)
    assert len(arr) == 14863


def test_payout_table_to_array_zeros_for_non_paying(dk_structure):
    """Positions 3856-14863 (0-indexed 3855-14862) should be zero."""
    arr = payout_table_to_array(dk_structure)
    # total_payout_positions = 3855, so from index 3855 onward should be 0
    assert np.all(arr[3855:] == 0.0)


def test_payout_table_to_array_first_place_payout(dk_structure):
    """Index 0 should be the first-place prize."""
    arr = payout_table_to_array(dk_structure)
    assert arr[0] == pytest.approx(5000.0)


def test_payout_table_to_array_last_paying_position(dk_structure):
    """Index 3854 (0-indexed position 3855) should be the minimum payout ($6)."""
    arr = payout_table_to_array(dk_structure)
    assert arr[3854] == pytest.approx(6.0)


def test_payout_table_to_array_min_payout_tier(dk_structure):
    """The tier covering positions 1741-3855 should all pay $6."""
    arr = payout_table_to_array(dk_structure)
    # 0-indexed: 1740 to 3854
    assert np.all(arr[1740:3855] == 6.0)


# ------------------------------------------------------------------ #
#  get_cash_line_score                                                 #
# ------------------------------------------------------------------ #

def test_get_cash_line_score_returns_float(dk_structure, score_percentiles):
    result = get_cash_line_score(dk_structure, score_percentiles)
    assert isinstance(result, float)


def test_get_cash_line_score_within_percentile_range(dk_structure, score_percentiles):
    """Cash line must be between the min and max of the percentile array."""
    result = get_cash_line_score(dk_structure, score_percentiles)
    assert score_percentiles[0] <= result <= score_percentiles[-1]


def test_get_cash_line_score_near_74th_percentile(dk_structure, score_percentiles):
    """With entry_fee=$4 and 3855 paying out of 14863, the break-even is at
    roughly the 74th percentile (top ~26% of entries cash).

    The last position paying > $4 is position 3855 (payout $6).
    That's index 3854 (0-indexed) out of 14863, i.e. the top 3855/14863 ≈ 25.9%
    of entrants, corresponding to the 74th percentile of scores.
    We verify the cash line is in the 60th-85th percentile range.
    """
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    # percentile array is linear from 80 to 220; 74th pctile ≈ 80 + 0.74*140 ≈ 183.6
    p60 = float(np.percentile(score_percentiles, 60))
    p85 = float(np.percentile(score_percentiles, 85))
    assert p60 <= cash_line <= p85, (
        f"Expected cash_line near 74th percentile, got {cash_line:.1f} "
        f"(range [{p60:.1f}, {p85:.1f}])"
    )


def test_get_cash_line_score_lower_than_target_percentile(dk_structure, score_percentiles):
    """Cash line should be well below the 97th percentile (optimization target)."""
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    p97 = float(np.percentile(score_percentiles, 97))
    assert cash_line < p97


def test_get_cash_line_score_fallback_when_no_payout_exceeds_fee():
    """When no payout exceeds entry_fee, function should return median."""
    structure = {
        "total_entries": 100,
        "entry_fee": 100.0,  # higher than any payout
        "payouts": [{"start": 1, "end": 10, "amount": 5.0}],
    }
    percentiles = np.linspace(50.0, 150.0, 100)
    result = get_cash_line_score(structure, percentiles)
    assert result == pytest.approx(float(np.median(percentiles)), rel=0.01)


def test_get_cash_line_score_single_paying_position():
    """Only position 1 pays out above entry_fee → cash line at top percentile."""
    structure = {
        "total_entries": 1000,
        "entry_fee": 10.0,
        "payouts": [
            {"start": 1, "end": 1, "amount": 100.0},   # > entry_fee
            {"start": 2, "end": 100, "amount": 5.0},   # < entry_fee
        ],
    }
    percentiles = np.linspace(50.0, 200.0, 100)
    result = get_cash_line_score(structure, percentiles)
    # Position 1 out of 1000 → 999/1000 = 99.9th percentile → near max
    assert result >= percentiles[95]


# ------------------------------------------------------------------ #
#  calibrate_beta                                                      #
# ------------------------------------------------------------------ #

def test_calibrate_beta_returns_float(dk_structure, score_percentiles):
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    result = calibrate_beta(dk_structure, score_percentiles, cash_line)
    assert isinstance(result, float)


def test_calibrate_beta_within_default_range(dk_structure, score_percentiles):
    """With the new default cap of 4.0, beta must be <= 4.0."""
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    beta = calibrate_beta(dk_structure, score_percentiles, cash_line)
    assert 1.5 <= beta <= 4.0


def test_calibrate_beta_respects_custom_range(dk_structure, score_percentiles):
    """Custom beta_range must be honoured."""
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    beta = calibrate_beta(
        dk_structure, score_percentiles, cash_line, beta_range=(2.0, 3.0)
    )
    assert 2.0 <= beta <= 3.0


def test_calibrate_beta_cap_prevents_large_beta(dk_structure, score_percentiles):
    """Old cap was 8.0; with new cap 4.0, beta should never reach 8."""
    cash_line = get_cash_line_score(dk_structure, score_percentiles)
    beta_new_cap = calibrate_beta(
        dk_structure, score_percentiles, cash_line, beta_range=(1.5, 4.0)
    )
    assert beta_new_cap <= 4.0


def test_calibrate_beta_fallback_on_zero_payout(score_percentiles):
    """If all payouts are zero, function should return the fallback value 2.5."""
    structure = {
        "total_entries": 100,
        "entry_fee": 10.0,
        "payouts": [],  # no payouts → payout array all zero
    }
    beta = calibrate_beta(structure, score_percentiles, cash_line=100.0)
    assert beta == pytest.approx(2.5)


# ------------------------------------------------------------------ #
#  power_law_payout                                                    #
# ------------------------------------------------------------------ #

def test_power_law_payout_zero_below_cash_line():
    scores = np.array([50.0, 80.0, 99.9])
    result = power_law_payout(scores, cash_line=100.0, beta=2.0)
    assert np.all(result == 0.0)


def test_power_law_payout_correct_value():
    scores = np.array([110.0])
    result = power_law_payout(scores, cash_line=100.0, beta=2.0)
    assert result[0] == pytest.approx(100.0)  # (110 - 100)^2 = 100


def test_power_law_payout_same_shape_as_input():
    scores = np.arange(10, dtype=np.float64)
    result = power_law_payout(scores, cash_line=5.0, beta=1.5)
    assert result.shape == scores.shape
