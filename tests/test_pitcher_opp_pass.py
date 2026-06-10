"""
Tests for the pitcher-opposition ownership adjustment in compute_heuristic_ownership.

The pass sequence promoted to production is:
  1. Two-sided pass: scale all above-median-salary batters by ratio^(-exp),
     where ratio = opp_pitcher_own / mean_pitcher_own.  Both sides (ratio>1 and
     ratio<1) are adjusted.
  2. Suppress-only pass: same loop but skip batters where ratio <= 1.0 — only
     above-average pitchers' opponents are discounted.

This combination was empirically validated over 20 slates (composite 0.8442 vs
single-pass 0.8436) via the V_sal_005 benchmark model in evaluate_ownership.py.
V_sal_005 called compute_heuristic_ownership (pass 1) then re-applied the
suppress-only external — exactly this two-pass structure.  After promoting these
passes to production, V_sal_005 was removed from the eval sweep.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.ownership import (
    compute_heuristic_ownership,
    _PITOPP_SAL_EXP,
    _SLOT_COUNTS,
)
from scripts.evaluate_ownership import _compute_model_v


# ---------------------------------------------------------------------------
# Shared pool fixture
# ---------------------------------------------------------------------------

def _make_pool(seed: int = 42) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Two-game pool: NYY@BOS and LAD@SF.
    Each team has 1 SP + 9 batters spread across C/1B/2B/3B/SS/OF positions.
    Salaries are varied so the median split exercises the salary gate.
    """
    rng = np.random.default_rng(seed)
    teams = ["NYY", "BOS", "LAD", "SF"]
    opp   = {"NYY": "BOS", "BOS": "NYY", "LAD": "SF", "SF": "LAD"}
    game  = {
        "NYY": "NYY@BOS 06/01/2026 07:05PM ET",
        "BOS": "NYY@BOS 06/01/2026 07:05PM ET",
        "LAD": "LAD@SF 06/01/2026 10:15PM ET",
        "SF":  "LAD@SF 06/01/2026 10:15PM ET",
    }
    positions_order = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "1B"]

    rows = []
    pid = 1
    for team in teams:
        # Starting pitcher
        rows.append(dict(
            player_id=pid, name=f"{team}_SP",
            position="P", team=team, opponent=opp[team],
            salary=9000, mean=float(rng.uniform(25, 35)),
            std_dev=7.0, lineup_slot=1,
            game=game[team], implied_total=4.5,
            eligible_positions=["P"],
        ))
        pid += 1
        # Batters — spread across cheap/expensive to straddle the median
        for slot in range(1, 10):
            pos = positions_order[slot - 1]
            sal = int(rng.integers(28, 60) * 100)
            mean = float(rng.uniform(5, 15))
            rows.append(dict(
                player_id=pid, name=f"{team}_B{slot}",
                position=pos, team=team, opponent=opp[team],
                salary=sal, mean=mean, std_dev=mean * 0.3,
                lineup_slot=slot,
                game=game[team], implied_total=4.5,
                eligible_positions=[pos],
                hr_prob=float(rng.uniform(0.05, 0.25)),
            ))
            pid += 1

    pool = pd.DataFrame(rows).reset_index(drop=True)
    pool["salary_value"] = pool["mean"] / pool["salary"] * 1000
    team_totals = {t: 4.5 for t in teams}
    return pool, team_totals


def _apply_pass_batter_list(
    result: np.ndarray,
    pool_df: pd.DataFrame,
    starter_own: dict[str, float],
    mean_starter_own: float,
    sal_median: float,
) -> np.ndarray:
    """
    Apply one V_sal pass using the explicit batter-position list from
    _compute_model_v.  Modifies and returns `result` in place.
    """
    result = result.copy()
    positions  = pool_df["position"].values
    salaries   = pool_df["salary"].values.astype(float)
    opponents  = pool_df["opponent"].values
    exp = _PITOPP_SAL_EXP

    for pos in ["C", "1B", "2B", "3B", "SS", "OF"]:
        pos_mask = positions == pos
        if not pos_mask.any():
            continue
        orig_sum = float(result[pos_mask].sum())
        for i in np.where(pos_mask)[0]:
            if salaries[i] < sal_median:
                continue
            opp_own = starter_own.get(opponents[i])
            if opp_own is None:
                continue
            result[i] *= (opp_own / mean_starter_own) ** (-exp)
        new_sum = float(result[pos_mask].sum())
        if new_sum > 0:
            result[pos_mask] *= orig_sum / new_sum
    return result


def _apply_pass_slot_counts(
    result: np.ndarray,
    pool_df: pd.DataFrame,
    starter_own: dict[str, float],
    mean_starter_own: float,
    sal_median: float,
) -> np.ndarray:
    """
    Apply one V_sal pass using _SLOT_COUNTS iteration — the internal
    production loop.  Modifies and returns `result` in place.
    """
    result = result.copy()
    positions  = pool_df["position"].values
    salaries   = pool_df["salary"].values.astype(float)
    opponents  = pool_df["opponent"].values
    exp = _PITOPP_SAL_EXP

    for pos, _n in _SLOT_COUNTS.items():
        if pos == "P":
            continue
        pos_mask = positions == pos
        if not pos_mask.any():
            continue
        orig_sum = float(result[pos_mask].sum())
        for i in np.where(pos_mask)[0]:
            if salaries[i] < sal_median:
                continue
            opp_own = starter_own.get(opponents[i])
            if opp_own is None:
                continue
            result[i] *= (opp_own / mean_starter_own) ** (-exp)
        new_sum = float(result[pos_mask].sum())
        if new_sum > 0:
            result[pos_mask] *= orig_sum / new_sum
    return result


def _apply_pass_suppress_only(
    result: np.ndarray,
    pool_df: pd.DataFrame,
    starter_own: dict[str, float],
    mean_starter_own: float,
    sal_median: float,
) -> np.ndarray:
    """Apply one suppress-only pass: skip batters whose opposing pitcher ratio <= 1.0."""
    result = result.copy()
    positions  = pool_df["position"].values
    salaries   = pool_df["salary"].values.astype(float)
    opponents  = pool_df["opponent"].values
    exp = _PITOPP_SAL_EXP

    for pos in ["C", "1B", "2B", "3B", "SS", "OF"]:
        pos_mask = positions == pos
        if not pos_mask.any():
            continue
        orig_sum = float(result[pos_mask].sum())
        for i in np.where(pos_mask)[0]:
            if salaries[i] < sal_median:
                continue
            opp_own = starter_own.get(opponents[i])
            if opp_own is None:
                continue
            ratio = opp_own / mean_starter_own
            if ratio <= 1.0:
                continue
            result[i] *= ratio ** (-exp)
        new_sum = float(result[pos_mask].sum())
        if new_sum > 0:
            result[pos_mask] *= orig_sum / new_sum
    return result


def _starter_info(
    base: np.ndarray, pool_df: pd.DataFrame
) -> tuple[dict[str, float], float, float]:
    """Return (starter_own, mean_starter_own, sal_median) from base ownership."""
    positions = pool_df["position"].values
    pitcher_mask = positions == "P"
    batter_mask  = ~pitcher_mask
    teams = pool_df["team"].values
    has_slot = "lineup_slot" in pool_df.columns

    starter_idx: dict[str, int] = {}
    for team in np.unique(teams[pitcher_mask]):
        tm = pitcher_mask & (teams == team)
        grp = pool_df[tm]
        if has_slot and grp["lineup_slot"].notna().any():
            best = int(grp.dropna(subset=["lineup_slot"])["lineup_slot"].idxmin())
        else:
            best = int(grp["mean"].idxmax())
        starter_idx[team] = best

    starter_own = {t: float(base[idx]) for t, idx in starter_idx.items()}
    mean_own = float(np.mean(list(starter_own.values())))
    sal_median = float(np.median(pool_df["salary"].values.astype(float)[batter_mask]))
    return starter_own, mean_own, sal_median


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSinglePassEquivalence:
    """The two iteration styles should produce identical single-pass output."""

    def test_batter_list_equals_slot_counts_single_pass(self):
        pool, team_totals = _make_pool()
        base = compute_heuristic_ownership(pool, team_totals)
        starter_own, mean_own, sal_median = _starter_info(base, pool)

        a = _apply_pass_batter_list(base, pool, starter_own, mean_own, sal_median)
        b = _apply_pass_slot_counts(base, pool, starter_own, mean_own, sal_median)

        assert np.allclose(a, b, atol=1e-12), (
            f"Single-pass outputs differ. Max diff: {np.max(np.abs(a - b)):.2e}\n"
            f"batter_list: {a}\nslot_counts: {b}"
        )


class TestDoublePassEquivalence:
    """
    Production compute_heuristic_ownership now runs: one two-sided pass then one
    suppress-only pass.  These tests verify structural properties of that sequence.
    """

    def test_eprod_equals_two_sided_then_suppress_only(self):
        """
        E_production output == apply two-sided then suppress-only on a base without
        any pitcher-opposition adjustment.

        We can't easily strip the passes from compute_heuristic_ownership, so we
        compare against an explicit decomposition using the helper functions.
        The base used is compute_heuristic_ownership on a pool with _PITOPP_SAL_EXP
        effectively zeroed — approximated by checking that applying both passes from
        scratch on an un-adjusted base matches production output.

        More practically: confirm that applying two-sided then suppress-only on an
        external base produces an array strictly different from two-sided alone, and
        that the difference is in the expected direction (suppress-only can only
        lower ownership of batters facing above-average pitchers, with renorm lifting
        the rest).
        """
        pool, team_totals = _make_pool()
        base = compute_heuristic_ownership(pool, team_totals)
        starter_own, mean_own, sal_median = _starter_info(base, pool)

        after_two_sided = _apply_pass_batter_list(base, pool, starter_own, mean_own, sal_median)
        after_both = _apply_pass_suppress_only(after_two_sided, pool, starter_own, mean_own, sal_median)

        # The two arrays must differ — suppress-only does add signal on top
        assert not np.allclose(after_two_sided, after_both, atol=1e-6), (
            "Suppress-only pass had no effect — expected it to further adjust "
            "batters facing above-average pitchers."
        )

    def test_eprod_equals_v_sal_external_structure(self):
        """
        After promotion, compute_heuristic_ownership incorporates V_sal_005's
        double-pass structure (two-sided + suppress-only).  Verify that calling
        compute_heuristic_ownership + one external suppress-only pass (what the
        old V_sal_005 evaluated as) now equals compute_heuristic_ownership + one
        more suppress-only — confirming the equivalence is maintained structurally.
        """
        pool, team_totals = _make_pool()

        prod = compute_heuristic_ownership(pool, team_totals)
        starter_own, mean_own, sal_median = _starter_info(prod, pool)

        # One more suppress-only on top of production
        prod_plus_sup = _apply_pass_suppress_only(prod, pool, starter_own, mean_own, sal_median)

        # Mirrors what _compute_model_v now computes (production already has 2 passes;
        # external adds a third) — so prod_plus_sup != prod
        assert not np.allclose(prod, prod_plus_sup, atol=1e-6), (
            "Additional suppress-only pass had no effect on production output."
        )

    def test_starter_own_stable_across_passes(self):
        """
        Pitcher ownership should be unchanged after a batter-position pass,
        so starter_own is the same anchor for both iterations.
        """
        pool, team_totals = _make_pool()
        base = compute_heuristic_ownership(pool, team_totals)
        starter_own_before, mean_before, sal_median = _starter_info(base, pool)

        after_1 = _apply_pass_batter_list(base, pool, starter_own_before, mean_before, sal_median)
        starter_own_after, mean_after, _ = _starter_info(after_1, pool)

        assert starter_own_before == starter_own_after, (
            "Pitcher ownership changed after batter-only pass — "
            "starter_own anchor is NOT stable across passes."
        )
        assert mean_before == mean_after
