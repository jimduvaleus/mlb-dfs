"""Heuristic ownership estimation for GPP contest simulation.

Model C: softmax(0.6*sqrt(mean) + 0.4*sqrt(salary_value)) within position group.
Model D: Model C + team implied-total weighting for batters and pitcher matchup
         factor (inverse opponent implied total) for pitchers + batting order
         multiplier + multi-position eligibility boost.

sqrt inputs: DFS ownership has diminishing returns at higher projections —
doubling projected points doesn't double ownership. The raw-mean formula
over-concentrates on elite outliers (e.g. 87% projected vs 23% actual for
Acuña on a typical slate). sqrt compression brings top-player estimates
into the realistic range while preserving rank order (Spearman unchanged).
"""
import numpy as np
import pandas as pd

# DK Classic roster slot counts per position — total = 10.
# ownership[j] is interpreted as P(player j appears in a random lineup),
# so each position group must sum to its slot count, not 1.
_SLOT_COUNTS: dict[str, int] = {
    "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3,
}

# Batting order multipliers applied to _raw before softmax.
# Slots 1-2 lead off more often (more PA); 8-9 are at the bottom of the order.
_BATTING_ORDER_MULT: dict[int, float] = {
    1: 1.25, 2: 1.18, 3: 1.12, 4: 1.06, 5: 1.00,
    6: 0.90, 7: 0.80, 8: 0.72, 9: 0.65,
}


def compute_heuristic_ownership(
    players_df: pd.DataFrame,
    team_totals: dict[str, float] | None = None,
) -> np.ndarray:
    """Return ownership probability array aligned with players_df row order.

    Parameters
    ----------
    players_df:
        Must contain columns: player_id, position, mean, salary, team.
        Optional columns used when present:
        - ``slot`` or ``lineup_slot``: batting order (1-9) for order multiplier.
        - ``opponent``: pitcher's opposing team for matchup boost (requires team_totals).
        - ``eligible_positions``: list of DK-eligible positions; multi-eligible
          players are included in each position group's softmax and their
          ownership contributions are summed.
    team_totals:
        Optional {team_abbrev: implied_run_total} dict.  When provided,
        batter ownership is boosted proportional to the team's implied total
        and pitchers are boosted inversely by opponent implied total (Model D).
        Falls back to Model C when None.

    Returns
    -------
    np.ndarray, shape (len(players_df),), dtype float64.
    """
    df = players_df.copy().reset_index(drop=True)
    df["salary_value"] = df["mean"] / (df["salary"] / 1000.0)
    df["_raw"] = (
        0.6 * np.sqrt(df["mean"].clip(lower=0))
        + 0.4 * np.sqrt(df["salary_value"].clip(lower=0))
    )

    # --- Batting order multiplier ------------------------------------------
    slot_col = "lineup_slot" if "lineup_slot" in df.columns else ("slot" if "slot" in df.columns else None)
    if slot_col:
        batter_mask = df["position"] != "P"
        for batting_slot, mult in _BATTING_ORDER_MULT.items():
            mask = batter_mask & (df[slot_col] == batting_slot)
            df.loc[mask, "_raw"] *= mult

    # --- Implied-total boosts (batters and pitchers) -----------------------
    if team_totals:
        vals = [v for v in team_totals.values() if v and v > 0]
        mean_total = float(np.mean(vals)) if vals else 1.0
        batter_mask = df["position"] != "P"
        pitcher_mask = df["position"] == "P"
        df["_boost"] = 1.0
        for team, total in team_totals.items():
            if total and total > 0 and mean_total > 0:
                # Batters: higher team implied total → more fantasy points expected
                mask = batter_mask & (df["team"] == team)
                df.loc[mask, "_boost"] = total / mean_total
                # Pitchers: lower opponent implied total → better matchup
                if "opponent" in df.columns:
                    mask_p = pitcher_mask & (df["opponent"] == team)
                    df.loc[mask_p, "_boost"] = mean_total / total
        df["_raw"] = df["_raw"] * df["_boost"]

    # --- Per-position softmax, respecting multi-position eligibility -------
    use_eligible = "eligible_positions" in df.columns
    df["ownership"] = 0.0

    if use_eligible:
        # Collect all positions that appear across eligible lists.
        all_pos: set[str] = set()
        for ep in df["eligible_positions"]:
            if isinstance(ep, list):
                all_pos.update(ep)
            else:
                all_pos.add(str(ep))
        pos_iter = all_pos
    else:
        pos_iter = set(df["position"].unique())

    for pos in pos_iter:
        if use_eligible:
            mask = df["eligible_positions"].apply(
                lambda ep, p=pos: p in ep if isinstance(ep, list) else str(ep) == p
            )
        else:
            mask = df["position"] == pos

        if not mask.any():
            continue

        vals = df.loc[mask, "_raw"].values.astype(float)
        std = vals.std()
        # sqrt inputs have std ≈ 0.16 vs ≈ 1.3 for raw mean; floor scaled accordingly.
        shifted = (vals - vals.mean()) / max(std, 0.4)
        exp_vals = np.exp(shifted)
        n_slots = _SLOT_COUNTS.get(pos, 1)
        # Scale so each position group sums to its slot count.
        # Multi-eligible players accumulate ownership from each eligible group.
        df.loc[mask, "ownership"] += (exp_vals / exp_vals.sum()) * n_slots

    return df["ownership"].values.astype(np.float64)
