"""Heuristic ownership estimation for GPP contest simulation.

Model C: softmax(0.6*sqrt(mean) + 0.4*sqrt(salary_value)) within position group.
Model D: Model C with team implied-total weighting for batters.

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


def compute_heuristic_ownership(
    players_df: pd.DataFrame,
    team_totals: dict[str, float] | None = None,
) -> np.ndarray:
    """Return ownership probability array aligned with players_df row order.

    Parameters
    ----------
    players_df:
        Must contain columns: player_id, position, mean, salary, team.
    team_totals:
        Optional {team_abbrev: implied_run_total} dict.  When provided,
        batter ownership is boosted proportional to the team's implied total
        (Model D).  Pitchers are unaffected.  Falls back to Model C when None.

    Returns
    -------
    np.ndarray, shape (len(players_df),), dtype float64, sums to 1.0.
    """
    df = players_df.copy().reset_index(drop=True)
    df["salary_value"] = df["mean"] / (df["salary"] / 1000.0)
    df["_raw"] = (
        0.6 * np.sqrt(df["mean"].clip(lower=0))
        + 0.4 * np.sqrt(df["salary_value"].clip(lower=0))
    )

    if team_totals:
        vals = [v for v in team_totals.values() if v and v > 0]
        mean_total = float(np.mean(vals)) if vals else 1.0
        batter_mask = df["position"] != "P"
        df["_boost"] = 1.0
        for team, total in team_totals.items():
            if total and total > 0 and mean_total > 0:
                mask = batter_mask & (df["team"] == team)
                df.loc[mask, "_boost"] = total / mean_total
        df["_raw"] = df["_raw"] * df["_boost"]

    df["ownership"] = 0.0
    for pos in df["position"].unique():
        mask = df["position"] == pos
        vals = df.loc[mask, "_raw"].values.astype(float)
        std = vals.std()
        # sqrt inputs have std ≈ 0.16 vs ≈ 1.3 for raw mean; floor scaled accordingly.
        shifted = (vals - vals.mean()) / max(std, 0.4)
        exp_vals = np.exp(shifted)
        n_slots = _SLOT_COUNTS.get(pos, 1)
        # Scale so each position group sums to its slot count.
        # Dot product with sim_matrix row then gives expected lineup score.
        df.loc[mask, "ownership"] = (exp_vals / exp_vals.sum()) * n_slots

    return df["ownership"].values.astype(np.float64)
