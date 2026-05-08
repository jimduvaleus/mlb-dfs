"""Heuristic ownership estimation for GPP contest simulation.

Batters: softmax(sqrt(mean)) within position group, with salary-cap pressure,
         batting-order multiplier, game start-time multiplier, and stack-value
         batter boost.
Pitchers: softmax(0.8*sqrt(mean) + 0.2*sqrt(salary_value)), with opponent
          implied-total matchup boost, own-team co-stack boost, and post-softmax
          compression toward uniform.

Key design choices:
- sqrt(mean): diminishing returns at high projections — doubling projected
  points doesn't double ownership.
- Salary_value dropped for batters (Spearman vs actual = 0.23; noise, not
  signal) but kept at 0.2 weight for pitchers (Spearman = 0.77).
- Salary-cap pressure on batters: expensive players are harder to pair with
  a full competitive lineup; (4500/salary)^0.5 penalty above the ~75th-pct
  batter salary. Accounts for field fading $6k+ superstars.
- Batter std floor raised to 0.7: prevents one outlier from consuming the
  entire position pool through softmax concentration.
- Game start-time multiplier: earlier games have confirmed batting lineups
  sooner, attracting DFS players building before late-game announcements.
  Games are bucketed into 30-minute windows; all games in the same window
  receive the same boost. Only applied to batters — pitcher slots are
  confirmed earlier in the day regardless of game time.
- Stack-value batter boost: teams with cheap top-of-order bats relative to
  their implied total are more attractive stacking targets — the field can
  pair them with expensive pitchers within the $50k cap. Only fires when the
  team's implied total is >= 4.0.
- Pitcher co-stack boost: pitchers on high-implied teams see elevated
  ownership because the field stacks that offense and often correlates
  the pitcher alongside their batters.
- Pitcher pool compression: post-softmax 20% blend toward uniform —
  empirical pitcher ownership is flatter than pure softmax produces.
"""
import re
import numpy as np
import pandas as pd

# DK Classic roster slot counts per position — total = 10.
# ownership[j] is interpreted as P(player j appears in a random lineup),
# so each position group must sum to its slot count, not 1.
_SLOT_COUNTS: dict[str, int] = {
    "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3,
}

# Batting order multipliers applied to raw score before softmax.
# Flattened vs initial version: actual field data shows slot 1/slot 9 ratio
# is ~2.6x, but the original 1.25/0.65 range was amplified by softmax into ~12x.
_BATTING_ORDER_MULT: dict[int, float] = {
    1: 1.12, 2: 1.08, 3: 1.05, 4: 1.02, 5: 1.00,
    6: 0.95, 7: 0.90, 8: 0.86, 9: 0.83,
}

# Fraction of secondary-pool ownership credited to multi-eligible players.
# A 3B/SS player gets full credit from their primary pool plus this fraction
# of whatever the SS pool would award them (and vice-versa).
_SECONDARY_POSITION_DISCOUNT = 0.5

# Batter implied-total boost cap: above ~5.5 runs the field diversifies enough
# that ownership stops climbing proportionally with implied total.
_BATTER_TOTAL_CAP = 5.5

# Pitcher opponent-matchup boost exponent. Higher values amplify the signal
# from low opponent implied totals — tuned against empirical ownership data.
_PITCHER_MATCHUP_EXP = 2.0

# Salary above which batters receive a soft cap penalty: (4500/salary)^0.5.
# Represents the ~75th percentile of batter salaries; above this, the $50k
# cap makes it progressively harder to build a full competitive lineup.
_BATTER_SALARY_PRESSURE_BASE = 4500

# Std floor for within-position softmax normalization.
# Higher floor → flatter distribution → less concentration on one player.
_BATTER_STD_FLOOR  = 0.7
_PITCHER_STD_FLOOR = 0.4

# Pitcher pool compression: blend this fraction of uniform into the
# post-softmax pitcher distribution. Empirical pitcher ownership is much
# flatter than pure softmax produces.
_PITCHER_COMPRESS = 0.20

# Pitcher own-team co-stack boost exponent. Pitchers on high-implied teams
# see elevated ownership because the field stacks that offense and often
# pairs the pitcher. (own_team_total / mean_total) ** this_exp.
_PITCHER_COSTACK_EXP = 0.40

# Stack-value batter boost: (mean_slot5_salary / team_slot5_salary) ** this_exp
# amplifies ownership for teams with cheap top-of-order bats.
_STACK_VALUE_EXP = 0.15

# Implied total threshold below which the stack-value signal does not fire.
_STACK_THRESHOLD = 4.0

# Maximum fractional boost for the earliest-bucket batter games on the slate.
# Applied only to batters; pitchers are confirmed before the slate regardless.
_TIME_FACTOR = 0.12

# Games are bucketed into this-minute windows from slate lock. All games in
# the same window receive the same time boost.
_TIME_NEUTRAL_WINDOW = 30


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
          players receive full credit from their primary pool and a discounted
          contribution from each secondary eligible pool.
    team_totals:
        Optional {team_abbrev: implied_run_total} dict.  When provided,
        batter ownership is boosted proportional to the team's implied total
        and pitchers are boosted inversely by opponent implied total.
        Falls back to projection-only model when None.

    Returns
    -------
    np.ndarray, shape (len(players_df),), dtype float64.
    """
    df = players_df.copy().reset_index(drop=True)
    salary_value = df["mean"] / (df["salary"] / 1000.0)

    pitcher_mask = df["position"] == "P"
    batter_mask  = ~pitcher_mask

    # --- Raw scores: mean-only for batters; mean + salary_value for pitchers ---
    df["_raw"] = np.sqrt(df["mean"].clip(lower=0))
    df.loc[pitcher_mask, "_raw"] = (
        0.8 * np.sqrt(df.loc[pitcher_mask, "mean"].clip(lower=0))
        + 0.2 * np.sqrt(salary_value.loc[pitcher_mask].clip(lower=0))
    )

    # --- Batting order multiplier --------------------------------------------
    slot_col = (
        "lineup_slot" if "lineup_slot" in df.columns
        else ("slot" if "slot" in df.columns else None)
    )
    if slot_col:
        for batting_slot, mult in _BATTING_ORDER_MULT.items():
            mask = batter_mask & (df[slot_col] == batting_slot)
            df.loc[mask, "_raw"] *= mult

    # --- Salary-cap pressure on batters --------------------------------------
    # Soft penalty for players priced above the ~75th-pct batter salary.
    df.loc[batter_mask, "_raw"] *= np.minimum(
        1.0,
        (_BATTER_SALARY_PRESSURE_BASE / df.loc[batter_mask, "salary"]) ** 0.5,
    )

    # --- Game start-time penalty (batters only) ------------------------------
    # Pitcher lineup slots are confirmed earlier in the day regardless of game
    # time. For batters, games within _TIME_NEUTRAL_WINDOW minutes of slate lock
    # (the earliest game) are treated as neutral. Only games starting beyond that
    # window get a penalty, scaling linearly to -_TIME_FACTOR at the latest game.
    if "game" in df.columns:
        def _game_minutes(game_str: str) -> float | None:
            m = re.search(r'(\d{1,2}):(\d{2})(AM|PM)', str(game_str))
            if not m:
                return None
            h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
            if ap == "PM" and h != 12:
                h += 12
            elif ap == "AM" and h == 12:
                h = 0
            return float(h * 60 + mi)

        game_mins = {g: _game_minutes(g) for g in df["game"].unique()}
        valid_mins = [v for v in game_mins.values() if v is not None]
        if len(valid_mins) > 1:
            min_t = min(valid_mins)
            # Snap each game to a 60-minute bucket from slate lock so that
            # games within the same hour window receive identical boosts.
            game_buckets = {
                g: min_t + _TIME_NEUTRAL_WINDOW * int((t - min_t) // _TIME_NEUTRAL_WINDOW)
                for g, t in game_mins.items() if t is not None
            }
            bucket_vals = list(game_buckets.values())
            min_b, max_b = min(bucket_vals), max(bucket_vals)
            bucket_range = max_b - min_b
            if bucket_range > 0:
                df["_time_mult"] = df["game"].map(
                    lambda g: 1.0 + _TIME_FACTOR * (max_b - game_buckets.get(g, max_b)) / bucket_range
                )
                df.loc[batter_mask, "_raw"] *= df.loc[batter_mask, "_time_mult"]

    # --- Implied-total boosts (batters and pitchers) -------------------------
    # Restrict team_totals to teams present in this player pool so that stale
    # data (e.g. implied totals for postponed games removed from the slate)
    # does not distort mean_total and bleed into boosts for active teams.
    if team_totals and "team" in df.columns:
        active_teams = set(df["team"].dropna().unique())
        team_totals = {t: v for t, v in team_totals.items() if t in active_teams}
    if team_totals:
        vals = [v for v in team_totals.values() if v and v > 0]
        mean_total = float(np.mean(vals)) if vals else 1.0
        df["_boost"] = 1.0

        # Pre-compute per-team average salary of top-5 batting-order slots
        # for the stack-value signal (cheap top-of-order → more stacking interest).
        bdf = df[batter_mask]
        team_slot5_salary: dict[str, float] = {}
        for team_s, grp in bdf.groupby("team"):
            if slot_col and grp[slot_col].notna().sum() >= 3:
                top5 = grp.dropna(subset=[slot_col]).sort_values(slot_col).head(5)
            else:
                top5 = grp.nlargest(min(5, len(grp)), "mean")
            if len(top5):
                team_slot5_salary[team_s] = float(top5["salary"].mean())
        mean_slot5_sal = float(np.mean(list(team_slot5_salary.values()))) if team_slot5_salary else 4500.0

        # Pass A: batter team-total boost + pitcher opponent-matchup boost
        for team, total in team_totals.items():
            if not (total and total > 0 and mean_total > 0):
                continue
            capped = min(total, _BATTER_TOTAL_CAP)
            mask_b = batter_mask & (df["team"] == team)
            slot5_sal = team_slot5_salary.get(team, mean_slot5_sal)
            stack_mult = (mean_slot5_sal / slot5_sal) ** _STACK_VALUE_EXP if total >= _STACK_THRESHOLD else 1.0
            df.loc[mask_b, "_boost"] = (capped / mean_total) * stack_mult
            if "opponent" in df.columns:
                mask_p = pitcher_mask & (df["opponent"] == team)
                df.loc[mask_p, "_boost"] = (mean_total / capped) ** _PITCHER_MATCHUP_EXP

        # Pass B: pitcher own-team co-stack boost (multiplicative)
        if "opponent" in df.columns:
            for team, total in team_totals.items():
                if not (total and total > 0 and mean_total > 0):
                    continue
                capped = min(total, _BATTER_TOTAL_CAP)
                mask_own = pitcher_mask & (df["team"] == team)
                df.loc[mask_own, "_boost"] *= (capped / mean_total) ** _PITCHER_COSTACK_EXP

        df["_raw"] *= df["_boost"]

    # --- Per-position softmax, respecting multi-position eligibility ---------
    use_eligible = "eligible_positions" in df.columns
    df["ownership"] = 0.0

    if use_eligible:
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

        raw_vals  = df.loc[mask, "_raw"].values.astype(float)
        n_slots   = _SLOT_COUNTS.get(pos, 1)
        is_p      = pos == "P"
        std_floor = _PITCHER_STD_FLOOR if is_p else _BATTER_STD_FLOOR

        std     = raw_vals.std()
        shifted = (raw_vals - raw_vals.mean()) / max(std, std_floor)
        exp_v   = np.exp(shifted)
        softmax = exp_v / exp_v.sum()

        if is_p:
            # Blend toward uniform to flatten the pitcher ownership distribution.
            uniform = np.ones(len(raw_vals)) / len(raw_vals)
            share = (1.0 - _PITCHER_COMPRESS) * softmax + _PITCHER_COMPRESS * uniform
        else:
            share = softmax

        contribution = share * n_slots

        if use_eligible:
            is_primary = (df.loc[mask, "position"] == pos).values
            discount   = np.where(is_primary, 1.0, _SECONDARY_POSITION_DISCOUNT)
            df.loc[mask, "ownership"] += contribution * discount
        else:
            df.loc[mask, "ownership"] += contribution

    return df["ownership"].values.astype(np.float64)
