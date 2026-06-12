"""Heuristic ownership estimation for GPP contest simulation.

Batters: softmax(sqrt(mean)) within position group, with salary-cap pressure,
         batting-order multiplier, HR-probability boost, game start-time
         multiplier, and stack-value batter boost.
Pitchers: softmax(0.98*sqrt(mean) + 0.02*sqrt(salary_value)), with opponent
          implied-total matchup boost and own-team co-stack boost.

Key design choices:
- sqrt(mean): diminishing returns at high projections — doubling projected
  points doesn't double ownership.
- team_ownership_reductions: per-team ownership reduction applied after the
  full softmax + power-law calibration.  Reduced-team players are locked at
  exactly (1 - pct/100) × baseline; the remaining (non-reduced) players in
  each position group are then scaled up to absorb the freed ownership,
  preserving slot-count sums.  Entered % maps directly to the output drop.
- Salary_value near-zero for pitchers (weight 0.02) and dropped entirely for
  batters: 10-slate DE regression found salary_value adds negligible signal
  for pitchers once projection quality is high.
- Batter std floor 1.19: field spreads batter ownership much more evenly than
  raw softmax predicts; raising the floor significantly flattens the
  distribution and matches empirical ownership patterns.
- Pitcher std floor 0.29: pitchers are more concentrated than batters — field
  tends to anchor on the top arm.
- Pitcher compression 0.00: actual pitcher ownership is not flat; the
  uniform-blend hurt accuracy. Disabled after 10-slate regression.
- Game start-time multiplier (factor 0.04): weak signal; field builds
  regardless of game time for most contests. Kept at low weight.
- Stack-value batter boost: teams with cheap top-of-order bats are more
  attractive stacking targets within the $50k cap. Fires only when implied
  total >= 4.0.
- Pitcher co-stack boost: pitchers on high-implied teams see elevated
  ownership because the field stacks that offense.
- Secondary position discount 0.10: multi-eligible players (e.g. 3B/SS) are
  almost entirely owned at their primary position; minimal credit for the
  secondary pool.

Constants last tuned: 10-slate differential-evolution regression, 2026-05-15.

Post-hoc isotonic calibration: production ownership is optionally mapped
through a walk-forward-fitted isotonic (PAVA) curve per group (pitchers /
batters) stored in data/processed/ownership_calibrator.json (built by
scripts/fit_ownership_calibrator.py).  21-slate walk-forward eval (W_resid,
2026-06-11): +0.066 log-RMSE vs uncalibrated, rank order preserved by
construction, raw RMSE unharmed.  The artifact stores the constants hash it
was fitted under; load_ownership_calibrator() refuses stale artifacts so a
constants tweak can never silently combine with an outdated curve.
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OWNERSHIP_CALIBRATOR_PATH = _PROJECT_ROOT / "data" / "processed" / "ownership_calibrator.json"

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
# of whatever the SS pool would award them. Regression found 0.10: field
# overwhelmingly owns multi-eligible players at their primary position.
_SECONDARY_POSITION_DISCOUNT = 0.10

# Batter implied-total boost cap: above this value the field diversifies enough
# that ownership stops climbing proportionally with implied total.
_BATTER_TOTAL_CAP = 6.0

# Pitcher opponent-matchup boost exponent. Higher values amplify the signal
# from low opponent implied totals. Swept 0.30–0.75 across 11 slates; 0.40
# minimises composite rank (Spearman + 2×RMSE + precision) — lower values
# flatten the matchup signal without improving RMSE further.
_PITCHER_MATCHUP_EXP = 0.40

# Salary above which batters receive a soft cap penalty: (4500/salary)^0.5.
# Represents the ~75th percentile of batter salaries; above this, the $50k
# cap makes it progressively harder to build a full competitive lineup.
_BATTER_SALARY_PRESSURE_BASE = 4500

# Std floor for within-position softmax normalization.
# Higher floor → flatter distribution → less concentration on one player.
# Batter floor 1.19: field spreads ownership far more evenly across batters
# than raw softmax predicts. Pitcher floor 0.29: pitchers are more
# concentrated — field tends to anchor on the top arm.
_BATTER_STD_FLOOR  = 1.19
_PITCHER_STD_FLOOR = 0.29

# Pitcher pool compression: blend this fraction of uniform into the
# post-softmax pitcher distribution. DE regression found 0.00: actual pitcher
# ownership is not flat — the field concentrates on the top arm. Kept as a
# tunable parameter for future regressions.
_PITCHER_COMPRESS = 0.00

# Post-softmax power-law calibration exponent for batters.
# ownership^b (renormalised per position) corrects the systematic magnitude
# compression: the softmax over-assigns to low-owned players and undershoots
# highly-owned ones.  b=1.12 chosen by 8-slate composite ranking evaluation
# (evaluate_ownership.py X_both_112); previous value was 1.02.
_BATTER_CALIB_EXP = 1.12

# Post-softmax power-law calibration exponent for pitchers.
# Same mechanism as _BATTER_CALIB_EXP.  b=1.12 promoted alongside batter
# exponent after 8-slate evaluation showing meaningful improvement on chalk
# pitcher predictions (notably top-owned starters).
_PITCHER_CALIB_EXP = 1.12

# Pitcher own-team co-stack boost exponent. Pitchers on high-implied teams
# see elevated ownership because the field stacks that offense and often
# pairs the pitcher. (own_team_total / mean_total) ** this_exp.
_PITCHER_COSTACK_EXP = 0.30

# Stack-value batter boost: (mean_slot5_salary / team_slot5_salary) ** this_exp
# amplifies ownership for teams with cheap top-of-order bats.
_STACK_VALUE_EXP = 0.17

# Implied total threshold below which the stack-value signal does not fire.
_STACK_THRESHOLD = 4.0

# Per-player ownership caps applied after all softmax + calibration steps.
# Any player exceeding their cap has the excess redistributed proportionally
# to the other players in the same position group, preserving the slot-count sum.
# _PITCHER_OWN_CAP: no SP is realistically in >85% of field lineups even as
# dominant chalk.  _BATTER_OWN_CAP: batters almost never approach 100%; 1.0
# acts as a pure safety valve against math overflow without distorting normal slates.
# _PITCHER_OWN_CAP_LARGE_SLATE: tighter cap for slates with ≥5 games. On larger
# slates ownership is distributed across more pitcher options, so exceeding 65% is
# rare and the lower cap prevents outlier softmax spikes from distorting the model.
_PITCHER_OWN_CAP = 0.85
_PITCHER_OWN_CAP_LARGE_SLATE = 0.65
_BATTER_OWN_CAP  = 1.0
_LARGE_SLATE_GAME_THRESHOLD = 5

# Maximum fractional boost for the earliest-bucket batter games on the slate.
# Applied only to batters; pitchers are confirmed before the slate regardless.
# Regression found 0.04: signal is real but weak — field builds regardless of
# game time for most contests.
_TIME_FACTOR = 0.04

# Games are bucketed into this-minute windows from slate lock. All games in
# the same window receive the same time boost.
_TIME_NEUTRAL_WINDOW = 30

# Pitcher hot-streak boost exponent — used by evaluate_ownership.py (Model H)
# but not applied in production until validated on more slates.
_PITCHER_AVG_RATIO_EXP = 0.50

# Fraction of sqrt(mean) in the pitcher raw-score composite.
# The remaining (1 - fraction) weight goes to sqrt(salary_value). Regression
# found 0.98: salary_value adds negligible signal for pitchers.
_PITCHER_MEAN_FRAC = 0.98

# HR-probability batter boost exponent, applied directly to _raw = sqrt(mean).
# (hr_prob / mean_hr_prob) ** _HR_PROB_EXP on _raw ≡ ratio ** (2·exp) on mean.
_HR_PROB_EXP = 0.11

# Pitcher-opposition batter discount (salary-gated).
# Batters at or above median salary facing above-average starters are scaled by
# (starter_own / mean_starter_own)^(-_PITOPP_SAL_EXP).  Calibrated from V_sal_005
# eval over 20 slates; revisit if slate-size distribution changes significantly.
_PITOPP_SAL_EXP = 0.05


def compute_heuristic_ownership(
    players_df: pd.DataFrame,
    team_totals: dict[str, float] | None = None,
    batting_order_mult: dict[int, float] | None = None,
    batter_std_floor: float | None = None,
    team_ownership_reductions: dict[str, float] | None = None,
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
        - ``hr_prob``: market fair implied probability of 0.5+ HRs (from
          market_odds_fair_odds.json).  When present, each batter's raw score
          is multiplied by (hr_prob / mean_hr_prob)^_HR_PROB_EXP before softmax.
          Batters with null hr_prob are left unchanged.  Pitchers are unaffected.
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
        _PITCHER_MEAN_FRAC * np.sqrt(df.loc[pitcher_mask, "mean"].clip(lower=0))
        + (1.0 - _PITCHER_MEAN_FRAC) * np.sqrt(salary_value.loc[pitcher_mask].clip(lower=0))
    )

    # --- Batting order multiplier --------------------------------------------
    slot_col = (
        "lineup_slot" if "lineup_slot" in df.columns
        else ("slot" if "slot" in df.columns else None)
    )
    if slot_col:
        bom = batting_order_mult if batting_order_mult is not None else _BATTING_ORDER_MULT
        for batting_slot, mult in bom.items():
            mask = batter_mask & (df[slot_col] == batting_slot)
            df.loc[mask, "_raw"] *= mult

    # --- Salary-cap pressure on batters --------------------------------------
    # Soft penalty for players priced above the ~75th-pct batter salary.
    df.loc[batter_mask, "_raw"] *= np.minimum(
        1.0,
        (_BATTER_SALARY_PRESSURE_BASE / df.loc[batter_mask, "salary"]) ** 0.5,
    )

    # --- HR-probability batter boost -----------------------------------------
    # DFS players pay an ownership premium for HR upside beyond what E[fpts]
    # captures.  When hr_prob (fair implied prob of 0.5+ HRs) is available,
    # scale each batter's raw score by (hr_prob / mean_hr_prob)^0.125.
    # Exponent is 0.125 because _raw = sqrt(mean): multiplying mean by ratio^0.25
    # (the eval-tested strength) propagates as ratio^0.125 on _raw.
    # Calibrated from I_hr_025 eval across 05082026–05092026; revisit with more data.
    if "hr_prob" in df.columns:
        hr_valid = batter_mask & df["hr_prob"].notna()
        if hr_valid.any():
            mean_hr = float(df.loc[hr_valid, "hr_prob"].mean())
            if mean_hr > 0:
                df.loc[hr_valid, "_raw"] *= (df.loc[hr_valid, "hr_prob"] / mean_hr) ** _HR_PROB_EXP

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

        def _iso_minutes(iso_str: str) -> float | None:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(str(iso_str))
                return float(dt.hour * 60 + dt.minute)
            except (ValueError, TypeError):
                return None

        # Prefer game_start_time (ISO datetime written by dk_slate.py) over
        # trying to parse time out of the game ID string ("LAA@TOR").
        if "game_start_time" in df.columns:
            _time_lookup = (
                df[["game", "game_start_time"]]
                .drop_duplicates("game")
                .set_index("game")["game_start_time"]
            )
            game_mins = {g: _iso_minutes(t) for g, t in _time_lookup.items()}
        else:
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
        std_floor = _PITCHER_STD_FLOOR if is_p else (batter_std_floor if batter_std_floor is not None else _BATTER_STD_FLOOR)

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

    # Post-hoc power-law magnitude calibration.  Applied after the full
    # multi-position accumulation so all contributions are included before
    # sharpening.  Corrects systematic under-assignment to high-owned players.
    result = df["ownership"].values.astype(np.float64)
    pos_vals = df["position"].values
    for pos, n_slots in _SLOT_COUNTS.items():
        exp = _PITCHER_CALIB_EXP if pos == "P" else _BATTER_CALIB_EXP
        if exp == 1.0:
            continue
        pmask = pos_vals == pos
        if not pmask.any():
            continue
        vals = result[pmask]
        total = vals.sum()
        if total > 0:
            cal = vals ** exp
            result[pmask] = cal / cal.sum() * n_slots

    # --- Per-team ownership reductions (post-calibration) -------------------
    # Applied after the full softmax + power-law calibration.  Per-position:
    #   1. Lock reduced-team players at exactly (1 - pct/100) × baseline.
    #   2. Scale the non-reduced remainder to absorb the freed ownership so
    #      the position slot-count sum is preserved.
    # This gives an exact pct% drop for the affected players.  If no
    # non-reduced players exist in a position, the reduction is still applied
    # and the position sum is allowed to fall below its slot count.
    if team_ownership_reductions and "team" in df.columns:
        team_vals = df["team"].values
        active_reductions = {t: p for t, p in team_ownership_reductions.items() if 0 < p < 100}
        if active_reductions:
            for pos, n_slots in _SLOT_COUNTS.items():
                pmask = pos_vals == pos
                if not pmask.any():
                    continue
                pos_teams = team_vals[pmask]
                pos_result = result[pmask].copy()
                # Identify which position-group entries belong to a reduced team.
                in_reduced = np.zeros(pmask.sum(), dtype=bool)
                for _t, _pct in active_reductions.items():
                    tin = pos_teams == _t
                    pos_result[tin] *= (1.0 - _pct / 100.0)
                    in_reduced |= tin
                # Scale non-reduced players to absorb the freed ownership.
                not_reduced = ~in_reduced
                if not_reduced.any():
                    target = n_slots - pos_result[in_reduced].sum()
                    current = pos_result[not_reduced].sum()
                    if current > 0:
                        pos_result[not_reduced] *= target / current
                result[pmask] = pos_result

    # --- Per-player ownership cap with redistribution ------------------------
    # Iteratively clip any player above their position cap and spread the excess
    # proportionally among uncapped players, preserving the slot-count sum.
    n_games = df["game"].nunique() if "game" in df.columns else 0
    pitcher_cap = (
        _PITCHER_OWN_CAP_LARGE_SLATE
        if n_games >= _LARGE_SLATE_GAME_THRESHOLD
        else _PITCHER_OWN_CAP
    )
    for pos, n_slots in _SLOT_COUNTS.items():
        cap = pitcher_cap if pos == "P" else _BATTER_OWN_CAP
        pmask = pos_vals == pos
        if not pmask.any():
            continue
        vals = result[pmask].copy()
        for _ in range(20):
            over = vals > cap
            if not over.any():
                break
            excess = (vals[over] - cap).sum()
            vals[over] = cap
            under = ~over
            if under.any():
                under_sum = vals[under].sum()
                if under_sum > 0:
                    vals[under] += excess * vals[under] / under_sum
        result[pmask] = vals

    # --- Pitcher-opposition batter discount (salary-gated) --------------------
    # DFS players who roster a pitcher implicitly fade that pitcher's opponents.
    # Only applied to batters at or above the median batter salary — cheap batters
    # are low-owned for price/projection reasons unrelated to pitcher matchups.
    if "opponent" in df.columns and _PITOPP_SAL_EXP > 0:
        # Identify each team's starting pitcher (lowest slot if known, else highest mean).
        starter_idx: dict[str, int] = {}
        for team in df.loc[pitcher_mask, "team"].unique():
            tm = pitcher_mask & (df["team"] == team)
            grp = df[tm]
            if slot_col and grp[slot_col].notna().any():
                best = int(grp.dropna(subset=[slot_col])[slot_col].idxmin())
            else:
                best = int(grp["mean"].idxmax())
            starter_idx[team] = best

        if starter_idx:
            starter_own = {team: float(result[idx]) for team, idx in starter_idx.items()}
            mean_starter_own = float(np.mean(list(starter_own.values())))

            if mean_starter_own > 0:
                salaries_arr = df["salary"].values.astype(float)
                opponents_arr = df["opponent"].values
                sal_median = float(np.median(salaries_arr[batter_mask]))

                for pos, n_slots in _SLOT_COUNTS.items():
                    if pos == "P":
                        continue
                    pmask = pos_vals == pos
                    if not pmask.any():
                        continue
                    orig_sum = float(result[pmask].sum())

                    for i in np.where(pmask)[0]:
                        if salaries_arr[i] < sal_median:
                            continue
                        opp_own = starter_own.get(opponents_arr[i])
                        if opp_own is None:
                            continue
                        result[i] *= (opp_own / mean_starter_own) ** (-_PITOPP_SAL_EXP)

                    new_sum = float(result[pmask].sum())
                    if new_sum > 0:
                        result[pmask] *= orig_sum / new_sum

                # Second pass: suppress-only — only batters facing above-average
                # pitchers are further discounted.  Mirrors V_sal_005's external
                # reapplication, which empirically outperformed a single two-sided
                # pass on 20 slates (0.8442 vs 0.8436 composite Spearman+RMSE+prec).
                for pos, n_slots in _SLOT_COUNTS.items():
                    if pos == "P":
                        continue
                    pmask = pos_vals == pos
                    if not pmask.any():
                        continue
                    orig_sum = float(result[pmask].sum())

                    for i in np.where(pmask)[0]:
                        if salaries_arr[i] < sal_median:
                            continue
                        opp_own = starter_own.get(opponents_arr[i])
                        if opp_own is None:
                            continue
                        ratio = opp_own / mean_starter_own
                        if ratio <= 1.0:
                            continue
                        result[i] *= ratio ** (-_PITOPP_SAL_EXP)

                    new_sum = float(result[pmask].sum())
                    if new_sum > 0:
                        result[pmask] *= orig_sum / new_sum

    return result


# ---------------------------------------------------------------------------
# Constants provenance
# ---------------------------------------------------------------------------

def collect_ownership_constants() -> dict[str, float]:
    """
    Snapshot every module-level _UPPER_CASE scalar constant of this module
    (the 11 DE-tunable parameters plus calibration exponents, caps,
    thresholds).  Dict-valued tunables (_SLOT_COUNTS, _BATTING_ORDER_MULT)
    are excluded; the scalars are sufficient for change detection.
    """
    return {
        name: float(value)
        for name, value in globals().items()
        if re.fullmatch(r"_[A-Z][A-Z0-9_]*", name)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def ownership_constants_hash() -> str:
    """Short stable hash of the current scalar constants — used to detect
    when a fitted calibrator artifact has gone stale after a constants tweak."""
    constants_json = json.dumps(collect_ownership_constants(), sort_keys=True)
    return hashlib.md5(constants_json.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Post-hoc isotonic calibration
# ---------------------------------------------------------------------------

def apply_ownership_calibration(
    ownership: np.ndarray,
    positions: np.ndarray,
    calibrator: dict,
) -> np.ndarray:
    """
    Map raw ownership through the fitted isotonic curve (linear interpolation
    between PAVA knots, clamped at the edges), then renormalise each position
    group back to its DK slot count.

    calibrator holds per-group knot arrays: {"P": (x, y), "bat": (x, y)};
    a missing group key means identity for that group.  The curve is monotone
    non-decreasing, so within-group rank order can never invert.
    """
    positions = np.asarray(positions)
    adjusted = np.asarray(ownership, dtype=float).copy()

    for key, group_mask in (("P", positions == "P"), ("bat", positions != "P")):
        curve = calibrator.get(key)
        if curve is None or not group_mask.any():
            continue
        x, y = curve
        adjusted[group_mask] = np.interp(adjusted[group_mask], x, y)

    adjusted = np.maximum(adjusted, 1e-6)

    result = np.zeros_like(adjusted)
    for pos, n_slots in _SLOT_COUNTS.items():
        mask = positions == pos
        if not mask.any():
            continue
        vals = adjusted[mask]
        total = vals.sum()
        result[mask] = vals / total * n_slots if total > 0 else vals
    return result


def load_ownership_calibrator(
    path: Path | str = OWNERSHIP_CALIBRATOR_PATH,
    check_constants_hash: bool = True,
) -> dict | None:
    """
    Load the fitted isotonic calibrator artifact written by
    scripts/fit_ownership_calibrator.py.

    Returns {"P": (x, y), "bat": (x, y), ...metadata} ready for
    apply_ownership_calibration, or None when the artifact is missing,
    malformed, or stale (fitted under different model constants than the
    ones currently in this module) — callers fall back to uncalibrated
    ownership in that case.
    """
    path = Path(path)
    if not path.exists():
        logger.info("Ownership calibrator artifact not found (%s) — using uncalibrated ownership.", path)
        return None

    try:
        artifact = json.loads(path.read_text())
        groups = artifact["groups"]
        calibrator: dict = {
            "fitted_at": artifact.get("fitted_at"),
            "constants_hash": artifact.get("constants_hash"),
            "n_slates": artifact.get("n_slates"),
        }
        for key in ("P", "bat"):
            if key not in groups:
                continue
            x = np.asarray(groups[key]["x"], dtype=float)
            y = np.asarray(groups[key]["y"], dtype=float)
            if len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
                raise ValueError(f"invalid knots for group {key!r}")
            calibrator[key] = (x, y)
        if "P" not in calibrator and "bat" not in calibrator:
            raise ValueError("no fitted groups in artifact")
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ownership calibrator artifact %s is malformed (%s) — using uncalibrated ownership.", path, exc)
        return None

    if check_constants_hash:
        current = ownership_constants_hash()
        fitted = calibrator.get("constants_hash")
        if fitted != current:
            logger.warning(
                "Ownership calibrator is stale (fitted under constants %s, current %s) — "
                "using uncalibrated ownership. Re-run scripts/fit_ownership_calibrator.py.",
                fitted, current,
            )
            return None

    return calibrator
