"""
Benchmark the production ownership model against baselines and experimental
candidates using actual DraftKings %Drafted data from archived slates.

The primary evaluation question is: does compute_heuristic_ownership()
(the live optimizer's model, labeled E_production) achieve Spearman ≥ 0.60
across 5+ slates? All other rows in the output table exist to provide context
for how much each signal contributes.

Usage
-----
    python scripts/evaluate_ownership.py archive/04072026
    python scripts/evaluate_ownership.py archive/04072026 archive/04082026 ...
    python scripts/evaluate_ownership.py archive/*/

Output
------
  - Per-slate and aggregate table printed to stdout.
  - archive/MMDDYYYY/ownership_eval.csv  (per-player predictions vs actuals).
  - archive/ownership_summary.csv        (one row per slate, aggregate metrics).

Models
------
  Production — the model used by the live optimizer at runtime:
    E_production   compute_heuristic_ownership() from
                   src/optimization/ownership.py. Includes sqrt compression,
                   batting-order multiplier, salary-cap pressure, HR-probability
                   batter boost (exp=0.25, when hr_prob column present),
                   game-time multiplier, stack-value batter boost, pitcher
                   matchup and co-stack boosts, and post-softmax pitcher
                   compression.

  Experimental — candidates being evaluated as potential replacements:
    P_batter_avg   Production + AvgPPG ratio boost for hot-streak batters.
                   avg_ratio = clip(avg_pts/mean, 1, 5); mean scaled by
                   avg_ratio^exp (exp=0.15).  No fade
                   for cold batters.  Requires avg_pts column in pool.
    R_scoring      Production + batter scoring-participation composite boost.
                   scoring = runs_over_0.5_prob + rbi_over_0.5_prob;
                   boost = (scoring/mean_scoring)^exp clipped to [0.5, 3.0]
                   (exp ∈ {0.10, 0.20}).  Requires "Player Runs" and/or
                   "Player RBIs" markets in market_odds_fair_odds.json.
    R_boost        Same as R_scoring but boost-only: ratio clipped to [1.0, 3.0]
                   so below-average scoring batters are unchanged (no explicit
                   fade).  Mirrors the P_bavg "no fade for cold batters" design.
    R_salpres      R_scoring with salary-cap pressure applied to the scoring
                   composite before ranking.  Discounts expensive batters'
                   scoring props by the same cap-pressure factor used by
                   compute_heuristic_ownership: min(1, (4500/salary)^0.5).
                   Aligns the scoring-participation signal with DFS ownership
                   patterns and preserves bot_prec relative to R_scoring.
                   exp ∈ {0.10, 0.20}.

"""

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DK position tags that appear in lineup strings
_LINEUP_POSITIONS = {"P", "C", "1B", "2B", "3B", "SS", "OF"}

# Mapping from multi-position strings in DK salary CSV to a canonical single position
_POSITION_CANONICAL = {
    "SP": "P", "RP": "P",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
    "UTIL": "UTIL",
}


# ---------------------------------------------------------------------------
# Contest standings parser
# ---------------------------------------------------------------------------

def _parse_contest_zip(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a DraftKings contest-standings-XXXXXXX.zip into two DataFrames:

    standings_df — one row per contest entry:
        rank, entry_id, entry_name, points, lineup_str

    ownership_df — one row per player in the contest pool:
        player_name, roster_position, pct_drafted, fpts
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        content = zf.read(csv_name).decode("utf-8-sig")

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise ValueError(f"Empty contest CSV in {zip_path}")

    header = rows[0]
    # Expected: Rank, EntryId, EntryName, TimeRemaining, Points, Lineup, '', Player, Roster Position, %Drafted, FPTS
    try:
        rank_col    = header.index("Rank")
        entry_col   = header.index("EntryId")
        name_col    = header.index("EntryName")
        points_col  = header.index("Points")
        lineup_col  = header.index("Lineup")
        player_col  = header.index("Player")
        pos_col     = header.index("Roster Position")
        pct_col     = header.index("%Drafted")
        fpts_col    = header.index("FPTS")
    except ValueError as exc:
        raise ValueError(f"Unexpected contest CSV header in {zip_path}: {exc}") from exc

    standing_records = []
    ownership_records = []

    for row in rows[1:]:
        if len(row) <= lineup_col:
            continue

        # Left section — contest entry
        rank_raw = row[rank_col].strip()
        if rank_raw:
            try:
                standing_records.append({
                    "rank": int(rank_raw),
                    "entry_id": row[entry_col].strip(),
                    "entry_name": row[name_col].strip(),
                    "points": float(row[points_col]) if row[points_col].strip() else 0.0,
                    "lineup_str": row[lineup_col].strip(),
                })
            except (ValueError, IndexError):
                pass

        # Right section — player ownership (only present for the first N rows)
        if len(row) > player_col and row[player_col].strip():
            pct_raw = row[pct_col].strip().rstrip("%") if len(row) > pct_col else ""
            fpts_raw = row[fpts_col].strip() if len(row) > fpts_col else ""
            try:
                ownership_records.append({
                    "player_name": row[player_col].strip(),
                    "roster_position": row[pos_col].strip() if len(row) > pos_col else "",
                    "pct_drafted": float(pct_raw) / 100.0 if pct_raw else None,
                    "actual_fpts": float(fpts_raw) if fpts_raw else None,
                })
            except (ValueError, IndexError):
                pass

    standings_df = pd.DataFrame(standing_records)
    ownership_df = pd.DataFrame(ownership_records).dropna(subset=["pct_drafted"])
    return standings_df, ownership_df


# ---------------------------------------------------------------------------
# Player pool builder
# ---------------------------------------------------------------------------

def _canonical_position(raw: str) -> str:
    """Map a raw DK position string (may be multi like 'SP/RP') to a canonical tag."""
    for part in raw.split("/"):
        canon = _POSITION_CANONICAL.get(part.strip().upper())
        if canon and canon != "UTIL":
            return canon
    return raw.split("/")[0].strip().upper()


def _parse_eligible_positions(raw: str) -> list[str]:
    """Parse a DK position string like '3B/SS' into a list of canonical positions."""
    seen: set[str] = set()
    result: list[str] = []
    for part in str(raw).split("/"):
        canon = _POSITION_CANONICAL.get(part.strip().upper())
        if canon and canon != "UTIL" and canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result or [raw.split("/")[0].strip().upper()]


def _build_player_pool(
    archive_dir: Path,
) -> pd.DataFrame:
    """
    Load DKSalaries.csv and projections from archive_dir, join on player_id,
    and compute features needed for ownership models.

    Projection source priority (mirrors live app "Market Odds" mode):
      1. market_odds_projections.csv — mean/std_dev for players that have it.
      2. dff_projections.csv — mean/std_dev fallback + lineup_slot for all players.

    Returns a DataFrame with columns:
        player_id, name, salary, position, eligible_positions, team, opponent,
        game, mean, std_dev, lineup_slot, salary_value, implied_total (if available)
    """
    salary_path = archive_dir / "DKSalaries.csv"
    dff_path    = archive_dir / "dff_projections.csv"
    mo_path     = archive_dir / "market_odds_projections.csv"

    if not salary_path.exists():
        raise FileNotFoundError(f"DKSalaries.csv not found in {archive_dir}")
    if not dff_path.exists():
        raise FileNotFoundError(f"dff_projections.csv not found in {archive_dir}")

    sal_df = pd.read_csv(salary_path)
    sal_df.rename(
        columns={"ID": "player_id", "Name": "name", "Salary": "salary",
                 "TeamAbbrev": "team", "Game Info": "game", "Position": "raw_position",
                 "AvgPointsPerGame": "avg_pts"},
        inplace=True,
    )
    sal_df["player_id"] = sal_df["player_id"].astype(int)
    sal_df["eligible_positions"] = sal_df["raw_position"].apply(_parse_eligible_positions)
    sal_df["position"] = sal_df["eligible_positions"].str[0]

    # Derive opponent from game string (e.g. "TEX@DET 05/02/2026 07:15PM ET")
    def _opponent(row: pd.Series) -> str:
        m = re.match(r"(\w+)@(\w+)", str(row["game"]))
        if m:
            away, home = m.group(1), m.group(2)
            return home if row["team"] == away else away
        return ""

    sal_df["opponent"] = sal_df.apply(_opponent, axis=1)
    avg_pts_col = ["avg_pts"] if "avg_pts" in sal_df.columns else []
    sal_df = sal_df[
        ["player_id", "name", "salary", "position", "eligible_positions", "team", "opponent", "game"] + avg_pts_col
    ].drop_duplicates("player_id")

    # Load DFF for lineup_slot (and as mean/std_dev fallback).
    dff_df = pd.read_csv(dff_path)
    dff_df["player_id"] = dff_df["player_id"].astype(int)
    dff_cols = ["player_id", "mean", "std_dev"]
    if "lineup_slot" in dff_df.columns:
        dff_cols.append("lineup_slot")

    # Load market odds if present and overlay mean/std_dev (mirrors live "Market Odds" source).
    if mo_path.exists():
        mo_df = pd.read_csv(mo_path)
        mo_df["player_id"] = mo_df["player_id"].astype(int)
        mo_lookup = mo_df.set_index("player_id")[["mean", "std_dev"]]
        dff_df = dff_df.copy()
        has_mo = dff_df["player_id"].isin(mo_lookup.index)
        dff_df.loc[has_mo, "mean"]    = dff_df.loc[has_mo, "player_id"].map(mo_lookup["mean"])
        dff_df.loc[has_mo, "std_dev"] = dff_df.loc[has_mo, "player_id"].map(mo_lookup["std_dev"])

        # Players present in MO but absent from DFF were still in the live RW player
        # pool (market_odds_projections.csv is a copy of the final merged projections).
        # Add them so they're not silently dropped at the dropna step — without this,
        # exclusions.csv entries for MO-only players have no effect.
        mo_only_mask = ~mo_df["player_id"].isin(dff_df["player_id"])
        if mo_only_mask.any():
            mo_extra_cols = ["player_id", "mean", "std_dev"]
            for col in ("lineup_slot",):
                if col in mo_df.columns and col in dff_cols:
                    mo_extra_cols.append(col)
            mo_extra = mo_df.loc[mo_only_mask, [c for c in mo_extra_cols if c in mo_df.columns]]
            dff_df = pd.concat([dff_df, mo_extra], ignore_index=True)

        n_mo_only = mo_only_mask.sum()
        print(f"  Projections: market_odds for {has_mo.sum()} players, "
              f"DFF fallback for {(~has_mo).sum()}, "
              f"{n_mo_only} MO-only player(s) added")
    else:
        print(f"  Projections: DFF only (no market_odds_projections.csv)")

    merged = sal_df.merge(dff_df[dff_cols], on="player_id", how="left")
    merged = merged.dropna(subset=["mean"])  # only projected starters

    merged["salary_value"] = merged["mean"] / merged["salary"] * 1000

    # Optionally attach team implied totals — prefer FL (live), then CNO (legacy), then DFF.
    totals_df = None
    for totals_fname in ("team_totals.csv", "cno_team_totals.csv", "dff_team_totals.csv"):
        totals_path = archive_dir / totals_fname
        if totals_path.exists():
            totals_df = pd.read_csv(totals_path)
            totals_df.columns = [c.lower() for c in totals_df.columns]
            break
    if totals_df is not None:
        merged = merged.merge(totals_df[["team", "implied_total"]], on="team", how="left")
    else:
        merged["implied_total"] = np.nan

    # Apply exclusions — players who were scratched or otherwise invalid.
    exclusions_path = archive_dir / "exclusions.csv"
    if exclusions_path.exists():
        excl_df = pd.read_csv(exclusions_path)
        if "player_id" in excl_df.columns:
            excl_ids = set(excl_df["player_id"].dropna().astype(int))
            n_before = len(merged)
            merged = merged[~merged["player_id"].isin(excl_ids)]
            dropped = n_before - len(merged)
            if dropped:
                names = excl_df.loc[excl_df["player_id"].isin(excl_ids), "name"].tolist() if "name" in excl_df.columns else []
                print(f"  Excluded {dropped} player(s) via exclusions.csv: {names}")

    # Attach HR fair-odds implied probability for batters (I_hr_* models).
    hr_odds = _load_hr_fair_odds(archive_dir)
    if hr_odds:
        merged["hr_prob"] = merged["name"].apply(lambda n: hr_odds.get(_normalise(str(n))))
        n_matched = merged["hr_prob"].notna().sum()
        print(f"  HR odds: {n_matched}/{len(merged)} players matched")
    else:
        merged["hr_prob"] = np.nan

    # Pitcher K-prop expected-K score (Q models).
    k_props = _load_pitcher_k_props(archive_dir)
    if k_props:
        merged["k_prop"] = merged["name"].apply(lambda n: k_props.get(_normalise(str(n))))
        n_k = merged["k_prop"].notna().sum()
        print(f"  K-props: {n_k}/{len(merged)} players matched")
    else:
        merged["k_prop"] = np.nan

    # Batter scoring participation props: runs + RBIs over-0.5 (R models).
    scoring_props = _load_batter_scoring_props(archive_dir)
    if scoring_props:
        merged["runs_prop"] = merged["name"].apply(
            lambda n: scoring_props.get(_normalise(str(n)), (None, None))[0]
        )
        merged["rbi_prop"] = merged["name"].apply(
            lambda n: scoring_props.get(_normalise(str(n)), (None, None))[1]
        )
        n_sc = merged["runs_prop"].notna().sum()
        print(f"  Scoring props: {n_sc}/{len(merged)} players matched")
    else:
        merged["runs_prop"] = np.nan
        merged["rbi_prop"]  = np.nan

    # Load static handedness table (bats L/R/S, throws L/R).
    handedness_path = PROJECT_ROOT / "data" / "handedness.csv"
    if handedness_path.exists():
        hand_df = pd.read_csv(handedness_path, dtype=str)
        # Stage 1: (norm_name, team) exact match
        hand_by_team = hand_df.set_index(["name", "team"])[["bats", "throws"]]
        # Stage 2: name-only fallback for players with exactly one entry
        unique_mask = ~hand_df.duplicated(subset="name", keep=False)
        hand_by_name = hand_df[unique_mask].set_index("name")[["bats", "throws"]]

        def _lookup_hand(row: pd.Series) -> pd.Series:
            key = (_normalise(str(row["name"])), str(row["team"]))
            if key in hand_by_team.index:
                return hand_by_team.loc[key]
            norm = key[0]
            if norm in hand_by_name.index:
                return hand_by_name.loc[norm]
            return pd.Series({"bats": pd.NA, "throws": pd.NA})

        hand_cols = merged.apply(_lookup_hand, axis=1)
        merged["bats"]   = hand_cols["bats"].values
        merged["throws"] = hand_cols["throws"].values
        n_hand = merged["bats"].notna().sum()
        print(f"  Handedness: {n_hand}/{len(merged)} players matched")
    else:
        merged["bats"]   = pd.NA
        merged["throws"] = pd.NA

    # Derive opposing starting pitcher's throwing hand for each batter.
    pitchers = merged[merged["position"] == "P"]
    slot_col = "lineup_slot" if "lineup_slot" in merged.columns else None
    starter_throws: dict[str, str] = {}
    for team, grp in pitchers.groupby("team"):
        grp_h = grp.dropna(subset=["throws"])
        if grp_h.empty:
            continue
        if slot_col and grp_h[slot_col].notna().any():
            row = grp_h.dropna(subset=[slot_col]).sort_values(slot_col).iloc[0]
        else:
            row = grp_h.loc[grp_h["mean"].idxmax()]
        starter_throws[team] = str(row["throws"])
    merged["opp_pitcher_throws"] = merged["opponent"].map(starter_throws)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ownership models
# ---------------------------------------------------------------------------


def compute_models(
    pool_df: pd.DataFrame,
    historical_tier_spend: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Return a dict of {model_name: ownership_array} aligned to pool_df index.

    Ownership arrays are probabilities within each position group (sum to 1 per pos).
    Read all output rows relative to E_production — that is the live optimizer's model.

    historical_tier_spend: from fit_historical_tier_spend(); when provided,
        N_fwd_tier (forward-looking, production-eligible) is computed.
    """
    models: dict[str, np.ndarray] = {}

    pool_df = pool_df.copy()

    # Build team_totals dict for models that accept it (E–H)
    team_totals: dict[str, float] | None = None
    if pool_df["implied_total"].notna().any():
        team_totals = (
            pool_df[["team", "implied_total"]]
            .dropna(subset=["implied_total"])
            .drop_duplicates("team")
            .set_index("team")["implied_total"]
            .to_dict()
        )

    # ── Production ────────────────────────────────────────────────────────────
    # compute_heuristic_ownership is what the live optimizer calls at runtime.
    # This is the row all other results should be read relative to.
    try:
        from src.optimization.ownership import compute_heuristic_ownership
        models["E_production"] = compute_heuristic_ownership(pool_df, team_totals)
    except Exception as exc:
        print(f"  [E_production skipped: {exc}]")

    # ── Experimental ──────────────────────────────────────────────────────────
    # P: sweep avg_ratio_exp for batter AvgPointsPerGame hot-streak boost.
    _BATTER_AVG_EXPS = [0.15]
    for _exp in _BATTER_AVG_EXPS:
        _label = f"P_bavg_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_p(pool_df, team_totals, _exp)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    # J: sweep _PITCHER_MATCHUP_EXP around current production value (0.75).
    _MATCHUP_EXPS = [0.30, 0.50]
    for _exp in _MATCHUP_EXPS:
        _label = f"J_mexp_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_j(pool_df, team_totals, _exp)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    # L: salary-calibrated ownership — tilts E_production toward expensive
    # players until implied lineup salary hits the historical field mean from
    # contest_stats.json.  Silently skipped if the stats file is absent.
    try:
        models["L_sal_calibrated"] = _compute_model_l(pool_df, team_totals)
    except Exception as exc:
        print(f"  [L_sal_calibrated skipped: {exc}]")

    # U: post-hoc power-law calibration sweep to address chalk under-prediction.
    # Applies ownership^b per position group (renormalised), including pitchers.
    # Spearman is unchanged (monotone transform); targets RMSE and chalk accuracy.
    _CALIB_EXPS = [1.1, 1.2]
    for _exp in _CALIB_EXPS:
        _label = f"U_calib_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_u(pool_df, team_totals, _exp)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    # R: batter scoring-participation composite (runs + RBIs over-0.5 props).
    # Also sweep boost-only variant (no fade for below-average scoring batters).
    _SCORING_EXPS = [0.10, 0.20]
    for _exp in _SCORING_EXPS:
        _label = f"R_scoring_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_r(pool_df, team_totals, _exp)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    for _exp in _SCORING_EXPS:
        _label = f"R_boost_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_r(pool_df, team_totals, _exp, boost_only=True)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    # R_salpres: R_scoring with salary-cap pressure applied to scoring composite.
    # Expensive batters' scoring props are discounted to align with DFS ownership.
    for _exp in _SCORING_EXPS:
        _label = f"R_salpres_{int(_exp * 100):03d}"
        try:
            models[_label] = _compute_model_r(pool_df, team_totals, _exp, sal_pressure=True)
        except Exception as exc:
            print(f"  [{_label} skipped: {exc}]")

    return models



def _compute_model_p(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    avg_ratio_exp: float,
) -> np.ndarray:
    """
    P_batter_avg — production model + batter hot-streak boost via AvgPointsPerGame.

    When a batter's DK AvgPointsPerGame exceeds their current forward projection
    (avg_ratio > 1), their projected mean is scaled by avg_ratio^avg_ratio_exp
    before passing to compute_heuristic_ownership.  No fade for cold batters
    (avg_ratio <= 1) — underperformance vs. projection is treated as noise.

    avg_ratio = clip(avg_pts / mean, 1, 5):
      - floor at 1  : only boost, never fade
      - cap at 5    : prevents extreme outliers (injured player returning, tiny samples)

    avg_pts is dropped before calling compute_heuristic_ownership so the
    production function cannot double-apply the signal if it is ever introduced there.

    Falls back to unmodified production output when avg_pts column is absent.
    """
    from src.optimization.ownership import compute_heuristic_ownership

    if "avg_pts" not in pool_df.columns:
        return compute_heuristic_ownership(pool_df, team_totals)

    df = pool_df.copy().reset_index(drop=True)
    batter_mask = df["position"] != "P"

    avg_ratio = (df["avg_pts"] / df["mean"].clip(lower=0.5)).clip(upper=5.0)
    hot = batter_mask & df["avg_pts"].notna() & (df["avg_pts"] > 0) & (avg_ratio > 1.0)

    if hot.any():
        df.loc[hot, "mean"] = df.loc[hot, "mean"] * (avg_ratio[hot] ** avg_ratio_exp)
        df.loc[hot, "salary_value"] = df.loc[hot, "mean"] / df.loc[hot, "salary"] * 1000

    return compute_heuristic_ownership(df.drop(columns=["avg_pts"]), team_totals)


def _compute_model_i(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    boost: float,
) -> np.ndarray:
    """
    I_rhlhp — production model + right-handed batter vs left-handed pitcher boost.

    Batters with bats=="R" and opp_pitcher_throws=="L" have their projected mean
    scaled by `boost` before passing to compute_heuristic_ownership.  Scaling
    mean propagates through sqrt(mean) in the raw score, giving an effective
    raw-score multiplier of sqrt(boost).

    Falls back to unmodified production output when handedness columns are absent.
    """
    from src.optimization.ownership import compute_heuristic_ownership

    if "bats" not in pool_df.columns or "opp_pitcher_throws" not in pool_df.columns:
        return compute_heuristic_ownership(pool_df, team_totals)

    rh_vs_lhp = (pool_df["bats"] == "R") & (pool_df["opp_pitcher_throws"] == "L")
    if not rh_vs_lhp.any():
        return compute_heuristic_ownership(pool_df, team_totals)

    df = pool_df.copy()
    df.loc[rh_vs_lhp, "mean"] *= boost
    df.loc[rh_vs_lhp, "salary_value"] = df.loc[rh_vs_lhp, "mean"] / df.loc[rh_vs_lhp, "salary"] * 1000
    return compute_heuristic_ownership(df, team_totals)


def _compute_model_j(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    matchup_exp: float,
) -> np.ndarray:
    """
    J_mexp_* — production model with _PITCHER_MATCHUP_EXP swept to matchup_exp.

    Temporarily patches the module-level constant so compute_heuristic_ownership
    uses the candidate exponent, then restores the original value.  Isolates the
    effect of the matchup exponent without duplicating the rest of the model.

    Current production value is 2.0; sweep covers 0.25–1.75.
    """
    import src.optimization.ownership as _own_mod
    from src.optimization.ownership import compute_heuristic_ownership

    old_exp = _own_mod._PITCHER_MATCHUP_EXP
    try:
        _own_mod._PITCHER_MATCHUP_EXP = matchup_exp
        return compute_heuristic_ownership(pool_df, team_totals)
    finally:
        _own_mod._PITCHER_MATCHUP_EXP = old_exp


def _compute_model_k(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    params: dict[str, float],
) -> np.ndarray:
    """
    K_regressed — production model with constants tuned by optimize_ownership_params.py.

    Temporarily patches all module-level constants listed in params, calls
    compute_heuristic_ownership, then restores originals.  Mirrors the pattern
    used by _compute_model_j but accepts a full constant dict rather than a
    single exponent.

    params comes from archive/ownership_regression_results.json → "params".
    """
    import src.optimization.ownership as _own_mod

    old_vals = {name: getattr(_own_mod, name) for name in params if hasattr(_own_mod, name)}
    for name, val in params.items():
        if hasattr(_own_mod, name):
            setattr(_own_mod, name, float(val))
    try:
        from src.optimization.ownership import compute_heuristic_ownership
        return compute_heuristic_ownership(pool_df, team_totals)
    finally:
        for name, old_val in old_vals.items():
            setattr(_own_mod, name, old_val)


def _compute_model_l(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
) -> np.ndarray:
    """
    L_sal_calibrated — E_production with a post-hoc salary tilt to hit the
    historical field mean lineup salary from contest_stats.json.

    After computing base ownership from compute_heuristic_ownership, applies:

        o'_j = o_j × (s_j / mean_s)^α × C

    where C normalises to preserve Σ o'_j = 10 (total slots unchanged), and
    α is found by bisection so that Σ(o'_j × s_j) = target_mean_sal.

    α = 0 → identical to E_production.  α > 0 → expensive players gain share;
    α < 0 → cheap players gain share.  The bisection converges in 50 iterations
    (~1e-14 precision) over a search range of [−5, 5].

    Skipped (exception raised) if data/processed/contest_stats.json is absent —
    call analyze_contest_lineups.py first to generate it.
    """
    import json as _json

    stats_path = PROJECT_ROOT / "data" / "processed" / "contest_stats.json"
    with open(stats_path) as _f:
        target_mean_sal = float(_json.load(_f)["aggregate"]["salary_mean"])

    from src.optimization.ownership import compute_heuristic_ownership
    base     = compute_heuristic_ownership(pool_df, team_totals)
    salaries = pool_df["salary"].values.astype(float)
    mean_s   = salaries.mean()
    total    = base.sum()  # sum of all ownership values = total roster slots = 10

    def _tilt(alpha: float) -> np.ndarray:
        w = base * (salaries / mean_s) ** alpha
        return w * total / w.sum()

    lo, hi = -5.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if float(np.sum(_tilt(mid) * salaries)) < target_mean_sal:
            lo = mid
        else:
            hi = mid

    return _tilt((lo + hi) / 2.0)




def fit_batter_calibration_exp(training_dirs: list[Path]) -> float:
    """Fit power-law exponent b on training slates to minimise batter RMSE.

    For each position group on each slate, calibrated ownership =
    (pred^b) / sum(pred^b) * n_slots.  Pitchers are excluded — they are
    already well-calibrated.  Returns the scalar b that minimises the summed
    squared error across all training batter observations.
    """
    from scipy.optimize import minimize_scalar

    _SLOT_COUNTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}

    dfs = []
    for d in training_dirs:
        fp = d / "ownership_eval.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["_slate"] = d.name
            dfs.append(df)

    if not dfs:
        return 1.0

    all_df = pd.concat(dfs, ignore_index=True)
    batters = all_df[all_df["position"] != "P"].copy()

    def loss(b: float) -> float:
        total = 0.0
        for (slate, pos), grp in batters.groupby(["_slate", "position"]):
            n_slots = _SLOT_COUNTS.get(pos, 1)
            pred = grp["pred_E_production"].values.astype(float)
            actual = grp["pct_drafted"].values.astype(float)
            if len(pred) == 0 or pred.sum() == 0:
                continue
            cal = pred ** b
            cal = cal / cal.sum() * n_slots
            total += float(np.sum((cal - actual) ** 2))
        return total

    result = minimize_scalar(loss, bounds=(0.3, 5.0), method="bounded")
    return float(result.x)


def _compute_model_u(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    calib_exp: float,
) -> np.ndarray:
    """
    U_calibrated — E_production with post-hoc power-law magnitude calibration.

    Applies ownership^b per position group (renormalised to slot counts).
    b is fitted on held-out training slates to minimise batter RMSE.
    Spearman is unchanged (monotone transform preserves rank order exactly).
    Pitchers inherit the same transform for consistency, though their bias
    is small — could be excluded in a future refinement.
    """
    from src.optimization.ownership import compute_heuristic_ownership

    _SLOT_COUNTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}

    base = compute_heuristic_ownership(pool_df, team_totals)
    result = np.zeros_like(base)
    positions = pool_df["position"].values

    for pos, n_slots in _SLOT_COUNTS.items():
        mask = positions == pos
        if not mask.any():
            continue
        vals = base[mask]
        cal = vals ** calib_exp
        total = cal.sum()
        result[mask] = cal / total * n_slots if total > 0 else vals
    return result


def _compute_model_r(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    scoring_exp: float,
    boost_only: bool = False,
    sal_pressure: bool = False,
) -> np.ndarray:
    """
    R_scoring — production model + batter scoring-participation composite boost.

    Individual batter "Player Runs" and "Player RBIs" over-0.5 implied probs
    encode expected scoring participation beyond batting-order × team-implied-total.

    scoring_i = runs_prob + rbi_prob  (NA treated as 0; skipped if both NA)
    boost     = (scoring_i / mean_scoring)^scoring_exp

    When boost_only=False (default): ratio clipped to [0.5, 3.0] — below-average
    scoring batters are faded as well as boosted.
    When boost_only=True (R_boost_*): ratio clipped to [1.0, 3.0] — only batters
    with above-average scoring participation receive a boost; below-average are
    unchanged (implicit fade through softmax renormalisation only).  Mirrors the
    P_bavg "no fade for cold batters" design.
    When sal_pressure=True (R_salpres_*): the scoring composite is first discounted
    by the same salary-cap pressure factor used by compute_heuristic_ownership
    [min(1, (4500/salary)^0.5)].  Aligns scoring prop signal with DFS ownership
    patterns — expensive players' scoring participation is partially discounted to
    reflect their lower DFS ownership share from salary constraints.

    Falls back to unmodified production when both props are absent.
    """
    from src.optimization.ownership import compute_heuristic_ownership

    has_runs = "runs_prop" in pool_df.columns and pool_df["runs_prop"].notna().any()
    has_rbi  = "rbi_prop"  in pool_df.columns and pool_df["rbi_prop"].notna().any()
    if not has_runs and not has_rbi:
        return compute_heuristic_ownership(pool_df, team_totals)

    df = pool_df.copy().reset_index(drop=True)
    batter_mask = df["position"] != "P"

    runs_vals = df["runs_prop"].fillna(0) if has_runs else pd.Series(0.0, index=df.index)
    rbi_vals  = df["rbi_prop"].fillna(0)  if has_rbi  else pd.Series(0.0, index=df.index)
    df["_scoring"] = runs_vals + rbi_vals

    if sal_pressure:
        # Discount scoring composite for expensive batters (mirrors production's cap penalty).
        pressure = np.minimum(1.0, (4500.0 / df["salary"].clip(lower=3000)) ** 0.5)
        df["_scoring"] *= pressure

    scoring_valid = batter_mask & (df["_scoring"] > 0)
    if scoring_valid.any():
        mean_sc = float(df.loc[scoring_valid, "_scoring"].mean())
        if mean_sc > 0:
            if boost_only:
                # Only boost above-average scorers; leave below-average unchanged.
                hot = scoring_valid & (df["_scoring"] > mean_sc)
                if hot.any():
                    ratio = (df.loc[hot, "_scoring"] / mean_sc).clip(upper=3.0)
                    df.loc[hot, "mean"] *= ratio ** scoring_exp
                    df.loc[hot, "salary_value"] = (
                        df.loc[hot, "mean"] / df.loc[hot, "salary"] * 1000
                    )
            else:
                # Two-sided: boost above-average, fade below-average.
                ratio = (df.loc[scoring_valid, "_scoring"] / mean_sc).clip(lower=0.5, upper=3.0)
                df.loc[scoring_valid, "mean"] *= ratio ** scoring_exp
                df.loc[scoring_valid, "salary_value"] = (
                    df.loc[scoring_valid, "mean"] / df.loc[scoring_valid, "salary"] * 1000
                )

    return compute_heuristic_ownership(df.drop(columns=["_scoring"], errors="ignore"), team_totals)


def fit_historical_tier_spend(training_dirs: list[Path]) -> dict[str, float]:
    """
    Compute average per-lineup tier spend from prior contest standings ZIPs.

    For each tier (cheap/mid/expensive), returns the mean amount spent on that
    tier across all lineups in all training slates.  Used by N_fwd_tier to apply
    a forward-looking salary-tier tilt without look-ahead into the current slate.

    Returns empty dict if no training data is available.
    """
    tier_totals: dict[str, list[float]] = {"cheap": [], "mid": [], "expensive": []}
    for d in training_dirs:
        zips = sorted(d.glob("contest-standings-*.zip"))
        if not zips:
            continue
        try:
            standings_df, _ = _parse_contest_zip(zips[0])
            stats = _compute_actual_salary_stats(standings_df, d)
            if stats is None:
                continue
            for t in ("cheap", "mid", "expensive"):
                v = stats.get("tier_mean_spend", {}).get(t)
                if v is not None:
                    tier_totals[t].append(v)
        except Exception:
            continue

    return {
        t: float(np.mean(vals))
        for t, vals in tier_totals.items()
        if vals
    }


def _compute_model_n_forward(
    pool_df: pd.DataFrame,
    team_totals: dict[str, float] | None,
    historical_tier_spend: dict[str, float],
) -> np.ndarray:
    """
    N_fwd_tier — E_production with per-salary-tier tilt using historical averages.

    Same mechanism as N_tier_tilt but uses historical average tier spend from
    prior slates instead of the current slate's actuals.  Forward-looking:
    valid for pre-contest projection.

    Tier definitions match N_tier_tilt:
        cheap    : salary ≤ $3,500
        mid      : $3,501 – $5,000
        expensive: salary > $5,000
    """
    from src.optimization.ownership import compute_heuristic_ownership

    _TIER_CUTOFFS = (3_500, 5_000)

    def _tier(sal: float) -> str:
        return "cheap" if sal <= _TIER_CUTOFFS[0] else ("mid" if sal <= _TIER_CUTOFFS[1] else "expensive")

    base = compute_heuristic_ownership(pool_df, team_totals).copy()
    if not historical_tier_spend:
        return base

    salaries = pool_df["salary"].values.astype(float)
    positions = pool_df["position"].values
    batter_mask = positions != "P"
    tiers = np.array([_tier(s) for s in salaries])
    orig_batter_sum = float(base[batter_mask].sum())

    for tier_name in ("cheap", "mid", "expensive"):
        actual_spend = historical_tier_spend.get(tier_name)
        if actual_spend is None:
            continue
        mask = batter_mask & (tiers == tier_name)
        if not mask.any():
            continue
        implied_spend = float(np.sum(base[mask] * salaries[mask]))
        if implied_spend <= 0:
            continue
        base[mask] *= actual_spend / implied_spend

    new_batter_sum = float(base[batter_mask].sum())
    if new_batter_sum > 0:
        base[batter_mask] *= orig_batter_sum / new_batter_sum

    return base


# ---------------------------------------------------------------------------
# Name matching (ownership_df → pool_df)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    import re
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _load_hr_fair_odds(archive_dir: Path) -> dict[str, float]:
    """Return {normalised_name: hr_over_0.5_implied_prob} from market_odds_fair_odds.json.

    Only returns unflagged 0.5-line 'over' entries.  Returns empty dict if the
    file is absent or malformed.
    """
    odds_path = archive_dir / "market_odds_fair_odds.json"
    if not odds_path.exists():
        return {}
    try:
        import json as _json
        with open(odds_path) as f:
            d = _json.load(f)
        result: dict[str, float] = {}
        for players in d.get("data", {}).values():
            for name, markets in players.items():
                for entry in markets.get("Player Home Runs", []):
                    if entry["line"] == 0.5 and entry["outcome"] == "over" and not entry["flagged"]:
                        result[_normalise(name)] = float(entry["implied_prob"])
        return result
    except Exception:
        return {}


def _load_pitcher_k_props(archive_dir: Path) -> dict[str, float]:
    """Return {normalised_name: expected_k_score} from market_odds_fair_odds.json.

    Expected K score = Σ P(K > n.5) across all non-flagged over lines in
    "Player Pitching Strikeouts".  Uses the summation identity E[K] ≈ Σ P(K ≥ n)
    to aggregate multi-line props into a single comparable signal per pitcher.
    Returns empty dict if the file is absent or malformed.
    """
    odds_path = archive_dir / "market_odds_fair_odds.json"
    if not odds_path.exists():
        return {}
    try:
        import json as _json
        with open(odds_path) as f:
            d = _json.load(f)
        result: dict[str, float] = {}
        for players in d.get("data", {}).values():
            for name, markets in players.items():
                entries = [
                    e for e in markets.get("Player Pitching Strikeouts", [])
                    if e["outcome"] == "over" and not e["flagged"]
                ]
                if entries:
                    result[_normalise(name)] = sum(float(e["implied_prob"]) for e in entries)
        return result
    except Exception:
        return {}


def _load_batter_scoring_props(
    archive_dir: Path,
) -> dict[str, tuple[float | None, float | None]]:
    """Return {normalised_name: (runs_over_0.5_prob, rbi_over_0.5_prob)} from fair_odds.

    Returns empty dict if the file is absent or malformed.  Players with neither
    market are excluded from the result.
    """
    odds_path = archive_dir / "market_odds_fair_odds.json"
    if not odds_path.exists():
        return {}
    try:
        import json as _json
        with open(odds_path) as f:
            d = _json.load(f)
        result: dict[str, tuple[float | None, float | None]] = {}
        for players in d.get("data", {}).values():
            for name, markets in players.items():
                def _prob(market_name: str, _m: dict = markets) -> float | None:
                    for e in _m.get(market_name, []):
                        if e["line"] == 0.5 and e["outcome"] == "over" and not e["flagged"]:
                            return float(e["implied_prob"])
                    return None
                runs_p = _prob("Player Runs")
                rbi_p  = _prob("Player RBIs")
                if runs_p is not None or rbi_p is not None:
                    result[_normalise(name)] = (runs_p, rbi_p)
        return result
    except Exception:
        return {}


def _match_ownership(
    ownership_df: pd.DataFrame,
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join ownership_df to pool_df by player name using exact then fuzzy matching.

    Returns a merged DataFrame with columns from both, dropping unmatched players.
    """
    import difflib

    pool_norm = {_normalise(str(r["name"])): r["player_id"] for _, r in pool_df.iterrows()}
    pool_by_id = pool_df.set_index("player_id")

    matched = []
    for _, row in ownership_df.iterrows():
        raw_name = str(row["player_name"])
        key = _normalise(raw_name)
        pid = pool_norm.get(key)
        if pid is None:
            close = difflib.get_close_matches(key, list(pool_norm.keys()), n=1, cutoff=0.82)
            if close:
                pid = pool_norm[close[0]]
        if pid is not None and pid in pool_by_id.index:
            r = pool_by_id.loc[pid]
            matched.append({
                "player_id": pid,
                "player_name": raw_name,
                "position": r["position"],
                "team": r["team"],
                "opponent": r.get("opponent", ""),
                "salary": r["salary"],
                "mean": r["mean"],
                "lineup_slot": r.get("lineup_slot", np.nan),
                "salary_value": r["salary_value"],
                "implied_total": r.get("implied_total", np.nan),
                "pct_drafted": row["pct_drafted"],
                "actual_fpts": row.get("actual_fpts", np.nan),
            })

    return pd.DataFrame(matched).drop_duplicates("player_id")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    top_pct: float = 0.20,
) -> dict:
    """
    Compute evaluation metrics for a single ownership model.

    Returns dict with: spearman_r, rmse, top_precision, bottom_precision.
    """
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) < 5:
        return {"spearman_r": np.nan, "rmse": np.nan, "top_precision": np.nan, "bottom_precision": np.nan}

    r, _ = spearmanr(a, p)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    n_top = max(1, int(len(a) * top_pct))
    actual_top = set(np.argsort(a)[-n_top:])
    pred_top   = set(np.argsort(p)[-n_top:])
    top_prec   = len(actual_top & pred_top) / n_top

    actual_bot = set(np.argsort(a)[:n_top])
    pred_bot   = set(np.argsort(p)[:n_top])
    bot_prec   = len(actual_bot & pred_bot) / n_top

    return {
        "spearman_r": round(float(r), 4),
        "rmse": round(rmse, 6),
        "top_precision": round(top_prec, 3),
        "bottom_precision": round(bot_prec, 3),
    }


# ---------------------------------------------------------------------------
# Salary-bias diagnostics
# ---------------------------------------------------------------------------

def _compute_actual_salary_stats(
    standings_df: pd.DataFrame,
    archive_dir: Path,
) -> dict | None:
    """
    Parse contest lineup strings into per-lineup total salaries and per-position
    mean salaries.

    Returns None if DKSalaries.csv is missing or no lineups parse successfully.
    Lineups where more than 2 players can't be salary-matched are skipped.

    Return dict keys:
        lineup_salaries  — float array of per-lineup total salaries
        pos_mean_salary  — {pos: mean_salary_per_slot} for P/C/1B/2B/3B/SS/OF.
                           OF is averaged across all 3 OF slots ($/slot, not total).
        pos_n_samples    — {pos: int} number of lineup observations per position.
    """
    sal_path = archive_dir / "DKSalaries.csv"
    if not sal_path.exists():
        return None

    # Local import avoids circular-import risk (analyze_contest_lineups already
    # imports from evaluate_ownership at module level).
    from analyze_contest_lineups import _parse_lineup_string  # noqa: PLC0415

    sal_df = pd.read_csv(sal_path)
    sal_df.rename(
        columns={"Name": "name", "Salary": "salary", "TeamAbbrev": "team"},
        inplace=True,
    )
    salary_map: dict[str, int] = dict(
        zip(sal_df["name"].str.strip(), sal_df["salary"].astype(int))
    )
    team_map: dict[str, str] = (
        dict(zip(sal_df["name"].str.strip(), sal_df["team"].str.strip()))
        if "team" in sal_df.columns else {}
    )

    _BATTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF"}
    _TIER_CUTOFFS = (3_500, 5_000)  # cheap ≤ 3500, mid 3501-5000, expensive > 5000

    def _tier(sal: int) -> str:
        return "cheap" if sal <= _TIER_CUTOFFS[0] else ("mid" if sal <= _TIER_CUTOFFS[1] else "expensive")

    totals: list[float] = []
    pos_salary_lists: dict[str, list[float]] = {
        p: [] for p in ("P", "C", "1B", "2B", "3B", "SS", "OF")
    }
    tier_spend_per_lineup: dict[str, list[float]] = {
        "cheap": [], "mid": [], "expensive": [],
    }
    team_primary_stack_count: dict[str, int] = {}
    n_parsed_for_stack = 0

    for _, row in standings_df.iterrows():
        players = _parse_lineup_string(str(row.get("lineup_str", "")))
        if not players:
            continue
        miss = sum(1 for _, name in players if name not in salary_map)
        if miss > 2:
            continue
        total = sum(salary_map.get(name, 0) for _, name in players)
        if total <= 0:
            continue
        totals.append(float(total))
        n_parsed_for_stack += 1

        lineup_tier_spend: dict[str, float] = {"cheap": 0.0, "mid": 0.0, "expensive": 0.0}
        team_batter_counts: dict[str, int] = {}
        for pos, name in players:
            sal = salary_map.get(name, 0)
            if sal > 0 and pos in pos_salary_lists:
                pos_salary_lists[pos].append(float(sal))
            if sal > 0 and pos in _BATTER_POSITIONS:
                lineup_tier_spend[_tier(sal)] += sal
                team = team_map.get(name, "")
                if team:
                    team_batter_counts[team] = team_batter_counts.get(team, 0) + 1
        for t, spend in lineup_tier_spend.items():
            tier_spend_per_lineup[t].append(spend)

        # Record primary stack team if ≥ 4 batters from one team.
        if team_batter_counts:
            primary_team = max(team_batter_counts, key=team_batter_counts.get)
            if team_batter_counts[primary_team] >= 4:
                team_primary_stack_count[primary_team] = (
                    team_primary_stack_count.get(primary_team, 0) + 1
                )

    if not totals:
        return None

    pos_mean = {
        pos: float(np.mean(sals)) for pos, sals in pos_salary_lists.items() if sals
    }
    pos_n = {pos: len(sals) for pos, sals in pos_salary_lists.items() if sals}
    tier_mean_spend = {
        t: float(np.mean(spends)) for t, spends in tier_spend_per_lineup.items() if spends
    }
    team_stack_rate: dict[str, float] = (
        {team: count / n_parsed_for_stack for team, count in team_primary_stack_count.items()}
        if n_parsed_for_stack > 0 else {}
    )

    return {
        "lineup_salaries": np.array(totals),
        "pos_mean_salary":  pos_mean,
        "pos_n_samples":    pos_n,
        "tier_mean_spend":  tier_mean_spend,
        "team_stack_rate":  team_stack_rate,
    }


def _salary_bias_metrics(
    pool_df: pd.DataFrame,
    predicted_ownership: np.ndarray,
    salary_stats: dict,
) -> dict:
    """
    Compare implied lineup salary (from ownership predictions) to actual contest salary.

    implied_mean = Σ(o_j × s_j) — exact by linearity of expectation; no
    independence assumption needed for the mean.

    implied_std = √(Σ(o_j(1−o_j)s_j²)) — approximation under independence;
    will over-estimate the actual spread because the $50k cap anti-correlates
    expensive players.

    Also computes per-position implied salary ($/slot) vs. actual field mean
    salary ($/slot) stored under key "pos_bias".
    """
    actual_salaries = salary_stats["lineup_salaries"]

    o = predicted_ownership.astype(float)
    s = pool_df["salary"].values.astype(float)

    implied_mean = float(np.sum(o * s))
    implied_std  = float(np.sqrt(np.sum(o * (1.0 - o) * s ** 2)))

    actual_mean = float(actual_salaries.mean())
    actual_std  = float(actual_salaries.std())
    salary_bias = implied_mean - actual_mean
    salary_bias_pct = salary_bias / actual_mean if actual_mean > 0 else float("nan")

    # Per-position implied salary ($/slot) vs actual ($/slot).
    pos_mean_actual: dict[str, float] = salary_stats.get("pos_mean_salary", {})
    pos_bias: dict[str, dict] = {}
    slot_counts = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    positions = pool_df["position"].values
    for pos, n_slots in slot_counts.items():
        mask = positions == pos
        if not mask.any() or pos not in pos_mean_actual:
            continue
        implied_pos_total = float(np.sum(o[mask] * s[mask]))
        implied_per_slot  = implied_pos_total / n_slots
        actual_per_slot   = pos_mean_actual[pos]
        bias              = implied_per_slot - actual_per_slot
        pos_bias[pos] = {
            "implied": round(implied_per_slot),
            "actual":  round(actual_per_slot),
            "bias":    round(bias),
            "bias_pct": round(bias / actual_per_slot, 4) if actual_per_slot else float("nan"),
        }

    return {
        "implied_mean_sal": round(implied_mean),
        "actual_mean_sal":  round(actual_mean),
        "salary_bias":      round(salary_bias),
        "salary_bias_pct":  round(salary_bias_pct, 4),
        "implied_std_sal":  round(implied_std),
        "actual_std_sal":   round(actual_std),
        "pos_bias":         pos_bias,
    }


# ---------------------------------------------------------------------------
# Single-slate evaluation
# ---------------------------------------------------------------------------

def evaluate_slate(
    archive_dir: Path,
    historical_tier_spend: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    """
    Run ownership model evaluation for one archive directory.

    Returns a DataFrame with one row per model (metrics), or None on failure.
    Side effect: writes archive_dir/ownership_eval.csv with per-player data.

    historical_tier_spend: from fit_historical_tier_spend(); when provided,
        N_fwd_tier (forward-looking tier tilt) is computed alongside other models.
    """
    archive_dir = Path(archive_dir)
    slate_label = archive_dir.name

    # Find contest standings zip
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    if not zips:
        print(f"[{slate_label}] No contest-standings-*.zip found — skipping.")
        return None

    print(f"\n[{slate_label}] Loading contest standings from {zips[0].name}")
    try:
        standings_df, ownership_df = _parse_contest_zip(zips[0])
    except Exception as exc:
        print(f"[{slate_label}] Error parsing contest zip: {exc}")
        return None

    print(f"[{slate_label}] {len(ownership_df)} players with ownership data")

    # Build player pool
    try:
        pool_df = _build_player_pool(archive_dir)
    except FileNotFoundError as exc:
        print(f"[{slate_label}] {exc}")
        return None

    print(f"[{slate_label}] {len(pool_df)} projected starters in pool")

    # Match ownership to pool
    merged = _match_ownership(ownership_df, pool_df)
    if len(merged) < 10:
        print(f"[{slate_label}] Only {len(merged)} players matched — skipping (need ≥10).")
        return None

    print(f"[{slate_label}] Matched {len(merged)} players to DK IDs")

    # Pre-compute actual salary stats — needed by model M inside compute_models
    # and by salary-bias diagnostics afterward.
    salary_stats = _compute_actual_salary_stats(standings_df, archive_dir)
    if salary_stats is not None:
        actual_lineup_salaries = salary_stats["lineup_salaries"]
        print(f"[{slate_label}] Salary diagnostics: {len(actual_lineup_salaries):,} lineups parsed "
              f"(mean=${actual_lineup_salaries.mean():,.0f})")
    else:
        actual_lineup_salaries = None

    models = compute_models(pool_df, historical_tier_spend=historical_tier_spend or {})

    # Add predicted ownership columns to merged
    pid_to_model: dict[str, dict[int, float]] = {}
    for model_name, ownership_arr in models.items():
        pid_to_model[model_name] = dict(zip(pool_df["player_id"], ownership_arr))

    actual = merged["pct_drafted"].values
    rows = []
    e_prod_pos_bias: dict | None = None
    e_prod_ownership: np.ndarray | None = None
    for model_name, pid_map in pid_to_model.items():
        predicted = merged["player_id"].map(pid_map).values.astype(float)
        metrics = _evaluate(actual, predicted)
        if salary_stats is not None:
            sal_metrics = _salary_bias_metrics(pool_df, models[model_name], salary_stats)
            pos_bias = sal_metrics.pop("pos_bias", {})
            metrics.update(sal_metrics)
            if model_name == "E_production":
                e_prod_pos_bias = pos_bias
                e_prod_ownership = models[model_name]
        rows.append({"slate": slate_label, "model": model_name, **metrics})

    # Print per-position salary bias table for E_production.
    if e_prod_pos_bias:
        pos_order = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
        print(f"[{slate_label}] Per-position salary bias (E_production vs. actual field, $/slot):")
        print(f"  {'Pos':<4} {'Implied':>8} {'Actual':>8} {'Bias($)':>8} {'Bias%':>7}")
        for pos in pos_order:
            if pos not in e_prod_pos_bias:
                continue
            d = e_prod_pos_bias[pos]
            print(f"  {pos:<4} {d['implied']:>8,} {d['actual']:>8,} "
                  f"  {d['bias']:>+,}  {d['bias_pct']:>+.1%}")

    # Print per-tier batter salary bias table for E_production.
    if salary_stats is not None and e_prod_ownership is not None:
        tier_mean_spend = salary_stats.get("tier_mean_spend", {})
        if tier_mean_spend:
            _TIER_CUTOFFS = (3_500, 5_000)
            def _tier(sal: float) -> str:
                return "cheap" if sal <= _TIER_CUTOFFS[0] else ("mid" if sal <= _TIER_CUTOFFS[1] else "expensive")
            sals = pool_df["salary"].values.astype(float)
            poss = pool_df["position"].values
            tiers_arr = np.array([_tier(s) for s in sals])
            batter_mask = poss != "P"
            print(f"[{slate_label}] Per-tier batter salary bias (E_production vs. actual field):")
            print(f"  {'Tier':<10} {'Implied':>9} {'Actual':>9} {'Bias($)':>8} {'Bias%':>7}")
            for tier_name in ("cheap", "mid", "expensive"):
                actual_spend = tier_mean_spend.get(tier_name)
                if actual_spend is None:
                    continue
                mask = batter_mask & (tiers_arr == tier_name)
                if not mask.any():
                    continue
                implied_spend = float(np.sum(e_prod_ownership[mask] * sals[mask]))
                bias = implied_spend - actual_spend
                bias_pct = bias / actual_spend if actual_spend else float("nan")
                print(f"  {tier_name:<10} {implied_spend:>9,.0f} {actual_spend:>9,.0f} "
                      f"  {bias:>+,.0f}  {bias_pct:>+.1%}")

    # Print per-team stack rate diagnostic vs. E_production implied batter ownership.
    if salary_stats is not None and e_prod_ownership is not None:
        team_stack_rate: dict[str, float] = salary_stats.get("team_stack_rate", {})
        if team_stack_rate:
            sals = pool_df["salary"].values.astype(float)
            poss = pool_df["position"].values
            teams_arr = pool_df["team"].values if "team" in pool_df.columns else None
            top_teams = sorted(team_stack_rate, key=lambda t: team_stack_rate[t], reverse=True)[:8]
            print(f"[{slate_label}] Team stack rates (actual) vs. E_production implied batter ownership:")
            print(f"  {'Team':<6} {'StackRate':>10} {'ImplBatOwn':>11} {'Exp(×4.6)':>10}")
            for tm in top_teams:
                rate = team_stack_rate[tm]
                if teams_arr is not None:
                    bm = (poss != "P") & (teams_arr == tm)
                    impl_bat_own = float(e_prod_ownership[bm].sum()) if bm.any() else 0.0
                else:
                    impl_bat_own = float("nan")
                print(f"  {tm:<6} {rate:>10.3f} {impl_bat_own:>11.3f} {rate * 4.6:>10.3f}")

    results_df = pd.DataFrame(rows)

    # Save per-player evaluation data
    eval_df = merged.copy()
    for model_name, pid_map in pid_to_model.items():
        eval_df[f"pred_{model_name}"] = eval_df["player_id"].map(pid_map)
    eval_df.to_csv(archive_dir / "ownership_eval.csv", index=False)
    print(f"[{slate_label}] Wrote per-player data → {archive_dir / 'ownership_eval.csv'}")

    # Sanity check: top 3 players by E_production projected ownership
    if "E_production" in pid_to_model:
        prod_col = "pred_E_production"
        top3 = eval_df.nlargest(3, prod_col)[["player_name", "position", "team", prod_col, "pct_drafted"]]
        print(f"[{slate_label}] Top 3 by E_production projected ownership:")
        for _, r in top3.iterrows():
            print(f"  {r['player_name']:<22} {r['position']:<3} {r['team']:<4}"
                  f"  pred={r[prod_col]:.3f}  actual={r['pct_drafted']:.3f}")

    return results_df


# ---------------------------------------------------------------------------
# Dry run (projections only, no actuals required)
# ---------------------------------------------------------------------------

def dry_run_slate(archive_dir: Path) -> None:
    """
    Compute and display ownership projections without a contest standings file.

    Prints a per-position table ranked by E_production projected ownership and
    writes archive_dir/ownership_projections.csv with all model predictions.
    """
    archive_dir = Path(archive_dir)
    slate_label = archive_dir.name

    try:
        pool_df = _build_player_pool(archive_dir)
    except FileNotFoundError as exc:
        print(f"[{slate_label}] {exc}")
        return

    print(f"\n[{slate_label}] {len(pool_df)} projected starters in pool")

    models = compute_models(pool_df)
    if not models:
        print(f"[{slate_label}] No models computed.")
        return

    # Build output DataFrame with all model predictions attached
    base_cols = ["player_id", "name", "position", "team", "salary", "mean", "salary_value"]
    optional_cols = ["lineup_slot", "implied_total", "avg_pts"]
    out = pool_df[[c for c in base_cols + optional_cols if c in pool_df.columns]].copy()
    for model_name, arr in models.items():
        out[f"pred_{model_name}"] = arr

    prod_col = "pred_E_production" if "E_production" in models else f"pred_{next(iter(models))}"
    has_slot = "lineup_slot" in out.columns

    # Print per-position table sorted by E_production descending
    W = 70
    print(f"\n{'='*W}")
    print(f"DRY RUN — {slate_label} — projected ownership (sorted by E_production)")
    print(f"{'='*W}")
    print(f"{'Pos':<4} {'Name':<24} {'Team':<5} {'Salary':>7} {'Proj':>5}  {'Slot':>4}  {'E_prod':>7}")
    print("-" * W)

    pos_order = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
    all_positions = pos_order + [p for p in sorted(out["position"].unique()) if p not in pos_order]

    for pos in all_positions:
        grp = out[out["position"] == pos].sort_values(prod_col, ascending=False)
        if grp.empty:
            continue
        for _, row in grp.iterrows():
            slot_str = str(int(row["lineup_slot"])) if has_slot and pd.notna(row.get("lineup_slot")) else "-"
            print(
                f"{row['position']:<4} {str(row['name']):<24} {str(row['team']):<5}"
                f" {row['salary']:>7,.0f} {row['mean']:>5.1f}  {slot_str:>4}"
                f"  {row[prod_col]:>7.3f}"
            )
        print()

    out_path = archive_dir / "ownership_projections.csv"
    out.to_csv(out_path, index=False)
    print(f"[{slate_label}] Projections written → {out_path}")


# ---------------------------------------------------------------------------
# Multi-slate aggregate
# ---------------------------------------------------------------------------

def run_evaluation(archive_dirs: list[Path]) -> None:
    # Discover all full slates in the archive for computing historical tier spend.
    archive_root = PROJECT_ROOT / "archive"
    all_full_slates = sorted(
        d for d in archive_root.iterdir()
        if d.is_dir() and any(d.glob("contest-standings-*.zip"))
    )

    all_results = []
    for d in archive_dirs:
        # Forward-looking: train on all slates strictly older than the current one.
        # Leave-one-out keeps each slate out of its own calibration, which avoids
        # circular influence and gives better out-of-sample Spearman.
        prior_slates = [s for s in all_full_slates if s.name < d.name]
        if prior_slates:
            print(f"[{d.name}] Computing historical tier spend from {len(prior_slates)} prior slate(s)...")
            hist_tier = fit_historical_tier_spend(prior_slates)
            print(f"[{d.name}] Historical tier spend: {hist_tier}")
        else:
            hist_tier = {}
            print(f"[{d.name}] No prior slates for historical tier spend — N_fwd_tier will be skipped.")

        result = evaluate_slate(d, historical_tier_spend=hist_tier)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo slates evaluated successfully.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Per-slate table
    print("\n" + "=" * 70)
    print("PER-SLATE RESULTS")
    print("=" * 70)
    print(combined.to_string(index=False))

    # Aggregate table — include salary_bias_pct when available.
    base_metric_cols = ["spearman_r", "rmse", "top_precision", "bottom_precision"]
    sal_cols = [c for c in ["salary_bias_pct"] if c in combined.columns]
    agg = (
        combined.groupby("model")[base_metric_cols + sal_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    col_names = ["model", "mean_spearman_r", "mean_rmse", "mean_top_prec", "mean_bot_prec"]
    if sal_cols:
        col_names.append("mean_sal_bias_pct")
    agg.columns = col_names

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(all_results)} slate(s))")
    print("=" * 70)
    print(agg.to_string(index=False))

    # Composite rank score: rank each model per metric, sum ranks.
    # RMSE is double-weighted (magnitude accuracy drives opponent field quality).
    # Lower composite = better overall model.
    #
    # Tolerance-binned ranking: values within the tolerance band are treated as
    # ties (same rank) so that noise-level differences don't award rank points.
    #   Spearman  tol=0.005 — differences below this are measurement noise
    #   RMSE      tol=0.001 — e.g. 0.0308 vs 0.0309 are the same bin
    #   precision tol=0.025 — half the step of correctly calling one more player
    def _rank_tol(series: pd.Series, ascending: bool, tol: float) -> pd.Series:
        binned = (series / tol).round(0) * tol
        return binned.rank(ascending=ascending, method="min")

    agg["_r_spearman"] = _rank_tol(agg["mean_spearman_r"], ascending=False, tol=0.005)
    agg["_r_rmse"]     = _rank_tol(agg["mean_rmse"],       ascending=True,  tol=0.001)
    agg["_r_top_prec"] = _rank_tol(agg["mean_top_prec"],   ascending=False, tol=0.025)
    agg["_r_bot_prec"] = _rank_tol(agg["mean_bot_prec"],   ascending=False, tol=0.025)
    agg["composite_rank"] = (
        agg["_r_spearman"] + 2 * agg["_r_rmse"] + agg["_r_top_prec"] + agg["_r_bot_prec"]
    )
    agg = agg.drop(columns=["_r_spearman", "_r_rmse", "_r_top_prec", "_r_bot_prec"])

    print("\n" + "=" * 70)
    print("COMPOSITE RANKING (Spearman + 2×RMSE + top_prec + bot_prec, lower = better)")
    print("  Ties within: Spearman ±0.005, RMSE ±0.001, precision ±0.025")
    print("=" * 70)
    ranked = agg.sort_values("composite_rank")[["model", "mean_spearman_r", "mean_rmse", "mean_top_prec", "mean_bot_prec", "composite_rank"]]
    print(ranked.to_string(index=False))

    best = agg.loc[agg["composite_rank"].idxmin()]
    threshold = 0.60
    verdict = "GO ✓" if best["mean_spearman_r"] >= threshold else "NO-GO ✗"
    print(f"\nBest model: {best['model']}  "
          f"Spearman={best['mean_spearman_r']:.4f}  RMSE={best['mean_rmse']:.4f}  "
          f"composite_rank={best['composite_rank']:.1f}  →  Phase 2: {verdict}")

    # Save aggregate summary
    summary_path = PROJECT_ROOT / "archive" / "ownership_summary.csv"
    combined.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_recent_full_slates(n: int) -> list[Path]:
    """
    Return the N most recent archive subdirectories that contain a
    contest-standings-*.zip file, sorted oldest-first by directory name.
    """
    archive_root = PROJECT_ROOT / "archive"
    full = sorted(
        d for d in archive_root.iterdir()
        if d.is_dir() and any(d.glob("contest-standings-*.zip"))
    )
    return full[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic ownership models against actual DK %Drafted data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs",
        nargs="*",
        metavar="ARCHIVE_DIR",
        help="Archive directories to evaluate (e.g. archive/04072026). "
             "Omit when using --recent.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        metavar="N",
        help="Evaluate the N most recent slates that have a contest-standings zip. "
             "Mutually exclusive with positional ARCHIVE_DIR arguments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print projected ownership without comparing to actuals. "
            "No contest-standings zip required — useful for pre-contest sanity checks."
        ),
    )
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR arguments are mutually exclusive.")

    if args.recent:
        dirs = _find_recent_full_slates(args.recent)
        if not dirs:
            print(f"No full slates (with contest-standings zip) found in {PROJECT_ROOT / 'archive'}.")
            sys.exit(1)
        print(f"--recent {args.recent}: selected {[d.name for d in dirs]}")
    else:
        dirs = []
        for raw in args.archive_dirs:
            p = Path(raw)
            if not p.exists():
                print(f"Warning: {p} does not exist — skipping.")
                continue
            if not p.is_dir():
                print(f"Warning: {p} is not a directory — skipping.")
                continue
            dirs.append(p)
        if not dirs:
            print("No valid archive directories found.")
            sys.exit(1)

    if args.dry_run:
        for d in dirs:
            dry_run_slate(d)
    else:
        run_evaluation(dirs)


if __name__ == "__main__":
    main()
