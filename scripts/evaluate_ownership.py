"""
Evaluate heuristic ownership models against actual DraftKings %Drafted data.

Run this on 5+ archived slates to decide whether a heuristic ownership model
is accurate enough (Spearman ≥ 0.60) to justify adding a leverage_surplus
objective to the live optimizer (Phase 2).

Usage
-----
    python scripts/evaluate_ownership.py archive/04072026
    python scripts/evaluate_ownership.py archive/04072026 archive/04082026 ...
    python scripts/evaluate_ownership.py archive/*/   # all archived slates

Output
------
  - Per-slate and aggregate evaluation table printed to stdout.
  - archive/MMDDYYYY/ownership_eval.csv  (per-player data for each slate).
  - archive/ownership_summary.csv        (one row per slate, aggregate metrics).

Models evaluated
----------------
  A. Uniform — equal ownership within each position group (baseline).
  B. Salary-value softmax — softmax(salary_value) within position group.
  C. Mean + salary-value softmax — softmax(w_mean*mean + w_sv*sv) within position.
  D. Model C + team-total boost — batters on high-implied teams get a multiplier.
     Only runs when dff_team_totals.csv is present in the archive directory.
  E. compute_heuristic_ownership() — full model with sqrt compression, batting
     order multiplier, pitcher matchup boost, and multi-position eligibility.
     Requires dff_team_totals.csv and lineup_slot in dff_projections.csv.
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
    Load DKSalaries.csv and dff_projections.csv from archive_dir, join on
    player_id, and compute features needed for ownership models.

    Returns a DataFrame with columns:
        player_id, name, salary, position, eligible_positions, team, opponent,
        game, mean, std_dev, lineup_slot, salary_value, implied_total (if available)
    """
    salary_path = archive_dir / "DKSalaries.csv"
    dff_path    = archive_dir / "dff_projections.csv"

    if not salary_path.exists():
        raise FileNotFoundError(f"DKSalaries.csv not found in {archive_dir}")
    if not dff_path.exists():
        raise FileNotFoundError(f"dff_projections.csv not found in {archive_dir}")

    sal_df = pd.read_csv(salary_path)
    sal_df.rename(
        columns={"ID": "player_id", "Name": "name", "Salary": "salary",
                 "TeamAbbrev": "team", "Game Info": "game", "Position": "raw_position"},
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
    sal_df = sal_df[
        ["player_id", "name", "salary", "position", "eligible_positions", "team", "opponent", "game"]
    ].drop_duplicates("player_id")

    proj_df = pd.read_csv(dff_path)
    proj_df["player_id"] = proj_df["player_id"].astype(int)
    proj_cols = ["player_id", "mean", "std_dev"]
    if "lineup_slot" in proj_df.columns:
        proj_cols.append("lineup_slot")

    merged = sal_df.merge(proj_df[proj_cols], on="player_id", how="left")
    merged = merged.dropna(subset=["mean"])  # only projected starters

    merged["salary_value"] = merged["mean"] / merged["salary"] * 1000

    # Optionally attach team implied totals
    totals_path = archive_dir / "dff_team_totals.csv"
    if totals_path.exists():
        totals_df = pd.read_csv(totals_path)
        totals_df.columns = [c.lower() for c in totals_df.columns]
        merged = merged.merge(
            totals_df.rename(columns={"implied_total": "implied_total"}),
            on="team", how="left",
        )
    else:
        merged["implied_total"] = np.nan

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ownership models
# ---------------------------------------------------------------------------

def _softmax(vals: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature scaling."""
    v = np.asarray(vals, dtype=float)
    if v.std() < 1e-9:
        return np.ones(len(v)) / len(v)
    scaled = (v - v.mean()) / (v.std() * temperature)
    exp_v = np.exp(scaled)
    return exp_v / exp_v.sum()


def _apply_per_position(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Apply softmax within each position group using score_col, return ownership array."""
    ownership = np.zeros(len(df))
    for pos, grp in df.groupby("position"):
        idx = grp.index.to_numpy()
        scores = df.loc[idx, score_col].values
        ownership[idx] = _softmax(scores)
    return ownership


def compute_models(pool_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Return a dict of {model_name: ownership_array} aligned to pool_df index.

    Ownership arrays are probabilities within each position group (sum to 1 per pos).
    """
    models: dict[str, np.ndarray] = {}

    # Model A: uniform within position
    uniform = np.zeros(len(pool_df))
    for pos, grp in pool_df.groupby("position"):
        idx = grp.index.to_numpy()
        uniform[idx] = 1.0 / len(idx)
    models["A_uniform"] = uniform

    # Model B: softmax(salary_value) within position
    models["B_salary_value"] = _apply_per_position(pool_df, "salary_value")

    # Model C: softmax(0.6*mean + 0.4*salary_value) within position
    pool_df = pool_df.copy()
    pool_df["_score_C"] = 0.6 * pool_df["mean"] + 0.4 * pool_df["salary_value"]
    models["C_mean_sv"] = _apply_per_position(pool_df, "_score_C")

    # Model D: Model C + team-total boost for batters (only when implied_total available)
    if pool_df["implied_total"].notna().any():
        pool_df = pool_df.copy()
        is_batter = pool_df["position"] != "P"
        # Scale implied total to [0, 2] range as a multiplier
        total_vals = pool_df.loc[is_batter, "implied_total"].fillna(
            pool_df["implied_total"].median()
        )
        min_t, max_t = total_vals.min(), total_vals.max()
        if max_t > min_t:
            scaled = 1.0 + (total_vals - min_t) / (max_t - min_t)
        else:
            scaled = pd.Series(1.0, index=total_vals.index)
        pool_df.loc[is_batter, "_score_D"] = pool_df.loc[is_batter, "_score_C"] * scaled
        pool_df.loc[~is_batter, "_score_D"] = pool_df.loc[~is_batter, "_score_C"]
        models["D_team_total"] = _apply_per_position(pool_df, "_score_D")

    # Model E: compute_heuristic_ownership — sqrt compression, batting order
    # multiplier, pitcher matchup boost, multi-position eligibility.
    try:
        from src.optimization.ownership import compute_heuristic_ownership
        team_totals: dict[str, float] | None = None
        if pool_df["implied_total"].notna().any():
            team_totals = (
                pool_df[["team", "implied_total"]]
                .dropna(subset=["implied_total"])
                .drop_duplicates("team")
                .set_index("team")["implied_total"]
                .to_dict()
            )
        models["E_full"] = compute_heuristic_ownership(pool_df, team_totals)
    except Exception as exc:
        print(f"  [Model E skipped: {exc}]")

    return models


# ---------------------------------------------------------------------------
# Name matching (ownership_df → pool_df)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    import re
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


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
# Single-slate evaluation
# ---------------------------------------------------------------------------

def evaluate_slate(archive_dir: Path) -> pd.DataFrame | None:
    """
    Run ownership model evaluation for one archive directory.

    Returns a DataFrame with one row per model (metrics), or None on failure.
    Side effect: writes archive_dir/ownership_eval.csv with per-player data.
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
        _, ownership_df = _parse_contest_zip(zips[0])
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

    # Compute model predictions
    models = compute_models(pool_df)

    # Add predicted ownership columns to merged
    pid_to_model: dict[str, dict[int, float]] = {}
    for model_name, ownership_arr in models.items():
        pid_to_model[model_name] = dict(zip(pool_df["player_id"], ownership_arr))

    actual = merged["pct_drafted"].values
    rows = []
    for model_name, pid_map in pid_to_model.items():
        predicted = merged["player_id"].map(pid_map).values.astype(float)
        metrics = _evaluate(actual, predicted)
        rows.append({"slate": slate_label, "model": model_name, **metrics})

    results_df = pd.DataFrame(rows)

    # Save per-player evaluation data
    eval_df = merged.copy()
    for model_name, pid_map in pid_to_model.items():
        eval_df[f"pred_{model_name}"] = eval_df["player_id"].map(pid_map)
    eval_df.to_csv(archive_dir / "ownership_eval.csv", index=False)
    print(f"[{slate_label}] Wrote per-player data → {archive_dir / 'ownership_eval.csv'}")

    return results_df


# ---------------------------------------------------------------------------
# Multi-slate aggregate
# ---------------------------------------------------------------------------

def run_evaluation(archive_dirs: list[Path]) -> None:
    all_results = []
    for d in archive_dirs:
        result = evaluate_slate(d)
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

    # Aggregate table
    agg = (
        combined.groupby("model")[["spearman_r", "rmse", "top_precision", "bottom_precision"]]
        .mean()
        .round(4)
        .reset_index()
    )
    agg.columns = ["model", "mean_spearman_r", "mean_rmse", "mean_top_prec", "mean_bot_prec"]

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(all_results)} slate(s))")
    print("=" * 70)
    print(agg.to_string(index=False))

    go_nogo = agg.loc[agg["mean_spearman_r"].idxmax()]
    threshold = 0.60
    verdict = "GO ✓" if go_nogo["mean_spearman_r"] >= threshold else "NO-GO ✗"
    print(f"\nBest model: {go_nogo['model']}  "
          f"Spearman={go_nogo['mean_spearman_r']:.4f}  "
          f"Threshold={threshold:.2f}  →  Phase 2: {verdict}")

    # Save aggregate summary
    summary_path = PROJECT_ROOT / "archive" / "ownership_summary.csv"
    combined.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic ownership models against actual DK %Drafted data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs",
        nargs="+",
        metavar="ARCHIVE_DIR",
        help="One or more archive directories (e.g. archive/04072026)",
    )
    args = parser.parse_args()

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

    run_evaluation(dirs)


if __name__ == "__main__":
    main()
