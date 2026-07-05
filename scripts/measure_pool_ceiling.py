"""
Measure the candidate pool's 99th-percentile ceiling gap against real
DraftKings contest results.

The portfolio selector can only pick lineups the candidate pool contains, so
the pool's tail is a hard upper bound on GPP upside: no amount of diversity
seeking recovers a 99th-percentile finish the pool never generated. This
script quantifies that ceiling per archived slate, three ways:

1. Pool tail rate — what fraction of the dumped candidate pool
   (candidate_pool_debug.csv) would have finished at or above the real
   field's 99th percentile, and how the pool's best lineup compares to the
   real p99 / p99.9 / winning scores. Reuses the real-percentile machinery
   from analyze_candidate_pool.py. A random real entry clears p99 exactly 1%
   of the time by construction, so a pool rate at or below 1% means our
   generator produces ceiling lineups no more often than the field does.

2. Hindsight ceiling capture — the ILP-optimal lineup with each player's
   *actual* FPTS as the objective (generate_optimal_lineups with mean =
   actual score): the best lineup anyone could have entered. Solved twice —
   once under the generator's own construction constraints (min_stack=4 +
   optimizer.salary_floor), once relaxed — so the pool-best/hindsight-best
   "ceiling capture ratio" separates the structural cost of our constraints
   from the sampling gap. Actual FPTS come from contest_player_fpts.json
   when present, else name-matched from the standings zip's FPTS sidebar
   (players absent from both are treated as 0, which can slightly understate
   the relaxed optimum if an entirely undrafted player went off).

3. Composition gap — stack shape, salary usage, real %Drafted ownership and
   bring-back rate for (a) the whole real field, (b) the real field's top-1%
   entries, (c) our full candidate pool, (d) our pool's candidates that
   reached the real p95. Shows *what shapes* the pool is missing, which is
   what decides where a generation fix should go (tail-aware weights vs.
   per-sim optimal seeding vs. refinement parents).

Sections 1 and the pool side of 3 need the post-enrichment pool dump
(candidate_pool_debug.csv + contest_player_fpts.json — only slates run with
gpp.dump_candidate_pool since the dump moved post-sweep). Sections 2 and the
real-field side of 3 only need DKSalaries.csv + the contest-standings zip,
so they cover the whole archive; slates degrade gracefully to whichever
sections their files support.

Usage
-----
    python scripts/measure_pool_ceiling.py                    # all eligible archive slates
    python scripts/measure_pool_ceiling.py archive/06282026
    python scripts/measure_pool_ceiling.py --recent 5
    python scripts/measure_pool_ceiling.py --no-hindsight     # skip the ILP solves

Output
------
  - Per-slate report printed to stdout, plus an aggregate table.
  - archive/MMDDYYYY/pool_ceiling_eval.csv   (tidy section/metric/value rows).
  - archive/pool_ceiling_summary.csv         (one row per slate, appended).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_candidate_pool as acp  # noqa: E402 — real-field loaders, real-percentile machinery
from analyze_contest_lineups import (  # noqa: E402 — lineup-string parsing + salary maps
    _parse_lineup_string, _load_salary_maps, _BATTER_POSITIONS, SALARY_CAP,
)
from evaluate_ownership import _parse_contest_zip, _normalise  # noqa: E402

_DEFAULT_TOP_PERCENTILE = 0.99
_POOL_GROUP_PERCENTILE = 0.95  # pool composition group (d): candidates >= real p95


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_config_salary_floor() -> float | None:
    """optimizer.salary_floor from config.yaml — the floor the candidate
    generator actually ran under, so the constrained hindsight solve matches
    the pool's own construction rules."""
    try:
        import yaml
        with open(PROJECT_ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        floor = cfg.get("optimizer", {}).get("salary_floor")
        return float(floor) if floor is not None else None
    except Exception:
        return None


def _load_actual_fpts(archive_dir: Path, ownership_df: pd.DataFrame, slate_df: pd.DataFrame) -> dict[int, float]:
    """player_id -> actual FPTS. Prefers contest_player_fpts.json; fills any
    player_ids it lacks by normalised-name match against the standings zip's
    FPTS sidebar (which covers every player in the contest pool)."""
    try:
        fpts_map = acp.load_contest_player_fpts(archive_dir)
    except (FileNotFoundError, ValueError):
        fpts_map = {}
    sidebar = {
        _normalise(str(r.player_name)): float(r.actual_fpts)
        for r in ownership_df.itertuples(index=False)
        if r.actual_fpts is not None and not pd.isna(r.actual_fpts)
    }
    for r in slate_df.itertuples(index=False):
        pid = int(r.player_id)
        if pid not in fpts_map:
            v = sidebar.get(_normalise(str(r.name)))
            if v is not None:
                fpts_map[pid] = v
    return fpts_map


# ---------------------------------------------------------------------------
# Section 2: hindsight-optimal ILP
# ---------------------------------------------------------------------------

def solve_hindsight_optimal(
    slate_df: pd.DataFrame, fpts_map: dict[int, float],
    min_stack: int, salary_floor: float | None, n: int = 3,
) -> tuple[float, list[int]]:
    """Best achievable actual score under the given constraints: ILP over the
    full salary-file pool with mean = actual FPTS (missing players -> 0, so
    they are never preferred). Returns (best score, best lineup player_ids)."""
    from src.optimization.optimal_lineups import generate_optimal_lineups

    df = slate_df.copy()
    df["mean"] = df["player_id"].astype(int).map(fpts_map).fillna(0.0)
    lineups = generate_optimal_lineups(
        df, n=n, min_uniques=1, min_stack=min_stack, salary_floor=salary_floor,
    )
    if not lineups:
        return float("nan"), []
    scores = [sum(fpts_map.get(int(p), 0.0) for p in lu.player_ids) for lu in lineups]
    best = int(np.argmax(scores))
    return float(scores[best]), [int(p) for p in lineups[best].player_ids]


# ---------------------------------------------------------------------------
# Section 3: composition profiling
# ---------------------------------------------------------------------------

def _profile_lineups(records: list[dict]) -> dict:
    """Aggregate per-lineup composition records (primary/secondary stack,
    salary, own_sum, own_min, bringback) into one summary row."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    return {
        "n": len(df),
        "prim_stack_mean": float(df["primary"].mean()),
        "prim5_frac": float((df["primary"] >= 5).mean()),
        "sec_stack_mean": float(df["secondary"].mean()),
        "sec2_frac": float((df["secondary"] >= 2).mean()),
        "salary_mean": float(df["salary"].mean()),
        "at_cap_frac": float((df["salary"] >= SALARY_CAP).mean()),
        "own_sum_mean": float(df["own_sum"].mean()),
        "own_min_mean": float(df["own_min"].mean()),
        "bringback_rate": float(df["bringback"].mean()),
    }


def _real_entry_records(
    standings_df: pd.DataFrame, salary_map: dict, team_map: dict, opp_map: dict,
    own_by_name: dict[str, float], ambiguous_names: set[str] | None = None,
) -> list[dict]:
    """One composition record per parseable real contest entry (points kept
    so callers can slice the top 1%). Duplicated entries count as-is — the
    field's shape includes its dupes. Players whose name appears more than
    once in DKSalaries.csv (ambiguous_names) get no team/opponent attribution
    — the name-keyed maps silently keep the last salary row, which fabricates
    cross-game stack/bring-back assignments (verified: two Max Muncys)."""
    ambiguous_names = ambiguous_names or set()
    records = []
    for row in standings_df.itertuples(index=False):
        players = _parse_lineup_string(str(row.lineup_str))
        if not players:
            continue
        if sum(1 for _, name in players if name not in salary_map) > 2:
            continue
        team_counts: dict[str, int] = {}
        pitcher_opps: list[str] = []
        for pos, name in players:
            if pos in _BATTER_POSITIONS:
                t = team_map.get(name, "") if name not in ambiguous_names else ""
                if t:
                    team_counts[t] = team_counts.get(t, 0) + 1
            else:
                pitcher_opps.append(opp_map.get(name, "") if name not in ambiguous_names else "")
        counts = sorted(team_counts.values(), reverse=True)
        owns = [own_by_name.get(_normalise(name), 0.0) for _, name in players]
        records.append({
            "points": float(row.points),
            "primary": counts[0] if counts else 0,
            "secondary": counts[1] if len(counts) > 1 else 0,
            "salary": sum(salary_map.get(name, 0) for _, name in players),
            "own_sum": float(sum(owns)),
            "own_min": float(min(owns)),
            "bringback": bool(any(o and o in team_counts for o in pitcher_opps)),
        })
    return records


def _pool_records(
    pool_df: pd.DataFrame, opp_by_name: dict[str, str], own_by_name: dict[str, float],
) -> pd.DataFrame:
    """One composition record per candidate lineup in the pool dump, indexed
    by lineup_index so callers can slice by real percentile. Ownership uses
    the real %Drafted sidebar (same source as the real-entry records), not
    the model's own field estimate."""
    df = pool_df.copy()
    df["_norm_name"] = df["name"].astype(str).map(_normalise)
    df["_own"] = df["_norm_name"].map(own_by_name).fillna(0.0)
    df["_is_batter"] = df["position"].astype(str) != "P"
    df["_opp"] = df["_norm_name"].map(opp_by_name).fillna("")

    records = []
    for li, g in df.groupby("lineup_index"):
        batters = g[g["_is_batter"]]
        counts = batters.groupby("team").size().sort_values(ascending=False)
        batter_teams = set(counts.index)
        pitcher_opps = g.loc[~g["_is_batter"], "_opp"].tolist()
        records.append({
            "lineup_index": li,
            "primary": int(counts.iloc[0]) if len(counts) else 0,
            "secondary": int(counts.iloc[1]) if len(counts) > 1 else 0,
            "salary": float(g["salary"].sum()),
            "own_sum": float(g["_own"].sum()),
            "own_min": float(g["_own"].min()),
            "bringback": bool(any(o and o in batter_teams for o in pitcher_opps)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Per-slate analysis
# ---------------------------------------------------------------------------

def analyze_slate(
    archive_dir: Path, top_percentile: float, salary_floor: float | None,
    run_hindsight: bool, hindsight_n: int,
) -> dict | None:
    sal_path = archive_dir / "DKSalaries.csv"
    if not sal_path.exists() or not list(archive_dir.glob("contest-standings-*.zip")):
        print(f"Skipping {archive_dir.name}: missing DKSalaries.csv or contest ZIP")
        return None

    field_points = acp.load_real_field_points(archive_dir)
    n_field = len(field_points)
    p99_score = float(np.quantile(field_points, 0.99))
    metrics: dict = {
        "slate": archive_dir.name,
        "n_field": n_field,
        "field_p99_score": p99_score,
        "field_p999_score": float(np.quantile(field_points, 0.999)),
        "winner_score": float(field_points[-1]),
    }

    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    standings_df, ownership_df = _parse_contest_zip(zips[0])
    salary_map, team_map, opp_map = _load_salary_maps(sal_path)
    _sal_names = pd.read_csv(sal_path)["Name"].str.strip()
    ambiguous_names = set(_sal_names.value_counts().loc[lambda s: s > 1].index)
    own_by_name = {
        _normalise(str(r.player_name)): float(r.pct_drafted)
        for r in ownership_df.itertuples(index=False)
    }
    opp_by_name = {_normalise(name): opp for name, opp in opp_map.items()}

    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    slate_df = DraftKingsSlateIngestor(str(sal_path)).get_slate_dataframe()
    fpts_map = _load_actual_fpts(archive_dir, ownership_df, slate_df)

    # --- Section 3a: real field composition (all entries + top 1%) ---
    real_records = _real_entry_records(
        standings_df, salary_map, team_map, opp_map, own_by_name, ambiguous_names,
    )
    top1 = [r for r in real_records if r["points"] >= p99_score]
    profiles = {
        "real_field": _profile_lineups(real_records),
        "real_top1pct": _profile_lineups(top1),
    }

    # --- Sections 1 + 3b: candidate pool (when the dump exists) ---
    lineup_df = None
    try:
        pool_df = acp.load_candidate_pool(archive_dir)
        lineup_df = acp.build_lineup_table(pool_df, fpts_map)
        if "tail_bypass" in pool_df.columns:
            # Carry the Phase 2c tail-bypass marker (constant per lineup)
            # through so bypass candidates can be evaluated separately.
            _tb = pool_df.groupby("lineup_index")["tail_bypass"].first()
            lineup_df = lineup_df.merge(
                _tb.rename("tail_bypass"), on="lineup_index", how="left",
            )
        lineup_df = acp.add_real_percentile(
            lineup_df, field_points, acp.default_cash_threshold(), top_percentile,
        )
        complete = lineup_df.dropna(subset=["real_percentile"])
        metrics.update({
            "n_pool": len(lineup_df),
            "pool_coverage": len(complete) / len(lineup_df) if len(lineup_df) else float("nan"),
            "pool_p99_rate": float(complete["would_top_pct"].mean()) if len(complete) else float("nan"),
            "n_pool_p99": int(complete["would_top_pct"].sum()) if len(complete) else 0,
            "pool_best_score": float(complete["actual_score"].max()) if len(complete) else float("nan"),
            "pool_best_pct": float(complete["real_percentile"].max()) if len(complete) else float("nan"),
        })

        pool_rec = _pool_records(pool_df, opp_by_name, own_by_name)
        pool_rec = pool_rec.merge(
            complete[["lineup_index", "real_percentile"]], on="lineup_index", how="left",
        )
        profiles["pool_all"] = _profile_lineups(pool_rec.to_dict("records"))
        p95_rec = pool_rec[pool_rec["real_percentile"] >= _POOL_GROUP_PERCENTILE]
        profiles["pool_ge_p95"] = _profile_lineups(p95_rec.to_dict("records"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {archive_dir.name}: pool sections skipped ({exc})")

    # --- Section 2: hindsight-optimal ILP ---
    if run_hindsight and fpts_map:
        for label, ms, floor in (
            ("constrained", 4, salary_floor),
            ("relaxed", 0, None),
        ):
            try:
                score, pids = solve_hindsight_optimal(
                    slate_df, fpts_map, min_stack=ms, salary_floor=floor, n=hindsight_n,
                )
            except Exception as exc:
                print(f"  {archive_dir.name}: hindsight ({label}) failed — {exc}")
                score, pids = float("nan"), []
            metrics[f"hindsight_{label}"] = score
            if label == "constrained":
                metrics["_hindsight_constrained_pids"] = pids
        pb = metrics.get("pool_best_score")
        for label in ("constrained", "relaxed"):
            h = metrics.get(f"hindsight_{label}")
            metrics[f"capture_{label}"] = (
                pb / h if pb is not None and h and not np.isnan(h) and h > 0 else float("nan")
            )

    metrics["_profiles"] = profiles
    metrics["_slate_df"] = slate_df

    if lineup_df is not None:
        lineup_df.to_csv(archive_dir / "pool_ceiling_lineups.csv", index=False)
    _write_eval_csv(archive_dir, metrics, profiles)
    return metrics


def _write_eval_csv(archive_dir: Path, metrics: dict, profiles: dict) -> None:
    rows = [
        {"section": "ceiling", "metric": k, "value": v}
        for k, v in metrics.items()
        if not k.startswith("_") and k != "slate"
    ]
    for group, prof in profiles.items():
        rows.extend(
            {"section": f"composition/{group}", "metric": k, "value": v}
            for k, v in prof.items()
        )
    pd.DataFrame(rows).to_csv(archive_dir / "pool_ceiling_eval.csv", index=False)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_PROFILE_COLS = [
    ("n", "n", ",.0f"),
    ("prim_stack_mean", "prim", ".2f"),
    ("prim5_frac", "prim5%", ".1%"),
    ("sec_stack_mean", "sec", ".2f"),
    ("sec2_frac", "sec2%", ".1%"),
    ("salary_mean", "salary", ",.0f"),
    ("at_cap_frac", "@cap%", ".1%"),
    ("own_sum_mean", "Σown", ".2f"),
    ("own_min_mean", "min-own", ".3f"),
    ("bringback_rate", "brngbk", ".1%"),
]


def _print_profile_table(profiles: dict) -> None:
    header = f"{'group':<14}" + "".join(f"{lbl:>9}" for _, lbl, _ in _PROFILE_COLS)
    print(header)
    print("-" * len(header))
    for group, prof in profiles.items():
        if not prof:
            print(f"{group:<14}{'(empty)':>9}")
            continue
        cells = "".join(f"{prof[k]:>9{fmt}}" for k, _, fmt in _PROFILE_COLS)
        print(f"{group:<14}{cells}")


def print_slate_report(m: dict) -> None:
    print(f"\n=== {m['slate']} ===  field={m['n_field']:,} entries")
    print(
        f"  Real field:  p99={m['field_p99_score']:.2f}  "
        f"p99.9={m['field_p999_score']:.2f}  winner={m['winner_score']:.2f}"
    )
    if "n_pool" in m:
        print(
            f"  Pool ({m['n_pool']:,} candidates):  "
            f"≥p99 rate={m['pool_p99_rate'] * 100:.2f}% ({m['n_pool_p99']})  "
            f"best score={m['pool_best_score']:.2f} (pct={m['pool_best_pct']:.4f})"
        )
    if "hindsight_constrained" in m:
        print(
            f"  Hindsight optimal:  constrained={m['hindsight_constrained']:.2f}  "
            f"relaxed={m['hindsight_relaxed']:.2f}"
        )
        if not np.isnan(m.get("capture_constrained", float("nan"))):
            print(
                f"  Ceiling capture (pool best / hindsight):  "
                f"constrained={m['capture_constrained']:.3f}  "
                f"relaxed={m['capture_relaxed']:.3f}"
            )
        pids = m.get("_hindsight_constrained_pids") or []
        slate_df = m.get("_slate_df")
        if pids and slate_df is not None:
            sub = slate_df[slate_df["player_id"].astype(int).isin(pids)]
            names = ", ".join(f"{r.name} ({r.team})" for r in sub.itertuples(index=False))
            print(f"  Hindsight lineup (constrained): {names}")
    print()
    _print_profile_table(m["_profiles"])


def print_aggregate(rows: list[dict]) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    print(f"\n=== Aggregate across {len(df)} slates ===")
    for col, label in (
        ("pool_p99_rate", "pool ≥p99 rate (1% = field-random)"),
        ("pool_best_pct", "pool best real percentile"),
        ("capture_constrained", "ceiling capture (constrained)"),
        ("capture_relaxed", "ceiling capture (relaxed)"),
    ):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                print(f"  {label}: mean={vals.mean():.4f}  median={vals.median():.4f}  n_slates={len(vals)}")

    # Pooled composition: mean of per-slate profiles, per group.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        for g, prof in r["_profiles"].items():
            if prof:
                groups.setdefault(g, []).append(prof)
    pooled = {
        g: {k: float(np.mean([p[k] for p in profs])) for k, _, _ in _PROFILE_COLS}
        for g, profs in groups.items()
    }
    print("\nComposition (mean of per-slate profiles):")
    _print_profile_table(pooled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_slates(n: int | None) -> list[Path]:
    archive_root = PROJECT_ROOT / "archive"
    full = sorted(
        (
            d for d in archive_root.iterdir()
            if d.is_dir()
            and (d / "DKSalaries.csv").exists()
            and list(d.glob("contest-standings-*.zip"))
        ),
        key=lambda d: acp._slate_sort_key(d.name),
    )
    return full[-n:] if n else full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the candidate pool's ceiling against real DK contest results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
        help="Archive directories to evaluate (default: every slate with DKSalaries.csv "
             "and a contest-standings zip).",
    )
    parser.add_argument(
        "--recent", type=int, default=0, metavar="N",
        help="Evaluate only the N most recent eligible slates.",
    )
    parser.add_argument(
        "--top-percentile", type=float, default=_DEFAULT_TOP_PERCENTILE, metavar="FRACTION",
        help=f"Real-field percentile defining the ceiling (default: {_DEFAULT_TOP_PERCENTILE}).",
    )
    parser.add_argument(
        "--no-hindsight", action="store_true",
        help="Skip the hindsight-optimal ILP solves (fast pool/composition-only run).",
    )
    parser.add_argument(
        "--hindsight-n", type=int, default=3, metavar="N",
        help="Near-optimal lineups to enumerate per hindsight solve (default: 3; "
             "only the best is reported).",
    )
    args = parser.parse_args()

    if args.archive_dirs and args.recent:
        parser.error("positional ARCHIVE_DIR args and --recent are mutually exclusive")
    dirs = [Path(d) for d in args.archive_dirs] if args.archive_dirs else _find_slates(args.recent or None)
    if not dirs:
        print("No eligible archive slates found.")
        sys.exit(1)

    salary_floor = _load_config_salary_floor()
    print(
        f"Measuring pool ceiling on {len(dirs)} slate(s)  "
        f"(top_percentile={args.top_percentile}, constrained salary_floor={salary_floor})"
    )

    rows = []
    for d in dirs:
        try:
            m = analyze_slate(
                d, args.top_percentile, salary_floor,
                run_hindsight=not args.no_hindsight, hindsight_n=args.hindsight_n,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue
        if m:
            rows.append(m)
            print_slate_report(m)

    if not rows:
        print("No slates produced results.")
        sys.exit(1)

    if len(rows) > 1:
        print_aggregate(rows)
    summary_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
    ]
    summary_path = PROJECT_ROOT / "archive" / "pool_ceiling_summary.csv"
    acp._append_summary(summary_rows, summary_path)
    print(f"\nSummary appended -> {summary_path}")


if __name__ == "__main__":
    main()
