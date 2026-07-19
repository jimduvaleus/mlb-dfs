"""Seed-block forensics (ceiling-first redesign, Phase 1a).

Why did per-sim exact ILP seeds (sim_optimal) stop enriching real-p99 after
the 2026-07-06 copula overlay + mean calibration? For each seed_source block
in a replayed (or live) candidate-pool dump, report:

  - redundancy: pairwise player-overlap distribution within the block
    (H1: per-world argmax optima collapse onto near-duplicates)
  - concentration: primary-stack-team entropy + top-2-team share
    (H1b: the overlay's shared team factor makes one hot team dominate)
  - composition vs the real top-1%: prim5/sec2/salary/at-cap/Σown/bring-back
    via measure_pool_ceiling's profiler (H3: argmax optima are structurally
    unlike real winners — too extreme, too unowned)
  - outcome: hit99 / hit99.9 per block against the real field

Usage
-----
    python scripts/seed_block_forensics.py outputs/replay/06282026/simopt1000 archive/06282026
    python scripts/seed_block_forensics.py --variant simopt1000 --slates 06282026 07032026
        (expands to outputs/replay/<slate>/<variant> + archive/<slate>)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_candidate_pool as acp  # noqa: E402
import measure_pool_ceiling as mpc  # noqa: E402

_OVERLAP_SAMPLE = 1500  # lineups sampled per block for the pairwise histogram


def _block_overlap_stats(lineup_sets: list[frozenset], rng: np.random.Generator) -> dict:
    n = len(lineup_sets)
    if n < 2:
        return {"overlap_mean": float("nan"), "overlap_ge7_frac": float("nan"), "n_unique": n}
    uniq = len(set(lineup_sets))
    idx = rng.choice(n, size=min(n, _OVERLAP_SAMPLE), replace=False)
    pids = sorted({p for i in idx for p in lineup_sets[i]})
    col = {p: j for j, p in enumerate(pids)}
    onehot = np.zeros((len(idx), len(pids)), dtype=np.int8)
    for r, i in enumerate(idx):
        for p in lineup_sets[i]:
            onehot[r, col[p]] = 1
    ov = onehot.astype(np.int32) @ onehot.T  # (s, s) shared-player counts
    iu = np.triu_indices(len(idx), k=1)
    pair_ov = ov[iu]
    return {
        "n_unique": uniq,
        "overlap_mean": float(pair_ov.mean()),
        "overlap_ge7_frac": float((pair_ov >= 7).mean()),
    }


def _primary_team_stats(block_df: pd.DataFrame) -> dict:
    """Entropy / concentration of the primary-stack team across a block."""
    primaries = []
    for _, g in block_df.groupby("lineup_index"):
        batters = g[g["position"].astype(str) != "P"]
        counts = batters.groupby("team").size()
        if len(counts):
            primaries.append(counts.idxmax())
    if not primaries:
        return {"team_entropy": float("nan"), "top2_team_share": float("nan"), "n_teams": 0}
    vc = pd.Series(primaries).value_counts(normalize=True)
    entropy = float(-(vc * np.log(vc)).sum())
    return {
        "team_entropy": entropy,
        "top2_team_share": float(vc.iloc[:2].sum()),
        "n_teams": int(len(vc)),
    }


def analyze_run(run_dir: Path, archive_dir: Path) -> pd.DataFrame:
    pool_df = pd.read_csv(run_dir / "candidate_pool_debug.csv", low_memory=False)
    field_points = acp.load_real_field_points(archive_dir)
    p99 = float(np.quantile(field_points, 0.99))
    p999 = float(np.quantile(field_points, 0.999))

    # Real %Drafted ownership + opponent maps (same sources the composition
    # profiler uses for real entries, so pool vs real profiles are comparable).
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    standings_df, ownership_df = mpc._parse_contest_zip(zips[0])
    sal_path = archive_dir / "DKSalaries.csv"
    salary_map, team_map, opp_map = mpc._load_salary_maps(sal_path)
    _names = pd.read_csv(sal_path)["Name"].str.strip()
    ambiguous = set(_names.value_counts().loc[lambda s: s > 1].index)
    own_by_name = {
        mpc._normalise(str(r.player_name)): float(r.pct_drafted)
        for r in ownership_df.itertuples(index=False)
    }
    opp_by_name = {mpc._normalise(name): opp for name, opp in opp_map.items()}

    real_records = mpc._real_entry_records(
        standings_df, salary_map, team_map, opp_map, own_by_name, ambiguous,
    )
    top1 = [r for r in real_records if r["points"] >= p99]

    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    slate_df = DraftKingsSlateIngestor(str(sal_path)).get_slate_dataframe()
    fpts_map = mpc._load_actual_fpts(archive_dir, ownership_df, slate_df)
    lineup_df = acp.build_lineup_table(pool_df, fpts_map)

    rng = np.random.default_rng(0)
    rows = []
    ref = mpc._profile_lineups(top1)
    rows.append({"block": "REAL_TOP1%", "n": ref.get("n", 0), **{
        k: ref.get(k, float("nan"))
        for k in ("prim5_frac", "sec2_frac", "salary_mean", "at_cap_frac",
                  "own_sum_mean", "own_min_mean", "bringback_rate")
    }})

    for source, block in pool_df.groupby("seed_source"):
        block_lineups = lineup_df[lineup_df["seed_source"] == source]
        scored = block_lineups.dropna(subset=["actual_score"])
        pool_recs = mpc._pool_records(block, opp_by_name, own_by_name)
        prof = mpc._profile_lineups(pool_recs.to_dict("records"))
        sets = [
            frozenset(g["player_id"].astype(int))
            for _, g in block.groupby("lineup_index")
        ]
        rows.append({
            "block": source,
            "n": len(sets),
            **{k: prof.get(k, float("nan"))
               for k in ("prim5_frac", "sec2_frac", "salary_mean", "at_cap_frac",
                         "own_sum_mean", "own_min_mean", "bringback_rate")},
            **_block_overlap_stats(sets, rng),
            **_primary_team_stats(block),
            "hit99_pct": 100 * float((scored["actual_score"] >= p99).mean()) if len(scored) else float("nan"),
            "hit999_pct": 100 * float((scored["actual_score"] >= p999).mean()) if len(scored) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", metavar="RUN_DIR ARCHIVE_DIR",
                        help="explicit pair: run dir + archive dir")
    parser.add_argument("--variant", help="variant name under outputs/replay/<slate>/")
    parser.add_argument("--slates", nargs="+", default=[], metavar="MMDDYYYY")
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = []
    if args.variant and args.slates:
        for s in args.slates:
            pairs.append((
                PROJECT_ROOT / "outputs" / "replay" / s / args.variant,
                PROJECT_ROOT / "archive" / s,
            ))
    elif len(args.paths) == 2:
        pairs.append((Path(args.paths[0]), Path(args.paths[1])))
    else:
        parser.error("give RUN_DIR ARCHIVE_DIR, or --variant NAME --slates S1 S2 ...")

    all_tables = []
    for run_dir, archive_dir in pairs:
        if not (run_dir / "candidate_pool_debug.csv").exists():
            print(f"{run_dir}: no dump — skipping")
            continue
        t = analyze_run(run_dir, archive_dir)
        t.insert(0, "slate", archive_dir.name)
        all_tables.append(t)
        print(f"\n=== {archive_dir.name} / {run_dir.name} ===")
        print(t.drop(columns=["slate"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if len(all_tables) > 1:
        agg = pd.concat(all_tables, ignore_index=True)
        num_cols = [c for c in agg.columns if c not in ("slate", "block", "n", "n_teams", "n_unique")]
        pooled = agg.groupby("block")[num_cols].mean()
        print("\n=== Pooled means across slates ===")
        print(pooled.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
