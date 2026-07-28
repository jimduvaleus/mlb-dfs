"""
Round 1: external (SaberSim) candidate pool vs. an internally-generated
candidate pool of the same theoretical construction, size-matched, on a
settled slate.

Motivation
----------
The external pool is advertised as drawn from a larger pool of lineups each
of which won at least one simulated world, subject to three generation
rules: (1) a minimum stack size, (2) no pitcher rostered against an opposing
hitter, (3) a salary floor. We already have an ILP that produces exactly
that kind of lineup (`generate_sim_optimal_lineups`, one exact roster-ILP
solve per simulated world) and already enforces all three rules natively:

  rule                          | already-existing enforcement
  ------------------------------|--------------------------------------
  1. min_stack batters, 1 team  | min_stack param (default 4)
  2. no pitcher-vs-opp-hitter   | constraint C5 in generate_optimal_lineups,
                                 | always on, not configurable off
  3. salary floor               | salary_floor param

So round 1 needs no new ILP-constraint work — just: (a) size the internal
pool to match the external pool's *raw pre-dedup* candidate count (so the
comparison isn't skewed by SaberSim's own dedup step), (b) pick a salary
floor (this script infers it per-slate from the external pool's own
observed minimum lineup salary — no universal number is assumed), and
(c) run enough simulated worlds to have a shot at that many *unique*
per-sim-optimal lineups.

Known open risk, not resolved by design: it is not known in advance
whether a given sim budget can produce `target_n` UNIQUE per-sim-optimal
lineups — many worlds can converge on the same optimal roster. This script
grows the sim sample in batches and stops either at target_n unique
lineups or at --max-sims, whichever comes first, and always reports which
one it hit rather than silently padding to the target.

Resumability
------------
The unit of resumability is the SLATE, not the batch: a slate's internal
pool is only ever written to disk (outputs/pool_compare/<slate>/) once
finished (target reached or --max-sims exhausted), with a manifest
recording the outcome. Re-running the script skips any slate whose
manifest already exists. In-flight batch progress for a slate that gets
interrupted mid-run is NOT preserved -- that slate simply restarts from
sim 0 next time, per instruction (only completed slates need to survive
a pause).

Usage
-----
    python scripts/compare_candidate_pools.py --slate 07262026
    python scripts/compare_candidate_pools.py --recent 8
    python scripts/compare_candidate_pools.py --slate 07262026 --max-sims 150000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api import external_pool as ep  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.optimal_lineups import (  # noqa: E402
    generate_sim_optimal_lineups, stratified_sim_sample,
)
from src.optimization.lineup import Lineup  # noqa: E402
from analyze_candidate_pool import load_contest_player_fpts, load_real_field_points  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "pool_compare"

_POSITION_MAP = {"SP": "P", "RP": "P"}


def _parse_eligible_positions(raw: str) -> list[str]:
    """DK "1B/3B" -> ["1B", "3B"], de-duplicated, order preserved. Mirrors
    src/ingestion/dk_slate.py's _parse_positions -- generate_optimal_lineups
    needs a real list here, not the raw slash-joined string; without it a
    multi-eligible player silently collapses to single-position-only."""
    tokens = str(raw).strip().split("/")
    mapped = [_POSITION_MAP.get(t, t) for t in tokens]
    seen: set = set()
    out: list[str] = []
    for t in mapped:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _derive_opponent(team: str, game: str) -> str:
    m = str(game).split(" ")[0]
    away, _, home = m.partition("@")
    return home if team == away else away


def build_slate_df(archive_dir: Path) -> pd.DataFrame:
    """DK slate frame with a real eligible_positions column (list[str] per
    row) -- the one thing sim_evaluate_portfolios.py's build_slate() skips,
    since it never needed multi-position ILP eligibility until now."""
    sal = pd.read_csv(archive_dir / "DKSalaries.csv")
    df = pd.DataFrame({
        "player_id": sal["ID"].astype(int),
        "name": sal["Name"].astype(str).str.strip(),
        "team": sal["TeamAbbrev"].astype(str),
        "game": sal["Game Info"].astype(str),
        "salary": sal["Salary"].astype(int),
    })
    df["eligible_positions"] = sal["Position"].astype(str).apply(_parse_eligible_positions)
    df["position"] = df["eligible_positions"].str[0]
    return df


def build_players_df(archive_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """(players_df, quantile_grids, name->player_id), players_df carrying a
    real eligible_positions column so the ILP's multi-position handling
    works, unlike sim_evaluate_portfolios.py's build_slate()."""
    slate_df = build_slate_df(archive_dir)
    found = ep.discover_external_files(str(archive_dir))
    if not found["projections_path"]:
        raise FileNotFoundError(f"no SaberSim projections CSV in {archive_dir}")
    proj_ext = ep.parse_player_projections(found["projections_path"])
    name_to_id = dict(zip(slate_df["name"], slate_df["player_id"]))
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, set(slate_df["player_id"]), _derive_opponent,
    )
    # build_external_players_df keeps only pool players + confirmed
    # batters + all pitchers -- eligible_positions/opponent/game survive
    # the merge since they're carried on slate_df, but re-verify here since
    # a silently-missing column is exactly the kind of bug this file exists
    # to avoid repeating.
    for col in ("eligible_positions", "opponent", "game", "salary", "team", "position"):
        if col not in players_df.columns:
            raise RuntimeError(f"players_df missing required ILP column {col!r}")
    return players_df, ep.build_quantile_grids(proj_ext), name_to_id


def load_pool_and_target(archive_dir: Path, players_df: pd.DataFrame) -> tuple:
    """(pool, target_n, salary_floor). target_n is the external pool's raw
    row count BEFORE our own parse_lineup_pool dropped exact/near
    duplicates -- the size SaberSim's own generator actually produced,
    which is the fair count to match rather than the post-dedup one.
    salary_floor is inferred from the pool's own observed minimum lineup
    salary (no universal number assumed)."""
    found = ep.discover_external_files(str(archive_dir))
    valid_ids = set(players_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids)
    target_n = len(pool.lineups) + pool.n_dropped_duplicates + pool.n_dropped_near_duplicates

    sal_by_id = dict(zip(players_df["player_id"], players_df["salary"]))
    pool_salaries = np.array([
        sum(sal_by_id.get(int(p), 0) for p in lu.player_ids) for lu in pool.lineups
    ])
    salary_floor = float(pool_salaries.min())
    return pool, target_n, salary_floor


def generate_internal_pool(
    players_df: pd.DataFrame, sim_results, target_n: int, salary_floor: float,
    max_sims: int, batch_size: int, seed: int,
) -> tuple[list[Lineup], dict]:
    """Grow the per-sim-optimal pool in batches of stratified sim draws
    until target_n unique lineups are found or max_sims sim-solves have
    been spent, whichever comes first. Returns (lineups, stats) where
    stats always records which stopping condition actually fired --
    reaching target_n is not guaranteed and must never be assumed."""
    rng = np.random.default_rng(seed)
    unique: list[Lineup] = []
    seen: set = set()
    total_sims_used = 0
    t0 = time.perf_counter()

    while len(unique) < target_n and total_sims_used < max_sims:
        this_batch = min(batch_size, max_sims - total_sims_used)
        sampled = stratified_sim_sample(sim_results.results_matrix, this_batch, rng)
        new = generate_sim_optimal_lineups(
            players_df, sim_results.results_matrix, sim_results.player_ids,
            [s for s, _ in sampled], min_stack=4, salary_floor=salary_floor,
            seen=seen,
        )
        unique.extend(new)
        total_sims_used += this_batch
        print(f"    sims used {total_sims_used:,}/{max_sims:,}  "
              f"unique lineups {len(unique):,}/{target_n:,}  "
              f"({time.perf_counter() - t0:.0f}s elapsed)")

    hit_target = len(unique) >= target_n
    return unique[:target_n], {
        "target_n": target_n, "achieved_n": len(unique), "hit_target": hit_target,
        "sims_used": total_sims_used, "max_sims": max_sims,
        "salary_floor": salary_floor, "elapsed_s": time.perf_counter() - t0,
    }


def _slate_cache_dir(slate: str) -> Path:
    d = OUT_ROOT / slate
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_generate_internal_pool(
    slate: str, players_df: pd.DataFrame, sim_results, target_n: int,
    salary_floor: float, max_sims: int, batch_size: int, seed: int,
) -> tuple[list[Lineup], dict]:
    d = _slate_cache_dir(slate)
    manifest_path = d / "manifest.json"
    pool_path = d / "internal_pool.json"
    if manifest_path.exists() and pool_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("target_n") == target_n and manifest.get("salary_floor") == salary_floor:
            lineups = [Lineup(player_ids=ids) for ids in json.loads(pool_path.read_text())]
            print(f"  [{slate}] cached internal pool: {len(lineups):,} lineups "
                  f"(hit_target={manifest.get('hit_target')}, reusing -- delete "
                  f"{d} to force regeneration)")
            return lineups, manifest
        print(f"  [{slate}] cached pool params changed (target/floor) -- regenerating")

    lineups, manifest = generate_internal_pool(
        players_df, sim_results, target_n, salary_floor, max_sims, batch_size, seed,
    )
    pool_path.write_text(json.dumps([list(map(int, lu.player_ids)) for lu in lineups]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  [{slate}] internal pool complete: {manifest['achieved_n']:,}/{manifest['target_n']:,} "
          f"unique lineups (hit_target={manifest['hit_target']}), "
          f"{manifest['sims_used']:,} sims, {manifest['elapsed_s']:.0f}s -- cached to {d}")
    return lineups, manifest


def _real_score(ids: list[int], fpts: dict) -> float | None:
    vals = [fpts.get(int(p)) for p in ids]
    return sum(vals) if all(v is not None for v in vals) else None


def pool_metrics(lineups: list, fpts: dict, field_pts: np.ndarray) -> dict:
    scores = np.array([s for lu in lineups if (s := _real_score(
        lu.player_ids if hasattr(lu, "player_ids") else lu, fpts)) is not None])
    if len(scores) == 0:
        return {"n": 0, "n_scored": 0, "mean": float("nan"), "max": float("nan"), "p99_hit_rate": float("nan")}
    n_field = len(field_pts)
    pct = np.searchsorted(field_pts, scores, side="right") / n_field
    return {
        "n": len(lineups), "n_scored": len(scores),
        "mean": float(scores.mean()), "max": float(scores.max()),
        "p99_hit_rate": float((pct >= 0.99).mean()),
    }


def select_portfolio(
    lineups: list, own_by_id: dict, proj_by_id: dict, players_df: pd.DataFrame,
    sim_results, portfolio_size: int, ev_type: str, sharpness: float, admit_n: int,
) -> list:
    """Run `lineups` through the same production p_win selection path
    (allocate_contests) used for external pools, so 'after our pipeline'
    means the identical selector for both pool sources."""
    from src.optimization.gpp_portfolio import DeterminantPortfolioSelector  # noqa: F401

    pool = ep.ExternalPool(
        lineups=[Lineup(player_ids=list(lu.player_ids) if hasattr(lu, "player_ids") else list(lu))
                for lu in lineups],
        contests={}, n_dropped_unknown_players=0, n_dropped_duplicates=0,
        n_dropped_near_duplicates=0, source_paths=[],
    )
    corr = ep.compute_pool_corr(pool.lineups, sim_results)
    proj_scores = ep.compute_pool_proj_scores(pool.lineups, players_df)
    own_scores = ep.compute_pool_ownership(pool.lineups, players_df)

    group = ep.ContestGroup(
        contest_id="c0", contest_name="pool-compare", entry_fee_cents=400,
        prize_pool_cents=int(10_000 * 400), single_entry_tag=False, roi_key="",
        entries=[(Path("x"), None)] * portfolio_size,
    )
    if ev_type == "p_win":
        n_third = sim_results.results_matrix.shape[0] // 2
        lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
        scores_A, scores_B = lineup_scores[:, :n_third], lineup_scores[:, n_third:2 * n_third]
        cs = ContestSimulator()
        own_vec = players_df["ownership"].astype(float).to_numpy()
        field_A = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=100),
                                 sim_results.results_matrix[:n_third], {int(p): i for i, p in enumerate(sim_results.player_ids)})
        field_B = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=101),
                                 sim_results.results_matrix[n_third:2 * n_third], {int(p): i for i, p in enumerate(sim_results.player_ids)})
        exponent = max(1.0, sharpness * 10_000.0)
        p_win_cull = ep.compute_p_win(scores_A, field_A, {"c0": exponent})
        p_win_select = ep.compute_p_win(scores_B, field_B, {"c0": exponent})
        alloc = ep.allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.10, evw_max=0.40,
            ev_type="p_win", p_win_cull=p_win_cull, p_win_select=p_win_select,
            p_win_admit_n=admit_n,
        )
    else:
        alloc = ep.allocate_contests(
            pool, corr, [group], risk=3.0, evw_base=0.10, evw_max=0.40,
            ev_type="prj_own", proj_scores=proj_scores, own_scores=own_scores,
        )
    return [lu for lu, _ in alloc.portfolio]


def run_slate(
    slate: str, portfolio_size: int, max_sims: int, batch_size: int, seed: int,
    sharpness: float, admit_n: int,
) -> dict:
    d = PROJECT_ROOT / "archive" / slate
    print(f"\n=== {slate} ===")
    players_df, grids, name_to_id = build_players_df(d)
    pool, target_n, salary_floor = load_pool_and_target(d, players_df)
    print(f"  external pool: {len(pool.lineups):,} post-dedup "
          f"(+{pool.n_dropped_duplicates} dup +{pool.n_dropped_near_duplicates} near-dup "
          f"= {target_n:,} raw target)  salary_floor={salary_floor:.0f}")

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    n_sims_for_selection = 25_000
    sim_results = engine.simulate(max(max_sims, n_sims_for_selection) + n_sims_for_selection)
    # First slice funds internal-pool generation; a disjoint later slice
    # funds both pools' "after our pipeline" p_win selection, so selection
    # never reuses the exact sims the internal pool was built from.
    from src.simulation.results import SimulationResults
    gen_sims = SimulationResults(sim_results.player_ids, sim_results.results_matrix[:max_sims])
    select_sims = SimulationResults(sim_results.player_ids,
                                    sim_results.results_matrix[max_sims:max_sims + n_sims_for_selection])

    internal_lineups, manifest = load_or_generate_internal_pool(
        slate, players_df, gen_sims, target_n, salary_floor, max_sims, batch_size, seed,
    )

    fpts = load_contest_player_fpts(d)
    field_pts = load_real_field_points(d)

    ext_before = pool_metrics(pool.lineups, fpts, field_pts)
    int_before = pool_metrics(internal_lineups, fpts, field_pts)

    own_by_id = dict(zip(players_df["player_id"], players_df["ownership"]))
    proj_by_id = dict(zip(players_df["player_id"], players_df["mean"]))
    ext_portfolio = select_portfolio(pool.lineups, own_by_id, proj_by_id, players_df,
                                     select_sims, portfolio_size, "p_win", sharpness, admit_n)
    int_portfolio = select_portfolio(internal_lineups, own_by_id, proj_by_id, players_df,
                                     select_sims, portfolio_size, "p_win", sharpness, admit_n)
    ext_after = pool_metrics(ext_portfolio, fpts, field_pts)
    int_after = pool_metrics(int_portfolio, fpts, field_pts)

    print(f"\n  {'':22s} {'n':>6s} {'mean':>8s} {'max':>8s} {'p99_hit':>9s}")
    for label, m in (("external, before", ext_before), ("internal, before", int_before),
                     ("external, after", ext_after), ("internal, after", int_after)):
        print(f"  {label:22s} {m['n']:6d} {m['mean']:8.2f} {m['max']:8.2f} {m['p99_hit_rate']:9.4f}")

    return {"slate": slate, "manifest": manifest, "ext_before": ext_before,
            "int_before": int_before, "ext_after": ext_after, "int_after": int_after}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Round 1: external SaberSim pool vs. an internally ILP-generated "
                    "pool of the same theoretical construction, size-matched.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--slate", action="append", default=[], help="Archive dir name; repeatable.")
    p.add_argument("--recent", type=int, default=0)
    p.add_argument("--portfolio-size", type=int, default=150)
    p.add_argument("--max-sims", type=int, default=100_000,
                   help="Sim-solve budget cap for the internal pool (default: 100,000). "
                        "Reaching target_n unique lineups is not guaranteed within this "
                        "budget -- the manifest records which stopping condition fired.")
    p.add_argument("--batch-size", type=int, default=5_000,
                   help="Sim indices sampled per batch while growing toward target_n.")
    p.add_argument("--sharpness", type=float, default=0.05)
    p.add_argument("--admit-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.recent:
        candidates = []
        for dd in sorted((PROJECT_ROOT / "archive").iterdir()):
            if not dd.is_dir():
                continue
            found = ep.discover_external_files(str(dd))
            if (found["lineups_paths"] and (dd / "contest_player_fpts.json").exists()
                    and list(dd.glob("contest-standings-*.zip"))):
                candidates.append(dd.name)
        slates = candidates[-args.recent:]
    else:
        slates = args.slate
    if not slates:
        print("No slates given (use --slate or --recent N).")
        sys.exit(1)

    results = [run_slate(s, args.portfolio_size, args.max_sims, args.batch_size,
                         args.seed, args.sharpness, args.admit_n) for s in slates]

    print("\n\n=== Summary across all slates ===")
    rows = []
    for r in results:
        for source in ("ext", "int"):
            for stage in ("before", "after"):
                m = r[f"{source}_{stage}"]
                rows.append({"slate": r["slate"], "source": source, "stage": stage, **m})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
