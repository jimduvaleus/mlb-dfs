"""
Round 1 (v3): does augmenting the external (SaberSim) candidate pool with a
TIME-BOXED batch of internally-generated ceiling lineups help, hurt, or do
nothing -- on a settled slate. Two generation methods:

  ilp         per-simulated-world exact ILP optimum
              (generate_sim_optimal_lineups). Tested first; the diagnostic
              (scripts/diagnose_ilp_supplement_pwin.py) found these
              lineups' own p_win_select systematically LOWER than the
              external pool's at every percentile across all 7 test
              slates, and literally 0 of them ever landed in the combined
              pool's top 1% (106.2 expected under random placement). An
              exact per-world argmax over-concentrates on whatever players
              happened to score huge in that one world -- optimal for that
              world, not for the distribution p_win integrates over.

  sim_winner  CandidateGenerator.generate_sim_winners: samples through the
              normal stack-construction machinery with per-world
              rank-softmax weights instead of solving each world's exact
              argmax -- explicitly built as "the scaled, sampling-based
              successor to per-sim exact ILP seeding" for this reason. This
              round tests whether that softer objective actually produces
              lineups with better p_win/selection properties than the ILP
              did, or whether the same failure mode reappears.

Why time-boxed rather than a target count
------------------------------------------
An earlier version of this script sized the internal pool to match the
external pool's raw pre-dedup count and measured how long that took (the
07/26 pilot: ~28 minutes for ~12,000 unique lineups). That's backwards for
how this would actually be used: the real constraint is wall-clock time
between late-breaking slate news and lock (realistically ~10 minutes, most
of which the rest of the pipeline needs too), not a target lineup count --
and different slates' player pools generate at different speeds, so a
fixed N doesn't map to a fixed time cost anyway. This version generates as
many unique lineups as fit in a fixed time budget (default 3 minutes) and
reports how many that turned out to be, rather than the reverse.

Mechanism: both `generate_sim_optimal_lineups` and `generate_sim_winners`
check a `stop_check` callable from inside their sim loop (per-solve for the
ILP; every 200 worlds for sim_winner, which is cheap per-world so the
coarser check doesn't meaningfully overshoot). Submitting a generous
oversupply of sim indices up front and passing a time-based stop_check
lets already in-flight work finish while not-yet-started work returns
instantly once the budget is spent -- no batch-size tuning needed.

The comparison itself
----------------------
This tests AUGMENTATION, not a competing rival pool: does external +
(whatever the generator found in the time budget) score better, the same,
or worse than external alone? Four numbers per pool, pre- and
post-pipeline: mean real score, max real score, and p99 hit rate (pool)
--> mean/max/hit99 of the SELECTED portfolio (post-pipeline, same p_win
selector run on both). "Post" uses a disjoint later sim slice from
"pre"/generation, so selection never reuses the exact sims the supplement
was built against.

Resumability: the unit of resumability is the slate. A slate's generated
supplement is cached to outputs/pool_compare/<slate>/<method>/ once the
time budget is spent; re-running reuses it unless --time-budget-minutes or
--seed changed. In-flight progress within a still-running slate is not
preserved.

Usage
-----
    python scripts/compare_candidate_pools.py --slate 07262026 --method ilp
    python scripts/compare_candidate_pools.py --slate 07262026 --method sim_winner
    python scripts/compare_candidate_pools.py --slate 07192026 --slate 07202026 \\
        --slate 07212026 --slate 07222026 --slate 07242026 --slate 07252026 --slate 07262026 \\
        --method sim_winner
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
from src.simulation.results import SimulationResults  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.candidate_generator import CandidateGenerator  # noqa: E402
from src.optimization.optimal_lineups import (  # noqa: E402
    generate_sim_optimal_lineups, stratified_sim_sample,
)
from src.optimization.lineup import Lineup  # noqa: E402
from analyze_candidate_pool import load_contest_player_fpts, load_real_field_points  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "pool_compare"
_SIM_WINNER_TEMP = 0.15       # matches gpp.sim_winner_temp production default
_SIM_WINNER_OWN_BLEND = 0.25  # matches gpp.sim_winner_own_blend production default

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
    for col in ("eligible_positions", "opponent", "game", "salary", "team", "position"):
        if col not in players_df.columns:
            raise RuntimeError(f"players_df missing required ILP column {col!r}")
    return players_df, ep.build_quantile_grids(proj_ext), name_to_id


def salary_floor_from_pool(pool, players_df: pd.DataFrame) -> float:
    """Salary floor inferred from the external pool's own observed minimum
    lineup salary -- no universal number assumed."""
    sal_by_id = dict(zip(players_df["player_id"], players_df["salary"]))
    pool_salaries = np.array([
        sum(sal_by_id.get(int(p), 0) for p in lu.player_ids) for lu in pool.lineups
    ])
    return float(pool_salaries.min())


def generate_ilp_supplement_timeboxed(
    players_df: pd.DataFrame, sim_results, salary_floor: float,
    time_budget_s: float, seed: int,
) -> tuple[list[Lineup], dict]:
    """Per-sim-optimal ILP lineups (generate_sim_optimal_lineups), spending
    up to time_budget_s wall-clock seconds rather than targeting a lineup
    count. Submits every available sim index up front (oversupply -- the
    07/26 pilot ran at ~100% unique yield per sim, ~0.14s/lineup, so
    n_sims_available should comfortably outlast any reasonable time
    budget) and lets the time-based stop_check cut generation off from
    inside the thread pool once the budget is spent.
    """
    n_sims_available = sim_results.results_matrix.shape[0]
    rng = np.random.default_rng(seed)
    sampled = stratified_sim_sample(sim_results.results_matrix, n_sims_available, rng)
    sim_indices = [s for s, _ in sampled]

    t0 = time.perf_counter()

    def _stop_check() -> bool:
        return (time.perf_counter() - t0) >= time_budget_s

    def _progress(n_done: int) -> None:
        if n_done % 250 == 0:
            print(f"    {n_done:,} sim-solves resolved, "
                  f"{time.perf_counter() - t0:.0f}s / {time_budget_s:.0f}s budget")

    lineups = generate_sim_optimal_lineups(
        players_df, sim_results.results_matrix, sim_results.player_ids,
        sim_indices, min_stack=4, salary_floor=salary_floor,
        progress_cb=_progress, stop_check=_stop_check,
    )
    elapsed = time.perf_counter() - t0
    return lineups, {
        "method": "ilp", "time_budget_s": time_budget_s, "elapsed_s": elapsed,
        "achieved_n": len(lineups), "sims_available": n_sims_available,
        "salary_floor": salary_floor,
    }


def _primary_stack_size(lineup: Lineup, players_df: pd.DataFrame) -> int:
    team_of = dict(zip(players_df["player_id"], players_df["team"]))
    pos_of = dict(zip(players_df["player_id"], players_df["position"]))
    teams = [team_of.get(int(p)) for p in lineup.player_ids if pos_of.get(int(p)) != "P"]
    if not teams:
        return 0
    return int(pd.Series(teams).value_counts().iloc[0])


def generate_sim_winner_supplement_timeboxed(
    players_df: pd.DataFrame, sim_results, salary_floor: float,
    time_budget_s: float, seed: int,
) -> tuple[list[Lineup], dict]:
    """CandidateGenerator.generate_sim_winners, time-boxed the same way as
    the ILP path: submit every available sim index up front, let a
    time-based stop_check cut sampling off (checked every 200 worlds inside
    generate_sim_winners -- cheap per-world sampling, so the coarser check
    doesn't meaningfully overshoot the budget).

    Unlike the ILP path, generate_sim_winners samples across a 5/4/3-batter
    stack-size MIX (CandidateGenerator.GROUP_FRACTIONS, ~62/31/7%) rather
    than enforcing a hard minimum -- so a post-filter drops anything below
    a 4-man primary stack to match the same min_stack=4 rule the ILP path
    enforces natively, keeping the three generation rules identical across
    both methods being compared.
    """
    n_sims_available = sim_results.results_matrix.shape[0]
    rng = np.random.default_rng(seed)
    sampled = stratified_sim_sample(sim_results.results_matrix, n_sims_available, rng)
    sim_indices = [s for s, _ in sampled]

    own_vec = players_df["ownership"].astype(float).to_numpy()
    gen = CandidateGenerator(players_df, own_vec, rng_seed=seed, salary_floor=salary_floor)

    t0 = time.perf_counter()

    def _stop_check() -> bool:
        return (time.perf_counter() - t0) >= time_budget_s

    def _progress(n_done: int) -> None:
        print(f"    {n_done:,} lineups sampled, "
              f"{time.perf_counter() - t0:.0f}s / {time_budget_s:.0f}s budget")

    raw = gen.generate_sim_winners(
        sim_results.results_matrix, sim_results.player_ids, sim_indices,
        per_world=1, temp=_SIM_WINNER_TEMP, own_blend=_SIM_WINNER_OWN_BLEND,
        progress_cb=_progress, stop_check=_stop_check,
    )
    lineups = [lu for lu in raw if _primary_stack_size(lu, players_df) >= 4]
    elapsed = time.perf_counter() - t0
    return lineups, {
        "method": "sim_winner", "time_budget_s": time_budget_s, "elapsed_s": elapsed,
        "achieved_n": len(lineups), "raw_n_before_stack_filter": len(raw),
        "sims_available": n_sims_available, "salary_floor": salary_floor,
    }


def _slate_cache_dir(slate: str, method: str) -> Path:
    d = OUT_ROOT / slate / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_generate_supplement(
    slate: str, method: str, players_df: pd.DataFrame, sim_results, salary_floor: float,
    time_budget_s: float, seed: int,
) -> tuple[list[Lineup], dict]:
    d = _slate_cache_dir(slate, method)
    manifest_path = d / "manifest.json"
    pool_path = d / "supplement.json"
    if manifest_path.exists() and pool_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("time_budget_s") == time_budget_s
                and manifest.get("salary_floor") == salary_floor):
            lineups = [Lineup(player_ids=ids) for ids in json.loads(pool_path.read_text())]
            print(f"  [{slate}/{method}] cached supplement: {len(lineups):,} lineups "
                  f"(reusing -- delete {d} to force regeneration)")
            return lineups, manifest
        print(f"  [{slate}/{method}] cached params changed (time budget/floor) -- regenerating")

    generator_fn = (generate_ilp_supplement_timeboxed if method == "ilp"
                    else generate_sim_winner_supplement_timeboxed)
    lineups, manifest = generator_fn(players_df, sim_results, salary_floor, time_budget_s, seed)
    pool_path.write_text(json.dumps([list(map(int, lu.player_ids)) for lu in lineups]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  [{slate}/{method}] supplement complete: {manifest['achieved_n']:,} unique lineups "
          f"in {manifest['elapsed_s']:.0f}s (budget {manifest['time_budget_s']:.0f}s) "
          f"-- cached to {d}")
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


def augment_pool(base_lineups: list, supplement: list[Lineup]) -> list:
    """base_lineups + supplement, dropping any supplement lineup that
    exactly duplicates one already in base_lineups (cheap safety net --
    the ILP draws from a >5,000-candidate space, so an accidental exact
    match is unlikely, but silently double-counting one would bias the
    pool-size-dependent metrics)."""
    seen = {frozenset(int(p) for p in lu.player_ids) for lu in base_lineups}
    added = []
    for lu in supplement:
        key = frozenset(int(p) for p in lu.player_ids)
        if key not in seen:
            seen.add(key)
            added.append(lu)
    return list(base_lineups) + added


def select_portfolio(
    lineups: list, players_df: pd.DataFrame, sim_results,
    portfolio_size: int, sharpness: float, admit_n: int, admit_multiplier: float,
) -> list:
    """Run `lineups` through the same production p_win selection path
    (allocate_contests) used for external pools, so 'after our pipeline'
    means the identical selector regardless of which pool fed it."""
    pool = ep.ExternalPool(
        lineups=[Lineup(player_ids=list(lu.player_ids) if hasattr(lu, "player_ids") else list(lu))
                for lu in lineups],
        contests={}, n_dropped_unknown_players=0, n_dropped_duplicates=0,
        n_dropped_near_duplicates=0, source_paths=[],
    )
    corr = ep.compute_pool_corr(pool.lineups, sim_results)

    group = ep.ContestGroup(
        contest_id="c0", contest_name="pool-compare", entry_fee_cents=400,
        prize_pool_cents=int(10_000 * 400), single_entry_tag=False, roi_key="",
        entries=[(Path("x"), None)] * portfolio_size,
    )
    n_half = sim_results.results_matrix.shape[0] // 2
    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
    scores_A, scores_B = lineup_scores[:, :n_half], lineup_scores[:, n_half:2 * n_half]
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    cs = ContestSimulator()
    own_vec = players_df["ownership"].astype(float).to_numpy()
    field_A = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=100),
                             sim_results.results_matrix[:n_half], col_map)
    field_B = cs.score_field(cs.generate_field(players_df, own_vec, 10_000, rng_seed=101),
                             sim_results.results_matrix[n_half:2 * n_half], col_map)
    exponent = max(1.0, sharpness * 10_000.0)
    p_win_cull = ep.compute_p_win(scores_A, field_A, {"c0": exponent})
    p_win_select = ep.compute_p_win(scores_B, field_B, {"c0": exponent})
    alloc = ep.allocate_contests(
        pool, corr, [group], risk=3.0, evw_base=0.10, evw_max=0.40,
        ev_type="p_win", p_win_cull=p_win_cull, p_win_select=p_win_select,
        p_win_admit_n=admit_n, p_win_admit_multiplier=admit_multiplier,
    )
    return [lu for lu, _ in alloc.portfolio]


def run_slate(
    slate: str, method: str, portfolio_size: int, time_budget_s: float, n_sims: int, seed: int,
    sharpness: float, admit_n: int, admit_multiplier: float,
) -> dict:
    d = PROJECT_ROOT / "archive" / slate
    print(f"\n=== {slate} ({method}) ===")
    players_df, grids, name_to_id = build_players_df(d)

    found = ep.discover_external_files(str(d))
    valid_ids = set(players_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids)
    salary_floor = salary_floor_from_pool(pool, players_df)
    print(f"  external pool: {len(pool.lineups):,} lineups  salary_floor={salary_floor:.0f}")

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    # One simulation draw funds generation AND (a disjoint later slice)
    # "after our pipeline" p_win selection -- generation and selection never
    # share sims, so the supplement isn't graded on the exact worlds it was
    # built against.
    sim_results = engine.simulate(n_sims + n_sims)
    gen_sims = SimulationResults(sim_results.player_ids, sim_results.results_matrix[:n_sims])
    select_sims = SimulationResults(sim_results.player_ids, sim_results.results_matrix[n_sims:])

    supplement, manifest = load_or_generate_supplement(
        slate, method, players_df, gen_sims, salary_floor, time_budget_s, seed,
    )
    augmented = augment_pool(pool.lineups, supplement)
    n_added = len(augmented) - len(pool.lineups)
    print(f"  augmented pool: {len(pool.lineups):,} + {n_added:,} new "
          f"(of {manifest['achieved_n']:,} generated) = {len(augmented):,}")

    fpts = load_contest_player_fpts(d)
    field_pts = load_real_field_points(d)

    ext_before = pool_metrics(pool.lineups, fpts, field_pts)
    aug_before = pool_metrics(augmented, fpts, field_pts)

    ext_portfolio = select_portfolio(pool.lineups, players_df, select_sims,
                                     portfolio_size, sharpness, admit_n, admit_multiplier)
    aug_portfolio = select_portfolio(augmented, players_df, select_sims,
                                     portfolio_size, sharpness, admit_n, admit_multiplier)
    ext_after = pool_metrics(ext_portfolio, fpts, field_pts)
    aug_after = pool_metrics(aug_portfolio, fpts, field_pts)

    print(f"\n  {'':26s} {'n':>6s} {'mean':>8s} {'max':>8s} {'p99_hit':>9s}")
    for label, m in (("external, pre-pipeline", ext_before), ("augmented, pre-pipeline", aug_before),
                     ("external, post-pipeline", ext_after), ("augmented, post-pipeline", aug_after)):
        print(f"  {label:26s} {m['n']:6d} {m['mean']:8.2f} {m['max']:8.2f} {m['p99_hit_rate']:9.4f}")

    return {"slate": slate, "manifest": manifest, "n_added": n_added,
            "ext_before": ext_before, "aug_before": aug_before,
            "ext_after": ext_after, "aug_after": aug_after}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Does augmenting the external pool with a time-boxed batch of "
                    "internally ILP-generated ceiling lineups help, hurt, or do nothing?",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--slate", action="append", default=[], help="Archive dir name; repeatable.")
    p.add_argument("--recent", type=int, default=0)
    p.add_argument("--method", choices=["ilp", "sim_winner"], default="ilp",
                   help="Generation method for the supplement (default: ilp).")
    p.add_argument("--portfolio-size", type=int, default=150)
    p.add_argument("--time-budget-minutes", type=float, default=3.0,
                   help="Wall-clock budget for supplement generation per slate (default: 3.0). "
                        "The actual number of lineups this buys varies by slate -- that's "
                        "the point; report what it turns out to be rather than assume it.")
    p.add_argument("--n-sims", type=int, default=10_000,
                   help="Sims per stage (generation and, separately, selection) -- "
                        "simulation itself is cheap, this isn't the constrained resource.")
    p.add_argument("--sharpness", type=float, default=0.05)
    p.add_argument("--admit-n", type=int, default=250)
    p.add_argument("--admit-multiplier", type=float, default=12.0)
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

    time_budget_s = args.time_budget_minutes * 60.0
    results = [run_slate(s, args.method, args.portfolio_size, time_budget_s, args.n_sims, args.seed,
                         args.sharpness, args.admit_n, args.admit_multiplier) for s in slates]

    print("\n\n=== Summary across all slates ===")
    rows = []
    for r in results:
        for source in ("ext", "aug"):
            for stage in ("before", "after"):
                m = r[f"{source}_{stage}"]
                rows.append({"slate": r["slate"], "source": source, "stage": stage,
                            "n_added": r["n_added"], **m})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Mean across slates, external vs. augmented ===")
    print(df.groupby(["source", "stage"])[["mean", "max", "p99_hit_rate"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
