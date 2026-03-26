#!/usr/bin/env python3
"""
Benchmark: run individual (n_sims, percentile) scenarios with per-scenario timeout.

Each scenario times:
  1. Simulation         — SimulationEngine.simulate()
  2. Single-lineup opt  — BasinHoppingOptimizer.optimize()
  3. Portfolio (N=20)   — PortfolioConstructor.construct()

Scenarios are run one at a time.  A configurable wall-clock timeout (default
10 min) kills any step that overruns and records partial results.

Usage:
    source venv/bin/activate

    # Run all scenarios
    python scripts/benchmark_timing.py

    # Run a single scenario by index (0-based)
    python scripts/benchmark_timing.py --scenario 3

    # List scenarios without running them
    python scripts/benchmark_timing.py --list

    # Override the per-step timeout (seconds)
    python scripts/benchmark_timing.py --timeout 300
"""
import argparse
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.dk_slate import DraftKingsSlateIngestor
from src.models.batter_model import BatterPCAModel
from src.models.copula import EmpiricalCopula
from src.optimization.lineup import ROSTER_REQUIREMENTS
from src.optimization.optimizer import BasinHoppingOptimizer
from src.optimization.portfolio import PortfolioConstructor
from src.simulation.engine import SimulationEngine
from src.simulation.results import SimulationResults

# ── Load config.yaml ────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

_paths = _CFG.get("paths", {})
_opt   = _CFG.get("optimizer", {})
_port  = _CFG.get("portfolio", {})
_sim   = _CFG.get("simulation", {})

# ── Configuration ───────────────────────────────────────────────────────────

# Scenarios: every (n_sims, target_percentile) combination to benchmark.
SCENARIOS = [
    (n_sims, pct)
    for n_sims in [5_000, 10_000, 25_000, 50_000]
    for pct in [80, 85, 90, 95, 99]
]

# Optimizer settings — sourced from config.yaml
OPT_CHAINS_DEFAULT = _opt.get("n_chains", 250)
OPT_STEPS_DEFAULT  = _opt.get("n_steps", 100)
OPT_TEMP           = _opt.get("temperature", 0.1)
OPT_WORKERS        = _opt.get("n_workers", 1)
OPT_NITER_SUCCESS  = _opt.get("niter_success", 25)
OPT_ES_WINDOW      = _opt.get("early_stopping_window", 25)
OPT_ES_THRESHOLD   = _opt.get("early_stopping_threshold", 0.001)
OPT_SALARY_FLOOR   = float(_opt["salary_floor"]) if _opt.get("salary_floor") is not None else None

# Quick-mode defaults (fast enough to complete in seconds per scenario)
OPT_CHAINS_QUICK = 25
OPT_STEPS_QUICK  = 20

PORTFOLIO_SIZE     = _port.get("size", 20)
TARGET_PERCENTILE  = _port.get("target_percentile", 90)
RNG_SEED           = _opt.get("rng_seed", 42)

# Default timeout per phase (seconds).  Override with --timeout.
DEFAULT_TIMEOUT = 600  # 10 minutes

# Paths — sourced from config.yaml
DK_PATH    = _paths.get("dk_slate",          "data/raw/DKSalaries.csv")
COPULA_PATH = _paths.get("copula",           "data/processed/empirical_copula.parquet")
PROJ_PATH  = _paths.get("projections",       "data/processed/projections.csv")
PCA_PATH   = _paths.get("batter_pca_model",  "data/processed/batter_pca_model.npz")
GRID_PATH  = _paths.get("batter_score_grid", "data/processed/batter_score_grid.npy")
OUTPUT_DIR = _paths.get("output_dir",        "outputs")


# ── Timeout helper ──────────────────────────────────────────────────────────

class ScenarioTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: int, label: str = "step"):
    """Raise ScenarioTimeout if the body exceeds *seconds* wall-clock time."""
    def _handler(*_):
        raise ScenarioTimeout(f"{label} timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Data helpers ────────────────────────────────────────────────────────────

def _derive_opponent(team: str, game: str) -> str:
    if not game:
        return ""
    parts = game.split("@")
    if len(parts) != 2:
        return ""
    away, home = parts[0].strip(), parts[1].strip()
    return home if team == away else (away if team == home else "")


def build_players_df(slate_df, proj_df=None):
    df = slate_df.copy()
    df["opponent"] = df.apply(lambda r: _derive_opponent(r["team"], r["game"]), axis=1)
    is_pitcher = df["position"] == "P"
    df["slot"] = 10
    batter_mask = ~is_pitcher
    df.loc[batter_mask, "slot"] = (
        df[batter_mask].groupby(["team", "opponent"]).cumcount().mod(9).add(1)
    )
    if proj_df is not None:
        proj = proj_df.copy().rename(columns={"mu": "mean", "sigma": "std_dev"})
        proj_cols = ["player_id", "mean", "std_dev"]
        if "lineup_slot" in proj.columns:
            proj_cols.append("lineup_slot")
        df = df.merge(proj[proj_cols], on="player_id", how="left")
        # Restrict to projected starters only — bench/bullpen players not in
        # projections will have lineup_slot=NaN after the left join.
        if "lineup_slot" in df.columns:
            df = df[df["lineup_slot"].notna()].copy()
        if "lineup_slot" in df.columns:
            has_slot = (
                ~(df["position"] == "P")
                & df["lineup_slot"].notna()
                & df["lineup_slot"].between(1, 9)
            )
            df.loc[has_slot, "slot"] = df.loc[has_slot, "lineup_slot"].astype(int)
    else:
        df["mean"] = df["salary"] / 400.0
        df["std_dev"] = df["mean"] * 0.85
    return df[["player_id", "team", "opponent", "slot", "mean", "std_dev", "position", "salary", "game"]]


def compute_target_at_percentile(players_df, sim_results, percentile):
    """Target = *percentile*-th percentile of greedy-best-lineup totals."""
    col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
    sorted_df = players_df.sort_values("mean", ascending=False)
    counts = {pos: 0 for pos in ROSTER_REQUIREMENTS}
    selected = []
    for _, row in sorted_df.iterrows():
        pos = str(row["position"])
        if pos in counts and counts[pos] < ROSTER_REQUIREMENTS[pos]:
            selected.append(int(row["player_id"]))
            counts[pos] += 1
        if len(selected) == 10:
            break
    if len(selected) == 10:
        cols = [col_map[pid] for pid in selected if pid in col_map]
        if len(cols) == 10:
            totals = sim_results.results_matrix[:, cols].sum(axis=1)
            return float(np.percentile(totals, percentile))
    n = len(players_df)
    row_sums = sim_results.results_matrix.sum(axis=1)
    return float(np.percentile(row_sums * 10.0 / n, percentile))


# ── Per-scenario runner ─────────────────────────────────────────────────────

def run_scenario(
    idx: int,
    n_sims: int,
    percentile: int,
    players_df: pd.DataFrame,
    engine: SimulationEngine,
    sim_cache: dict,
    timeout: int,
    n_chains: int,
    n_steps: int,
    temperature: float = OPT_TEMP,
    n_workers: int = 1,
    niter_success: int = OPT_NITER_SUCCESS,
    early_stopping_window: int = OPT_ES_WINDOW,
    early_stopping_threshold: float = OPT_ES_THRESHOLD,
    portfolio_size: int = PORTFOLIO_SIZE,
    salary_floor: float = OPT_SALARY_FLOOR,
) -> dict:
    """Run one benchmark scenario and return a result dict."""
    label = f"scenario {idx}: n_sims={n_sims:,d}  p{percentile:02d}"
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")

    result = {
        "scenario_idx": idx,
        "n_sims": n_sims,
        "percentile": percentile,
        "temperature": temperature,
        "sim_time_s": None,
        "target": None,
        "single_opt_time_s": None,
        "single_opt_p_hit": None,
        "portfolio_time_s": None,
        "portfolio_avg_p_hit": None,
        "portfolio_min_p_hit": None,
        "portfolio_n_lineups": None,
        "portfolio_per_lineup_s": None,
        "timed_out": False,
        "timeout_phase": None,
    }

    # ── 1. Simulation ──────────────────────────────────────────────────
    if n_sims in sim_cache:
        sim_results = sim_cache[n_sims]
        result["sim_time_s"] = 0.0  # already cached
        print(f"  [sim]     n_sims={n_sims:>7,d}  →  (cached)")
    else:
        try:
            with time_limit(timeout, f"simulation n_sims={n_sims}"):
                # Suppress verbose print() output from engine
                _saved = sys.stdout
                try:
                    sys.stdout = open(os.devnull, "w")
                    t0 = time.perf_counter()
                    sim_results = engine.simulate(n_sims)
                    elapsed = time.perf_counter() - t0
                finally:
                    sys.stdout.close()
                    sys.stdout = _saved

            sim_cache[n_sims] = sim_results
            result["sim_time_s"] = elapsed
            print(
                f"  [sim]     n_sims={n_sims:>7,d}  →  {elapsed:6.2f}s"
                f"  (matrix: {sim_results.results_matrix.shape})"
            )
        except ScenarioTimeout as e:
            print(f"  [sim]     TIMEOUT — {e}")
            result["timed_out"] = True
            result["timeout_phase"] = "simulation"
            return result

    target = compute_target_at_percentile(players_df, sim_results, percentile)
    result["target"] = target
    print(f"  [target]  p{percentile:02d}  →  {target:.1f}")

    # ── 2. Single-lineup optimisation ─────────────────────────────────
    try:
        with time_limit(timeout, f"single-lineup opt n_sims={n_sims} p{percentile}"):
            t0 = time.perf_counter()
            optimizer = BasinHoppingOptimizer(
                sim_results=sim_results,
                players_df=players_df,
                target=target,
                n_chains=n_chains,
                temperature=temperature,
                n_steps=n_steps,
                niter_success=niter_success,
                n_workers=n_workers,
                early_stopping_window=early_stopping_window,
                early_stopping_threshold=early_stopping_threshold,
                rng_seed=RNG_SEED,
                salary_floor=salary_floor,
            )
            lineup, score = optimizer.optimize()
            elapsed = time.perf_counter() - t0

        result["single_opt_time_s"] = elapsed
        result["single_opt_p_hit"] = score
        print(
            f"  [opt-1]   time={elapsed:6.1f}s  P(hit)={score:.4f}"
            f"  hit_rows={int(score * n_sims):>6,d}"
        )
    except ScenarioTimeout as e:
        print(f"  [opt-1]   TIMEOUT — {e}")
        result["timed_out"] = True
        result["timeout_phase"] = "single_opt"
        return result

    # ── 3. Portfolio construction (PORTFOLIO_SIZE lineups) ─────────────
    try:
        with time_limit(timeout, f"portfolio n_sims={n_sims} p{percentile}"):
            t0 = time.perf_counter()
            constructor = PortfolioConstructor(
                sim_results=sim_results,
                players_df=players_df,
                target=target,
                portfolio_size=portfolio_size,
                n_chains=n_chains,
                temperature=temperature,
                n_steps=n_steps,
                niter_success=niter_success,
                n_workers=n_workers,
                early_stopping_window=early_stopping_window,
                early_stopping_threshold=early_stopping_threshold,
                rng_seed=RNG_SEED,
                salary_floor=salary_floor,
            )
            portfolio = constructor.construct()
            elapsed = time.perf_counter() - t0

        scores = [s for _, s in portfolio]
        per_lineup = elapsed / max(len(portfolio), 1)
        result["portfolio_time_s"] = elapsed
        result["portfolio_n_lineups"] = len(portfolio)
        result["portfolio_avg_p_hit"] = float(np.mean(scores)) if scores else None
        result["portfolio_min_p_hit"] = float(np.min(scores)) if scores else None
        result["portfolio_per_lineup_s"] = per_lineup
        print(
            f"  [port]    time={elapsed:6.1f}s  lineups={len(portfolio):>2d}"
            f"  avg_P={np.mean(scores):.4f}  {per_lineup:.1f}s/lineup"
        )
        if elapsed > timeout * 0.9:
            print(f"  [port]    WARNING: close to timeout limit ({timeout}s)")
    except ScenarioTimeout as e:
        print(f"  [port]    TIMEOUT — {e}")
        result["timed_out"] = True
        result["timeout_phase"] = "portfolio"

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB DFS timing benchmark")
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        metavar="IDX",
        help="Run only the scenario at this 0-based index (omit to run all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all scenarios and exit",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"Per-phase timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Use reduced optimizer settings ({OPT_CHAINS_QUICK} chains / "
            f"{OPT_STEPS_QUICK} steps) to get relative timing without waiting "
            "for full production runs"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=OPT_WORKERS,
        metavar="N",
        help=f"Number of parallel chain workers (default: {OPT_WORKERS}; 1 = sequential)",
    )
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=PORTFOLIO_SIZE,
        metavar="N",
        help=f"Number of lineups to construct per scenario (default: {PORTFOLIO_SIZE})",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=TARGET_PERCENTILE,
        metavar="PCT",
        help=(
            f"Target percentile for score threshold "
            f"(default: {TARGET_PERCENTILE} from config.yaml; use 0 to sweep all)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=OPT_TEMP,
        metavar="T",
        help=f"Metropolis acceptance temperature (default: {OPT_TEMP} from config.yaml)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'IDX':>4}  {'n_sims':>10}  {'percentile':>10}")
        print("-" * 30)
        for i, (n, p) in enumerate(SCENARIOS):
            print(f"{i:>4}  {n:>10,d}  p{p:>02d}")
        return

    n_chains = OPT_CHAINS_QUICK if args.quick else OPT_CHAINS_DEFAULT
    n_steps = OPT_STEPS_QUICK if args.quick else OPT_STEPS_DEFAULT
    n_workers = args.workers

    print("=" * 72)
    print("MLB DFS Timing Benchmark")
    print(f"  config            : {_CONFIG_PATH}")
    print(f"  mode              : {'QUICK' if args.quick else 'PRODUCTION'}")
    print(f"  timeout per phase : {args.timeout}s ({args.timeout / 60:.1f} min)")
    print(f"  chains / steps    : {n_chains} / {n_steps}")
    print(f"  temperature       : {args.temperature}")
    print(f"  niter_success     : {OPT_NITER_SUCCESS}")
    print(f"  early_stop window : {OPT_ES_WINDOW}  threshold: {OPT_ES_THRESHOLD}")
    print(f"  salary_floor      : {OPT_SALARY_FLOOR if OPT_SALARY_FLOOR is not None else 'disabled'}")
    print(f"  workers           : {n_workers}")
    print(f"  portfolio size    : {args.portfolio_size}")
    print(f"  target percentile : {'all' if args.percentile == 0 else f'p{args.percentile}'}  (config default: p{TARGET_PERCENTILE})")
    print("=" * 72)

    # Load data once
    print("\nLoading data...")
    ingestor = DraftKingsSlateIngestor(DK_PATH)
    slate_df = ingestor.get_slate_dataframe()
    proj_df = pd.read_csv(PROJ_PATH)
    players_df = build_players_df(slate_df, proj_df)
    if "name" in slate_df.columns:
        players_df = players_df.merge(
            slate_df[["player_id", "name"]], on="player_id", how="left"
        )

    copula = EmpiricalCopula(COPULA_PATH)

    pca_model = None
    score_grid = None
    if os.path.exists(PCA_PATH) and os.path.exists(GRID_PATH):
        pca_model = BatterPCAModel.load(PCA_PATH)
        score_grid = np.load(GRID_PATH)

    engine = SimulationEngine(
        copula, players_df, batter_pca_model=pca_model, score_grid=score_grid
    )
    print(f"Player pool: {len(players_df)} players")

    # Select which scenarios to run
    if args.scenario is not None:
        if args.scenario < 0 or args.scenario >= len(SCENARIOS):
            print(f"ERROR: --scenario {args.scenario} out of range (0–{len(SCENARIOS)-1})")
            sys.exit(1)
        selected = [(args.scenario, *SCENARIOS[args.scenario])]
    else:
        selected = [(i, n, p) for i, (n, p) in enumerate(SCENARIOS)]
        if args.percentile != 0:
            selected = [(i, n, p) for i, n, p in selected if p == args.percentile]
            if not selected:
                print(f"ERROR: no scenarios match --percentile {args.percentile}")
                sys.exit(1)

    print(f"\nRunning {len(selected)} scenario(s)...\n")

    sim_cache: dict = {}
    all_results = []

    script_start = time.perf_counter()

    for idx, n_sims, percentile in selected:
        res = run_scenario(
            idx=idx,
            n_sims=n_sims,
            percentile=percentile,
            players_df=players_df,
            engine=engine,
            sim_cache=sim_cache,
            timeout=args.timeout,
            n_chains=n_chains,
            n_steps=n_steps,
            temperature=args.temperature,
            n_workers=n_workers,
            niter_success=OPT_NITER_SUCCESS,
            early_stopping_window=OPT_ES_WINDOW,
            early_stopping_threshold=OPT_ES_THRESHOLD,
            portfolio_size=args.portfolio_size,
            salary_floor=OPT_SALARY_FLOOR,
        )
        all_results.append(res)

        # Save incrementally so results are available even if the run is aborted
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(all_results).to_csv(
            f"{OUTPUT_DIR}/benchmark_results.csv", index=False
        )

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - script_start
    print(f"\n{'=' * 72}")
    print(f"SUMMARY  (total wall time: {total_elapsed:.1f}s = {total_elapsed/60:.1f} min)")
    print(f"{'=' * 72}")

    df = pd.DataFrame(all_results)
    cols_to_show = [
        "scenario_idx", "n_sims", "percentile", "target",
        "sim_time_s", "single_opt_time_s", "single_opt_p_hit",
        "portfolio_time_s", "portfolio_n_lineups", "portfolio_per_lineup_s",
        "timed_out", "timeout_phase",
    ]
    cols_present = [c for c in cols_to_show if c in df.columns]
    print(df[cols_present].to_string(index=False))

    # Flag scenarios that came close to the limit
    timed_out = df[df["timed_out"] == True]
    feasible = df[df["portfolio_time_s"].notna() & (df["timed_out"] == False)]
    if not timed_out.empty:
        print(f"\n  {len(timed_out)} scenario(s) exceeded the {args.timeout}s timeout.")
    if not feasible.empty:
        slowest = feasible.loc[feasible["portfolio_time_s"].idxmax()]
        print(
            f"\n  Slowest feasible portfolio: scenario {int(slowest['scenario_idx'])}"
            f"  (n_sims={int(slowest['n_sims']):,d}  p{int(slowest['percentile']):02d})"
            f"  →  {slowest['portfolio_time_s']:.1f}s"
        )

    print(f"\nFull results saved to {OUTPUT_DIR}/benchmark_results.csv")


if __name__ == "__main__":
    main()
