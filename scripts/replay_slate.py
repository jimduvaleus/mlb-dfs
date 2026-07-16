"""
Offline end-to-end pipeline replay harness.

Re-runs the REAL production pipeline (PipelineRunner: simulate → ownership →
candidate generation → contest scoring → fresh re-score → Det-EV risk sweep)
on an archived slate's inputs, then grades every funnel stage against the
real contest field from the archived standings zip. This is the empirical
engine for the ceiling-first redesign: every generation/ranking/selection
intervention is judged here on many past slates before it touches a live run.

Inputs per slate (all under archive/MMDDYYYY/):
  DKSalaries.csv                 — slate (exclusions/twitter overrides keyed
                                   by this file's fingerprint are inherited
                                   from the live run automatically)
  market_odds_projections.csv    — projections (identical schema to
                                   data/processed/projections.csv)
  contest-standings-*.zip        — real per-entry Points table + FPTS sidebar
  team_totals.csv / dff_team_totals.csv — resolved automatically by the
                                   pipeline from the slate date
  projections_mo_dist.parquet    — optional; copied next to the prepped
                                   projections when present so the sim uses
                                   market-implied marginals (not archived
                                   before 2026-07-12 — those replays fall
                                   back to the parametric marginal path)

Mean-calibration cutover: archives from 2026-07-06 onward already contain
calibrated means. For earlier slates the prep step applies
MEAN_CALIB_BATTER/PITCHER at load (means linearly; batter std √-style,
pitcher std linearly — mirroring the fetcher) so pre-cutover replays run the
same projection scale as today's pipeline. --no-precalib disables.

Usage
-----
    # single slate, current config:
    python scripts/replay_slate.py archive/07082026

    # all replay-capable slates (salaries + market odds + standings):
    python scripts/replay_slate.py --all

    # ad-hoc config overrides (dotted keys, YAML-parsed values):
    python scripts/replay_slate.py archive/07082026 \
        --name simopt --set gpp.seed_sim_optimal_lineups=true --set gpp.n_sim_optimals=1000

    # a pre-registered variant matrix (see outputs/replay/variants.example.yaml):
    python scripts/replay_slate.py --all --variants variants.yaml

    # coarse screening profile (10k sims / 10k candidates, ~3x faster):
    python scripts/replay_slate.py --all --screen

    # regrade existing run dirs without re-running the pipeline:
    python scripts/replay_slate.py archive/07082026 --name baseline --grade-only

Grading output: one row per (slate, variant, stage) appended to
outputs/replay/replay_summary.csv, a per-run stage table on stdout, and a
pooled per-variant pivot across slates at the end of a batch.

Stages graded from the run's candidate_pool_debug.csv:
  pool            — every generated candidate
  floor           — mined EV ≥ gpp.ev_floor (the legacy admission lane)
  fresh_surv      — fresh-rescore survivors (fresh_ev ≥ its lane's floor)
  bypass          — tail-bypass admits (tail_bypass=1)
  seed:<source>   — per seed_source block (random / optimal / sim_optimal / ...)
  risk<r>         — the selected portfolio for each risk tier
  rank:<col>      — top --top-n by any numeric dump column (--rank-col)

Metrics per stage: n, hit99 / hit99.9 (fraction at/above the real field's
p99 / p99.9), best real percentile, mean real percentile, cash rate
(≥ cash-threshold fraction of the real field beaten), and mean realized
net dollars (candidate's actual score placed in the real field; the
reference dk_classic_gpp.json payout curve rank-scaled to the real field
size, ties paid as the tie-band mean, entry fee subtracted). The pool stage
also reports Spearman(col, real percentile) for every numeric column —
mined EV, fresh EV, projected score, ownership, and any new tail-metric
columns the pipeline dumps.

Lookahead caveat (recorded per row, disclosed once here): the ownership
calibrator, dupe-model coefficients, copula/PCA artifacts, and the copula
dependence overlay constants were fitted on archives that overlap this
replay set. Treat variant deltas as the signal, absolute levels with care.
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_candidate_pool as acp  # noqa: E402
import measure_pool_ceiling as mpc  # noqa: E402
from src.models.projection_calibration import (  # noqa: E402
    MEAN_CALIB_BATTER, MEAN_CALIB_PITCHER,
)

MEAN_CALIB_CUTOVER = datetime(2026, 7, 6)
REPLAY_ROOT = PROJECT_ROOT / "outputs" / "replay"
SIM_CACHE_DIR = REPLAY_ROOT / "sim_cache"
SUMMARY_PATH = REPLAY_ROOT / "replay_summary.csv"
PAYOUT_JSON = PROJECT_ROOT / "data" / "payout_structures" / "dk_classic_gpp.json"
DIST_FILENAME = "projections_mo_dist.parquet"

# Columns that are identifiers/labels rather than ranking signals.
_NON_SIGNAL_COLS = {"lineup_index", "n_players", "n_missing", "actual_score", "real_pct"}


def _slate_date(name: str) -> datetime:
    return datetime.strptime(name[:8], "%m%d%Y")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prep_projections(archive_dir: Path, run_dir: Path, precalib: bool) -> Path:
    """Copy the archived projections into the run dir, applying the mean
    calibration for pre-cutover slates so every replay runs on the same
    projection scale as the current pipeline. Also drops the market-implied
    dist parquet next to it when the archive has one."""
    src = archive_dir / "market_odds_projections.csv"
    dst = run_dir / "projections.csv"
    df = pd.read_csv(src)
    is_precutover = _slate_date(archive_dir.name) < MEAN_CALIB_CUTOVER
    if precalib and is_precutover:
        pitcher = df["lineup_slot"].fillna(0).astype(int) == 10
        df.loc[pitcher, "mean"] *= MEAN_CALIB_PITCHER
        df.loc[~pitcher, "mean"] *= MEAN_CALIB_BATTER
        if "std_dev" in df.columns:
            df.loc[pitcher, "std_dev"] *= MEAN_CALIB_PITCHER
            df.loc[~pitcher, "std_dev"] *= np.sqrt(MEAN_CALIB_BATTER)
    df.to_csv(dst, index=False)
    dist_src = archive_dir / DIST_FILENAME
    if dist_src.exists():
        shutil.copy(dist_src, run_dir / DIST_FILENAME)
    return dst


def _dotted_set(cfg: dict, dotted_key: str, value) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_set_arg(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise SystemExit(f"--set expects dotted.key=value, got: {raw}")
    key, _, val = raw.partition("=")
    return key.strip(), yaml.safe_load(val)


def synth_config(
    archive_dir: Path, run_dir: Path, proj_path: Path,
    overrides: dict, portfolio_size: int, screen: bool,
) -> tuple[Path, dict]:
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["platform"] = "draftkings"
    cfg["paths"]["dk_slate"] = str(archive_dir / "DKSalaries.csv")
    cfg["paths"]["projections"] = str(proj_path)
    cfg["paths"]["output_dir"] = str(run_dir)
    cfg["optimizer"]["rng_seed"] = cfg["optimizer"].get("rng_seed") or 42
    cfg["portfolio"]["size"] = portfolio_size
    cfg["gpp"]["dump_candidate_pool"] = True
    if screen:
        cfg["simulation"]["n_sims"] = 10_000
        cfg["gpp"]["n_candidates"] = 10_000
    for key, value in overrides.items():
        _dotted_set(cfg, key, value)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path, cfg


def _sim_cache_key(archive_dir: Path, cfg: dict, proj_path: Path) -> Path:
    """Sim reuse is only valid across variants that share the simulation
    inputs — key on slate, n_sims, seed, and the prepped projections bytes
    (mean overrides / precalib change the marginals)."""
    proj_hash = hashlib.md5(proj_path.read_bytes()).hexdigest()[:8]
    n_sims = cfg["simulation"]["n_sims"]
    seed = cfg["optimizer"]["rng_seed"]
    return SIM_CACHE_DIR / f"{archive_dir.name}_{n_sims}_{seed}_{proj_hash}.npz"


def run_variant(
    archive_dir: Path, variant: str, overrides: dict,
    portfolio_size: int, screen: bool, precalib: bool,
    force: bool, grade_only: bool,
) -> Path | None:
    run_dir = REPLAY_ROOT / archive_dir.name / variant
    dump_path = run_dir / "candidate_pool_debug.csv"
    if grade_only:
        if not dump_path.exists():
            print(f"  {archive_dir.name}/{variant}: no existing run to grade — skipping.")
            return None
        return run_dir
    if dump_path.exists() and not force:
        print(f"  {archive_dir.name}/{variant}: run exists (use --force to re-run) — regrading only.")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    proj_path = prep_projections(archive_dir, run_dir, precalib)
    cfg_path, cfg = synth_config(archive_dir, run_dir, proj_path, overrides, portfolio_size, screen)
    SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sim_cache = _sim_cache_key(archive_dir, cfg, proj_path)

    from src.api.pipeline import PipelineRunner
    t0 = time.perf_counter()
    runner = PipelineRunner(
        str(cfg_path),
        persist_caches=False,
        sim_cache_path=str(sim_cache),
    )
    runner.run()
    print(f"  {archive_dir.name}/{variant}: pipeline done in {time.perf_counter() - t0:.0f}s")
    if not dump_path.exists():
        print(f"  WARNING: {dump_path} was not produced — nothing to grade.")
        return None
    return run_dir


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _payout_curve(n_field: int) -> tuple[np.ndarray, float]:
    """Per-rank gross payout (rank 1..n_field): the reference contest's
    payout curve sampled at each rank's percentile, renormalized so the
    paid fraction of collected fees matches the reference exactly (DK's
    ~16% rake is fixed across contest sizes). The previous rank-interval
    scaling let the single-rank top tiers overwrite each other at scaled
    indices, destroying 20-50% of the top-heavy prize mass (implied rake
    24-29% at common field sizes) and understating every realized-net
    figure's tail."""
    pj = json.loads(PAYOUT_JSON.read_text())
    fee = float(pj.get("entry_fee", 4.0))
    ref_n = int(pj["total_entries"])
    ref = np.zeros(ref_n)
    for tier in pj["payouts"]:
        ref[tier["start"] - 1: tier["end"]] = tier["amount"]
    idx = np.minimum((np.arange(n_field) * ref_n) // n_field, ref_n - 1)
    curve = ref[idx].astype(np.float64)
    ref_pool_frac = ref.sum() / (ref_n * fee)
    if curve.sum() > 0:
        curve *= (n_field * fee * ref_pool_frac) / curve.sum()
    return curve, fee


def _load_actuals(archive_dir: Path) -> dict[int, float]:
    """player_id -> actual FPTS: contest_player_fpts.json when the live
    'Analyze Contest' snapshot exists, standings-zip FPTS sidebar (name
    match) for anything it lacks — so slates without the JSON still grade."""
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    slate_df = DraftKingsSlateIngestor(str(archive_dir / "DKSalaries.csv")).get_slate_dataframe()
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    _, ownership_df = mpc._parse_contest_zip(zips[0])
    return mpc._load_actual_fpts(archive_dir, ownership_df, slate_df)


def build_graded_lineups(run_dir: Path, archive_dir: Path) -> tuple[pd.DataFrame, np.ndarray, float]:
    """One row per candidate lineup with actual_score, real_pct, and
    realized net dollars, from the run's pool dump vs the real field."""
    pool_df = pd.read_csv(run_dir / "candidate_pool_debug.csv", low_memory=False)
    fpts_map = _load_actuals(archive_dir)
    lu = acp.build_lineup_table(pool_df, fpts_map)
    field_points = acp.load_real_field_points(archive_dir)
    n_field = len(field_points)
    curve, entry_fee = _payout_curve(n_field)

    lu = lu.dropna(subset=["actual_score"]).reset_index(drop=True)
    right = np.searchsorted(field_points, lu["actual_score"].values, side="right")
    left = np.searchsorted(field_points, lu["actual_score"].values, side="left")
    n_above = n_field - right
    n_tied = right - left
    gross = np.empty(len(lu))
    for i, (a, t) in enumerate(zip(n_above, n_tied)):
        band = curve[a: a + max(int(t), 1) + 1]  # the candidate joins the tie band
        gross[i] = band.mean() if len(band) else 0.0
    lu["real_pct"] = right / n_field
    lu["realized_net"] = gross - entry_fee
    return lu, field_points, entry_fee


def _stage_masks(lu: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    gpp = cfg.get("gpp", {})
    ev_floor = float(gpp.get("ev_floor", 0.25))
    bypass_floor = float(gpp.get("tail_bypass_ev_floor", -1.0))
    stages: dict[str, pd.Series] = {"pool": pd.Series(True, index=lu.index)}
    stages["floor"] = lu["projected_ev"] >= ev_floor
    is_bypass = lu["tail_bypass"].fillna(0).astype(int) == 1
    fresh = lu["fresh_ev"]
    stages["fresh_surv"] = fresh.notna() & (
        (~is_bypass & (fresh >= ev_floor)) | (is_bypass & (fresh >= bypass_floor))
    )
    if is_bypass.any():
        stages["bypass"] = is_bypass
    for source, grp in lu.groupby("seed_source"):
        if source and source != "not_tracked":
            stages[f"seed:{source}"] = lu["seed_source"] == source
    risks = lu["selected_risks"].fillna("").astype(str)
    all_risk_labels = sorted({r for cell in risks if cell for r in cell.split(",") if r})
    for r in all_risk_labels:
        stages[f"risk{r}"] = risks.str.split(",").apply(lambda labels, _r=r: _r in labels)
    if all_risk_labels:
        stages["risk_all"] = risks.str.len() > 0
    return stages


def _stage_metrics(sub: pd.DataFrame, field_points: np.ndarray, cash_threshold: float) -> dict:
    n_field = len(field_points)
    p99 = field_points[int(np.ceil(0.99 * n_field)) - 1]
    p999 = field_points[int(np.ceil(0.999 * n_field)) - 1]
    return {
        "n": len(sub),
        "hit99": float((sub["actual_score"] >= p99).mean()) if len(sub) else float("nan"),
        "hit999": float((sub["actual_score"] >= p999).mean()) if len(sub) else float("nan"),
        "best_pct": float(sub["real_pct"].max()) if len(sub) else float("nan"),
        "mean_pct": float(sub["real_pct"].mean()) if len(sub) else float("nan"),
        "cash_rate": float((sub["real_pct"] >= cash_threshold).mean()) if len(sub) else float("nan"),
        "mean_net_$": float(sub["realized_net"].mean()) if len(sub) else float("nan"),
    }


def grade_run(
    run_dir: Path, archive_dir: Path, variant: str,
    rank_cols: list[str], top_n: int, cash_threshold: float,
) -> list[dict]:
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    lu, field_points, _ = build_graded_lineups(run_dir, archive_dir)
    if lu.empty:
        print(f"  {archive_dir.name}/{variant}: no complete-coverage lineups — skipped.")
        return []

    stages = _stage_masks(lu, cfg)
    for col in rank_cols:
        if col in lu.columns and pd.api.types.is_numeric_dtype(lu[col]):
            thresh = lu[col].nlargest(top_n)
            if len(thresh):
                stages[f"rank:{col}"] = lu.index.isin(thresh.index)
        else:
            print(f"  {archive_dir.name}/{variant}: --rank-col {col} missing/non-numeric — skipped.")

    rows = []
    for stage, mask in stages.items():
        m = _stage_metrics(lu[mask], field_points, cash_threshold)
        rows.append({
            "slate": archive_dir.name, "variant": variant, "stage": stage, **m,
        })

    # Ranking-signal report on the full pool: every numeric column vs outcome.
    signal = {}
    for col in lu.columns:
        if col in _NON_SIGNAL_COLS or not pd.api.types.is_numeric_dtype(lu[col]):
            continue
        vals = lu[[col, "real_pct"]].dropna()
        if len(vals) >= 100 and vals[col].nunique() > 1:
            signal[col] = float(spearmanr(vals[col], vals["real_pct"]).correlation)
    for col, r in signal.items():
        rows.append({
            "slate": archive_dir.name, "variant": variant, "stage": f"spearman:{col}",
            "n": int(lu[col].notna().sum()), "hit99": float("nan"), "hit999": float("nan"),
            "best_pct": float("nan"), "mean_pct": float("nan"),
            "cash_rate": float("nan"), "mean_net_$": r,
        })
    lu.to_csv(run_dir / "graded_lineups.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_run_table(rows: list[dict]) -> None:
    df = pd.DataFrame([r for r in rows if not r["stage"].startswith("spearman:")])
    if df.empty:
        return
    for slate, sub in df.groupby("slate"):
        print(f"\n=== {slate} / {sub['variant'].iloc[0]} ===")
        out = sub.drop(columns=["slate", "variant"]).copy()
        for c in ("hit99", "hit999"):
            out[c] = out[c] * 100
        print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    sp = pd.DataFrame([r for r in rows if r["stage"].startswith("spearman:")])
    if not sp.empty:
        for slate, sub in sp.groupby("slate"):
            pairs = ", ".join(
                f"{s['stage'][9:]}={s['mean_net_$']:+.3f}" for _, s in sub.iterrows()
            )
            print(f"  spearman vs real_pct [{slate}]: {pairs}")


def print_pooled_pivot(all_rows: list[dict]) -> None:
    df = pd.DataFrame([r for r in all_rows if not r["stage"].startswith("spearman:")])
    if df.empty or df["slate"].nunique() < 2:
        return
    print(f"\n=== Pooled across {df['slate'].nunique()} slates (lineup-weighted hit99 %, per variant × stage) ===")
    df["hits99"] = df["hit99"] * df["n"]
    g = df.groupby(["variant", "stage"]).agg(
        n=("n", "sum"), hits=("hits99", "sum"),
        net=("mean_net_$", "mean"), slates=("slate", "nunique"),
    )
    g["hit99_%"] = 100 * g["hits"] / g["n"]
    print(
        g.reset_index()[["variant", "stage", "n", "hit99_%", "net", "slates"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


def append_summary(all_rows: list[dict]) -> None:
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    df["run_ts"] = datetime.now().isoformat(timespec="seconds")
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    if SUMMARY_PATH.exists():
        old = pd.read_csv(SUMMARY_PATH, dtype={"slate": str})
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary appended -> {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Slate discovery / CLI
# ---------------------------------------------------------------------------

def replay_capable_slates() -> list[Path]:
    out = []
    for d in sorted((PROJECT_ROOT / "archive").iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        try:
            _slate_date(d.name)
        except ValueError:
            continue
        if d.name[8:].startswith("r"):  # r/r2/... re-run dirs duplicate their base contest
            continue
        if (
            (d / "DKSalaries.csv").exists()
            and (d / "market_odds_projections.csv").exists()
            and list(d.glob("contest-standings-*.zip"))
        ):
            out.append(d)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the full pipeline on archived slates and grade vs the real field.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("archive_dirs", nargs="*", metavar="ARCHIVE_DIR")
    parser.add_argument("--all", action="store_true", help="all replay-capable archive slates")
    parser.add_argument("--recent", type=int, default=0, metavar="N", help="N most recent replay-capable slates")
    parser.add_argument("--name", default="baseline", help="variant name for --set runs (default: baseline)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="dotted.key=value")
    parser.add_argument("--variants", metavar="YAML", help="variant-matrix file: {name: {dotted.key: value}}")
    parser.add_argument("--screen", action="store_true", help="10k sims / 10k candidates screening profile")
    parser.add_argument("--portfolio-size", type=int, default=20)
    parser.add_argument("--no-precalib", action="store_true", help="don't apply mean calibration to pre-2026-07-06 archives")
    parser.add_argument("--force", action="store_true", help="re-run even when the run dir already has a dump")
    parser.add_argument("--grade-only", action="store_true", help="regrade existing run dirs, run nothing")
    parser.add_argument("--rank-col", dest="rank_cols", action="append", default=[], metavar="COL",
                        help="also grade the top --top-n candidates by this dump column")
    parser.add_argument("--top-n", type=int, default=3000)
    parser.add_argument("--cash-threshold", type=float, default=None)
    args = parser.parse_args()

    if sum(bool(x) for x in (args.archive_dirs, args.all, args.recent)) != 1:
        parser.error("give ARCHIVE_DIR(s), or exactly one of --all / --recent N")
    if args.all:
        dirs = replay_capable_slates()
    elif args.recent:
        dirs = replay_capable_slates()[-args.recent:]
    else:
        dirs = [Path(p) for p in args.archive_dirs]
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            parser.error(f"not a directory: {missing}")

    if args.variants:
        with open(args.variants) as f:
            variants: dict[str, dict] = yaml.safe_load(f) or {}
        if args.sets:
            parser.error("--variants and --set are mutually exclusive")
    else:
        variants = {args.name: dict(parse_set_arg(s) for s in args.sets)}

    cash_threshold = args.cash_threshold if args.cash_threshold is not None else acp.default_cash_threshold()
    screen_tag = " [screen]" if args.screen else ""
    print(f"Replaying {len(dirs)} slate(s) × {len(variants)} variant(s){screen_tag}: "
          f"{[d.name for d in dirs]} × {list(variants)}")

    all_rows: list[dict] = []
    for d in dirs:
        for variant, overrides in variants.items():
            vname = f"{variant}_screen" if args.screen else variant
            try:
                run_dir = run_variant(
                    d, vname, overrides, args.portfolio_size, args.screen,
                    precalib=not args.no_precalib, force=args.force,
                    grade_only=args.grade_only,
                )
            except Exception as exc:
                print(f"  {d.name}/{vname}: pipeline FAILED — {exc}")
                continue
            if run_dir is None:
                continue
            try:
                rows = grade_run(run_dir, d, vname, args.rank_cols, args.top_n, cash_threshold)
            except Exception as exc:
                print(f"  {d.name}/{vname}: grading FAILED — {exc}")
                continue
            print_run_table(rows)
            all_rows.extend(rows)

    print_pooled_pivot(all_rows)
    append_summary(all_rows)


if __name__ == "__main__":
    main()
