#!/usr/bin/env python3
"""Run the Marginal-Reward Portfolio allocator on a slate and write DK uploads.

The runnable entry point for the MRP track. Production is untouched: this reads
the same inputs, produces the same `upload_*.csv` files, and writes its own
portfolio JSON alongside.

    # inspect what MRP would do with a live slate, writing nothing
    python scripts/run_mrp.py --dry-run

    # allocate every purchased entry and write uploads
    python scripts/run_mrp.py --out outputs/mrp

    # A/B: production already shipped its half, MRP fills the rest
    python scripts/run_mrp.py --preassign-from outputs/portfolio_sweep_draftkings.json

    # replay an archived slate
    python scripts/run_mrp.py --slate archive/08182026 --dry-run

THE A/B, and why --preassign-from is the whole design. Splitting a contest
between production and MRP puts two of OUR portfolios in the same prize pool,
competing with each other. `--preassign-from` reads production's shipped
lineups and commits them into each contest's state as incumbents before MRP
picks, so MRP sees them displacing its candidates rather than pretending the
contest is empty. Operationally this also means production keeps running
exactly as today and MRP fills what is left -- no change to the live path.

Every entry MRP does not fill is reported, never silently dropped: an arm that
quietly grades fewer entries looks better per-entry for free.

--publish AND THE UI. Passing --publish writes the three artifacts the app
reads, all into the same directory, and verified against the real loaders:

    portfolio_sweep_<platform>.json   Portfolio tab (GET /api/portfolio/sweep).
                                      Carries upload_tag / entry_fee /
                                      contest_name / entry_sort_order per
                                      lineup, which is what renders the
                                      lineup -> contest mapping.
    portfolio_<platform>.csv          GET /api/portfolio?platform=, a DIFFERENT
                                      endpoint with its own loader.
    upload_*Entries*.csv              the DK upload files, and what the Late
                                      Swap tab operates on -- it globs
                                      <output_dir>/*Entries*.csv and never
                                      looks at the portfolio at all.

They must share one directory. Splitting them would leave the Portfolio tab
showing MRP while Late Swap edited production's submitted lineups. Existing
files are backed up first, since they are the real-money record of what was
sent to DK.

The Metrics tab needs nothing extra -- it derives from the portfolio array the
Portfolio tab already holds. Its TIMING block stays empty, because that reads
SSE events from a live pipeline run and a CLI run emits none.

TWO THINGS DO NOT WORK on a published MRP portfolio, both by construction:
`POST /api/portfolio/activate_risk` and `POST /api/portfolio/replace/{i}` need
the in-memory PipelineRunner from a UI-driven run, so they return a clear 400
rather than acting on stale state. There is also only ONE risk tier: risk is an
EVw dial belonging to the Det selector, and dR has no such knob, so the sweep
carries a single entry (the UI falls back to sweep[0] and renders it normally).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import parse_entry_file, scan_entry_files, write_upload_files  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.optimization.mrp.runner import (  # noqa: E402
    MRPConfig,
    allocate_marginal_reward,
    publish_portfolio,
)
from src.optimization.mrp.slate_inputs import (  # noqa: E402
    SimCalibration,
    build_slate_inputs,
)


def load_preassigned(path: Path, pool, groups) -> dict:
    """{contest_id: [pool index]} from a shipped portfolio JSON.

    Matched by exact 10-player set. A shipped lineup absent from the pool
    (late swap, or a pool file captured at a different moment) cannot be
    committed as an incumbent, so it is reported rather than ignored -- it
    means MRP is modelling a slightly emptier contest than it will face.
    """
    sw = json.loads(Path(path).read_text())
    entries = (next((x for x in sw["sweep"] if x.get("risk") == 1.0), sw["sweep"][0])["lineups"]
               if "sweep" in sw else sw.get("lineups", []))
    idx_by_set = {frozenset(int(p) for p in lu.player_ids): i
                  for i, lu in enumerate(pool.lineups)}
    name_to_id = {g.contest_name: g.contest_id for g in groups}

    out: dict = {}
    missed = 0
    for e in entries:
        key = frozenset(int(p["player_id"]) for p in e["players"])
        i = idx_by_set.get(key)
        if i is None:
            missed += 1
            continue
        cid = name_to_id.get(str(e.get("contest_name", "")))
        if cid is None:
            for nm, c in name_to_id.items():
                if nm and str(e.get("contest_name", "")) and (
                        nm in e["contest_name"] or e["contest_name"] in nm):
                    cid = c
                    break
        if cid is not None:
            out.setdefault(cid, []).append(i)
    if missed:
        print(f"  WARNING: {missed} shipped lineups are not in the pool and cannot be "
              f"modelled as incumbents -- MRP will see those contests as emptier "
              f"than they really are")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate", default="data/raw", help="slate dir (default data/raw)")
    ap.add_argument("--entries", default=None, help="dir holding *Entries.csv (default: --slate)")
    ap.add_argument("--out", default="outputs/mrp", help="output dir")
    ap.add_argument("--preassign-from", default=None,
                    help="portfolio JSON whose lineups are the OTHER arm's entries")
    ap.add_argument("--n-sims", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma-in", type=int, default=7)
    ap.add_argument("--gamma-out", type=int, default=8)
    ap.add_argument("--smooth-tau", type=float, default=0.0)
    ap.add_argument("--max-sims-per-contest", type=int, default=12_500)
    ap.add_argument("--field-pool", type=int, default=25_000)
    ap.add_argument("--frontier", action="store_true",
                    help="generate along the Haugh & Singal line-2 mean-variance "
                         "frontier and add the result to the candidate pool")
    ap.add_argument("--frontier-lambdas", type=int, default=12)
    ap.add_argument("--frontier-target", type=int, default=4000,
                    help="TOTAL generated lineups to aim for; the per-team cap "
                         "is derived from this and the emergent lambda* count")
    ap.add_argument("--frontier-min-per-team", type=int, default=4)
    ap.add_argument("--frontier-sample", type=int, default=30_000,
                    help="candidates sampled before exact ranking")
    ap.add_argument("--frontier-anchors", type=int, default=2,
                    help="exact CP-SAT solves; 0 = solver-free generation")
    ap.add_argument("--frontier-generations", type=int, default=2)
    ap.add_argument("--frontier-mutants", type=int, default=4)
    ap.add_argument("--frontier-salary-floor", type=float, default=47_500.0,
                    help="min salary for generated lineups; match what SaberSim "
                         "was given. 0 disables")
    ap.add_argument("--frontier-timeout", type=float, default=8.0)
    ap.add_argument("--proj-score-pct", type=float, default=None,
                    help="pool-wide ceiling floor: cull the bottom N%% of lineups "
                         "by SaberSim 99th (default: gpp.external_pool_proj_score_pct "
                         "from config.yaml, so the CLI and the pipeline cull alike; "
                         "0 disables)")
    ap.add_argument("--no-calibration", action="store_true",
                    help="force raw SaberSim grids, overriding config. By default the "
                         "gpp.external_pool_* calibration keys are read from config.yaml "
                         "so this path simulates identically to the UI/pipeline run.")
    ap.add_argument("--sim-cache", default="outputs/replay/sim_cache")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    ap.add_argument("--publish", action="store_true",
                    help="ALSO write outputs/portfolio_sweep_<platform>.json so the UI's "
                         "Portfolio tab shows this portfolio. OVERWRITES production's "
                         "shipped portfolio (a backup is taken first).")
    ap.add_argument("--publish-dir", default="outputs",
                    help="where the UI reads the sweep from (default outputs)")
    args = ap.parse_args()

    slate_dir = PROJECT_ROOT / args.slate if not Path(args.slate).is_absolute() else Path(args.slate)
    entries_dir = Path(args.entries) if args.entries else slate_dir

    print(f"slate      {slate_dir}")

    si = build_slate_inputs(
        slate_dir, n_sims=args.n_sims, seed=args.seed,
        # None = read config, matching the pipeline. Only --no-calibration
        # overrides, and it says so on the line below so a run that does not
        # match the UI is never silent about it.
        calibration=SimCalibration() if args.no_calibration else None,
        sim_cache_dir=PROJECT_ROOT / args.sim_cache,
    )
    print(f"n_sims     {args.n_sims}  seed {args.seed}")
    print(f"sim calib  {si.calibration.describe()}"
          f"{'   [OVERRIDDEN by --no-calibration]' if args.no_calibration else '   [from config.yaml]'}")
    print(f"pool       {len(si.pool.lineups):,} lineups   players {len(si.players_df)}")

    entry_files = scan_entry_files(str(entries_dir))
    if not entry_files:
        raise SystemExit(f"no *Entries.csv found in {entries_dir}")
    all_file_entries = [(p, parse_entry_file(p)) for p in entry_files]
    groups = ep.group_and_match_contests(all_file_entries, si.pool)
    total_slots = sum(len(g.entries) for g in groups)
    print(f"entries    {total_slots} across {len(groups)} contests "
          f"({len(entry_files)} file(s))")

    preassigned = None
    if args.preassign_from:
        preassigned = load_preassigned(Path(args.preassign_from), si.pool, groups)
        n_pre = sum(len(v) for v in preassigned.values())
        print(f"preassign  {n_pre} entries from the other arm, treated as incumbents")

    cfg = MRPConfig(
        gamma_in=args.gamma_in, gamma_out=args.gamma_out,
        smooth_tau_scale=args.smooth_tau, seed=args.seed,
        field_pool_size=args.field_pool,
        max_sims_per_contest=args.max_sims_per_contest,
        frontier_enabled=args.frontier,
        frontier_n_lambdas=args.frontier_lambdas,
        frontier_target_lineups=args.frontier_target,
        frontier_min_per_team=args.frontier_min_per_team,
        frontier_sample_n=args.frontier_sample,
        frontier_n_anchors=args.frontier_anchors,
        frontier_n_generations=args.frontier_generations,
        frontier_mutants_per_parent=args.frontier_mutants,
        frontier_salary_floor=args.frontier_salary_floor,
        frontier_solver_timeout_s=args.frontier_timeout,
    )
    _floor_pct = args.proj_score_pct
    if _floor_pct is None:
        _floor_pct = float(
            (yaml.safe_load(open(PROJECT_ROOT / "config.yaml")).get("gpp", {}) or {})
            .get("external_pool_proj_score_pct", 0.0)
        )
    _floor_scores = (
        ep.compute_pool_ceiling_scores(si.pool, si.players_df) if _floor_pct > 0 else None
    )
    def _progress(d: dict) -> None:
        """Not every stage is a counter.

        `mrp_floor`, `mrp_preflight` and the `mrp_frontier_*` pair report a
        verdict rather than progress and carry no done/total, so keying on
        them unconditionally raises KeyError partway through a real run.
        """
        stage = d.get("stage", "")
        done, total = d.get("done"), d.get("total")
        if done is None or total is None:
            print(f"  {stage}: " + ", ".join(
                f"{k}={v}" for k, v in d.items() if k != "stage"))
            return
        if done % 25 == 0 or done == total:
            print(f"  {stage} {done}/{total}", end="\r")

    alloc, diag = allocate_marginal_reward(
        si.pool, si.players_df, si.sim_results, groups, cfg, preassigned=preassigned,
        floor_scores=_floor_scores, proj_score_floor_percentile=_floor_pct,
        progress_cb=_progress,
    )
    print()
    print(diag.summary())
    if alloc.unfilled:
        print(f"UNFILLED: {len(alloc.unfilled)} entries -- the constraints "
              f"(gamma_in={cfg.gamma_in}, gamma_out={cfg.gamma_out}) exhausted the pool")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # When publishing, the upload CSVs must land in the SAME directory as the
    # portfolio artifacts. Late swap does not read the portfolio at all --
    # `late_swap.scan_swap_entry_files` globs `<output_dir>/*Entries*.csv` --
    # so writing uploads elsewhere would leave the Portfolio tab showing MRP
    # while the Late Swap tab edited production's submitted lineups.
    publish_dir = PROJECT_ROOT / args.publish_dir
    out_dir = publish_dir if args.publish else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    slate_df = DraftKingsSlateIngestor(str(slate_dir / "DKSalaries.csv")).get_slate_dataframe()

    assignments: dict = {}
    for (lineup, _d), (file_path, rec) in zip(alloc.portfolio, alloc.entry_plan):
        assignments.setdefault(file_path, []).append((rec, lineup))

    if args.publish:
        # Back up production's submitted uploads before replacing them; these
        # are what was actually sent to DK.
        for p in sorted(out_dir.glob("upload_*Entries*.csv")):
            shutil.copy2(p, p.with_name(p.name.replace("upload_", "prod-backup_upload_", 1)))

    written = write_upload_files(all_file_entries, assignments, slate_df, str(out_dir))

    payload = {
        "mode": "marginal_reward",
        "slate": slate_dir.name,
        "config": vars(args),
        "sim_calibration": si.calibration.__dict__,
        "total_reward": diag.total_reward,
        "n_unfilled": diag.n_unfilled,
        "per_contest": diag.per_contest,
        "lineups": [
            {"contest_name": rec.contest_name, "contest_id": rec.contest_id,
             "entry_id": rec.entry_id, "delta_reward": d,
             "players": [int(p) for p in lu.player_ids]}
            for (lu, d), (_fp, rec) in zip(alloc.portfolio, alloc.entry_plan)
        ],
    }
    (out_dir / "mrp_portfolio.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(written)} upload file(s) + mrp_portfolio.json to {out_dir}")

    if args.publish:
        pub = publish_portfolio(
            alloc, diag, si.players_df,
            slate_path=slate_dir / "DKSalaries.csv",
            output_dir=publish_dir,
        )
        print(f"\npublished {pub['n_lineups']} lineups, tagged mode=marginal_reward:")
        print(f"  Portfolio tab (sweep)   {pub['sweep_path']}")
        print(f"  GET /api/portfolio      {pub['csv_path']}")
        print(f"  Late Swap / DK upload   {len(written)} upload_*.csv in {out_dir}")
        print("  Metrics tab derives from the portfolio, so it follows automatically "
              "(its timing block stays empty -- that needs SSE events from a live run).")
        for b in pub["backup_paths"]:
            print(f"  backed up: {b}")
    else:
        print("(not published to the UI -- pass --publish to show this in the Portfolio tab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
