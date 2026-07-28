"""
Sweep the p_win stage-A cull's contest-size scaling
(external_pool_pwin_admit_multiplier) against a flat admit_n, using each
slate's REAL multi-contest group-size breakdown.

Why this needed its own harness
--------------------------------
sim_evaluate_portfolios.py --build tests the p_win pipeline with a single
synthetic 150-entry contest group, which is exactly the wrong shape to test
per-contest-size scaling: the whole point of the fix is that a flat
admit_n gives a 72-entry contest a much tighter *relative* reservoir than a
14-entry one, so the test needs the real spread of contest sizes, not one
uniform group.

Real per-contest sizes without an entries.csv
-----------------------------------------------
Archived slates don't keep the user's personal DK entries file, but they do
keep portfolio_sweep_draftkings.json from the last live run -- grouping its
risk-1 lineups by contest_name recovers the exact real entry-count
breakdown for that slate (e.g. 07/26: mini-MAX=72, Solo Shot=29, ...). This
script rebuilds synthetic ContestGroups at those sizes and runs the real
allocate_contests/compute_p_win path against them.

p_win_cull/p_win_select are computed ONCE per slate (shared across every
admit_n/multiplier combo in the sweep) precisely so the sweep isolates the
cull-size variable -- the underlying sim/field draws never change between
combos being compared. The per-contest exponent uses a flat
implied_entries=10,000 assumption for every contest (matching the
methodology sim_evaluate_portfolios.py --build already uses), since
reconstructing each contest's true prize pool/entry fee isn't available
without the entries.csv either, and holding it flat/constant across the
whole sweep is what actually matters for isolating admit_n's effect.

Usage
-----
    python scripts/sweep_admit_n_scaling.py --slate 07262026
    python scripts/sweep_admit_n_scaling.py --recent 8
"""
import argparse
import sys
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
from analyze_candidate_pool import load_contest_player_fpts, load_real_field_points  # noqa: E402
from sim_evaluate_portfolios import build_slate  # noqa: E402

_FLAT_IMPLIED_ENTRIES = 10_000.0


def real_contest_sizes(archive_dir: Path) -> dict[str, int]:
    """{contest_name: n_entries} from the slate's own persisted sweep
    (risk 1), i.e. what was actually entered live -- no entries.csv needed."""
    import json
    sweep_path = archive_dir / "portfolio_sweep_draftkings.json"
    if not sweep_path.exists():
        raise FileNotFoundError(f"no portfolio_sweep_draftkings.json in {archive_dir}")
    sweep = json.loads(sweep_path.read_text())
    risk1 = next((e for e in sweep["sweep"] if e["risk"] == 1.0), sweep["sweep"][0])
    sizes: dict[str, int] = {}
    for lu in risk1["lineups"]:
        c = lu.get("contest_name") or "unknown"
        sizes[c] = sizes.get(c, 0) + 1
    return sizes


def build_groups(sizes: dict[str, int]) -> list:
    return [
        ep.ContestGroup(
            contest_id=name, contest_name=name, entry_fee_cents=400,
            prize_pool_cents=int(_FLAT_IMPLIED_ENTRIES * 400), single_entry_tag=(n == 1),
            roi_key="", entries=[(Path("x"), None)] * n,
        )
        for name, n in sizes.items()
    ]


def portfolio_metrics(lineups: list, fpts: dict, field_pts: np.ndarray) -> dict:
    scores = np.array([
        s for lu in lineups
        if (s := sum(fpts.get(int(p), 0) for p in lu.player_ids)) is not None
    ])
    n_field = len(field_pts)
    pct = np.searchsorted(field_pts, scores, side="right") / n_field
    return {"n": len(lineups), "mean_pctl": float(pct.mean()),
            "hit99": float((pct >= 0.99).mean()), "hit95": float((pct >= 0.95).mean())}


def run_slate(slate: str, sharpness: float, n_sims: int, field_size: int, seed: int) -> pd.DataFrame:
    d = PROJECT_ROOT / "archive" / slate
    sizes = real_contest_sizes(d)
    biggest_contest = max(sizes, key=sizes.get)
    print(f"\n=== {slate} ===  {len(sizes)} contests, sizes {sorted(sizes.values(), reverse=True)}, "
          f"biggest={biggest_contest!r} ({sizes[biggest_contest]} entries)")

    players_df, grids, name_to_id = build_slate(d)
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    sim = engine.simulate(n_sims)
    pid_index = {int(p): i for i, p in enumerate(sim.player_ids)}

    found = ep.discover_external_files(str(d))
    valid_ids = set(players_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids)
    print(f"  pool: {len(pool.lineups):,} lineups")

    groups = build_groups(sizes)
    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim)
    corr = ep.compute_pool_corr(pool.lineups, sim, scores=lineup_scores)

    n_half = n_sims // 2
    scores_A, scores_B = lineup_scores[:, :n_half], lineup_scores[:, n_half:2 * n_half]
    sims_A, sims_B = sim.results_matrix[:n_half], sim.results_matrix[n_half:2 * n_half]

    own_vec = players_df["ownership"].astype(float).to_numpy()
    cs = ContestSimulator()
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_size, rng_seed=seed + 100)
    field_B = cs.generate_field(players_df, own_vec, n_lineups=field_size, rng_seed=seed + 101)
    field_scores_A = cs.score_field(field_A, sims_A, pid_index)
    field_scores_B = cs.score_field(field_B, sims_B, pid_index)

    exponent = max(1.0, sharpness * _FLAT_IMPLIED_ENTRIES)
    exponents = {g.contest_id: exponent for g in groups}
    p_win_cull = ep.compute_p_win(scores_A, field_scores_A, exponents)
    p_win_select = ep.compute_p_win(scores_B, field_scores_B, exponents)

    fpts = load_contest_player_fpts(d)
    field_pts = load_real_field_points(d)

    rows = []
    for label, admit_n, mult in [
        ("no cull (disabled)", 0, 0.0),
        ("flat 250 (old default)", 250, 0.0),
        ("flat 500", 500, 0.0),
        ("flat 1000", 1000, 0.0),
        ("flat 1500", 1500, 0.0),
        ("flat 2000", 2000, 0.0),
        ("flat 3000", 3000, 0.0),
        ("flat 5000", 5000, 0.0),
        ("floor250 x12 (best scaled)", 250, 12.0),
    ]:
        alloc = ep.allocate_contests(
            pool, corr, groups, risk=3.0, evw_base=0.10, evw_max=0.40,
            ev_type="p_win", p_win_cull=p_win_cull, p_win_select=p_win_select,
            p_win_admit_n=admit_n, p_win_admit_multiplier=mult,
        )
        # allocate_contests doesn't tag lineups with contest_id, but its
        # portfolio list is one contiguous block per group IN GROUP ORDER,
        # each block sized by how many entries that group actually filled
        # -- NOT necessarily len(g.entries), if a contest ran dry. Offset
        # advance must use the real fill count, so only trust the
        # per-contest breakdown when nothing was left unfilled (true for
        # every admit_n tested here, since even the tightest reservoir
        # comfortably exceeds any single contest's need at this slate
        # scale -- but assumed, not hardcoded).
        agg = portfolio_metrics([lu for lu, _ in alloc.portfolio], fpts, field_pts)
        big_metrics = None
        if len(alloc.unfilled) == 0:
            offset = 0
            for g in groups:
                k = len(g.entries)
                block = [lu for lu, _ in alloc.portfolio[offset:offset + k]]
                if g.contest_id == biggest_contest:
                    big_metrics = portfolio_metrics(block, fpts, field_pts)
                offset += k
        else:
            print(f"    WARNING: {len(alloc.unfilled)} unfilled entries -- "
                  f"per-contest breakdown skipped for {label!r} (offsets would misalign)")
        rows.append({
            "slate": slate, "rule": label, "admit_n": admit_n, "multiplier": mult,
            "n_unfilled": len(alloc.unfilled), **{f"agg_{k}": v for k, v in agg.items()},
            **({f"big_{k}": v for k, v in big_metrics.items()} if big_metrics else {}),
        })
        print(f"  {label:20s} agg mean_pctl={agg['mean_pctl']:.3f} hit99={agg['hit99']:.3f}  "
              f"| big-contest mean_pctl={big_metrics['mean_pctl'] if big_metrics else float('nan'):.3f} "
              f"hit99={big_metrics['hit99'] if big_metrics else float('nan'):.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slate", action="append", default=[])
    p.add_argument("--recent", type=int, default=0)
    p.add_argument("--sharpness", type=float, default=0.05)
    p.add_argument("--n-sims", type=int, default=25_000)
    p.add_argument("--field-size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.recent:
        candidates = []
        for dd in sorted((PROJECT_ROOT / "archive").iterdir()):
            if not dd.is_dir():
                continue
            if ((dd / "portfolio_sweep_draftkings.json").exists()
                    and (dd / "contest_player_fpts.json").exists()
                    and list(dd.glob("contest-standings-*.zip"))):
                candidates.append(dd.name)
        slates = candidates[-args.recent:]
    else:
        slates = args.slate
    if not slates:
        print("No slates given (use --slate or --recent N).")
        sys.exit(1)

    all_rows = [run_slate(s, args.sharpness, args.n_sims, args.field_size, args.seed)
                for s in slates]
    df = pd.concat(all_rows, ignore_index=True)

    print("\n\n=== Aggregate portfolio: mean across slates, by rule ===")
    print(df.groupby("rule", sort=False)[["agg_mean_pctl", "agg_hit99"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Biggest contest only: mean across slates, by rule ===")
    print(df.groupby("rule", sort=False)[["big_mean_pctl", "big_hit99"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out = PROJECT_ROOT / "outputs" / "admit_n_scaling_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nWritten -> {out}")


if __name__ == "__main__":
    main()
