"""Would a win-space diversity term have REJECTED the lineups that actually won?

[[project-winspace-diversity]] established the signal is estimable at a top-1%
bar (rho_full 0.979, inside bulk Dn's own band) and genuinely independent of
bulk (rho_vs_bulk 0.853). That says it is MEASURABLE. It says nothing about
whether it is GOOD -- and there is a specific known counterexample: on the
08/11 slate, pool lineup #102 (the best realized lineup of 5,179) was
bulk-orthogonal (rank 1/1881) but tail-REDUNDANT (21.7th pctile), so a
win-space term would likely never have picked it.

THE QUESTION MUST BE FRAMED AS A PAIRED COMPARISON, or the test is rigged. A
diversity term is SUPPOSED to reject some high-EV lineups -- that is what
trading EV for spread means. The fair question is whether win-space rejects the
eventual winners MORE THAN BULK DOES, on the same lineups, with the same
reference set. So both arms are computed against one shared, currency-neutral
reference set (top-N by the pool's own ceiling score), and only the DELTA is
interpreted.

Metric: each elite lineup's PERCENTILE within the redundancy distribution.
Higher = more redundant = more likely a diversity term suppresses it. Reported
for both arms; `delta = winspace - bulk`, so POSITIVE delta means win-space is
the more hostile of the two to lineups that actually won.

WHAT THIS CAN AND CANNOT CONCLUDE. Selecting the top-K by REALIZED score
conditions on the outcome, so this is look-ahead by construction. It is a
SAFETY SCREEN: it can kill the idea, it can never validate it. A pass here
means "no evidence of harm", and benefit still has to be shown prospectively
per [[project-season-ev-program]].

Why it has power where the ROI work did not: it grades on realized-FPTS
ORDERING WITHIN THE POOL, not dollars -- no payout resolution, no standings
window, no ~0.47pp floor ([[project-8slate-cannot-measure-roi]]). 23 archived
slates carry both a pool export and a realized-FPTS map.

Checkpoint / resume per CLAUDE.md: one row per slate appended to
outputs/winspace_validity/results.csv; slates already on disk are skipped.

Usage
-----
    source venv/bin/activate
    python scripts/eval_winspace_validity.py 08102026 08112026 ...

Env vars
--------
    WSV_BAR        win-space bar as a field percentile (default 0.01 = top 1%)
    WSV_TAU        smoothing tau scale (default 1.0)
    WSV_REF        reference "already-picked" set size (default 50)
    WSV_TOPK       how many realized-elite lineups to grade (default 10)
    WSV_FIELD      opponent field size (default 10000)
    WSV_NSIMS      sim worlds (default 25000)
    WSV_FORCE      "1" re-runs slates already in results.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import wilcoxon

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402

BAR = float(os.environ.get("WSV_BAR", "0.01"))
TAU = float(os.environ.get("WSV_TAU", "1.0"))
REF_SIZE = int(os.environ.get("WSV_REF", "50"))
TOPK = int(os.environ.get("WSV_TOPK", "10"))
FIELD_N = int(os.environ.get("WSV_FIELD", "10000"))
N_SIMS = int(os.environ.get("WSV_NSIMS", "25000"))
FORCE = os.environ.get("WSV_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "winspace_validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def _append_and_reload(csv_path: Path, slate: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"slate": str})
        old = old[old["slate"] != slate]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"slate": str})


def _redundancy(X: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    """production's `sum_i max(r_i, 0)^2` against a fixed reference set."""
    A = X.astype(np.float32)
    A = A - A.mean(axis=1, keepdims=True)
    sd = np.sqrt((A * A).sum(axis=1))
    sd[sd <= 0] = np.inf
    A /= sd[:, None]
    R = A @ A[ref_idx].T
    np.maximum(R, 0.0, out=R)
    return np.square(R).sum(axis=1)


def _pctile_of(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Percentile rank (0-100) of `idx`'s entries within `values`."""
    order = np.argsort(np.argsort(values))
    return 100.0 * order[idx] / max(len(values) - 1, 1)


def run_slate(slate: str, cfg: dict) -> list[dict]:
    gpp, paths = cfg["gpp"], cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)
    adir = PROJECT_ROOT / "archive" / slate
    fpts_map = {
        int(k): float(v)
        for k, v in json.loads((adir / "contest_player_fpts.json").read_text())["player_fpts"].items()
    }
    found = ep.discover_external_files(str(adir))
    slate_df = DraftKingsSlateIngestor(str(adir / "DKSalaries.csv")).get_slate_dataframe()
    pool = ep.parse_lineup_pool(
        found["lineups_paths"], set(slate_df["player_id"].astype(int)), require_roi_blocks=False,
    )
    proj_ext = ep.parse_player_projections(found["projections_path"])
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, {int(p) for lu in pool.lineups for p in lu.player_ids},
        PipelineRunner._derive_opponent,
    )

    # Grade every pool lineup on realized FPTS. Lineups with any unmapped
    # player are dropped and COUNTED -- a silent drop here is exactly the
    # failure mode [[project-dksalaries-staleness-pool-drop]] records (an
    # archived morning DKSalaries silently losing 6.9% of the pool, including
    # its three best lineups), so it has to be visible in the output.
    graded, keep = [], []
    for i, lu in enumerate(pool.lineups):
        vals = [fpts_map.get(int(p)) for p in lu.player_ids]
        if any(v is None for v in vals):
            continue
        graded.append(float(sum(vals)))
        keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    realized = np.array(graded, dtype=np.float64)
    n_dropped = len(pool.lineups) - len(keep)
    lineups = [pool.lineups[i] for i in keep]
    if len(lineups) < REF_SIZE + TOPK + 10:
        raise RuntimeError(f"{slate}: only {len(lineups)} gradeable lineups")

    cache = OUT_DIR / f"sim_{slate}_{N_SIMS}_{seed}.npz"
    if cache.exists():
        with np.load(cache) as z:
            pid, mat = [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64)
    else:
        grids = ep.build_quantile_grids(
            proj_ext,
            zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
            scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
            mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
            mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)),
        )
        engine = SimulationEngine(
            EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
            batter_pca_model=None, score_grid=None, quantile_grids=grids,
        )
        st = np.random.get_state()
        np.random.seed(seed)
        sr = engine.simulate(N_SIMS)
        np.random.set_state(st)
        pid, mat = sr.player_ids, sr.results_matrix
        np.savez_compressed(cache, player_ids=np.asarray(pid, dtype=np.int64),
                            results_matrix=mat.astype(np.float32))

    class _SR:
        player_ids = pid
        results_matrix = mat

    cand = ep.compute_lineup_scores(lineups, _SR).astype(np.float32)      # (M, S)
    own = players_df["ownership"].astype(float).to_numpy()
    fpool = ep.build_topn_field_pool(players_df, own, FIELD_N, seed)
    col_map = {int(p): i for i, p in enumerate(pid)}
    fcols = np.array([[col_map[int(p)] for p in r] for r in fpool], dtype=np.int32)
    fs = ep._score_field_cols_batched(mat.astype(np.float32), fcols)      # (S, F) transient
    F = fs.shape[1]
    N = max(1, min(F, int(round(BAR * F))))
    lo, hi = ep._rung_bracket_ranks(N, F)
    part = np.partition(fs, np.unique(-np.array(sorted({N, lo, hi}))), axis=1)
    del fs
    thr = part[:, -N].astype(np.float32)
    tau = np.maximum(ep.smoothing_tau(part, N, F, TAU), ep._SMOOTH_TAU_FLOOR)
    del part
    X_win = expit(ep._LOGISTIC_NORMAL_SCALE * (cand - thr[None, :]) / tau[None, :]).astype(np.float32)

    # CURRENCY-NEUTRAL shared reference set: the pool's own ceiling score,
    # which is neither arm's currency. Both arms measure redundancy against
    # the SAME set, so only the delta is interpreted.
    ceil_scores = ep.compute_pool_ceiling_scores(
        type("P", (), {"lineups": lineups, "p99": getattr(pool, "p99", None)})(), players_df,
    ) if hasattr(pool, "p99") else None
    if ceil_scores is None or not np.isfinite(ceil_scores).any():
        ceil_scores = ep.compute_pool_proj_scores(lineups, players_df)
    ref_idx = np.argsort(-np.nan_to_num(ceil_scores, nan=-np.inf))[:REF_SIZE]

    red_bulk = _redundancy(cand, ref_idx)
    red_win = _redundancy(X_win, ref_idx)

    rest = np.setdiff1d(np.arange(len(lineups)), ref_idx)
    r_sorted = rest[np.argsort(-realized[rest])]
    elite = r_sorted[:TOPK]

    pb = _pctile_of(red_bulk[rest], np.searchsorted(np.sort(rest), elite))
    pw = _pctile_of(red_win[rest], np.searchsorted(np.sort(rest), elite))
    # searchsorted maps global -> position within `rest` (rest is sorted)
    rows = [{
        "slate": slate, "n_pool": len(pool.lineups), "n_graded": len(lineups),
        "n_dropped_ungradeable": n_dropped, "field_n": F, "bar_rank": N,
        "topk": TOPK, "ref_size": REF_SIZE,
        "best_realized": round(float(realized[elite[0]]), 2),
        "best_pctile_bulk": round(float(pb[0]), 1),
        "best_pctile_winspace": round(float(pw[0]), 1),
        "mean_pctile_bulk": round(float(pb.mean()), 1),
        "mean_pctile_winspace": round(float(pw.mean()), 1),
        "delta_mean_pctile": round(float(pw.mean() - pb.mean()), 1),
        "n_elite_worse_under_winspace": int((pw > pb).sum()),
    }]
    return rows


def main() -> None:
    slates = sys.argv[1:]
    if not slates:
        raise SystemExit("usage: eval_winspace_validity.py <slate> [<slate> ...]")
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"slate": str})["slate"])
    for sl in slates:
        if sl in done:
            print(f"[skip] {sl}")
            continue
        t0 = time.time()
        rows = run_slate(sl, cfg)
        _append_and_reload(RESULTS_CSV, sl, rows)
        r = rows[0]
        print(f"{sl}  pool {r['n_pool']} graded {r['n_graded']} "
              f"(dropped {r['n_dropped_ungradeable']})  bar N={r['bar_rank']}/{r['field_n']}  "
              f"({time.time()-t0:.0f}s)")
        print(f"    best realized {r['best_realized']}  redundancy pctile: "
              f"bulk {r['best_pctile_bulk']}  winspace {r['best_pctile_winspace']}")
        print(f"    top-{TOPK} mean pctile: bulk {r['mean_pctile_bulk']}  "
              f"winspace {r['mean_pctile_winspace']}  "
              f"delta {r['delta_mean_pctile']:+.1f}  "
              f"({r['n_elite_worse_under_winspace']}/{TOPK} worse under winspace)")

    df = pd.read_csv(RESULTS_CSV)
    print("\n=== VALIDITY SCREEN: is win-space more hostile to the winners than bulk? ===")
    print(df[["slate", "n_graded", "best_pctile_bulk", "best_pctile_winspace",
              "mean_pctile_bulk", "mean_pctile_winspace", "delta_mean_pctile",
              "n_elite_worse_under_winspace"]].to_string(index=False))
    d = df["delta_mean_pctile"].to_numpy()
    print(f"\n  mean delta (winspace - bulk) = {d.mean():+.1f} pctile points   "
          f"slates worse under winspace: {int((d > 0).sum())}/{len(d)}")
    if len(d) >= 5 and np.ptp(d) > 0:
        print(f"  Wilcoxon p = {wilcoxon(d).pvalue:.3f}")
    print("\n  POSITIVE delta = win-space suppresses eventual winners MORE than bulk (bad).")
    print("  This screen can only KILL the idea; benefit needs prospective testing.")


if __name__ == "__main__":
    main()
