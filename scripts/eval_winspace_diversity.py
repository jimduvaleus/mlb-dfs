"""Is win-space diversity ESTIMABLE once the crossing indicator is smoothed?

Production's spread mechanism is `Dn`, built on `compute_pool_corr` -- a BULK
POINTS-SPACE correlation over every simulated world. But the payout only cares
about co-movement in the worlds that pay. Two lineups can be bulk-uncorrelated
and still boom together in exactly those worlds (same leverage stack), or be
bulk-correlated and never co-boom. Bulk correlation cannot tell them apart.

Moving diversity into win space was proposed before and DISCONFIRMED, but the
recorded root cause was ESTIMABILITY, not validity: "bulk corr rests on 25k
worlds, #102's win-world signature on 15 crossing events" -- and the note ends
"Never measured: split-half reliability of the tail-space overlap." That is the
measurement this script takes, now that smoothed exceedance exists to attack
exactly the 15-events problem.

Arms, all differing ONLY in the per-world quantity whose correlation is taken:

    bulk             the lineup's raw simulated score        (production)
    winspace_hard    1[score >= rank-N field threshold]      (the disconfirmed form)
    winspace_smooth  P(threshold <= score) via external_pool.smoothing_tau
                     at tau_scale 1.0 and 2.0

PROTOCOL mirrors how bulk Dn's reliability was established, so the numbers are
comparable to its 0.976-0.999: hold an already-picked reference set FIXED, then
recompute every remaining candidate's redundancy to that set --
`sum_i max(r_i, 0)^2`, production's own formula -- from two DISJOINT halves of
the sim worlds, and correlate the two orderings (Spearman, then Spearman-Brown
stepped up to the full world budget).

TWO-AXIS VERDICT, because reliability alone is not enough:

  rho_full        is the win-space ordering estimable at all? Must approach
                  bulk's 0.976-0.999 to be usable.
  rho_vs_bulk     is it measuring anything NEW? If it correlates ~1.0 with the
                  bulk ordering it is reliable but redundant, and there is no
                  point in replacing anything. The #102 anecdote (bulk-
                  orthogonal, tail-redundant) predicts real divergence.

A usable result needs BOTH: high rho_full AND rho_vs_bulk well below 1.

`degenerate_pct` is reported alongside -- the share of candidates whose
per-world vector has zero variance within a half (a hard indicator that never
fires, so no correlation is even defined). That is the estimability failure
made visible, and smoothing should drive it to zero by construction.

Checkpoint / resume per CLAUDE.md: rows appended per contest to
outputs/winspace_diversity/results.csv; contests already on disk are skipped.

Usage
-----
    source venv/bin/activate
    python scripts/eval_winspace_diversity.py 2>&1 | tee /tmp/ws.log

Env vars
--------
    TOPN_REQ_RAW      slate input dir (default data/raw)
    WS_REF_SIZE       reference "already-picked" set size (default 50)
    WS_TAUS           comma-separated tau scales (default "1.0,2.0")
    WS_FORCE          "1" re-runs contests already in results.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import parse_entry_file  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
REF_SIZE = int(os.environ.get("WS_REF_SIZE", "50"))
TAUS = [float(x) for x in os.environ.get("WS_TAUS", "1.0,2.0").split(",")]
FORCE = os.environ.get("WS_FORCE") == "1"
# Bar sweep. "prod" = the effective rank the topn allocator uses; a float is a
# PERCENTILE of the contest's own field (0.01 = top 1%). Reliability turned out
# to track how extreme the bar is rather than how rare the crossings are, so
# the open question is whether a bar exists that is loose enough to estimate
# and still tight enough to be payout-relevant.
BARS = [x.strip() for x in os.environ.get("WS_BARS", "prod").split(",")]

OUT_DIR = PROJECT_ROOT / "outputs" / "winspace_diversity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"


def _append_and_reload(csv_path: Path, contest_id: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"contest_id": str})


def _spearman_brown(r: float) -> float:
    if not np.isfinite(r):
        return float("nan")
    d = 1.0 + r
    return float(2.0 * r / d) if d > 1e-9 else float("nan")


def _redundancy_to_ref(X: np.ndarray, ref_idx: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """production's `sum_i max(r_i, 0)^2` for every row of `X` against the
    fixed reference set, computed on the world subset `cols`.

    Zero-variance rows correlate with nothing, so they contribute 0 -- the
    honest reading of "this candidate never moves in these worlds", and the
    case a hard indicator produces en masse."""
    A = X[:, cols].astype(np.float32)
    A = A - A.mean(axis=1, keepdims=True)
    sd = np.sqrt((A * A).sum(axis=1))
    sd[sd <= 0] = np.inf                      # -> correlation 0, not NaN
    A /= sd[:, None]
    R = A @ A[ref_idx].T                       # (M, n_ref) correlations
    np.maximum(R, 0.0, out=R)
    return np.square(R).sum(axis=1)


def _degenerate_pct(X: np.ndarray, cols: np.ndarray) -> float:
    sub = X[:, cols]
    return float(np.mean(sub.std(axis=1) <= 0) * 100.0)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp, paths = cfg["gpp"], cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)
    n_sims = int(cfg["simulation"].get("n_sims", 25_000))

    found = ep.discover_external_files(RAW_DIR)
    slate_df = DraftKingsSlateIngestor(str(PROJECT_ROOT / paths["dk_slate"])).get_slate_dataframe()
    pool = ep.parse_lineup_pool(
        found["lineups_paths"], set(slate_df["player_id"].astype(int)), require_roi_blocks=False,
    )
    proj_ext = ep.parse_player_projections(found["projections_path"])
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, {int(p) for lu in pool.lineups for p in lu.player_ids},
        PipelineRunner._derive_opponent,
    )
    fe = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            fe.append((p, recs))
    groups = ep.group_and_match_contests(fe, pool)

    big = sorted((SHARED / "sim_cache").glob(f"{found['projections_path'].stem}_*_{seed}.npz"))
    if not big:
        raise SystemExit("no cached sim -- run scripts/eval_topn_smoothed_exceedance.py first")
    with np.load(big[-1]) as z:
        sim_results = SimulationResults(
            [int(p) for p in z["player_ids"]], z["results_matrix"][:n_sims].astype(np.float64),
        )

    floor_scores = ep.compute_pool_ceiling_scores(pool, players_df)
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)))
    elig = np.ones(len(pool.lineups), dtype=bool)
    if floor is not None:
        elig &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    elig_lineups = [lu for lu, e in zip(pool.lineups, elig) if e]
    cand = ep.compute_lineup_scores(elig_lineups, sim_results).astype(np.float32)  # (M, S)
    M = cand.shape[0]
    print(f"sim {sim_results.results_matrix.shape}  post-floor candidates {M}  ref_size {REF_SIZE}")

    fp_size = int(gpp.get("external_pool_topn_field_pool_size", 25_000))
    own_vec = players_df["ownership"].astype(float).to_numpy()
    fp_cache = SHARED / f"field_pool_{found['projections_path'].stem}_{fp_size}_{seed}.npy"
    field_lineups = np.load(fp_cache) if fp_cache.exists() else ep.build_topn_field_pool(
        players_df, own_vec, fp_size, seed)
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    fcols = np.array([[col_map[int(p)] for p in r] for r in field_lineups], dtype=np.int32)

    rank = int(gpp.get("external_pool_topn_rank", 10))
    pct_floor = float(gpp.get("external_pool_topn_percentile_floor", 0.001))
    rng = np.random.default_rng(seed + 909)
    perm = rng.permutation(n_sims)
    h1, h2 = perm[: n_sims // 2], perm[n_sims // 2:]

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"])

    for g in groups:
        if not g.entries or g.contest_id in done:
            continue
        t0 = time.time()
        F = ep._topn_field_size_for_group(g, fcols.shape[0])
        sub = rng.choice(fcols.shape[0], size=F, replace=False)
        # (n_sims, F) transient -- built, used for thresholds, discarded.
        fs = ep._score_field_cols_batched(
            sim_results.results_matrix.astype(np.float32), fcols[sub])
        bar_ranks = {}
        for b in BARS:
            Nb = (ep._topn_effective_rank(rank, F, pct_floor) if b == "prod"
                  else max(1, min(F, int(round(float(b) * F)))))
            bar_ranks[b] = Nb
        kths = set()
        for Nb in bar_ranks.values():
            lo, hi = ep._rung_bracket_ranks(Nb, F)
            kths |= {Nb, lo, hi}
        part = np.partition(fs, np.unique(-np.array(sorted(kths))), axis=1)
        del fs

        arms = {"bulk": cand}
        arm_bar = {"bulk": "-"}
        for b, Nb in bar_ranks.items():
            thr = part[:, -Nb].astype(np.float32)
            nm = f"hard@{b}"
            arms[nm] = (cand >= thr[None, :]).astype(np.float32)
            arm_bar[nm] = f"{b}(N={Nb})"
            for ts in TAUS:
                tau = np.maximum(ep.smoothing_tau(part, Nb, F, ts), ep._SMOOTH_TAU_FLOOR)
                nm = f"smooth{ts:g}@{b}"
                arms[nm] = expit(
                    ep._LOGISTIC_NORMAL_SCALE * (cand - thr[None, :]) / tau[None, :]
                ).astype(np.float32)
                arm_bar[nm] = f"{b}(N={Nb})"
        del part
        N = bar_ranks[BARS[0]]

        # Reference "already-picked" set: the top candidates by mean smoothed
        # crossing rate -- a stand-in for what the selector fills early.
        rank_by = arms[f"smooth{TAUS[0]:g}@{BARS[0]}"].mean(axis=1)
        ref_idx = np.argsort(-rank_by)[:REF_SIZE]
        rest = np.setdiff1d(np.arange(M), ref_idx)

        red = {}
        rows = []
        for name, X in arms.items():
            r1 = _redundancy_to_ref(X, ref_idx, h1)[rest]
            r2 = _redundancy_to_ref(X, ref_idx, h2)[rest]
            red[name] = _redundancy_to_ref(X, ref_idx, perm)[rest]
            rho = float(spearmanr(r1, r2).statistic) if r1.std() > 0 and r2.std() > 0 else float("nan")
            rows.append({
                "contest_id": g.contest_id, "contest": g.contest_name,
                "k": len(g.entries), "field_size": F, "N": N, "arm": name,
                "bar": arm_bar[name],
                "rho_half": round(rho, 4), "rho_full": round(_spearman_brown(rho), 4),
                "degenerate_pct": round(_degenerate_pct(X, h1), 2),
                "ref_size": REF_SIZE,
            })
        for r in rows:
            b = red["bulk"]
            x = red[r["arm"]]
            r["rho_vs_bulk"] = round(
                float(spearmanr(x, b).statistic) if x.std() > 0 and b.std() > 0 else float("nan"), 4)
        _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        print(f"\n{g.contest_name[:46]:<48} k={len(g.entries):<4} F={F} N={N} "
              f"({time.time()-t0:.0f}s)")
        for r in rows:
            print(f"    {r['arm']:<20} rho_full={r['rho_full']:.4f}  "
                  f"vs_bulk={r['rho_vs_bulk']:+.3f}  degenerate={r['degenerate_pct']:.1f}%")
        del arms, red

    df = pd.read_csv(RESULTS_CSV)
    print("\n=== WIN-SPACE DIVERSITY: estimable, and is it new? (entry-weighted) ===")
    for arm, s in df.groupby("arm"):
        w = s["k"]
        print(f"  {arm:<20} rho_full {np.average(s['rho_full'], weights=w):.4f}   "
              f"vs_bulk {np.average(s['rho_vs_bulk'], weights=w):+.3f}   "
              f"degenerate {np.average(s['degenerate_pct'], weights=w):5.1f}%")
    print("\n  Bulk Dn's established reliability is 0.976-0.999. A usable win-space")
    print("  term needs rho_full near that AND vs_bulk well below 1 (reliable AND new).")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
