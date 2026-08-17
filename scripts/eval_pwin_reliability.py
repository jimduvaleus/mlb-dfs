"""How settled is the p_win currency the selector actually acts on?

p_win is `mean_over_worlds(q**n)` where `q` is the candidate's percentile
against a SAMPLED opponent field of F lineups (`_field_percentiles`:
`searchsorted(sorted_field, score) / F`, a step-function empirical CDF) and
`n = sharpness * implied_entries`.

Propagating the field-sampling error of `q` through the exponentiation gives

    per-world relative error of q**n  ~  sqrt(n / F)

because `1 - q ~ 1/n` in the region that matters, so only `F/n` field lineups
separate "wins this world" from "doesn't". At production settings that is ~17
field lineups for a 29,762-entry contest -- the same rare-event estimability
problem `smooth_tau_scale` was built for in allocate_contests_topn_coverage,
in a different coordinate. This script measures whether the prediction is real.

THE METHODOLOGICAL TRAP THIS SCRIPT EXISTS TO AVOID: the obvious experiment --
split the SIM WORLDS in half and compare -- would miss the dominant error
entirely and report a falsely reassuring rho. `compute_p_win` scores ONE field
of F lineups across every world, so the component of the error driven by WHICH
lineups are in the field sample is common to all worlds and does not average
away with more sims. The field axis has to be split too.

Production already builds exactly the two independent estimates needed
(pipeline.py's p_win branch): `p_win_cull` from (field A, sims A) and
`p_win_select` from (field B, sims B), with disjoint sim halves and two
independently generated fields (rng_seed, rng_seed + 1). Their correlation IS
the reliability of the estimate production acts on, at production's real
per-stage budget -- no step-up needed, because the selector ranks on
`p_win_select` alone rather than on a pooled A+B average.

Four (field, sims) combinations give the full decomposition:

    AA vs BB   BOTH changed -- production's actual cull-vs-select guard
    AA vs AB   WORLDS only  -- same field, disjoint sim halves
    AA vs BA   FIELD only   -- same worlds, independently generated field

If the field-only arm degrades as much as the both arm, the error lives on the
field axis and no amount of n_sims fixes it.

Metrics, per contest per comparison:
  rho             Spearman over the post-floor pool
  top{admit_n}    overlap of the stage-A cull window (external_pool_pwin_
                  admit_n) -- the operative one: two independent draws
                  disagreeing here means the reservoir the diversity term gets
                  to draw from is itself partly noise
  top50           overlap of the very top, which the greedy eats first

Checkpoint / resume per CLAUDE.md: rows appended per contest to
outputs/pwin_reliability/results.csv; contests already on disk are skipped.

Usage
-----
    source venv/bin/activate
    python scripts/eval_pwin_reliability.py 2>&1 | tee /tmp/pwin.log

Env vars
--------
    TOPN_REQ_RAW      slate input dir (default data/raw)
    PWIN_REL_FORCE    "1" re-runs contests already in results.csv
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import parse_entry_file  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
FORCE = os.environ.get("PWIN_REL_FORCE") == "1"

OUT_DIR = PROJECT_ROOT / "outputs" / "pwin_reliability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"  # reuse the sim cache


def _append_and_reload(csv_path: Path, contest_id: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"contest_id": str})
        old = old[old["contest_id"] != contest_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"contest_id": str})


def _overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    if k <= 0:
        return float("nan")
    return len(set(np.argsort(-a)[:k].tolist()) & set(np.argsort(-b)[:k].tolist())) / k


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
    all_file_entries = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            all_file_entries.append((p, recs))
    groups = ep.group_and_match_contests(all_file_entries, pool)

    # Reuse the topn eval's cached sim and take the first n_sims worlds. Worlds
    # are exchangeable, so a 25,000-world slice of a 135,655-world draw is the
    # same object production simulates -- this only avoids re-running the sim.
    big = sorted((SHARED / "sim_cache").glob(f"{found['projections_path'].stem}_*_{seed}.npz"))
    if not big:
        raise SystemExit(
            "no cached sim -- run scripts/eval_topn_smoothed_exceedance.py first"
        )
    with np.load(big[-1]) as z:
        mat = z["results_matrix"][:n_sims].astype(np.float64)
        sim_results = SimulationResults([int(p) for p in z["player_ids"]], mat)
    print(f"sim {sim_results.results_matrix.shape} (sliced from {big[-1].name})")

    # --- production's exact p_win setup (pipeline.py p_win branch) --------
    sharpness = float(gpp.get("external_pool_pwin_sharpness", 0.05))
    flat_ref = float(gpp.get("external_pool_pwin_flat_reference", 0.0))
    field_size_cfg = int(gpp.get("external_pool_pwin_field_size", 0))
    field_n = field_size_cfg if field_size_cfg > 0 else ep.pwin_field_size(
        groups, floor=int(gpp.get("n_field_lineups", 5_000)),
    )
    exponents = ep.pwin_exponents(groups, sharpness, flat_ref)
    admit_n = int(gpp.get("external_pool_pwin_admit_n", 2000))
    admit_mult = float(gpp.get("external_pool_pwin_admit_multiplier", 0.0))

    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)  # (M, S)
    # Same post-floor eligibility the allocator applies before ranking, so rho
    # is measured over the population the cull actually sees.
    floor_scores = ep.compute_pool_ceiling_scores(pool, players_df)
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)),
    )
    elig = np.ones(len(pool.lineups), dtype=bool)
    if floor is not None:
        elig &= np.isfinite(floor_scores) & (floor_scores >= floor[0])
    eidx = np.where(elig)[0]
    print(f"pool {len(pool.lineups)}, post-floor {len(eidx)}, field_n {field_n}, "
          f"sharpness {sharpness}, flat_ref {flat_ref}, admit_n {admit_n}")

    n_half = n_sims // 2
    scores = {"A": lineup_scores[eidx][:, :n_half], "B": lineup_scores[eidx][:, n_half:2 * n_half]}
    sims = {"A": sim_results.results_matrix[:n_half],
            "B": sim_results.results_matrix[n_half:2 * n_half]}
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}

    cs = ContestSimulator()
    fields = {}
    for tag, s in (("A", seed), ("B", seed + 1)):
        fc = OUT_DIR / f"field_{found['projections_path'].stem}_{field_n}_{s}.npy"
        if fc.exists():
            fields[tag] = np.load(fc)
        else:
            t0 = time.time()
            fields[tag] = cs.generate_field(
                players_df, players_df["ownership"].astype(float).to_numpy(),
                n_lineups=field_n, rng_seed=s,
            )
            np.save(fc, fields[tag])
            print(f"  field {tag}: {len(fields[tag])} lineups ({time.time()-t0:.0f}s)")
    print(f"fields: A={len(fields['A'])}, B={len(fields['B'])}")

    # Four (field, sims) combinations. Each field-score array is
    # (n_half x field_n) float32 -- ~1.25 GB at production scale -- so they are
    # built and released ONE AT A TIME, keeping only the reduced (M,) p_win
    # vectors. Materializing all four would be ~5 GB for no reason.
    pw, se2 = {}, {}
    for ftag in ("A", "B"):
        for stag in ("A", "B"):
            t0 = time.time()
            fsc = cs.score_field(fields[ftag], sims[stag], col_map)   # (n_half, field_n)
            pw[(ftag, stag)], se2[(ftag, stag)] = ep.compute_p_win(
                scores[stag], fsc, exponents, return_var=True,
            )
            del fsc
            print(f"  p_win field={ftag} sims={stag}: {time.time()-t0:.0f}s")

    # SHRINKAGE ARMS. Each is a deterministic per-draw transform, applied
    # independently to both draws, so the comparison stays honest -- nothing
    # here lets one draw see the other.
    #
    # Both metrics below (Spearman rho, top-k overlap) are RANK-based and so
    # are invariant to the uniform component of the shrinkage. That is what
    # makes this test well posed: shrinking everything harder cannot inflate
    # them, and any movement must come from the heteroscedastic reordering.
    #
    # `shuffled` is the NEGATIVE CONTROL: identical marginal shrinkage, but
    # the candidate<->variance pairing is destroyed. If it moves the metrics
    # as much as the real arms, the effect is an artifact of shrinking rather
    # than of shrinking the right candidates.
    ARMS = ["raw", "shrunk_raw", "shrunk_log", "shuffled_raw"]
    shrunk: dict = {a: {} for a in ARMS}
    ctrl_rng = np.random.default_rng(seed + 77)
    for combo, per_contest in pw.items():
        for cid, vec in per_contest.items():
            v = se2[combo][cid]
            shrunk["raw"][(combo, cid)] = vec
            shrunk["shrunk_raw"][(combo, cid)] = ep.shrink_p_win(vec, v, "raw")[0]
            shrunk["shrunk_log"][(combo, cid)] = ep.shrink_p_win(vec, v, "log")[0]
            shrunk["shuffled_raw"][(combo, cid)] = ep.shrink_p_win(
                vec, v, "raw", se2_override=ctrl_rng.permutation(v),
            )[0]

    COMPARISONS = [
        ("both",       ("A", "A"), ("B", "B")),   # production's cull vs select
        ("worlds_only", ("A", "A"), ("A", "B")),
        ("field_only",  ("A", "A"), ("B", "A")),
    ]

    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        done = set(pd.read_csv(RESULTS_CSV, dtype={"contest_id": str})["contest_id"])

    implied = ep.pwin_implied_entries(groups)
    for g in groups:
        if not g.entries or g.contest_id in done:
            if g.entries:
                print(f"[skip] {g.contest_name}")
            continue
        n_exp = exponents[g.contest_id]
        eff_admit = max(admit_n, int(round(admit_mult * len(g.entries)))) if admit_mult > 0 else admit_n
        rows = []
        for label, ka, kb in COMPARISONS:
          for arm in ARMS:
            a = shrunk[arm][(ka, g.contest_id)]
            b = shrunk[arm][(kb, g.contest_id)]
            rho = float(spearmanr(a, b).statistic) if a.std() > 0 and b.std() > 0 else float("nan")
            _w = ep.shrink_p_win(
                pw[ka][g.contest_id], se2[ka][g.contest_id],
                "log" if arm == "shrunk_log" else "raw",
            )[1]
            rows.append({
                "arm": arm,
                "mean_shrink_w": round(float(np.mean(_w)), 4),
                "contest_id": g.contest_id, "contest": g.contest_name,
                "k": len(g.entries), "implied_entries": round(implied[g.contest_id]),
                "exponent": round(n_exp, 1), "field_n": field_n,
                # F/n = how many field lineups separate "wins" from "doesn't"
                "field_lineups_above_bar": round(field_n / n_exp, 1),
                "predicted_rel_err": round(float(np.sqrt(n_exp / field_n)), 4),
                "comparison": label,
                "rho": round(rho, 4),
                "top_admit_overlap": round(_overlap(a, b, eff_admit), 4),
                "top50_overlap": round(_overlap(a, b, 50), 4),
                "admit_n": eff_admit,
            })
        _append_and_reload(RESULTS_CSV, g.contest_id, rows)
        d = {(r["comparison"], r["arm"]): r for r in rows}
        print(f"{g.contest_name[:40]:<42} k={len(g.entries):<4} n={n_exp:<7.0f} "
              f"F/n={field_n/n_exp:<7.1f}")
        for arm in ARMS:
            r = d[("both", arm)]
            base = d[("both", "raw")]
            print(f"    both/{arm:<13} rho={r['rho']:.4f} ({r['rho']-base['rho']:+.4f})  "
                  f"top50={r['top50_overlap']*100:5.1f}% ({(r['top50_overlap']-base['top50_overlap'])*100:+5.1f})  "
                  f"top{eff_admit}={r['top_admit_overlap']*100:.1f}%  w={r['mean_shrink_w']:.3f}")

    df = pd.read_csv(RESULTS_CSV)
    print("\n=== SHRINKAGE TEST: agreement between the two independent draws ===")
    print("    (comparison='both' = production's p_win_cull vs p_win_select)")
    b = df[df.comparison == "both"]
    base_rho = np.average(b[b.arm == "raw"]["rho"], weights=b[b.arm == "raw"]["k"])
    base_t50 = np.average(b[b.arm == "raw"]["top50_overlap"], weights=b[b.arm == "raw"]["k"])
    for arm in ["raw", "shrunk_raw", "shrunk_log", "shuffled_raw"]:
        sub = b[b.arm == arm]
        if sub.empty:
            continue
        w = sub["k"]
        r = np.average(sub["rho"], weights=w)
        t = np.average(sub["top50_overlap"], weights=w)
        a = np.average(sub["top_admit_overlap"], weights=w)
        print(f"  {arm:<14} rho {r:.4f} ({r-base_rho:+.4f})   "
              f"top50 {t*100:5.1f}% ({(t-base_t50)*100:+5.1f})   "
              f"top_admit {a*100:5.1f}%   mean w {sub['mean_shrink_w'].mean():.3f}")
    print("\n  VERDICT KEY: a real effect needs shrunk_* > raw AND "
          "shrunk_* > shuffled_raw.\n  Rank metrics are invariant to uniform "
          "shrinkage, so movement can only come from reordering.")
    print("\n=== per-contest, by exponent (comparison='both') ===")
    piv = b.pivot_table(index=["contest", "k", "exponent"], columns="arm",
                        values="rho").sort_index(level="exponent")
    print(piv.round(4).to_string())
    print("\n=== axis decomposition (arm='raw') ===")
    for lab, sub in df[df.arm == "raw"].groupby("comparison"):
        w = sub["k"]
        print(f"  {lab:<12} rho {np.average(sub['rho'], weights=w):.3f}   "
              f"top_admit {np.average(sub['top_admit_overlap'], weights=w)*100:5.1f}%   "
              f"top50 {np.average(sub['top50_overlap'], weights=w)*100:5.1f}%")
    both = df[(df.comparison == "both") & (df.arm == "raw")].sort_values("exponent")
    print("\n=== predicted vs observed, by contest (comparison='both') ===")
    print(both[["contest", "k", "exponent", "field_lineups_above_bar",
                "predicted_rel_err", "rho", "top_admit_overlap"]].to_string(index=False))
    if both["predicted_rel_err"].std() > 0:
        r = spearmanr(both["predicted_rel_err"], both["rho"]).statistic
        print(f"\nSpearman(predicted_rel_err, observed rho) = {r:.3f}  "
              f"(strongly negative = the sqrt(n/F) prediction holds)")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
