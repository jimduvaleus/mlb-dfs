"""Could a purpose-built pool have reached the good quadrant BEFORE lock?

`analyze_ev_drivers.py` showed the +EV corner (high lineup ceiling, low
ownership) held 21 of the contest's 3,335 real entries. A follow-on question:
that emptiness might be a property of what the field chose to build, not of
what was buildable. This generates our own pool from the same pre-lock inputs,
places it on the same two axes, and scores the selections against the REAL
3,335-entry field with the real payout ladder.

Everything used to SELECT is pre-lock:
  ceiling  -- lineup p99.9 from SimulationEngine (projections + copula; the sim
              never sees the contest). NOT the SaberSim per-player p99 column,
              which ignores correlation and was measured dead (rho +0.010).
  ownership-- SaberSim's projected `Adj Own`, summed over the roster.
Only the GRADING uses post-contest data (the real field's lineups + ladder).

Each candidate is graded as a marginal 3,336th entry: ranked against the 3,335
real entries, ladder applied. The real entries in the comparison table were
ranked within a 3,335-entry field, a one-slot difference that is immaterial
here but worth knowing when reading the two side by side.

Usage
-----
    source venv/bin/activate
    python scripts/prelock_quadrant_pool.py --slate archive/08252026 \
        --entries outputs/contest_sim_roi/entries_sim_roi.csv \
        --payout-table outputs/contest_sim_roi/me_warmup_payouts.txt \
        --n-candidates 20000 --n-sims 200000
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

from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization.candidate_generator import CandidateGenerator  # noqa: E402
from src.optimization.optimal_lineups import (  # noqa: E402
    generate_sim_optimal_lineups, stratified_sim_sample,
)
from analyze_contest_sim_roi import build_slate, parse_payout_table  # noqa: E402
from analyze_ev_drivers import parse_names  # noqa: E402


def z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--payout-table", required=True)
    ap.add_argument("--entry-fee", type=float, default=333.0)
    ap.add_argument("--n-candidates", type=int, default=20_000)
    ap.add_argument("--n-sims", type=int, default=200_000)
    ap.add_argument("--sim-batch", type=int, default=25_000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--select", type=int, default=150)
    ap.add_argument("--mutants-per-anchor", type=int, default=4,
                    help="shape-preserving mutants grown off each ILP anchor "
                         "(candidate_generator.generate_shape_mutants). CBC "
                         "solves were ~half the wall clock of the first build; "
                         "mutating a smaller anchor set is far cheaper and "
                         "keeps the anchors' stack shape exactly.")
    ap.add_argument("--n-sim-optimals", type=int, default=3_000,
                    help="per-world ILP ceiling seeds mixed into the pool; the "
                         "raw generator is a diversity engine and on its own "
                         "produces a pool weaker than the real field")
    ap.add_argument("--salary-floor", type=float, default=48_700.0,
                    help="real entrants play near the cap; without this the pool "
                         "fills with lineups nobody would submit and the "
                         "comparison against the field is meaningless")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", default="outputs/contest_sim_roi")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = PROJECT_ROOT / args.out_dir

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    print("[1/5] slate")
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    own_pct = players_df["ownership"].fillna(0.0).to_numpy()   # projected, pct points

    print(f"[2/5] generating {args.n_candidates:,} candidates from projected "
          f"ownership (salary floor ${args.salary_floor:,.0f})")
    gen = CandidateGenerator(players_df, own_pct, rng_seed=args.seed,
                             salary_floor=args.salary_floor)
    cands = gen.generate(n_candidates=args.n_candidates)
    print(f"      {len(cands):,} from the sampler")
    if args.n_sim_optimals > 0:
        seed_sim = engine.simulate(max(args.n_sim_optimals * 5, 10_000))
        df_ilp = players_df.copy()
        df_ilp["eligible_positions"] = df_ilp["eligible_positions"].map(
            lambda x: list(dict.fromkeys(
                "P" if t.strip().upper() in ("SP", "RP", "P") else t.strip().upper()
                for t in str(x).split("/"))))
        rng = np.random.default_rng(args.seed)
        idx = [i for i, _ in stratified_sim_sample(
            seed_sim.results_matrix, args.n_sim_optimals, rng)]
        seen = {frozenset(int(p) for p in lu.player_ids) for lu in cands}
        seeds = generate_sim_optimal_lineups(
            df_ilp, seed_sim.results_matrix, list(seed_sim.player_ids), idx,
            min_stack=5, salary_floor=49_500.0, seen=seen,
            progress_cb=lambda n: (n % 500 == 0) and print(f"      ILP {n}/{len(idx)}"),
        )
        print(f"      +{len(seeds):,} per-world ILP ceiling seeds")
        cands = list(cands) + list(seeds)
        if args.mutants_per_anchor > 0 and seeds:
            seen2 = {frozenset(int(p) for p in lu.player_ids) for lu in cands}
            muts = gen.generate_shape_mutants(
                seeds, n_per_parent=args.mutants_per_anchor, seen=seen2,
                rng_seed=args.seed + 5, salary_floor=args.salary_floor,
                n_workers=0,
            )
            print(f"      +{len(muts):,} shape-preserving mutants of those seeds")
            cands = cands + list(muts)
        del seed_sim
    print(f"      {len(cands):,} candidates in the pool")

    entries = pd.read_csv(args.entries)
    entries["names"] = entries["lineup_raw"].map(parse_names)
    payout = parse_payout_table(Path(args.payout_table).read_text(), len(entries))
    n_field = len(entries)
    n_paid = int((payout > 0).sum())

    n_players = len(pid_index)
    Ffield = np.zeros((n_field, n_players), dtype=np.float32)
    for r, ns in enumerate(entries["names"]):
        for n in ns:
            Ffield[r, pid_index[name_to_id[n]]] = 1.0
    C = np.zeros((len(cands), n_players), dtype=np.float32)
    for r, lu in enumerate(cands):
        for p in lu.player_ids:
            C[r, pid_index[int(p)]] = 1.0
    cand_own = C @ own_pct
    field_own_proj = Ffield @ own_pct
    sal_v = players_df.set_index("player_id")["salary"].reindex(
        [int(p) for p in engine.players_df["player_id"]]).to_numpy(float)
    print(f"      mean salary used -- pool ${(C @ sal_v).mean():,.0f} vs "
          f"real field ${(Ffield @ sal_v).mean():,.0f}")

    print(f"[3/5] simulating {args.n_sims:,} worlds for ceilings")
    # Ceiling needs order statistics, so a fixed sample is retained rather than
    # streamed; 25k worlds resolves p99.9 to ~25 exceedances per lineup.
    np.random.seed(args.seed)
    keep_c, keep_f, kept = [], [], 0
    done = 0
    while done < args.n_sims:
        b = min(args.sim_batch, args.n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix.astype(np.float32)
        if kept < 25_000:
            take = min(25_000 - kept, b)
            keep_c.append(sc[:take] @ C.T)
            keep_f.append(sc[:take] @ Ffield.T)
            kept += take
        done += b
        print(f"      {done:,}/{args.n_sims:,}")
        del sim, sc
    CS_s = np.concatenate(keep_c, 0)
    FS_s = np.concatenate(keep_f, 0)
    cand_ceiling = np.percentile(CS_s, 99.9, axis=0)
    field_ceiling = np.percentile(FS_s, 99.9, axis=0)
    del keep_c, keep_f

    print("[4/5] selecting on PRE-LOCK axes only")
    score = z(cand_ceiling) - z(cand_own)
    k = args.select
    # The quadrant is a CONJUNCTION ("high ceiling AND low ownership"), not a
    # difference of z-scores. The difference degenerates: ownership can always
    # be lowered by rostering worse players, so -z(own) walks the selection
    # straight out of the buildable region. Gate on ceiling first, against the
    # REAL field's ceiling distribution, then take the least-owned survivors.
    gate = np.percentile(field_ceiling, 60)
    elig = np.where(cand_ceiling >= gate)[0]
    print(f"      {len(elig):,}/{len(cands):,} candidates clear the field's "
          f"60th-pct ceiling ({gate:.1f})")
    quad = elig[np.argsort(cand_own[elig])[:k]] if len(elig) >= k else elig
    med = np.where(cand_ceiling >= np.median(field_ceiling))[0]
    rng2 = np.random.default_rng(args.seed)
    picks = {
        "pool: QUADRANT (ceiling gate, then lowest proj own)": quad,
        "pool: highest ceiling only": np.argsort(-cand_ceiling)[:k],
        "pool: z(ceiling) - z(own)": np.argsort(-score)[:k],
        "pool: lowest proj own only": np.argsort(cand_own)[:k],
        "pool: field-quality control (ceiling >= field median)":
            rng2.choice(med, min(k, len(med)), replace=False),
        "pool: random control": rng2.choice(len(cands), k, replace=False),
    }
    sel = sorted({int(i) for v in picks.values() for i in v})
    sel_pos = {c: j for j, c in enumerate(sel)}
    Csel = C[sel]
    print(f"      {len(sel)} distinct candidates to grade")

    print(f"[5/5] grading against the real {n_field:,}-entry field")
    gross = np.zeros(len(sel))
    np.random.seed(args.seed + 1)
    done = 0
    BIG = 1e6
    while done < args.n_sims:
        b = min(args.sim_batch, args.n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix.astype(np.float32)
        for st in range(0, b, args.chunk):
            s = sc[st:st + args.chunk]
            c = s.shape[0]
            FS = np.sort((s @ Ffield.T).astype(np.float64), axis=1)   # ascending
            CSb = (s @ Csel.T).astype(np.float64)
            # One global searchsorted instead of a per-world Python loop: each
            # world's block is offset into its own disjoint value range, so the
            # concatenation is already globally sorted.
            offs = (np.arange(c) * BIG)[:, None]
            idx = np.searchsorted((FS + offs).ravel(), (CSb + offs).ravel(),
                                  side="right")
            n_le = idx - np.repeat(np.arange(c) * n_field, Csel.shape[0])
            rank = n_field - n_le            # 0-based rank among the field
            pay = np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0)
            gross += pay.reshape(c, -1).sum(axis=0)
        done += b
        print(f"      {done:,}/{args.n_sims:,}")
        del sim, sc
    cand_roi_sel = (gross / args.n_sims - args.entry_fee) / args.entry_fee

    fee = args.entry_fee
    real = pd.read_csv(args.entries)
    print("\n=== WHERE EACH GROUP LANDS ON THE PRE-LOCK AXES, AND WHAT IT'S WORTH ===")
    cq = pd.Series(cand_ceiling).rank(pct=True)
    oq = pd.Series(cand_own).rank(pct=True)
    rows = []
    for label, idx in picks.items():
        r = np.array([cand_roi_sel[sel_pos[int(i)]] for i in idx])
        rows.append({
            "group": label, "n": len(idx),
            "ceiling": cand_ceiling[idx].mean(),
            "proj_own": cand_own[idx].mean(),
            "mean_roi": r.mean(), "pct_pos": 100 * (r > 0).mean(),
            "best_roi": r.max(),
        })
    rows.append({"group": f"REAL FIELD (all {n_field:,} entries)", "n": n_field,
                 "ceiling": field_ceiling.mean(), "proj_own": field_own_proj.mean(),
                 "mean_roi": real["roi"].mean(),
                 "pct_pos": 100 * (real["roi"] > 0).mean(),
                 "best_roi": real["roi"].max()})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== DID THE POOL REACH DEEPER INTO THE QUADRANT THAN THE FIELD? ===")
    fc = pd.Series(field_ceiling).rank(pct=True)
    fo = pd.Series(field_own_proj).rank(pct=True)
    good_f = int(((fc >= 0.6) & (fo <= 0.4)).sum())
    good_c = int(((cq >= 0.6) & (oq <= 0.4)).sum())
    print(f"  real field : {good_f:5d} / {n_field:,} entries "
          f"({100*good_f/n_field:.1f}%) in top-40% ceiling and bottom-40% proj own")
    print(f"  our pool   : {good_c:5d} / {len(cands):,} candidates "
          f"({100*good_c/len(cands):.1f}%) in the same box")

    pd.DataFrame({
        "cand_index": sel,
        "ceiling_p999": cand_ceiling[sel],
        "proj_own_sum": cand_own[sel],
        "prelock_score": score[sel],
        "roi": cand_roi_sel,
        "lineup": ["|".join(str(int(p)) for p in cands[i].player_ids) for i in sel],
    }).sort_values("roi", ascending=False).to_csv(
        out_dir / "prelock_pool_graded.csv", index=False)
    print(f"\nwrote {out_dir}/prelock_pool_graded.csv")


if __name__ == "__main__":
    main()
