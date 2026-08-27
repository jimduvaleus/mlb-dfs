"""Build a 150-lineup portfolio from pre-lock inputs, several ways, and grade them.

Six selection arms plus three controls, all fed the SAME pool, the SAME ceiling
gate and the SAME simulated contest context, so the only thing that differs
between them is how they price self-competition:

    dr            exact marginal reward with the demotion term
    kelly         E[log(bankroll + portfolio payout)]
    emax          E[max_i payout_i] -- best-entry coverage
    coverage      greedy max-coverage of ceiling worlds
    determinant   EV + low pairwise correlation (the proxy production ships)
    gate_then_own the measured incumbent: gate, then lowest projected ownership
    zdiff         NEGATIVE CONTROL: z(ceiling) - z(own)   -- expect ~ -15.8%
    lowown        NEGATIVE CONTROL: lowest ownership only -- expect ~ -99%
    random        control

Grading (--contest) inserts all 150 at once against a real archived field, so
demotion between your own entries is priced. Marginal ROI is reported alongside
for comparability with the earlier analysis, but it is structurally blind to
everything the arms differ on -- 50 identical lineups grade +7.6% marginal and
-60.1% portfolio.

Grading worlds are drawn with a different seed from the build worlds, so the
arms are graded out-of-sample relative to the selection.

Usage
-----
    source venv/bin/activate
    python scripts/build_fast_portfolio.py --slate archive/08252026 \
        --payout-table outputs/contest_sim_roi/me_warmup_payouts.txt \
        --field-size 3335 --entry-fee 333 \
        --contest ME:outputs/contest_sim_roi/entries_sim_roi.csv:outputs/contest_sim_roi/me_warmup_payouts.txt:333
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization import fast_portfolio as fp  # noqa: E402
from src.api import external_pool as ep  # noqa: E402
from analyze_contest_sim_roi import build_slate, parse_payout_table  # noqa: E402
import portfolio_grading as pg  # noqa: E402
from analyze_rival_portfolio import (  # noqa: E402
    overlap_profile, exposure_profile, cluster_decomposition, primary_teams,
)

ALL_ARMS = ["dr", "kelly", "emax", "coverage", "determinant",
            "gate_then_own", "zdiff", "lowown", "random"]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class Clock:
    def __init__(self):
        self.rows, self.t0 = [], time.perf_counter()

    def mark(self, stage: str):
        now = time.perf_counter()
        self.rows.append({"stage": stage, "seconds": now - self.t0,
                          "peak_rss_gb": rss_gb()})
        print(f"      [{stage}] {now - self.t0:6.1f}s  peak {rss_gb():.2f}GB")
        self.t0 = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--payout-table", required=True)
    ap.add_argument("--field-size", type=int, default=15_000)
    ap.add_argument("--entry-fee", type=float, default=333.0)
    ap.add_argument("--n-candidates", type=int, default=30_000)
    ap.add_argument("--n-anchors", type=int, default=800)
    ap.add_argument("--shortlist", type=int, default=4_000)
    ap.add_argument("--ceiling-worlds", type=int, default=25_000)
    ap.add_argument("--contest-worlds", type=int, default=12_500)
    ap.add_argument("--portfolio-size", type=int, default=150)
    ap.add_argument("--arm", action="append", default=None,
                    help=f"repeatable; default all of {ALL_ARMS}")
    ap.add_argument("--contest", action="append", default=[],
                    help="LABEL:ENTRIES_CSV:PAYOUT_TXT:FEE — grade against a real field")
    ap.add_argument("--grade-sims", type=int, default=40_000)
    ap.add_argument("--exclude-overlap", default=None,
                    help="PORTFOLIOS_CSV:ARM:GAMMA — drop shortlist candidates "
                         "sharing more than GAMMA players with any lineup in "
                         "that already-built portfolio. Cross-contest "
                         "diversification: REBUILDS to the full portfolio size "
                         "under the constraint rather than deleting entries, so "
                         "the marginal distribution is preserved and the "
                         "decorrelation effect is not confounded by simply "
                         "playing fewer lineups.")
    ap.add_argument("--gate-currency", choices=["abs", "rank"], default="abs",
                    help="what the ceiling gate concentrates on: 'abs' = p99.9 "
                         "of raw points (current); 'rank' = worlds cleared "
                         "against a per-world pool bar. The ladder pays rank.")
    ap.add_argument("--mutants-per-anchor", type=int, default=10,
                    help="shape-preserving mutants grown off each ILP anchor. "
                         "0 disables. Measured 08/26: mutants hit the pool's "
                         "top-5%% ceiling band at 3.3%%, BELOW the sampler's "
                         "5.1%% and far below the anchors' 17.5%% — perturbing "
                         "a per-world optimum mostly moves you off it.")
    ap.add_argument("--shortlist-mode", choices=["gate", "random"], default="gate",
                    help="'random' bypasses BOTH gates and hands the arms an "
                         "unbiased sample of the pool — the control for whether "
                         "the objectives find the ceiling/ownership quadrant on "
                         "their own.")
    ap.add_argument("--own-gate-pct", type=float, default=40.0,
                    help="Gate B width: keep this %% of the ceiling survivors by "
                         "ownership. 100 = no ownership gate at all.")
    ap.add_argument("--own-metric", choices=["sum", "log"], default="sum",
                    help="ownership currency for the gate and the "
                         "ownership-ranked arms; see fast_portfolio."
                         "ownership_currency")
    ap.add_argument("--real-own-csv", default=None,
                    help="player_sim_roi.csv supplying REALIZED dk_drafted_pct")
    ap.add_argument("--oracle", choices=["none", "field", "all"], default="none",
                    help="none: projected ownership everywhere (the only "
                         "honest, playable setting). field: projected for pool "
                         "and gate, REALIZED for the simulated opponent field "
                         "— isolates what a perfect FIELD model would buy. "
                         "all: realized everywhere — also lets the gate and the "
                         "pool sampler see the future, so it upper-bounds the "
                         "total value of a perfect ownership projection. Both "
                         "oracle modes are hindsight and unplayable; they exist "
                         "to bound the headroom, not to be used.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", default="outputs/fast_portfolio")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    arms = args.arm or ALL_ARMS
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clk = Clock()

    cfg = fp.FastPortfolioConfig(
        n_candidates=args.n_candidates, n_anchors=args.n_anchors,
        target_shortlist=args.shortlist, ceiling_worlds=args.ceiling_worlds,
        contest_worlds=args.contest_worlds, field_size=args.field_size,
        portfolio_size=args.portfolio_size, seed=args.seed,
        own_gate_pct=args.own_gate_pct,
        mutants_per_anchor=args.mutants_per_anchor,
    )

    print("[1/7] slate + engine")
    ycfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), ycfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / ycfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    own_pct = players_df["ownership"].fillna(0.0).to_numpy()

    # Ownership enters in THREE places — the pool sampler's weights, the gate's
    # ranking, and the simulated opponent field — so a projection error
    # compounds rather than acting once. The oracle modes swap in realized
    # %Drafted to bound how much of the loss is recoverable, and which stage
    # it is recoverable at.
    own_pool, own_field = own_pct, own_pct
    if args.oracle != "none":
        if not args.real_own_csv:
            raise SystemExit("--oracle needs --real-own-csv")
        d = pd.read_csv(args.real_own_csv)
        real = np.zeros(len(own_pct), dtype=float)
        for nm, pctd in zip(d["player"], d["dk_drafted_pct"]):
            pid = name_to_id.get(nm)
            if pid is not None and int(pid) in pid_index:
                real[pid_index[int(pid)]] = float(pctd)
        own_field = real
        if args.oracle == "all":
            own_pool = real
        print(f"      ORACLE={args.oracle}: realized ownership sums to "
              f"{real.sum():.0f} vs projected {own_pct.sum():.0f}; "
              f"MAE {np.abs(real - own_pct).mean():.2f}pp, "
              f"max {np.abs(real - own_pct).max():.1f}pp")
    clk.mark("slate")

    print(f"[2/7] pool  (candidates={cfg.n_candidates:,}, anchors={cfg.n_anchors:,})")
    n_sampler = cfg.n_candidates
    cands = fp.build_pool(players_df, engine, own_pool, cfg,
                          progress=lambda m: print(f"      {m}"))
    C = fp.indicator_matrix(cands, pid_index)
    # The gate and the ownership-ranked arms use the selected currency; the
    # REPORTED mean ownership always stays additive so runs stay comparable.
    own_sum = fp.ownership_currency(C, own_pool, args.own_metric)
    sal = players_df.set_index("player_id")["salary"].reindex(
        [int(p) for p in engine.players_df["player_id"]]).to_numpy(float)
    print(f"      pool {len(cands):,} lineups, mean salary ${(C @ sal).mean():,.0f}")
    clk.mark("pool")

    print(f"[3/7] ceiling + coverage bits over {cfg.ceiling_worlds:,} worlds")
    ceiling, bits, sim32, bar = fp.lineup_ceilings(
        engine, C, cfg, gate_currency=args.gate_currency)
    print(f"      currency={args.gate_currency}; coverage bar {bar:.1f} pts; "
          f"gate-currency p50 {np.median(ceiling):.1f}")
    clk.mark("ceiling")

    print("[4/7] conjunctive gate")
    if args.shortlist_mode == "random":
        shortlist, gd = fp.random_shortlist(len(cands), cfg.target_shortlist,
                                            args.seed + 31)
    else:
        shortlist, gd = fp.conjunctive_gate(ceiling, own_sum, cfg)
    anchor_ref = fp.anchor_ceiling_reference(ceiling, slice(n_sampler, len(cands)))
    print(f"      mode={args.shortlist_mode}  metric={args.own_metric}  "
          f"C*={gd['c_star']:.1f}  O*={gd['o_star']:.1f}  "
          f"gateA={gd['n_gate_a']:,}  shortlist={gd['n_shortlist']:,} "
          f"({gd['pool_pct_admitted']:.2f}% of pool)")
    # Where the shortlist sits in the POOL's own distributions — so a portfolio
    # can be read against what was available, not just against its shortlist.
    _cr = pd.Series(ceiling).rank(pct=True).to_numpy()
    _orr = pd.Series(own_sum).rank(pct=True).to_numpy()
    print(f"      shortlist sits at pool ceiling pctile "
          f"{100*_cr[shortlist].mean():.0f}, ownership pctile "
          f"{100*_orr[shortlist].mean():.0f}")
    print(f"      cross-check: ILP anchors' median ceiling {anchor_ref:.1f} "
          f"vs C* {gd['c_star']:.1f}"
          + ("  [OK]" if abs(anchor_ref - gd["c_star"]) < 0.15 * gd["c_star"]
             else "  [WARN: calibration disagrees on this slate]"))
    clk.mark("gate")

    print(f"[5/7] contest context: simulated field of {cfg.field_size:,}")
    cw = min(cfg.contest_worlds, cfg.ceiling_worlds)
    if args.exclude_overlap:
        _xp, _xarm, _xg = args.exclude_overlap.split(":")
        _xg = int(_xg)
        _xdf = pd.read_csv(_xp)
        _xg_rows = _xdf[_xdf.arm == _xarm].sort_values("slot")
        _X = np.zeros((len(_xg_rows), len(pid_index)), dtype=np.float32)
        for _r, _ids in enumerate(_xg_rows["player_ids"]):
            for _p in str(_ids).split("|"):
                _X[_r, pid_index[int(_p)]] = 1.0
        _ov = C[shortlist] @ _X.T                      # (S, K) shared players
        _ok = np.where(_ov.max(axis=1) <= _xg)[0]
        print(f"      exclude-overlap: {len(_ok):,}/{len(shortlist):,} shortlist "
              f"candidates share <= {_xg} players with {_xarm} in {_xp}")
        if len(_ok) < cfg.portfolio_size:
            raise SystemExit(
                f"only {len(_ok)} candidates clear overlap <= {_xg}; cannot "
                f"build {cfg.portfolio_size} lineups. Loosen the cap or widen "
                "the shortlist."
            )
        shortlist = shortlist[_ok]
    short_cands = [cands[i] for i in shortlist]
    Cshort = C[shortlist]
    cand_scores = (Cshort @ sim32[:cw].T)                       # (S_cand, cw)
    payout = parse_payout_table(Path(args.payout_table).read_text(), cfg.field_size)
    cs = ContestSimulator()
    fl = cs.generate_field(players_df, own_field, n_lineups=cfg.field_size,
                           rng_seed=args.seed + 2)
    field_scores = cs.score_field(fl, sim32[:cw], pid_index)     # (cw, F)
    field_sorted = np.sort(field_scores, axis=1)
    del field_scores
    print(f"      field {fl.shape[0]:,} lineups; ladder pays {int((payout>0).sum()):,}"
          f" of {cfg.field_size:,}; pool ${payout.sum():,.0f}")
    cand_payout = fp.candidate_payout_matrix(cand_scores, field_sorted, payout)
    print(f"      cand_payout {cand_payout.shape} "
          f"mean ${cand_payout.mean():.2f}/entry vs fee ${args.entry_fee:.0f}")
    clk.mark("context")

    print(f"[6/7] selection — {len(arms)} arms")
    portfolios: dict[str, list] = {}
    for arm in arms:
        a = time.perf_counter()
        if arm == "dr":
            lus, _ = fp.select_dr(cand_scores, field_sorted, payout, short_cands, cfg,
                                  progress=lambda m: print(f"      {m}"))
        elif arm == "kelly":
            lus, _ = fp.select_kelly(cand_payout, short_cands, cfg, args.entry_fee)
        elif arm == "emax":
            lus, _ = fp.select_emax(cand_payout, short_cands, cfg)
        elif arm == "coverage":
            lus, _ = fp.select_coverage(cand_payout, bits[shortlist],
                                        own_sum[shortlist], short_cands, cfg)
        elif arm == "determinant":
            corr = ep.compute_pool_corr(short_cands, None, scores=cand_scores,
                                        max_sims=cfg.corr_max_sims)
            lus, _ = fp.select_determinant(cand_payout, corr, own_sum[shortlist],
                                           short_cands, cfg)
            del corr
        elif arm == "gate_then_own":
            lus, _ = fp.select_gate_then_own(own_sum[shortlist], short_cands, cfg)
        elif arm == "zdiff":
            def _z(v):
                v = np.asarray(v, float)
                return (v - v.mean()) / v.std()
            sc = _z(ceiling) - _z(own_sum)
            lus = [cands[i] for i in np.argsort(-sc)[:cfg.portfolio_size]]
        elif arm == "lowown":
            lus = [cands[i] for i in np.argsort(own_sum)[:cfg.portfolio_size]]
        elif arm == "random":
            rng = np.random.default_rng(args.seed + 7)
            lus = [cands[i] for i in rng.choice(len(cands), cfg.portfolio_size,
                                                replace=False)]
        else:
            raise SystemExit(f"unknown arm {arm!r}")
        portfolios[arm] = lus
        print(f"      {arm:<14} {len(lus):>4} lineups  {time.perf_counter()-a:6.1f}s")
    del field_sorted, cand_payout, cand_scores
    clk.mark("selection")

    print("[7/7] diagnostics" + (" + grading" if args.contest else ""))
    pos_map = dict(zip(engine.players_df["player_id"].astype(int),
                       engine.players_df["position"].astype(str)))
    team_map = dict(zip(engine.players_df["player_id"].astype(int),
                        engine.players_df["team"].astype(str)))
    pitchers = {p for p, v in pos_map.items() if v == "P"}

    key_to_pool = {frozenset(int(p) for p in lu.player_ids): i
                   for i, lu in enumerate(cands)}
    rows = []
    for arm, lus in portfolios.items():
        tup = [tuple(int(p) for p in lu.player_ids) for lu in lus]
        # Pairwise diagnostics need a pair: a single-entry contest (DK Skipper
        # is 1-max) has no within-portfolio overlap to measure, and the
        # upper-triangle index set is empty rather than zero-length-safe.
        if len(tup) > 1:
            ov = overlap_profile(tup)
            cl = cluster_decomposition(tup, primary_teams(tup, team_map, pitchers))
        else:
            ov = {"mean_overlap": float("nan"), "pct_ge7": float("nan")}
            cl = {"within": float("nan"), "between": float("nan"),
                  "ratio": float("nan")}
        exp = exposure_profile(tup)
        A = fp.indicator_matrix(lus, pid_index)
        rows.append({
            "arm": arm,
            "mean_overlap": ov["mean_overlap"], "pct_ge7": ov["pct_ge7"],
            "within": cl["within"], "between": cl["between"], "ratio": cl["ratio"],
            "max_exposure": float(exp.max()), "n_distinct_players": int(len(exp)),
            "ceiling_pctile": float(100 * np.mean([
                _cr[key_to_pool[frozenset(int(p) for p in lu.player_ids)]]
                for lu in lus])),
            "own_pctile": float(100 * np.mean([
                _orr[key_to_pool[frozenset(int(p) for p in lu.player_ids)]]
                for lu in lus])),
            "mean_ceiling": float(np.mean([
                ceiling[key_to_pool[frozenset(int(p) for p in lu.player_ids)]]
                for lu in lus])),
            "mean_proj_own": float((A @ own_pool).mean()),
        })
    diag = pd.DataFrame(rows)
    print("\n=== PORTFOLIO STRUCTURE ===")
    print(diag.to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))

    for spec in args.contest:
        label, entries_csv, pay_txt, fee = spec.split(":")
        fee = float(fee)
        print(f"\n=== GRADING vs {label} (fee ${fee:.0f}) ===")
        Ff, pay_real, n_field, n_paid = pg.build_field(
            entries_csv, pay_txt, pid_index, name_to_id, len(pid_index))
        print(f"      real field {n_field:,} entries, {n_paid:,} paid")
        mats = {arm: fp.indicator_matrix(lus, pid_index)
                for arm, lus in portfolios.items()}
        # One simulation pass for every arm and both modes: the per-chunk field
        # sort is the expensive shared term, and grading arm-by-arm repeats it
        # 2 x n_arms times over identical worlds.
        port, marg = pg.grade_portfolios_multi(
            engine, mats, Ff, pay_real, args.grade_sims,
            sim_batch=20_000, chunk=500, seed=args.seed + 1000, progress=True)
        gr = []
        off = 0
        for arm, lus in portfolios.items():
            k = len(lus)
            pm, mm = port[arm], marg[off:off + k]
            off += k
            sp, sm = pg.summarize(pm, fee), pg.summarize(mm, fee)
            gr.append({
                "arm": arm,
                "portfolio_net": sp["net"], "portfolio_roi": sp["roi"],
                "marginal_net": sm["net"], "marginal_roi": sm["roi"],
                "self_comp_cost": sm["gross"] - sp["gross"],
                "pct_lineups_pos": sp["pct_lineups_positive"],
            })
            print(f"      {arm:<14} portfolio {sp['roi']:+7.1%}  "
                  f"marginal {sm['roi']:+7.1%}  "
                  f"self-competition ${sm['gross']-sp['gross']:>9,.0f}")
        g = pd.DataFrame(gr).sort_values("portfolio_roi", ascending=False)
        g.insert(0, "oracle", args.oracle)
        g.to_csv(out_dir / f"grade_{label}.csv", index=False)
        print()
        print(g.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    diag.to_csv(out_dir / "portfolio_structure.csv", index=False)
    recs = []
    for arm, lus in portfolios.items():
        for r, lu in enumerate(lus):
            recs.append({"arm": arm, "slot": r,
                         "player_ids": "|".join(str(int(p)) for p in lu.player_ids)})
    pd.DataFrame(recs).to_csv(out_dir / "portfolios.csv", index=False)
    pd.DataFrame(clk.rows).to_csv(out_dir / "timings.csv", index=False)
    total = sum(r["seconds"] for r in clk.rows)
    print(f"\nTOTAL {total/60:.1f} min   peak RSS {rss_gb():.2f} GB")
    print(f"wrote {out_dir}/portfolios.csv, portfolio_structure.csv, timings.csv")



if __name__ == "__main__":
    main()
