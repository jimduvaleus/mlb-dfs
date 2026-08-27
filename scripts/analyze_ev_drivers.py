"""What actually makes a lineup +EV in the contest simulation?

`scripts/analyze_contest_sim_roi.py` answers *which* real entries were +EV.
This answers *why*: it builds a feature matrix over the same 3,335 real
entries and tests the candidate explanations against their simulated ROI --
leverage (optimal ownership vs the field's real ownership), raw projection,
lineup shape (chalk/contrarian mix), and specific players and pairs.

Two independent notions of "optimal ownership" are computed, because they
answer different questions and can disagree:

  opt_own  -- UNCONSTRAINED. For each of `--n-opt-worlds` stratified sim
              worlds, solve the roster ILP on that world's realized scores
              over the WHOLE slate pool; opt_own[p] is the share of those
              world-optimal lineups containing p. This is the industry
              "optimal ownership": what a player should have been rostered
              at, ignoring what anyone actually built.
  win_own  -- FIELD-CONDITIONAL. P(the winning REAL entry contains p), which
              is exactly sum of p_win over the entries rostering him -- no
              extra simulation needed, it falls out of the 1M-world payout
              run. This asks what the winner had, given the field that
              actually showed up, and is bounded by what people built.

leverage = optimal - real field ownership, under either definition.

Usage
-----
    source venv/bin/activate
    python scripts/analyze_ev_drivers.py \
        --slate archive/08252026 \
        --entries outputs/contest_sim_roi/entries_sim_roi.csv \
        --n-sims 200000 --n-opt-worlds 2000
"""
import argparse
import collections
import re
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
from src.optimization.optimal_lineups import (  # noqa: E402
    generate_sim_optimal_lineups, stratified_sim_sample,
)
from analyze_contest_sim_roi import build_slate  # noqa: E402

_POS = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
_SPLIT = re.compile(r"\s*\b(" + "|".join(_POS) + r")\b\s+")
SALARY_CAP = 50_000


def parse_names(lineup_raw: str) -> list[str]:
    return _SPLIT.split(lineup_raw)[1:][1::2]


def to_eligible_list(s: str) -> list[str]:
    """'3B/OF' -> ['3B','OF']; 'SP'/'RP' -> ['P']."""
    out = []
    for tok in str(s).split("/"):
        tok = tok.strip().upper()
        out.append("P" if tok in ("SP", "RP", "P") else tok)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Per-entry score distribution
# ---------------------------------------------------------------------------

def entry_score_stats(engine, used, F, n_sims, batch, sample_worlds, seed):
    """(mean, sd, p99, p999) per entry, plus a per-player mean.

    Mean/sd stream over every world; the percentiles come from a retained
    sample, since order statistics can't be accumulated incrementally and the
    full (n_sims x n_entries) matrix is 2.7GB at 200k x 3335.
    """
    n_e = F.shape[0]
    s1 = np.zeros(n_e); s2 = np.zeros(n_e)
    psum = np.zeros(len(used))
    keep, kept = [], 0
    np.random.seed(seed)
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix[:, used].astype(np.float32)
        psum += sc.sum(axis=0)
        for st in range(0, b, 5000):
            FS = sc[st:st + 5000] @ F.T
            s1 += FS.sum(axis=0)
            s2 += (FS.astype(np.float64) ** 2).sum(axis=0)
            if kept < sample_worlds:
                take = min(sample_worlds - kept, FS.shape[0])
                keep.append(FS[:take].copy())
                kept += take
            del FS
        done += b
        print(f"      {done:,}/{n_sims:,} worlds")
        del sim, sc
    mean = s1 / n_sims
    sd = np.sqrt(np.maximum(s2 / n_sims - mean ** 2, 0.0))
    samp = np.concatenate(keep, axis=0)
    p99 = np.percentile(samp, 99, axis=0)
    p999 = np.percentile(samp, 99.9, axis=0)
    return mean, sd, p99, p999, psum / n_sims


# ---------------------------------------------------------------------------
# Optimal ownership
# ---------------------------------------------------------------------------

def unconstrained_optimal_ownership(players_df, sim, n_worlds, seed, min_stack):
    """opt_own[player_id] = share of per-world ILP-optimal lineups rostering
    him. Solved over the whole slate pool, so it is a statement about the
    slate, not about the field."""
    df = players_df.copy()
    df["eligible_positions"] = df["eligible_positions"].map(to_eligible_list)
    rng = np.random.default_rng(seed)
    idx = [i for i, _ in stratified_sim_sample(sim.results_matrix, n_worlds, rng)]
    done = [0]

    def _cb(n):
        done[0] = n
        if n % 200 == 0:
            print(f"      ILP {n}/{len(idx)} worlds")

    lus = generate_sim_optimal_lineups(
        df, sim.results_matrix, list(sim.player_ids), idx,
        min_stack=min_stack, salary_floor=None, progress_cb=_cb,
    )
    cnt = collections.Counter(int(p) for lu in lus for p in lu.player_ids)
    n = len(lus)
    print(f"      {n} unique world-optimal lineups from {len(idx)} worlds")
    return {p: 100.0 * c / n for p, c in cnt.items()}, lus


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def decile_table(feat: pd.DataFrame, cols, target="roi", q=10):
    rows = []
    for c in cols:
        d = pd.qcut(feat[c].rank(method="first"), q, labels=False)
        g = feat.groupby(d)[target].mean()
        rows.append({
            "feature": c,
            "spearman": spearman(feat[c], feat[target]),
            "bottom_decile_roi": g.iloc[0],
            "top_decile_roi": g.iloc[-1],
            "spread": g.iloc[-1] - g.iloc[0],
            "monotone": bool(np.all(np.diff(g.values) > 0) or np.all(np.diff(g.values) < 0)),
        })
    return pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["spearman"].abs().sort_values(ascending=False).index)


def partial_spearman(feat: pd.DataFrame, cols, control, target="roi"):
    """Spearman(feature, ROI) after rank-residualising BOTH against `control`.

    The decisive test for every theory here: ceiling alone explains most of
    the ROI spread, so the question is never "does X correlate with ROI" but
    "does X still say anything once ceiling is held fixed".
    """
    def rz(v):
        r = pd.Series(v).rank().to_numpy(float)
        return (r - r.mean()) / r.std()
    c = rz(feat[control])
    y = rz(feat[target])
    y_r = y - c * float(np.dot(c, y) / np.dot(c, c))
    rows = []
    for col in cols:
        if col == control:
            continue
        x = rz(feat[col])
        x_r = x - c * float(np.dot(c, x) / np.dot(c, c))
        rows.append({"feature": col,
                     "raw_spearman": spearman(feat[col], feat[target]),
                     f"partial_given_{control}": spearman(x_r, y_r)})
    out = pd.DataFrame(rows)
    return out.reindex(out[f"partial_given_{control}"].abs()
                       .sort_values(ascending=False).index)


def standardized_ols(feat: pd.DataFrame, cols, target="roi", lam=1e-6):
    X = feat[cols].to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    y = feat[target].to_numpy(float)
    y = (y - y.mean()) / y.std()
    A = np.hstack([X, np.ones((len(X), 1))])
    beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    resid = y - A @ beta
    r2 = 1.0 - resid.var() / y.var()
    # Condition number of the standardized design: anything past ~30 means the
    # individual betas are trading mass against each other and must not be
    # read one at a time.
    cond = float(np.linalg.cond(X))
    tbl = pd.DataFrame({"feature": cols, "std_beta": beta[:-1]}).reindex(
        pd.Series(beta[:-1]).abs().sort_values(ascending=False).index)
    return tbl, r2, cond


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--players", default="outputs/contest_sim_roi/player_sim_roi.csv")
    ap.add_argument("--n-sims", type=int, default=200_000)
    ap.add_argument("--sim-batch", type=int, default=50_000)
    ap.add_argument("--sample-worlds", type=int, default=25_000)
    ap.add_argument("--n-opt-worlds", type=int, default=2_000)
    ap.add_argument("--opt-min-stack", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="outputs/contest_sim_roi")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse lineup_features.csv / player_leverage.csv and "
                         "re-run only the analysis (no sim, no ILP)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = PROJECT_ROOT / args.out_dir
    archive = Path(args.slate)

    if args.from_cache:
        print("[cache] reusing lineup_features.csv / _cache_players.csv / _cache_F.npy")
        f = pd.read_csv(out_dir / "lineup_features.csv")
        P = pd.read_csv(out_dir / "_cache_players.csv")
        F = np.load(out_dir / "_cache_F.npy")
        used = list(range(F.shape[1]))
        inv_pid = {k: (P["player"].iloc[k], int(P["player_id"].iloc[k]))
                   for k in range(len(P))}
        team_v = P["team"].to_numpy()
        run_analysis(f, P, F, used, inv_pid, team_v, out_dir)
        return

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    print("[1/6] slate + sim engine")
    players_df, grids, name_to_id = build_slate(archive, cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}

    entries = pd.read_csv(args.entries)
    entries["names"] = entries["lineup_raw"].map(parse_names)
    pl = pd.read_csv(args.players)
    print(f"      {len(entries):,} entries, {len(players_df):,} slate players")

    used = sorted({pid_index[name_to_id[n]] for ns in entries["names"] for n in ns})
    col_of = {j: k for k, j in enumerate(used)}
    F = np.zeros((len(entries), len(used)), dtype=np.float32)
    for r, ns in enumerate(entries["names"]):
        for n in ns:
            F[r, col_of[pid_index[name_to_id[n]]]] = 1.0

    print(f"[2/6] per-entry score distribution over {args.n_sims:,} worlds")
    e_mean, e_sd, e_p99, e_p999, p_mean = entry_score_stats(
        engine, used, F, args.n_sims, args.sim_batch, args.sample_worlds, args.seed)

    print(f"[3/6] unconstrained optimal ownership "
          f"({args.n_opt_worlds:,} ILP solves, min_stack={args.opt_min_stack})")
    sim_small = engine.simulate(max(args.n_opt_worlds * 5, 10_000))
    opt_own, opt_lus = unconstrained_optimal_ownership(
        players_df, sim_small, args.n_opt_worlds, args.seed, args.opt_min_stack)

    print("[4/6] player table: ownership, optimal%, leverage")
    field_own = dict(zip(pl["player"].map(name_to_id), pl["dk_drafted_pct"]))
    # win_own[p] = P(the winning real entry rosters p) = sum of p_win over the
    # entries that roster him -- exactly one entry wins each world.
    wp = (F * entries["p_win"].to_numpy()[:, None]).sum(axis=0) * 100.0
    t100 = (F * entries["p_top100"].to_numpy()[:, None]).sum(axis=0)
    inv_pid = {}
    for name, pid in name_to_id.items():
        j = pid_index.get(int(pid))
        if j is not None and j in col_of:
            inv_pid[col_of[j]] = (name, int(pid))
    prow = []
    for k in range(len(used)):
        name, pid = inv_pid[k]
        prow.append({
            "player": name, "player_id": pid,
            "team": players_df.set_index("player_id")["team"].get(pid, ""),
            "salary": float(players_df.set_index("player_id")["salary"].get(pid, np.nan)),
            "field_own": float(field_own.get(pid, 0.0)),
            "opt_own": float(opt_own.get(pid, 0.0)),
            "win_own": float(wp[k]),
            "top100_own": float(t100[k]),
            "sim_mean": float(p_mean[k]),
        })
    P = pd.DataFrame(prow)
    P["leverage"] = P["opt_own"] - P["field_own"]
    P["win_leverage"] = P["win_own"] - P["field_own"]
    P.sort_values("leverage", ascending=False).to_csv(
        out_dir / "player_leverage.csv", index=False)
    # Cache in `used` column order (not the sorted view) plus the indicator
    # matrix, so --from-cache can rebuild the analysis without the 5-minute
    # sim + ILP.
    P.to_csv(out_dir / "_cache_players.csv", index=False)
    np.save(out_dir / "_cache_F.npy", F)

    print("[5/6] lineup features")
    own_v = P["field_own"].to_numpy()
    lev_v = P["leverage"].to_numpy()
    wlev_v = P["win_leverage"].to_numpy()
    optv = P["opt_own"].to_numpy()
    sal_v = P["salary"].to_numpy()
    mu_v = P["sim_mean"].to_numpy()
    pid_v = P["player_id"].to_numpy()
    team_v = P["team"].to_numpy()
    is_p = np.array([str(players_df.set_index("player_id")["position"].get(int(p), "")) == "P"
                     for p in pid_v])

    f = pd.DataFrame(index=entries.index)
    f["roi"] = entries["roi"]
    f["ev_net"] = entries["ev_net"]
    f["proj_total"] = F @ mu_v
    f["ceiling_p99"] = e_p99
    f["ceiling_p999"] = e_p999
    f["score_sd"] = e_sd
    f["ceiling_over_mean"] = e_p99 - e_mean
    f["own_sum"] = F @ own_v
    f["own_log_sum"] = F @ np.log(np.clip(own_v, 0.1, None))
    f["own_max"] = (F * own_v).max(axis=1)
    f["own_std"] = np.array([own_v[r > 0].std() for r in F])
    f["n_chalk_15"] = (F * (own_v > 15)).sum(axis=1)
    f["n_contrarian_5"] = (F * (own_v < 5)).sum(axis=1)
    f["leverage_sum"] = F @ lev_v
    f["win_leverage_sum"] = F @ wlev_v
    f["opt_own_sum"] = F @ optv
    f["salary_used"] = F @ sal_v
    f["salary_unused"] = SALARY_CAP - f["salary_used"]
    f["p_salary"] = F @ np.where(is_p, sal_v, 0.0)
    f["p_proj"] = F @ np.where(is_p, mu_v, 0.0)
    f["p_own"] = F @ np.where(is_p, own_v, 0.0)
    f["bat_proj"] = f["proj_total"] - f["p_proj"]

    stack1, stack2, n_teams = [], [], []
    for r in F:
        idx = np.where(r > 0)[0]
        c = collections.Counter(team_v[i] for i in idx if not is_p[i])
        top = c.most_common()
        stack1.append(top[0][1] if top else 0)
        stack2.append(top[1][1] if len(top) > 1 else 0)
        n_teams.append(len(c))
    f["stack1"] = stack1
    f["stack2"] = stack2
    f["n_bat_teams"] = n_teams
    f.to_csv(out_dir / "lineup_features.csv", index=False)

    print("[6/6] analysis\n")
    run_analysis(f, P, F, used, inv_pid, team_v, out_dir)


def run_analysis(f, P, F, used, inv_pid, team_v, out_dir):

    pd.set_option("display.width", 220)
    feats = ["ceiling_p999", "ceiling_p99", "proj_total", "bat_proj", "p_proj",
             "score_sd", "ceiling_over_mean", "own_sum", "own_log_sum",
             "own_max", "own_std", "n_chalk_15", "n_contrarian_5",
             "leverage_sum", "win_leverage_sum", "opt_own_sum",
             "salary_used", "salary_unused", "p_salary", "p_own",
             "stack1", "stack2", "n_bat_teams"]

    print("=== UNIVARIATE: each theory vs simulated ROI ===")
    print(decile_table(f, feats).to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== MULTIVARIATE: standardized OLS on ROI ===")
    # Two specs. Spec A has no algebraic redundancy. Spec B substitutes
    # leverage for its two components -- including own_sum, opt_own_sum AND
    # leverage_sum together makes the design singular by construction
    # (leverage = optimal - own) and produced betas that flipped sign between
    # runs; that was a spec error, not a finding.
    for label, core in (
        ("A: raw drivers", ["ceiling_p999", "proj_total", "score_sd", "own_sum",
                            "opt_own_sum", "salary_unused", "stack1", "p_proj"]),
        ("B: leverage framing", ["ceiling_p999", "leverage_sum", "score_sd",
                                 "salary_unused", "stack1", "p_proj"]),
    ):
        tbl, r2, cond = standardized_ols(f, core)
        print(f"  -- spec {label}  (R^2 = {r2:.3f}, condition number = {cond:.1f})")
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        print()

    print("=== PARTIAL: does each theory survive holding CEILING fixed? ===")
    print(partial_spearman(f, feats, "ceiling_p999").to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== THE TWO-AXIS PICTURE: ceiling x ownership ===")
    cb = pd.qcut(f["ceiling_p999"].rank(method="first"), 5, labels=[
        "ceil Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
    ob = pd.qcut(f["own_sum"].rank(method="first"), 5, labels=[
        "own Q1(low)", "Q2", "Q3", "Q4", "Q5(chalky)"])
    piv = f.groupby([cb, ob], observed=True)["roi"].mean().unstack()
    print(piv.to_string(float_format=lambda x: f"{x:+.1%}"))
    print("\n  n per cell:")
    print(f.groupby([cb, ob], observed=True)["roi"].size().unstack().to_string())

    print("\n=== SHAPE: chalk/contrarian mix (barbell test) ===")
    sh = (f.groupby(["n_chalk_15", "n_contrarian_5"], observed=True)["roi"]
          .agg(["size", "mean"]).reset_index())
    sh = sh[sh["size"] >= 30].sort_values("mean", ascending=False)
    sh.columns = ["n_chalk(>15%)", "n_contrarian(<5%)", "n", "roi"]
    print(sh.head(12).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("  ...")
    print(sh.tail(8).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== TOP 25 PLAYERS BY LEVERAGE (optimal% - field%) ===")
    print(P.sort_values("leverage", ascending=False).head(25)[
        ["player", "team", "salary", "field_own", "opt_own", "leverage",
         "win_own", "win_leverage", "sim_mean"]].to_string(
        index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n=== BOTTOM 15 BY LEVERAGE (over-owned vs optimal) ===")
    print(P.sort_values("leverage").head(15)[
        ["player", "team", "salary", "field_own", "opt_own", "leverage",
         "win_own", "win_leverage", "sim_mean"]].to_string(
        index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== PAIRS: biggest ROI lift over the additive expectation ===")
    roi = f["roi"].to_numpy()
    base = roi.mean()
    solo = np.array([roi[F[:, k] > 0].mean() - base for k in range(len(used))])
    rows = []
    for a in range(len(used)):
        ma = F[:, a] > 0
        if ma.sum() < 40:
            continue
        for b in range(a + 1, len(used)):
            m = ma & (F[:, b] > 0)
            n = int(m.sum())
            if n < 40:
                continue
            rows.append({"a": inv_pid[a][0], "b": inv_pid[b][0], "n": n,
                         "same_team": bool(team_v[a] == team_v[b]),
                         "roi": roi[m].mean(),
                         "additive": base + solo[a] + solo[b],
                         "lift": roi[m].mean() - (base + solo[a] + solo[b])})
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs.to_csv(out_dir / "pair_lift.csv", index=False)
        # Same-team pairs are mechanically sub-additive: two players from one
        # team share a team-level ROI penalty/bonus that the additive baseline
        # counts twice, so "lift" for them measures stacking, not chemistry.
        # Report the split, then rank genuine cross-team combinations.
        g = pairs.groupby("same_team")["lift"].agg(["size", "mean"])
        print("  mean lift by pair type (the stacking artifact, isolated):")
        print(g.to_string(float_format=lambda x: f"{x:,.4f}"))
        cross = pairs[~pairs.same_team]
        print(f"\n  best CROSS-TEAM combinations ({len(cross)} pairs >= 40 entries):")
        print(cross.sort_values("lift", ascending=False).head(12).to_string(
            index=False, float_format=lambda x: f"{x:,.3f}"))
        print("\n  worst CROSS-TEAM combinations:")
        print(cross.sort_values("lift").head(8).to_string(
            index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nwrote {out_dir}/player_leverage.csv, lineup_features.csv, pair_lift.csv")


if __name__ == "__main__":
    main()
