"""
Which per-lineup signal, if any, predicts a realized top-1% (hit99) finish?

The external-pool selector needs a "value" currency. `--own-sweep` in
analyze_external_pool.py calibrates one particular formula
(proj - own*field/own_scale); this script asks the prior question: does
*anything* in the SaberSim export rank lineups by their realized hit99
probability, and by how much?

Method
------
Pools every settled slate that has a lineups_*.csv export, a
contest_player_fpts.json and a contest-standings zip. Each lineup gets its
realized score (from the fpts map) and its percentile against the real
contest field, so hit99 = "would have finished in the field's top 1%".

Three readouts, in increasing order of how much they should be trusted:

  --mode univariate  within-slate ROC AUC of each candidate signal vs hit99,
                     plus the hit99 rate of a top-N selection by that signal.
                     Descriptive only — an AUC computed on the same slates
                     you'd pick a formula from is optimistic.

  --mode loso        the honest test: per-slate top-N hit99 for each signal
                     against a random-draw baseline from the same pool, with
                     a paired t across slates. A signal that cannot beat a
                     random draw here is not a value currency.

  --mode cull        does the pool-wide projected-score floor
                     (gpp.external_pool_proj_score_pct) raise or lower the
                     pool's hit99 rate? Highest-powered question available,
                     since it uses every lineup rather than a top-N slice.

  --mode coverage    150-entry portfolios built three ways — top-N by value,
                     uniformly random, and greedy max-diversity (minimum
                     player overlap) — compared on hits/slate and
                     P(at least one hit99). Tests whether spreading entries
                     apart beats concentrating them on the best-scoring ones.

Caveat that dominates every readout: hit99 is a ~1% event, so a 150-entry
portfolio experiences 0-3 of them per slate. With a handful of settled
slates the per-slate numbers are counts, not rates, and the power section
printed by --mode loso says how many slates a given effect would need.

Usage
-----
    python scripts/analyze_hit99_signals.py --mode loso
    python scripts/analyze_hit99_signals.py --mode univariate --top-n 150
    python scripts/analyze_hit99_signals.py --mode cull
    python scripts/analyze_hit99_signals.py --mode coverage
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.dk_entries import _parse_prize_pool_cents  # noqa: E402
from src.api.external_pool import discover_external_files  # noqa: E402
from analyze_candidate_pool import (  # noqa: E402
    add_real_percentile, default_cash_threshold, load_contest_player_fpts,
    load_real_field_points,
)
from analyze_external_pool import (  # noqa: E402
    add_actual_score, discover_contest_blocks, load_external_lineups,
)

_PCTL_COLS = ["25th", "50th", "75th", "85th", "95th", "99th"]
_BUCKETS = ["Large Slate | 100-1k", "Large Slate | 1k-10k",
            "Large Slate | 10k-50k", "Large Slate | 50k+"]


def _extra_columns(paths: list[Path]) -> pd.DataFrame:
    """Per-lineup export columns load_external_lineups drops: Saber's own
    simulated score percentiles, the slate-size bucket scores, Saber Score,
    and the primary contest's win/cash/dupes/ROI-StDev block. Row order is
    the raw combined-file order, i.e. what lineup_index indexes into."""
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    out = pd.DataFrame(index=df.index)
    for c in _PCTL_COLS + _BUCKETS + ["Saber Score"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    blocks = discover_contest_blocks(df.columns)
    primary = max(blocks, key=lambda n: _parse_prize_pool_cents(blocks[n]["raw_name"]) or 0)
    b = blocks[primary]
    out["c_roi"] = pd.to_numeric(df[b["roi"]], errors="coerce")
    out["c_win"] = pd.to_numeric(df[b["win_rate"]], errors="coerce")
    out["c_cash"] = pd.to_numeric(df[b["cash_rate"]], errors="coerce")
    for key, suffix in (("c_dupes", " Sim Dupes"), ("c_roisd", " ROI StDev")):
        col = f"{b['raw_name']}{suffix}"
        out[key] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    out["primary_contest"] = b["raw_name"]
    return out.reset_index(drop=True)


def _player_level_features(df: pd.DataFrame, archive_dir: Path) -> pd.DataFrame:
    """Per-lineup aggregates that are NOT recoverable from the lineup's
    summed projection and summed ownership.

    A per-player *linear* delta is not one of them: Sum(proj_i - k*own_i) is
    identically Sum(proj) - k*Sum(own), so player-level scoring only says
    something new once it is nonlinear in the player's own numbers. These
    are the nonlinear ones — concentration (hhi/max/chalk counts), per-player
    ratios, and ownership residualized against what a player of that
    projection is normally owned (isotonic, fitted within the slate)."""
    from src.api.external_pool import parse_player_projections
    found = discover_external_files(str(archive_dir))
    ppath = found.get("projections_path")
    cols = ["own_hhi", "own_max", "n_chalk20", "n_lev5", "sum_ratio",
            "sum_log_own", "resid_own", "min_proj"]
    if not ppath:
        for c in cols:
            df[c] = np.nan
        return df
    proj = parse_player_projections(Path(ppath))
    pm = dict(zip(proj["player_id"], proj["mean"]))
    om = dict(zip(proj["player_id"], proj["ownership"]))

    # Expected ownership for a given projection, fitted on this slate's own
    # players via PAVA (ownership rises monotonically with projection).
    from src.api.external_pool import _pava
    ok = proj.dropna(subset=["mean", "ownership"]).sort_values("mean")
    fit_x = ok["mean"].to_numpy()
    fit_y = _pava(ok["ownership"].to_numpy()) if len(ok) else np.array([])

    rows = []
    for pids in df["player_ids"]:
        p = np.array([pm.get(int(i), np.nan) for i in pids], dtype=float)
        o = np.array([om.get(int(i), np.nan) for i in pids], dtype=float)
        if np.isnan(p).any() or np.isnan(o).any():
            rows.append([np.nan] * len(cols))
            continue
        expected = np.interp(p, fit_x, fit_y) if len(fit_x) else np.zeros_like(p)
        rows.append([
            float((o ** 2).sum()),                       # own_hhi
            float(o.max()),                              # own_max
            float((o > 20).sum()),                       # n_chalk20
            float((o < 5).sum()),                        # n_lev5
            float((p / np.maximum(o, 0.5)).sum()),       # sum_ratio
            float(np.log(o + 1.0).sum()),                # sum_log_own
            float((o - expected).sum()),                 # resid_own
            float(p.min()),                              # min_proj
        ])
    return pd.concat([df, pd.DataFrame(rows, columns=cols, index=df.index)], axis=1)


def build_dataset(archive_dirs: list[Path]) -> pd.DataFrame:
    cash_thr = default_cash_threshold()
    frames = []
    for d in archive_dirs:
        try:
            found = discover_external_files(str(d))
            if not found["lineups_paths"]:
                raise FileNotFoundError("no lineups_*.csv")
            lineup_df, _, _, _ = load_external_lineups(found["lineups_paths"])
            fpts = load_contest_player_fpts(d)
            field = load_real_field_points(d)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {d.name}: {exc}")
            continue
        # load_external_lineups drops exact duplicates, so realign by
        # lineup_index rather than assuming positional parity.
        extra = _extra_columns(found["lineups_paths"]).loc[lineup_df["lineup_index"].to_numpy()]
        df = pd.concat([lineup_df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        df = add_actual_score(df, fpts)
        df = add_real_percentile(df, field, cash_thr, 0.95)
        df = _player_level_features(df, d)
        df["slate"] = d.name
        df["n_field"] = len(field)
        frames.append(df)
    if not frames:
        raise SystemExit("No settled slates with all three required files found.")
    out = pd.concat(frames, ignore_index=True)
    out["hit99"] = np.where(out["real_percentile"].isna(), np.nan,
                            (out["real_percentile"] >= 0.99).astype(float))
    return out.dropna(subset=["hit99"]).reset_index(drop=True)


def candidate_scores() -> dict:
    """Every value currency worth testing, including the ones the selector
    already uses. Each maps a per-slate frame to a higher-is-better array."""
    c = {
        "proj_score": lambda g: g["proj_score"].to_numpy(),
        "-ownership": lambda g: -g["ownership"].to_numpy(),
        "salary": lambda g: g["salary"].to_numpy(),
        "saber 99th": lambda g: g["99th"].to_numpy(),
        "saber 95th": lambda g: g["95th"].to_numpy(),
        "ceil spread 99-50": lambda g: (g["99th"] - g["50th"]).to_numpy(),
        "ceil ratio 99/50": lambda g: (g["99th"] / g["50th"]).to_numpy(),
        "99th - proj": lambda g: (g["99th"] - g["proj_score"]).to_numpy(),
        "saber score": lambda g: g["Saber Score"].to_numpy(),
        "contest ROI": lambda g: g["c_roi"].to_numpy(),
        "contest WinRate": lambda g: g["c_win"].to_numpy(),
        "contest CashRate": lambda g: g["c_cash"].to_numpy(),
        "contest ROI StDev": lambda g: g["c_roisd"].to_numpy(),
        "-contest SimDupes": lambda g: -g["c_dupes"].to_numpy(),
        "bucket 10k-50k": lambda g: g["Large Slate | 10k-50k"].to_numpy(),
    }
    for k in (0.1, 0.2, 1 / 3, 0.5, 0.8):
        c[f"proj - {k:.2f}*own"] = (
            lambda kk: lambda g: (g["proj_score"] - kk * g["ownership"]).to_numpy())(k)
        c[f"99th - {k:.2f}*own"] = (
            lambda kk: lambda g: (g["99th"] - kk * g["ownership"]).to_numpy())(k)

    # --- ratios (scale-free) rather than deltas -------------------------
    c["proj / own"] = lambda g: (g["proj_score"] / g["ownership"]).to_numpy()
    c["proj / sqrt(own)"] = lambda g: (g["proj_score"] / np.sqrt(g["ownership"])).to_numpy()
    c["proj / own^0.25"] = lambda g: (g["proj_score"] / g["ownership"] ** 0.25).to_numpy()
    c["log proj - log own"] = lambda g: (np.log(g["proj_score"]) - np.log(g["ownership"])).to_numpy()

    # --- nonlinear penalties on the ownership term ----------------------
    c["proj - 2*sqrt(own)"] = lambda g: (g["proj_score"] - 2 * np.sqrt(g["ownership"])).to_numpy()
    c["proj - 12*log(own)"] = lambda g: (g["proj_score"] - 12 * np.log(g["ownership"])).to_numpy()
    c["proj - own^2/300"] = lambda g: (g["proj_score"] - g["ownership"] ** 2 / 300).to_numpy()
    c["proj^2/100 - own/3"] = lambda g: (g["proj_score"] ** 2 / 100 - g["ownership"] / 3).to_numpy()

    # --- player-level nonlinear (not recoverable from the two sums) -----
    c["-own hhi"] = lambda g: -g["own_hhi"].to_numpy()
    c["-own max"] = lambda g: -g["own_max"].to_numpy()
    c["-n chalk>20%"] = lambda g: -g["n_chalk20"].to_numpy()
    c["n lev<5%"] = lambda g: g["n_lev5"].to_numpy()
    c["sum proj_i/own_i"] = lambda g: g["sum_ratio"].to_numpy()
    c["-sum log own_i"] = lambda g: -g["sum_log_own"].to_numpy()
    c["-resid own (isotonic)"] = lambda g: -g["resid_own"].to_numpy()
    c["min player proj"] = lambda g: g["min_proj"].to_numpy()
    c["proj - 0.33*own - hhi/60"] = lambda g: (
        g["proj_score"] - g["ownership"] / 3 - g["own_hhi"] / 60).to_numpy()
    return c


def _cull(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    q = df.groupby("slate")["proj_score"].rank(pct=True)
    return df[q >= pct / 100.0]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _top_n_rate(g: pd.DataFrame, score: np.ndarray, n: int) -> float:
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-1e18)
    return g["hit99"].to_numpy()[np.argsort(-score)[:n]].mean()


def run_univariate(df: pd.DataFrame, n_top: int, proj_pct: float) -> None:
    pool = _cull(df, proj_pct)
    rows = []
    for name, fn in candidate_scores().items():
        aucs, tops = [], []
        for _, g in pool.groupby("slate"):
            sc = fn(g)
            if not np.isfinite(np.asarray(sc, dtype=float)).any():
                continue
            aucs.append(_auc(g["hit99"].to_numpy(),
                             np.nan_to_num(np.asarray(sc, dtype=float), nan=-1e18)))
            tops.append(_top_n_rate(g, sc, n_top))
        rows.append({"score": name, "auc": np.nanmean(aucs), "auc_sd": np.nanstd(aucs),
                     f"top{n_top}": np.mean(tops),
                     "auc>0.5 on": int(np.nansum(np.array(aucs) > 0.5))})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(f"pool = {len(pool):,} lineups after a {proj_pct:.0f}% projection cull, "
          f"{int(pool['hit99'].sum())} hit99, base rate {pool['hit99'].mean():.4f}\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAUC 0.50 = coin flip. In-sample and descriptive — use --mode loso to decide.")


def run_loso(df: pd.DataFrame, n_top: int, proj_pct: float, seed: int) -> None:
    pool = _cull(df, proj_pct)
    slates = sorted(pool["slate"].unique())
    rng = np.random.default_rng(seed)
    base = {}
    for s in slates:
        g = pool[pool["slate"] == s]
        y = g["hit99"].to_numpy()
        base[s] = np.mean([y[rng.choice(len(g), min(n_top, len(g)), replace=False)].mean()
                           for _ in range(400)])
    rows = {"random draw": base}
    for name, fn in candidate_scores().items():
        rows[name] = {s: _top_n_rate(pool[pool["slate"] == s], fn(pool[pool["slate"] == s]), n_top)
                      for s in slates}
    t = pd.DataFrame(rows).T[slates]
    t["MEAN"] = t.mean(axis=1)
    print(f"hit99 rate of a top-{n_top} selection, per slate "
          f"(a {proj_pct:.0f}% projection cull applied first)\n")
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))

    # Multiple comparisons: with this many formulas over this few slates,
    # the best-looking one is expected to look good even under pure noise.
    # Shuffle hit99 within each slate, redo the whole search, and record the
    # best |t| — that distribution is the bar a real signal has to clear.
    print(f"\nPermutation null: best |t| across all {len(t) - 1} formulas when "
          f"hit99 is shuffled within slate")
    null_best = []
    pools = {s: pool[pool["slate"] == s] for s in slates}
    scores = {name: {s: np.nan_to_num(np.asarray(fn(pools[s]), dtype=float), nan=-1e18)
                     for s in slates} for name, fn in candidate_scores().items()}
    for _ in range(200):
        perm = {s: rng.permutation(pools[s]["hit99"].to_numpy()) for s in slates}
        best = 0.0
        for name in scores:
            d = np.array([perm[s][np.argsort(-scores[name][s])[:n_top]].mean() - base[s]
                          for s in slates])
            sd = d.std(ddof=1)
            if sd > 0:
                best = max(best, abs(d.mean() / (sd / np.sqrt(len(d)))))
        null_best.append(best)
    null_best = np.array(null_best)
    print(f"  median {np.median(null_best):.2f}   90th pct {np.percentile(null_best, 90):.2f}"
          f"   95th pct {np.percentile(null_best, 95):.2f}")
    print("  -> treat any |t| below the 95th percentile as indistinguishable from noise.")

    b = t.loc["random draw", slates].to_numpy()
    print(f"\nPaired against the random draw ({len(slates)} slates):")
    print(f"  {'score':20s} {'mean diff':>10s} {'se':>8s} {'t':>7s} {'wins':>6s} {'slates for 80% power':>22s}")
    for name in t.index:
        if name == "random draw":
            continue
        d = t.loc[name, slates].to_numpy() - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        need = ((1.96 + 0.84) ** 2 * d.std(ddof=1) ** 2 / d.mean() ** 2) if d.mean() else np.inf
        print(f"  {name:20s} {d.mean():+10.5f} {se:8.5f} {d.mean()/se if se else 0:+7.2f} "
              f"{int((d > 0).sum()):3d}/{len(d)} {np.ceil(need) if np.isfinite(need) else np.inf:22.0f}")


def run_cull(df: pd.DataFrame, grid: list[float]) -> None:
    slates = sorted(df["slate"].unique())
    rows = {}
    for pct in grid:
        sub = _cull(df, pct)
        rows[f"cull {pct:.0f}%"] = {s: sub[sub["slate"] == s]["hit99"].mean() for s in slates}
    t = pd.DataFrame(rows).T[slates]
    t["MEAN"] = t.mean(axis=1)
    print("Pool hit99 rate by projected-score floor (gpp.external_pool_proj_score_pct)\n")
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    ref = t.loc[f"cull {grid[0]:.0f}%", slates].to_numpy()
    print(f"\nPaired against a {grid[0]:.0f}% cull:")
    for pct in grid[1:]:
        d = t.loc[f"cull {pct:.0f}%", slates].to_numpy() - ref
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  cull {pct:4.0f}%: mean {d.mean():+.5f}  se {se:.5f}  "
              f"t {d.mean()/se if se else 0:+.2f}  helps on {int((d > 0).sum())}/{len(d)} slates")
    print("\nA random entry in the real field hits99 exactly 1.00% by construction — "
          "compare the pool's rate to that.")


def _indicator(g: pd.DataFrame) -> np.ndarray:
    ids = sorted({p for lst in g["player_ids"] for p in lst})
    pos = {p: i for i, p in enumerate(ids)}
    m = np.zeros((len(g), len(ids)), dtype=np.float32)
    for r, lst in enumerate(g["player_ids"]):
        for p in lst:
            m[r, pos[p]] = 1.0
    return m


def _greedy_diverse(indicator: np.ndarray, n: int, seed_row: int) -> np.ndarray:
    """Repeatedly take the lineup with the smallest max player-overlap
    against the already-picked set, ties broken by smallest total overlap."""
    picked = [seed_row]
    max_ov = indicator @ indicator[seed_row]
    tot_ov = max_ov.copy()
    for _ in range(n - 1):
        key = max_ov * 1000.0 + tot_ov
        key[picked] = np.inf
        j = int(np.argmin(key))
        picked.append(j)
        ov = indicator @ indicator[j]
        max_ov = np.maximum(max_ov, ov)
        tot_ov += ov
    return np.array(picked)


def run_coverage(df: pd.DataFrame, n_top: int, proj_pct: float, seed: int) -> None:
    pool = _cull(df, proj_pct)
    rng = np.random.default_rng(seed)
    print(f"{n_top}-entry portfolios: top-N by value vs uniformly random vs max-diversity\n")
    print(f"{'slate':12s} {'value hits':>11s} {'random hits':>12s} {'diverse hits':>13s}")
    acc = {k: [] for k in ("v", "r", "d", "v_any", "r_any", "d_any")}
    for s in sorted(pool["slate"].unique()):
        g = pool[pool["slate"] == s].reset_index(drop=True)
        y = g["hit99"].to_numpy()
        ev = (g["proj_score"] - g["ownership"] / 3.0).to_numpy()
        ind = _indicator(g)
        v = np.argsort(-ev)[:n_top]
        d = _greedy_diverse(ind, n_top, int(np.argmax(ev)))
        draws = [rng.choice(len(g), n_top, replace=False) for _ in range(400)]
        acc["v"].append(y[v].sum()); acc["d"].append(y[d].sum())
        acc["r"].append(np.mean([y[r].sum() for r in draws]))
        acc["v_any"].append(float(y[v].sum() > 0)); acc["d_any"].append(float(y[d].sum() > 0))
        acc["r_any"].append(np.mean([y[r].sum() > 0 for r in draws]))
        print(f"{s:12s} {y[v].sum():11.0f} {acc['r'][-1]:12.2f} {y[d].sum():13.0f}")
    print(f"\n  mean hits/slate  value {np.mean(acc['v']):.2f}   random {np.mean(acc['r']):.2f}"
          f"   diverse {np.mean(acc['d']):.2f}")
    print(f"  P(>=1 hit99)     value {np.mean(acc['v_any']):.2f}   random {np.mean(acc['r_any']):.2f}"
          f"   diverse {np.mean(acc['d_any']):.2f}")


def _settled_slates() -> list[Path]:
    out = []
    for d in sorted((PROJECT_ROOT / "archive").iterdir()):
        if not d.is_dir():
            continue
        found = discover_external_files(str(d))
        if (found["lineups_paths"] and (d / "contest_player_fpts.json").exists()
                and list(d.glob("contest-standings-*.zip"))):
            out.append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test whether any per-lineup signal predicts a realized top-1% finish.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
                   help="Slates to use (default: every settled slate in archive/).")
    p.add_argument("--mode", choices=["univariate", "loso", "cull", "coverage"], default="loso")
    p.add_argument("--top-n", type=int, default=150,
                   help="Entries a selection rule picks (default: 150).")
    p.add_argument("--proj-score-pct", type=float, default=25.0,
                   help="Projection cull applied before ranking, matching "
                        "gpp.external_pool_proj_score_pct (default: 25.0).")
    p.add_argument("--cull-grid", type=str, default="0,10,25,40,55",
                   help="Percentiles for --mode cull (default: 0,10,25,40,55).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dirs = [Path(x) for x in args.archive_dirs] if args.archive_dirs else _settled_slates()
    df = build_dataset(dirs)
    n_slates = df["slate"].nunique()
    share = df.groupby("slate")["hit99"].sum() / df["hit99"].sum()
    print(f"{len(df):,} lineups over {n_slates} settled slates, "
          f"{int(df['hit99'].sum())} hit99 ({df['hit99'].mean():.4f}); "
          f"effective independent slates {1 / np.sum(share.to_numpy() ** 2):.2f}\n")

    if args.mode == "univariate":
        run_univariate(df, args.top_n, args.proj_score_pct)
    elif args.mode == "loso":
        run_loso(df, args.top_n, args.proj_score_pct, args.seed)
    elif args.mode == "cull":
        run_cull(df, [float(v) for v in args.cull_grid.split(",")])
    else:
        run_coverage(df, args.top_n, args.proj_score_pct, args.seed)


if __name__ == "__main__":
    main()
