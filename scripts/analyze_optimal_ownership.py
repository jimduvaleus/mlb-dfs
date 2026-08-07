"""Phase A validation for the game-theoretic "optimal ownership" / leverage
idea (see plan doc need-a-plan-to-squishy-sutherland.md for the full design).

Core mechanism: score the external candidate pool against ITSELF as a
self-referential field -- ep.compute_p_win(scores, scores.T, exponents) --
instead of an ownership-sampled opponent field (ContestSimulator.generate_field)
or a per-simulated-world ILP solve. This gives, per real contest's field size,
a per-candidate "probability of being optimal" (p_opt) with no field
simulation and no per-world optimizer call. Aggregating p_opt through the
same player-indicator matrix compute_lineup_scores already builds gives a
smooth, field-size-conditioned "optimal ownership" O[p] -- diffed against
the pool's own projected ownership (SaberSim's "Adj Own"/"My Own" column,
players_df["ownership"]) to get leverage = O - projected.

Two checks, no portfolio construction:

  1. STABILITY -- O[p] is estimated off a field with only M (~8,000) discrete
     support points (the pool scored against itself), worse resolution than
     production's real p_win (opponent field grown to 25,000). Spearman
     correlation of O[p] across 3 sim seeds x 2 independent-draw halves (6
     draws) per real contest checks how much that resolution ceiling matters
     at each contest's real field size.
  2. REAL-OUTCOME VALIDATION -- does leverage[p] predict which players were
     actually good to be overweight on in the REAL field, better than a
     naive -own_pct[p] contrarian baseline? Per real contest: real_profit_lift
     per player = (mean $/entry among real entrants who rostered them) -
     (mean $/entry across that contest's whole field), from the standings
     zip's own Lineup/points columns. Spearman(leverage, real_profit_lift)
     vs Spearman(-own_pct, real_profit_lift) is the go/no-go signal for
     Phase B/C (see plan: do not proceed unless this is directionally
     positive for the leverage currency).

Checkpointed per slate under tests/backtest_output/optimal_ownership/
(--force to recompute) -- a slate's sim + pool-scoring + real-zip-parsing
work is not cheap, and stdout is flushed per print so a `tee`'d run shows
live progress instead of buffering until exit.

Usage
-----
    source venv/bin/activate
    python scripts/analyze_optimal_ownership.py
    python scripts/analyze_optimal_ownership.py --slates 07222026,08032026
    python scripts/analyze_optimal_ownership.py --force   # ignore checkpoints
"""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tests.bt_core import build_slate_context, load_real_contests  # noqa: E402
from tests.backtest_oracle import N_SIMS, SEEDS, SHARPNESS, SIM_CACHE_DIR  # noqa: E402
from scripts.analyze_rival_portfolio import parse_standings_rows  # noqa: E402
from src.api import external_pool as ep  # noqa: E402

TENTH_SLATE = "08032026"  # full named zips present, not yet in bt_core.BACKTEST_SLATES
CHECKPOINT_DIR = PROJECT_ROOT / "tests" / "backtest_output" / "optimal_ownership"

# players_df["ownership"] is percentage points (0-100, "My Own" scale --
# build_external_players_df). Floor before a ratio division mirrors
# tests/backtest_oracle.py::lineup_features' own.fillna(0.1).clip(lower=0.01).
OWN_FLOOR_PCT = 0.1
# Real per-contest profit-lift needs enough entrants rostering a player to
# not be pure noise -- 5 is a diagnostic-script floor, not a calibrated value.
MIN_REAL_ENTRIES = 5


def compute_optimal_ownership(pool, players_df, scores_half: np.ndarray,
                               exponents: dict) -> dict[str, np.ndarray]:
    """{bucket_label: (P,) percentage-point optimal ownership}.

    field_scores = scores_half.T makes compute_p_win's "field" the same M
    pool lineups the candidates are drawn from -- no ContestSimulator field,
    no per-simulated-world ILP. p_opt[j] = mean_over_worlds(percentile_in_
    pool(j) ** exponent); O[p] is the p_opt-weighted share of pool
    composition attributable to lineups containing player p (via the same
    player-indicator matmul compute_lineup_scores/compute_pool_corr use),
    scaled by 100 to match players_df["ownership"]'s percentage-point units.
    """
    I = ep._lineup_indicator_matrix(pool.lineups, players_df["player_id"].tolist())
    p_opt = ep.compute_p_win(scores_half, scores_half.T, exponents)
    out = {}
    for lbl, w in p_opt.items():
        w = w.astype(np.float64)
        denom = w.sum()
        frac = (I @ w) / denom if denom > 0 else np.zeros(I.shape[0])
        out[lbl] = 100.0 * frac
    return out


def _payout_for_field_entries(scores: np.ndarray, sorted_scores: np.ndarray,
                              payout_arr: np.ndarray) -> np.ndarray:
    """Vectorized twin of scripts/analyze_rival_roi.py::payout_for, for
    entries ALREADY IN sorted_scores (no insertion) -- real standings-zip
    entrants are already part of the field they're being graded against.

    Deliberately NOT bt_core.grade_pool/grade_pick/grade_portfolio: those
    insert an ADDITIONAL hypothetical entry on top of the real field (the
    right semantics for grading one of OUR candidate lineups, which isn't
    already in the standings), and their tie band is [n_above, n_above +
    n_tied + 1) -- the "+1" accounts for the inserted entry itself. Here the
    entry is already counted within n_tied (it's part of sorted_scores), so
    adding that +1 would double-count it and shift every real entrant's
    payout band by one rank.
    """
    n = len(sorted_scores)
    right = np.searchsorted(sorted_scores, scores, side="right")
    left = np.searchsorted(sorted_scores, scores, side="left")
    n_above = n - right
    n_tied = right - left
    L = len(payout_arr)
    cum = np.concatenate(([0.0], np.cumsum(payout_arr, dtype=np.float64)))
    lo = np.clip(n_above, 0, L)
    hi = np.clip(n_above + n_tied, 0, L)
    width = hi - lo
    return np.where(width > 0, (cum[hi] - cum[lo]) / np.maximum(width, 1), 0.0)


def real_contest_player_stats(d: Path, c: dict, name_to_id: dict) -> pd.DataFrame:
    """Per-player (real_pct_drafted, real_profit_lift, real_n_entries) for
    ONE real contest. real_profit_lift = (mean $/entry among entrants who
    rostered this player) - (mean $/entry across the whole real field).

    Fully vectorized: _payout_for_field_entries replaces a per-entry Python
    call to payout_for, and pandas explode/map/groupby replaces a per-entry
    per-rostered-player Python double loop -- both were the dominant cost on
    the slate's largest real fields (mini-MAX/Rally Cap, up to ~30k entries)
    in the first version of this script. DKSalaries names shared by two real
    players that slate are dropped entirely (mirrors bt_core.verify_slate's
    conservative default -- name_to_id already excludes them).
    """
    z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(
            zf.read(name).decode("utf-8-sig", errors="replace"))))
    e, fp = parse_standings_rows(rows)
    e = e[e["points"].notna()].copy()
    empty = pd.DataFrame(columns=["real_pct_drafted", "real_profit_lift", "real_n_entries"])
    if e.empty:
        return empty

    scores = e["points"].to_numpy(dtype=np.float64)
    gross = _payout_for_field_entries(scores, c["sorted_scores"], c["payout_arr"])
    e["profit"] = gross - c["fee"]
    field_mean = float(e["profit"].mean())

    long = e[["names", "profit"]].explode("names").rename(columns={"names": "name"})
    long["player_id"] = long["name"].map(name_to_id)
    long = long.dropna(subset=["player_id"])
    if long.empty:
        return empty
    long["player_id"] = long["player_id"].astype(int)
    agg = long.groupby("player_id")["profit"].agg(["sum", "count"])

    # Multi-position-eligible players get one sidebar row PER roster position
    # they were used at, each carrying that position's share of %Drafted (not
    # a repeated total) -- e.g. 07/22's Luis Rengifo is 12.76% at 2B + 1.79%
    # at 3B, summing to his real total ownership. Sum, not mean/last-wins.
    own = fp.assign(player_id=fp["player"].map(name_to_id)).dropna(subset=["player_id"])
    own["player_id"] = own["player_id"].astype(int)
    own_by_id = own.groupby("player_id")["pct_drafted"].sum()

    out = pd.DataFrame({
        "real_pct_drafted": own_by_id,
        "real_profit_lift": agg["sum"] / agg["count"] - field_mean,
        "real_n_entries": agg["count"],
    })
    out["real_n_entries"] = out["real_n_entries"].fillna(0).astype(int)
    return out


def resolve_ambiguous_names(d: Path, dup_names: set[str], sal: pd.DataFrame) -> dict[str, int]:
    """{name: player_id} for DKSalaries names shared by two real MLB players
    that slate (e.g. Max Muncy LAD/ATH, Jose Fermin STL/LAA -- both teams on
    the slate that day) where exactly one candidate id shows a signal of
    having actually been active: nonzero projected ownership ("My Own") or a
    batting Order, either present for one id and absent/zero for the other.
    A benched/inactive player draws ~0% projected ownership and no Order, so
    if only one of the two ids clears that bar, every real entrant who
    rostered this name that slate can be safely attributed to it.

    Deliberately weaker evidence than bt_core.resolve_duplicate_names (which
    resolves which id gets a REALIZED FPTS value, using the actual set of
    FPTS values observed across contests -- stronger evidence, but answering
    a different question). This script needs to attribute a real ENTRANT's
    roster slot to one specific id, and there's no per-entrant id column in
    the standings zip to cross-reference regardless, so pre-lock projection
    data is the only signal available either way.

    Names that don't resolve this way (both candidates active, both
    inactive, more than 2 candidates, or the projections export doesn't
    cover this slate) are left OUT of the returned dict -- callers should
    keep excluding those, same conservative default as before.
    """
    found = ep.discover_external_files(str(d))
    proj_path = found.get("projections_path")
    if proj_path is None:
        return {}
    proj = ep.parse_player_projections(proj_path)
    own_by_id = dict(zip(proj["player_id"], proj["ownership"]))
    order_by_id = dict(zip(proj["player_id"], proj["order"]))

    out: dict[str, int] = {}
    for name in dup_names:
        ids = [int(i) for i in sal.loc[sal["Name"] == name, "ID"]]
        if len(ids) != 2:
            continue  # 3+-way collisions: stay conservative, don't attempt
        active = []
        for pid in ids:
            own = own_by_id.get(pid)
            has_own = pd.notna(own) and own > 0
            has_order = pd.notna(order_by_id.get(pid))
            if has_own or has_order:
                active.append(pid)
        if len(active) == 1:
            out[name] = active[0]
    return out


def _load_checkpoint(slate: str) -> list[dict] | None:
    path = CHECKPOINT_DIR / f"{slate}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.to_dict("records")


def _save_checkpoint(slate: str, rows: list[dict]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CHECKPOINT_DIR / f"{slate}.csv", index=False)


def run_slate(slate: str, force: bool = False) -> list[dict]:
    if not force:
        cached = _load_checkpoint(slate)
        if cached is not None:
            print(f"\n=== {slate}: loaded from checkpoint "
                  f"({CHECKPOINT_DIR / f'{slate}.csv'}) ===", flush=True)
            for r in cached:
                print(f"{r['contest']:<16s}{int(r['n_field']):>9d}{r['stability']:>14.3f}"
                      f"{r['r_lev']:>13.3f}{r['r_base']:>18.3f}{r['r_O']:>14.3f}{int(r['n_pairs']):>9d}",
                      flush=True)
            return cached

    d = PROJECT_ROOT / "archive" / slate
    if not d.exists():
        print(f"{slate}: archive dir not found, skipping", flush=True)
        return []
    real = load_real_contests(d)
    if not real:
        print(f"{slate}: no real contests found, skipping", flush=True)
        return []

    # One slate context per seed -> per-half score matrices. pool/players_df
    # are pure functions of the static archive files (no seed dependence),
    # so only the sim draw differs seed to seed; want_pwin=False skips
    # production's own ContestSimulator.generate_field call entirely, since
    # this mechanism explicitly doesn't use an opponent field.
    per_seed_scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pool = players_df = None
    for seed in SEEDS:
        print(f"  {slate}: building context for seed={seed}...", flush=True)
        ctx = build_slate_context(d, seed, False, real, n_sims=N_SIMS,
                                  sharpness=SHARPNESS, sim_cache_dir=SIM_CACHE_DIR,
                                  want_corr=False, want_pwin=False)
        pool, players_df = ctx["pool"], ctx["players_df"]
        per_seed_scores[seed] = (ctx["scores_A"], ctx["scores_B"])
    M = len(pool.lineups)
    print(f"\n=== {slate}: M={M} pool lineups, {len(real)} real contests ===", flush=True)

    # Resolution cap: min(field_size, M) -- the self-referential field only
    # has M discrete support points, so an exponent built off the literal
    # (possibly much larger) real field size over-trusts resolution the
    # construction doesn't have. See plan doc's "pool size is a resolution
    # ceiling" section.
    exponents = {c["contest_id"]: max(1.0, SHARPNESS * min(c["n_field"], M)) for c in real}

    draws: dict[tuple, np.ndarray] = {}
    for seed, (scores_A, scores_B) in per_seed_scores.items():
        for half, scores in (("A", scores_A), ("B", scores_B)):
            O = compute_optimal_ownership(pool, players_df, scores, exponents)
            for cid, arr in O.items():
                draws[(cid, seed, half)] = arr
    print(f"  {slate}: p_opt/O[p] computed for {len(exponents)} field-size buckets "
          f"x {len(SEEDS)} seeds x 2 halves", flush=True)

    pids = players_df["player_id"].tolist()
    own_pct = players_df.set_index("player_id")["ownership"].astype(float).reindex(pids)

    sal = pd.read_csv(d / "DKSalaries.csv")
    sal["Name"] = sal["Name"].astype(str).str.strip()
    dup = set(sal["Name"][sal["Name"].duplicated(keep=False)])
    name_to_id = {n: int(i) for n, i in zip(sal["Name"], sal["ID"]) if n not in dup}
    resolved = resolve_ambiguous_names(d, dup, sal)
    if resolved:
        print(f"  {slate}: resolved {len(resolved)}/{len(dup)} ambiguous name(s) "
              f"via active-player signal: {sorted(resolved)}", flush=True)
    name_to_id.update(resolved)

    hdr = (f"{'contest':<16s}{'n_field':>9s}{'stability(r)':>14s}"
           f"{'lev_vs_real':>13s}{'baseline_vs_real':>18s}{'O_vs_realown':>14s}{'n_pairs':>9s}")
    print(hdr, flush=True)
    rows: list[dict] = []
    for c in real:
        cid, n_field = c["contest_id"], c["n_field"]
        six_draws = [draws[(cid, s, h)] for s in SEEDS for h in ("A", "B")]
        stack = np.vstack(six_draws)  # (6, P)
        rmat = pd.DataFrame(stack.T).corr(method="spearman").to_numpy()
        iu = np.triu_indices(6, k=1)
        stability = float(np.nanmean(rmat[iu]))

        O_series = pd.Series(stack.mean(axis=0), index=pids)
        leverage_diff = O_series - own_pct
        real_stats = real_contest_player_stats(d, c, name_to_id)
        real_stats = real_stats[real_stats["real_n_entries"] >= MIN_REAL_ENTRIES]

        merged = pd.DataFrame({
            "leverage_diff": leverage_diff, "O_pct": O_series, "own_pct": own_pct,
        }).join(real_stats, how="inner")

        n = len(merged)
        label = c["contest"]
        if n < 5:
            print(f"{label:<16s}{n_field:>9d}{stability:>14.3f}"
                  f"{'n/a':>13s}{'n/a':>18s}{'n/a':>14s}{n:>9d}", flush=True)
            continue

        r_lev = merged["leverage_diff"].corr(merged["real_profit_lift"], method="spearman")
        r_base = (-merged["own_pct"]).corr(merged["real_profit_lift"], method="spearman")
        r_O = merged["O_pct"].corr(merged["real_pct_drafted"], method="spearman")
        print(f"{label:<16s}{n_field:>9d}{stability:>14.3f}"
              f"{r_lev:>13.3f}{r_base:>18.3f}{r_O:>14.3f}{n:>9d}", flush=True)
        rows.append({"slate": slate, "contest": label, "n_field": n_field,
                     "stability": stability, "r_lev": r_lev, "r_base": r_base,
                     "r_O": r_O, "n_pairs": n})

    _save_checkpoint(slate, rows)
    print(f"  {slate}: checkpoint written -> {CHECKPOINT_DIR / f'{slate}.csv'}", flush=True)
    return rows


def print_summary(all_rows: list[dict]) -> None:
    """Pooled go/no-go read across every graded real contest: does leverage
    beat the naive -own_pct contrarian baseline more often than chance (a
    binomial sign test, since correlation magnitudes are individually noisy
    at one-contest sample sizes but the WIN/LOSE comparison is a cheap,
    robust aggregate), plus per-slate win counts so a single slate can't
    dominate the headline number the way one huge-field lottery win
    dominates a raw dollar total elsewhere in this codebase's evaluations."""
    from scipy.stats import binomtest

    if not all_rows:
        print("\nno gradeable contests found -- nothing to summarize", flush=True)
        return
    df = pd.DataFrame(all_rows)
    wins = int((df["r_lev"] > df["r_base"]).sum())
    n = len(df)
    pval = binomtest(wins, n, 0.5).pvalue

    print(f"\n===== POOLED SUMMARY: {n} real contests across {df['slate'].nunique()} slates =====",
          flush=True)
    print(f"  leverage beats naive -own_pct baseline: {wins}/{n} contests "
          f"(sign test p={pval:.4g})", flush=True)
    print(f"  mean r_lev={df['r_lev'].mean():+.3f}   mean r_base={df['r_base'].mean():+.3f}"
          f"   mean stability={df['stability'].mean():.3f}", flush=True)
    print("\n  per-slate win rate:", flush=True)
    per_slate = df.groupby("slate").apply(
        lambda g: pd.Series({
            "n_contests": len(g), "wins": int((g.r_lev > g.r_base).sum()),
            "mean_r_lev": g.r_lev.mean(), "mean_r_base": g.r_base.mean(),
        }), include_groups=False)
    print(per_slate.to_string(float_format=lambda v: f"{v:+.3f}"), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slates", default=f"07222026,{TENTH_SLATE}",
                    help="comma-separated MMDDYYYY (default: 07222026,08032026)")
    p.add_argument("--force", action="store_true",
                    help="ignore per-slate checkpoints and recompute everything")
    args = p.parse_args()
    all_rows: list[dict] = []
    for slate in args.slates.split(","):
        all_rows.extend(run_slate(slate.strip(), force=args.force))
    print_summary(all_rows)


if __name__ == "__main__":
    main()
