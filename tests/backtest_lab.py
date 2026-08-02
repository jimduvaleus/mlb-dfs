"""Fast strategy evaluation on top of the oracle tables.

tests/backtest_oracle.py precomputed, for every lineup in every pool, its
realized payout in every real contest. So a strategy here is just "produce
{contest_id: [lineup indices]}" and grading is an array lookup -- no pipeline
run, no simulation. That makes it affordable to compare dozens of arms, to
bootstrap, and to fit anything leave-one-slate-out.

WHY THIS EXISTS, i.e. what the existing harness measured:

  Production (`flat2000_uc`) shows +108.5% ROI over 3,804 entries, but a
  single $20,000 payout on 07/26 is 71.6% of everything it won. Drop that one
  slate and it is -46.9%, indistinguishable from every other arm. On the
  metrics that actually carry statistical power it is beaten by a coin flip:
  cash rate 17.9% vs random's 23.2%, top-1% rate 1.00% vs 1.28%.

  So the headline number is a lottery draw, and "beat +108.5%" is trivially
  achievable by luck. Every arm here is therefore judged on the protocol in
  `report()`: pooled $/entry, but gated on surviving leave-one-slate-out and
  the drop-largest-payout check, with the low-variance rate ladder reported
  alongside because that is what can actually distinguish two strategies at
  n=8 slates.

DENOMINATORS: lineups whose realized score is unknowable (they roster a
DKSalaries name shared by two real players that slate -- see
bt_core.verify_slate) are removed from the selectable pool up front, rather
than selected and then dropped at grading time as tests/backtest.py does.
Every arm therefore fills exactly the same number of entries, so per-entry
figures are comparable across arms without the harness's 2.6-6.1% denominator
spread.

    source venv/bin/activate
    python tests/backtest_lab.py verify      # substrate equivalence checks
    python tests/backtest_lab.py currencies  # Phase B: whole-pool decile lift
    python tests/backtest_lab.py arms        # portfolio arms + report
    python tests/backtest_lab.py adjudicate  # Amendment A1: model-light adjudication
"""
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import BACKTEST_SLATES  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / "tests" / "backtest_output" / "oracle"

# Currencies stored per (slate, seed, calib) by backtest_oracle.build_currencies.
PER_CONTEST_CURRENCIES = ("p_win", "ev_dollars", "p_cash", "ev_tail")
GLOBAL_CURRENCIES = ("p_beat99", "sim_mean", "sim_std", "sim_p99")


@dataclass
class SlateData:
    slate: str
    seed: int
    calib: bool
    cids: list                      # contest ids, in the order arms fill them
    contest: np.ndarray             # display names
    k: np.ndarray                   # entries per contest (fixed, real)
    fee: np.ndarray
    n_field: np.ndarray
    gross: np.ndarray               # (C, M) realized $ for every lineup
    rank: np.ndarray                # (C, M) realized rank
    actual: np.ndarray              # (M,) realized FPTS
    ok: np.ndarray                  # (M,) bool, gradeable (non-ambiguous)
    feats: dict                     # static per-lineup features
    cur: dict                       # {half: {name: (C, M) or (M,)}}

    @property
    def M(self) -> int:
        return len(self.actual)

    def currency(self, name: str, half: str = "B") -> np.ndarray:
        """(C, M) currency matrix. Global (contest-independent) currencies are
        broadcast to every contest so callers never special-case them."""
        v = self.cur[half][name]
        return np.broadcast_to(v, (len(self.cids), self.M)) if v.ndim == 1 else v


def load_slate(slate: str, seed: int = 42, calib: bool = False) -> SlateData:
    real = np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False)
    cur_f = np.load(ORACLE_DIR / f"{slate}_s{seed}_c{int(calib)}.npz", allow_pickle=False)
    cids = [str(x) for x in real["contest_id"]]
    assert [str(x) for x in cur_f["contest_id"]] == cids, f"{slate}: contest order drift"

    cur = {}
    for half in ("A", "B"):
        cur[half] = {n: cur_f[f"{half}_{n}"] for n in PER_CONTEST_CURRENCIES}
        cur[half].update({n: cur_f[f"{half}_{n}"] for n in GLOBAL_CURRENCIES})
    feats = {k: real[k] for k in real.files
             if real[k].ndim == 1 and len(real[k]) == len(real["actual_score"])
             and k not in ("actual_score",)}
    feats["proj_score"] = cur_f["proj_score"]
    return SlateData(
        slate=slate, seed=seed, calib=calib, cids=cids,
        contest=real["contest"], k=real["k"], fee=real["fee"], n_field=real["n_field"],
        gross=real["gross"], rank=real["rank"], actual=real["actual_score"],
        ok=np.isfinite(real["actual_score"]), feats=feats, cur=cur,
    )


def load_all(seed: int = 42, calib: bool = False, slates=None) -> list:
    return [load_slate(s, seed, calib) for s in (slates or BACKTEST_SLATES)]


@functools.lru_cache(maxsize=None)
def load_field(slate: str) -> dict:
    """{contest_id: (sorted_scores, payout_arr)} from
    tests/backtest_oracle.py's {slate}_field.npz sidecar -- the whole real
    ladder per contest, needed by grade_joint/bt_core.grade_portfolio (unlike
    {slate}_real.npz, which only carries the realized payout for pool
    lineups, not the field itself)."""
    z = np.load(ORACLE_DIR / f"{slate}_field.npz", allow_pickle=False)
    cids = [str(x) for x in z["contest_id"]]
    return {cid: (z[f"scores_{j}"], z[f"payout_{j}"]) for j, cid in enumerate(cids)}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def proj_floor_mask(sd: SlateData, floor_pct: float) -> np.ndarray:
    """Production's pool-wide projected-score cull, plus the gradeability
    filter (see the DENOMINATORS note in the module docstring)."""
    proj = sd.feats["proj_score"]
    mask = sd.ok & np.isfinite(proj)
    if floor_pct > 0:
        mask &= proj >= np.percentile(proj[np.isfinite(proj)], floor_pct)
    return mask


def select_greedy(sd: SlateData, sel: np.ndarray, cull: np.ndarray, *,
                  floor_pct: float = 30.0, admit_n: int = 2000,
                  admit_mult: float = 0.0, evw: float = 1.0,
                  corr: np.ndarray = None, corr_fn=None, order: list = None,
                  rng: np.random.Generator = None) -> dict:
    """Production's shape: per contest, cull to a window on the independent
    `cull` draw, then rank survivors on `sel`, removing picks from a shared
    mask so a lineup is used at most once per slate.

    evw=1.0 is pure EV ranking (no correlation matrix needed, which is most
    arms here); evw<1.0 delegates to the real DeterminantPortfolioSelector so
    the diversity/hedge terms are production's, not a reimplementation.
    rng set (with evw=None) draws uniformly from the survivors instead.

    `corr_fn`, if given, is a callable `rem -> (len(rem), len(rem))` matrix
    computed lazily for just that contest's admit window, for correlation
    sources too large to hold densely at (M, M) (e.g. composition overlap --
    see `_composition_overlap_fn`). Takes precedence over `corr` when both
    are supplied; existing callers pass only `corr` and are unaffected.
    """
    mask = proj_floor_mask(sd, floor_pct)
    picks: dict = {}
    for ci in (order if order is not None else range(len(sd.cids))):
        k = int(sd.k[ci])
        if k <= 0:
            continue
        rem = np.where(mask & np.isfinite(sel[ci]))[0]
        eff = max(admit_n, int(round(admit_mult * k))) if admit_mult > 0 else admit_n
        if eff > 0 and len(rem) > eff:
            rem = np.sort(rem[np.argsort(-cull[ci][rem])[:eff]])
        k = min(k, len(rem))
        if k == 0:
            continue
        if rng is not None:
            chosen = rem[rng.choice(len(rem), size=k, replace=False)]
        elif evw >= 1.0:
            chosen = rem[np.argsort(-sel[ci][rem])[:k]]
        else:
            from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
            sub_corr = (corr_fn(rem) if corr_fn is not None
                       else corr[np.ix_(rem, rem)])
            s = DeterminantPortfolioSelector(
                robust_payout=None, candidates=list(range(sd.M)), portfolio_size=k,
                risk=3.0, evw_base=evw, evw_max=evw, ev_floor=float("-inf"),
                precomputed=(rem, sel[ci][rem].astype(np.float64),
                             np.ascontiguousarray(sub_corr)),
            )
            chosen = np.array([i for i, _ in s.select()], dtype=np.int64)
        picks[sd.cids[ci]] = list(map(int, chosen))
        mask[chosen] = False
    return picks


def select_assign(sd: SlateData, sel: np.ndarray, cull: np.ndarray, *,
                  floor_pct: float = 30.0, admit_n: int = 2000,
                  admit_mult: float = 0.0) -> dict:
    """Global assignment instead of sequential greedy.

    With per-contest entry counts fixed and each lineup usable once, routing
    lineups to contests is a transportation problem -- maximize
    sum_c sum_{i in S_c} v[i,c] subject to |S_c| = k_c. Production instead
    walks contests in a fixed order and lets each take its best remaining
    lineups, so whichever contest happens to go first gets the pick of the
    pool and later ones are left with what survives. Nothing about that order
    reflects where a marginal lineup buys the most dollars.

    Only the top (total slots) lineups per contest can appear in an optimal
    solution, so the problem reduces to a few thousand rows and
    linear_sum_assignment solves it exactly in well under a second.
    """
    from scipy.optimize import linear_sum_assignment

    mask = proj_floor_mask(sd, floor_pct)
    total = int(sd.k.sum())
    cand = set()
    per_contest_rem = {}
    for ci in range(len(sd.cids)):
        rem = np.where(mask & np.isfinite(sel[ci]))[0]
        eff = max(admit_n, int(round(admit_mult * sd.k[ci]))) if admit_mult > 0 else admit_n
        if eff > 0 and len(rem) > eff:
            rem = np.sort(rem[np.argsort(-cull[ci][rem])[:eff]])
        per_contest_rem[ci] = set(map(int, rem))
        cand.update(map(int, rem[np.argsort(-sel[ci][rem])[:total]]))
    rows = np.array(sorted(cand), dtype=np.int64)

    # One column per entry slot; a lineup not admitted to that contest's
    # window is barred with -inf so the solver can never route it there.
    cols, col_ci = [], []
    for ci in range(len(sd.cids)):
        v = sel[ci][rows].astype(np.float64)
        v[[j for j, r in enumerate(rows) if int(r) not in per_contest_rem[ci]]] = -np.inf
        for _ in range(int(sd.k[ci])):
            cols.append(v)
            col_ci.append(ci)
    cost = -np.column_stack(cols)
    cost[~np.isfinite(cost)] = 1e18
    r_idx, c_idx = linear_sum_assignment(cost)

    picks: dict = {c: [] for c in sd.cids}
    for r, c in zip(r_idx, c_idx):
        if cost[r, c] >= 1e17:
            continue
        picks[sd.cids[col_ci[c]]].append(int(rows[r]))
    return picks


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(sd: SlateData, picks: dict, arm: str) -> pd.DataFrame:
    rows = []
    for ci, cid in enumerate(sd.cids):
        idx = picks.get(cid, [])
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        g = sd.gross[ci][idx]
        r = sd.rank[ci][idx]
        nf = int(sd.n_field[ci])
        rows.append(pd.DataFrame({
            "slate": sd.slate, "seed": sd.seed, "calib": sd.calib, "arm": arm,
            "contest": str(sd.contest[ci]), "cid": cid, "fee": float(sd.fee[ci]),
            "won": g, "rank": r, "n_field": nf,
            "cash": (g > 0).astype(int),
            "top1": (r <= max(1, nf // 100)).astype(int),
            "top01": (r <= max(1, nf // 1000)).astype(int),
        }))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    unfilled = int(sd.k.sum()) - len(df)
    if unfilled:
        print(f"    WARNING {arm} {sd.slate}: {unfilled} entries unfilled "
              "-- per-entry figures are on a smaller denominator")
    return df


def grade_joint(sd: SlateData, picks: dict, arm: str) -> pd.DataFrame:
    """Same output schema as `grade` (same columns, same handling of
    contests we placed nothing in / an all-empty picks dict), but per-contest
    gross/rank come from bt_core.grade_portfolio -- ALL of that contest's
    picked entries inserted into the real field JOINTLY -- instead of
    `grade`'s per-lineup lookup into {slate}_real.npz's single-insertion
    table. Needs the {slate}_field.npz sidecar (load_field) for the whole
    real ladder per contest, not just our lineups' realized payouts."""
    from tests.bt_core import grade_portfolio

    field = load_field(sd.slate)
    rows = []
    for ci, cid in enumerate(sd.cids):
        idx = picks.get(cid, [])
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        sorted_scores, payout_arr = field[cid]
        g, r = grade_portfolio(sd.actual[idx], sorted_scores, payout_arr)
        nf = int(sd.n_field[ci])
        rows.append(pd.DataFrame({
            "slate": sd.slate, "seed": sd.seed, "calib": sd.calib, "arm": arm,
            "contest": str(sd.contest[ci]), "cid": cid, "fee": float(sd.fee[ci]),
            "won": g, "rank": r, "n_field": nf,
            "cash": (g > 0).astype(int),
            "top1": (r <= max(1, nf // 100)).astype(int),
            "top01": (r <= max(1, nf // 1000)).astype(int),
        }))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    unfilled = int(sd.k.sum()) - len(df)
    if unfilled:
        print(f"    WARNING {arm} {sd.slate}: {unfilled} entries unfilled "
              "-- per-entry figures are on a smaller denominator")
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame, baseline: str = "flat2000_uc", n_boot: int = 20000) -> pd.DataFrame:
    """The adjudication protocol. Dollars are the stated objective, but at
    n=8 slates they are noise-dominated, so every dollar figure is shown with
    a paired bootstrap CI and two robustness columns:

      LOSO_min  worst leave-one-slate-out $/entry -- an arm that only wins
                with one slate in the sample fails here.
      drop_max  $/entry with that arm's single largest payout removed.

    The rate ladder (cash/top1%/top0.1%) is what actually discriminates at
    this sample size: a p~1-2% Bernoulli over ~3,800 entries has far lower
    relative variance than a payout with a CV of 30-50.
    """
    df = df.copy()
    slates = sorted(df.slate.unique())
    out = []
    per = {a: g for a, g in df.groupby("arm")}
    base = per.get(baseline)

    for arm, g in per.items():
        fees = g.fee.sum()
        net = g.won.sum() - fees
        loso = []
        for s in slates:
            h = g[g.slate != s]
            loso.append((h.won.sum() - h.fee.sum()) / max(len(h), 1))
        gmax = g.won.max()
        drop = (g.won.sum() - gmax - fees) / max(len(g), 1)
        row = {
            "arm": arm, "entries": len(g), "fees": fees, "won": g.won.sum(),
            "net": net, "ROI%": 100 * net / fees if fees else np.nan,
            "$/entry": net / len(g),
            "LOSO_min": min(loso), "LOSO_max": max(loso), "drop_max": drop,
            "cash%": 100 * g.cash.mean(),
            "top1%": 100 * g.top1.mean(), "top01%": 100 * g.top01.mean(),
            "best": int(g["rank"].min()),
        }
        if base is not None and arm != baseline:
            # Paired by slate: resample slates with replacement, which is the
            # unit that actually varies (seeds only redraw the sim).
            a_by = g.groupby("slate").apply(
                lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
            b_by = base.groupby("slate").apply(
                lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
            common = a_by.index.intersection(b_by.index)
            d = (a_by[common] - b_by[common]).to_numpy()
            rng = np.random.default_rng(0)
            bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
            row["d$/entry"] = d.mean()
            row["d_lo95"] = np.percentile(bs, 2.5)
            row["d_hi95"] = np.percentile(bs, 97.5)
            row["win_slates"] = f"{int((d > 0).sum())}/{len(d)}"
        out.append(row)

    res = pd.DataFrame(out).sort_values("$/entry", ascending=False)
    return res.reset_index(drop=True)


def print_report(res: pd.DataFrame, title: str) -> None:
    cols_money = ["arm", "entries", "ROI%", "$/entry", "LOSO_min", "LOSO_max", "drop_max"]
    cols_rate = ["arm", "cash%", "top1%", "top01%", "best"]
    cols_cmp = [c for c in ("arm", "d$/entry", "d_lo95", "d_hi95", "win_slates")
                if c in res.columns]
    print(f"\n===== {title} =====")
    print("\n-- dollars (noise-dominated; LOSO_min < 0 means the win is one slate) --")
    print(res[cols_money].round(3).to_string(index=False))
    print("\n-- rate ladder (this is what has the statistical power) --")
    print(res[cols_rate].round(3).to_string(index=False))
    if len(cols_cmp) > 1:
        print("\n-- paired vs baseline, bootstrap over slates --")
        print(res[cols_cmp].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Substrate verification
# ---------------------------------------------------------------------------

def _brute_grade_portfolio(actual_scores: np.ndarray, sorted_real: np.ndarray,
                           payout_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent O((n+k) log(n+k)) reference for bt_core.grade_portfolio,
    used only by cmd_verify's check 4.

    Deliberately NOT the searchsorted/prefix-sum algebra grade_portfolio
    itself uses: this literally merges our k scores into the field's score
    list (tagging which entries are ours), sorts the merged list descending,
    walks it top-down assigning 1-indexed ranks, groups consecutive equal-
    score entries into one tie band, and averages the clipped payout window
    over that band -- same clipped-width convention as grade_pool /
    grade_portfolio (band = payout_arr[lo:hi] clipped to the table length,
    mean over the CLIPPED width). A genuinely different code path from the
    vectorized implementation, so it can catch a bug in that implementation's
    own logic rather than merely re-deriving the same formula.

    NaN entries are excluded from the merge (they don't displace/tie with
    anyone) and come back as gross=NaN, rank=-1, matching grade_portfolio.
    """
    k = len(actual_scores)
    L = len(payout_arr)
    cum = np.concatenate(([0.0], np.cumsum(payout_arr, dtype=np.float64)))
    gross = np.full(k, np.nan)
    rank = np.full(k, -1, dtype=np.int64)

    tagged = [(float(v), i) for i, v in enumerate(actual_scores) if np.isfinite(v)]
    field = [(float(v), -1) for v in sorted_real]
    merged = sorted(tagged + field, key=lambda t: -t[0])  # descending by score
    n = len(merged)
    idx = 0
    while idx < n:
        v = merged[idx][0]
        j = idx
        while j < n and merged[j][0] == v:
            j += 1
        # 0-indexed positions [idx, j) tie at this score -> 1-indexed ranks
        # idx+1 .. j. lo/hi mirror grade_pool's clipped-width band exactly.
        lo, hi = min(idx, L), min(j, L)
        width = hi - lo
        band_mean = (cum[hi] - cum[lo]) / width if width > 0 else 0.0
        for p in range(idx, j):
            _, tag = merged[p]
            if tag >= 0:
                gross[tag] = band_mean
                rank[tag] = idx + 1
        idx = j
    return gross, rank


def cmd_verify(slate: str = "07222026", seed: int = 42) -> None:
    """Two independent checks that the fast substrate is the real thing.

    1. grade_pool == grade_pick for EVERY (lineup, contest) on every slate --
       an exhaustive equivalence proof of the vectorization, not a spot check.
    2. The lab's selection + oracle grading reproduces, to the cent, what
       production's own ep.allocate_contests + bt_core.grade_pick produce for
       the same arm. If this fails nothing downstream can be trusted.
    """
    from src.api import external_pool as ep
    from tests.bt_core import (build_slate_context, grade_pick, grade_pool,
                               grade_portfolio, load_real_contests, _FakeGroup)

    print("== check 1: grade_pool == grade_pick, exhaustively ==")
    worst = 0.0
    for s in BACKTEST_SLATES:
        if not (ORACLE_DIR / f"{s}_real.npz").exists():
            print(f"  {s}: no oracle table yet, skipped")
            continue
        sd = load_slate(s, seed, False)
        real = load_real_contests(PROJECT_ROOT / "archive" / s)
        by_id = {c["contest_id"]: c for c in real}
        for ci, cid in enumerate(sd.cids):
            c = by_id[cid]
            ref = np.array([
                grade_pick(a, c["sorted_scores"], c["payout_arr"])[0] if np.isfinite(a) else np.nan
                for a in sd.actual
            ])
            d = np.nanmax(np.abs(ref - sd.gross[ci]))
            worst = max(worst, float(d))
        print(f"  {s}: {len(sd.cids)} contests x {sd.M} lineups, max |diff| = {worst:.3g}")
    print(f"  => max |grade_pool - grade_pick| over everything = {worst:.3g}"
          f"  {'PASS' if worst == 0 else 'FAIL'}")

    print(f"\n== check 2: lab reproduces production's allocate_contests ({slate} s{seed}) ==")
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    ctx = build_slate_context(d, seed, False, real, n_sims=25000, sharpness=0.05,
                              sim_cache_dir=PROJECT_ROOT / "tests/backtest_output/sim_cache")
    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ctx["contests"] if c["k"] > 0]
    alloc = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0, evw_base=0.25, evw_max=0.25,
        proj_scores=ctx["proj_scores"], proj_score_floor_percentile=30.0,
        ev_type="p_win", p_win_cull=ctx["p_win_cull"], p_win_select=ctx["p_win_select"],
        p_win_admit_n=2000, p_win_admit_multiplier=0.0,
    )
    idx_of = {id(lu): i for i, lu in enumerate(ctx["pool"].lineups)}
    prod_picks: dict = {}
    i = 0
    for g in groups:
        prod_picks[g.contest_id] = [idx_of[id(lu)] for lu, _ in alloc.portfolio[i:i + len(g.entries)]]
        i += len(g.entries)
    prod_total = 0.0
    # grade production's picks through the ORACLE table
    sd = load_slate(slate, seed, False)
    oracle_total = 0.0
    for ci, cid in enumerate(sd.cids):
        for j in prod_picks.get(cid, []):
            g = sd.gross[ci][j]
            if np.isfinite(g):
                oracle_total += g
    # ... and through the harness's own per-pick path
    by_id = {c["contest_id"]: c for c in real}
    actual = sd.actual
    for cid, idxs in prod_picks.items():
        c = by_id[cid]
        for j in idxs:
            if np.isfinite(actual[j]):
                prod_total += grade_pick(actual[j], c["sorted_scores"], c["payout_arr"])[0]
    print(f"  production picks graded via grade_pick : ${prod_total:,.2f}")
    print(f"  same picks graded via the oracle table : ${oracle_total:,.2f}")
    print(f"  => {'PASS' if abs(prod_total - oracle_total) < 1e-6 else 'FAIL'}")

    # 3. does the lab's own selection reproduce production's picks?
    lab_picks = select_greedy(
        sd, sd.currency("p_win", "B"), sd.currency("p_win", "A"),
        floor_pct=30.0, admit_n=2000, admit_mult=0.0, evw=0.25, corr=ctx["corr"],
    )
    same = sum(len(set(lab_picks.get(c, [])) & set(prod_picks.get(c, [])))
               for c in sd.cids)
    tot = sum(len(v) for v in prod_picks.values())
    print(f"\n  lab select_greedy(evw=0.25) vs production: {same}/{tot} identical picks")
    print("  (the lab drops ambiguous-name lineups up front, so a small gap here "
          "is expected on slates that have them)")

    print("\n== check 3: grade_portfolio([v]) == grade_pick(v), k=1 reduction, "
          "2000-lineup sample per slate ==")
    rng3 = np.random.default_rng(20260802)
    n_checked3 = mismatches3 = 0
    for s in BACKTEST_SLATES:
        if not (ORACLE_DIR / f"{s}_real.npz").exists():
            print(f"  {s}: no oracle table yet, skipped")
            continue
        sd3 = load_slate(s, seed, False)
        real3 = load_real_contests(PROJECT_ROOT / "archive" / s)
        by_id3 = {c["contest_id"]: c for c in real3}
        gradeable3 = np.where(sd3.ok)[0]
        n_sample = min(2000, len(gradeable3))
        sample = rng3.choice(gradeable3, size=n_sample, replace=False)
        sample_scores = sd3.actual[sample]
        slate_mismatch = 0
        for cid in sd3.cids:
            c = by_id3[cid]
            # grade_pool (already proven == grade_pick in check 1) as the bulk
            # reference, so the loop only pays for grade_portfolio's own call.
            ref_g, ref_r = grade_pool(sample_scores, c["sorted_scores"], c["payout_arr"])
            for j, v in enumerate(sample_scores):
                g, r = grade_portfolio(np.array([v]), c["sorted_scores"], c["payout_arr"])
                n_checked3 += 1
                if g[0] != ref_g[j] or r[0] != ref_r[j]:
                    mismatches3 += 1
                    slate_mismatch += 1
        print(f"  {s}: {n_sample} lineups x {len(sd3.cids)} contests, "
              f"{slate_mismatch} mismatches")
    print(f"  => {n_checked3:,} k=1 checks, {mismatches3} mismatches "
          f"{'PASS' if mismatches3 == 0 else 'FAIL'}")

    print("\n== check 4: brute-force cross-check, 50 random portfolios/slate "
          "(k in [2,300], dupes injected) ==")
    rng4 = np.random.default_rng(9182736)
    n_portfolios4 = n_cells4 = mismatches4 = property_violations4 = 0
    worst4 = 0.0
    for s in BACKTEST_SLATES:
        if not (ORACLE_DIR / f"{s}_real.npz").exists():
            print(f"  {s}: no oracle table yet, skipped")
            continue
        sd4 = load_slate(s, seed, False)
        real4 = load_real_contests(PROJECT_ROOT / "archive" / s)
        gradeable4 = np.where(sd4.ok)[0]
        for _trial in range(50):
            k = int(rng4.integers(2, 301))
            idx = rng4.choice(gradeable4, size=k, replace=True)  # replace=True injects dupes
            scores = sd4.actual[idx]
            n_portfolios4 += 1
            for c in real4:
                joint_g, joint_r = grade_portfolio(scores, c["sorted_scores"], c["payout_arr"])
                brute_g, brute_r = _brute_grade_portfolio(scores, c["sorted_scores"], c["payout_arr"])
                d = np.max(np.abs(joint_g - brute_g))
                worst4 = max(worst4, float(d))
                n_cells4 += k
                if not (np.array_equal(joint_r, brute_r) and
                        np.allclose(joint_g, brute_g, atol=1e-9)):
                    mismatches4 += 1
                # property: joint total gross <= sum of single-insertion grosses.
                single_total = sum(
                    grade_pick(v, c["sorted_scores"], c["payout_arr"])[0] for v in scores
                )
                joint_total = float(joint_g.sum())
                if joint_total > single_total + 1e-6:
                    property_violations4 += 1
        print(f"  {s}: 50 portfolios x {len(real4)} contests checked", flush=True)
    print(f"  => {n_portfolios4} portfolios x contests, {n_cells4:,} entry-cells, "
          f"max |diff| {worst4:.3g}, {mismatches4} mismatches "
          f"{'PASS' if mismatches4 == 0 else 'FAIL'}")
    print(f"  => joint-total <= sum-of-single-insertions property: "
          f"{property_violations4} violations "
          f"{'PASS' if property_violations4 == 0 else 'FAIL'}")


# ---------------------------------------------------------------------------
# SaberSim's own per-lineup columns
# ---------------------------------------------------------------------------

def load_saber(slate: str) -> dict:
    """{column: (M,) array} of SaberSim's own simulated per-lineup columns,
    aligned to the oracle table's lineup order.

    The exports carry `<contest> ROI`, `Win Rate`, `Cash Rate` and `Sim Dupes`
    per contest block, but src/api/external_pool.py's parser keeps only ROI
    and ROI StDev -- under `ev_type: p_win` (production) it doesn't read any
    of them. That's a whole independent simulation's opinion, from a vendor
    whose sim we don't control, going unused. Worth measuring before assuming
    ours is better. Sim Dupes matters separately: duplicate entries dilute
    top-band payouts, and nothing in the external-pool path models that.

    Aligned by frozenset of player ids (exact duplicates are already removed
    from the pool, so the key is unique) rather than by row order, which the
    parser's dedup/near-dup passes do not preserve.
    """
    import csv as csv_mod
    from src.api import external_pool as ep

    path = ORACLE_DIR / f"{slate}_saber.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    real = np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False)
    pids = real["player_ids"]
    key_to_row = {frozenset(map(int, row)): i for i, row in enumerate(pids)}
    M = len(pids)

    found = ep.discover_external_files(str(PROJECT_ROOT / "archive" / slate))
    cols: dict = {}
    for p in found["lineups_paths"]:
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv_mod.reader(f))
        header = rows[0]
        hset = set(header)
        wanted: dict = {}
        for j, col in enumerate(header):
            for suffix, tag in ((" ROI", "roi"), (" Win Rate", "win_rate"),
                                (" Cash Rate", "cash_rate"), (" Sim Dupes", "dupes")):
                if col.endswith(suffix) and f"{col[:-len(suffix)]} Sim Dupes" in hset:
                    wanted.setdefault(tag, []).append(j)
            if col in ("Proj Score", "Ownership", "Salary"):
                wanted[col.lower().replace(" ", "_")] = j
        for tag in wanted:
            cols.setdefault(tag, np.full(M, np.nan))
        for r in rows[1:]:
            if len(r) < 10:
                continue
            try:
                k = frozenset(int(x) for x in r[:10])
            except ValueError:
                continue
            i = key_to_row.get(k)
            if i is None:
                continue
            for tag, j in wanted.items():
                try:
                    if isinstance(j, list):
                        # average the per-contest blocks: one lineup-level
                        # summary rather than a currency for one contest
                        vals = [float(r[jj]) for jj in j if r[jj] not in ("", None)]
                        cols[tag][i] = np.mean(vals) if vals else np.nan
                    else:
                        cols[tag][i] = float(r[j])
                except (ValueError, IndexError):
                    pass
    np.savez_compressed(path, **cols)
    return cols


# ---------------------------------------------------------------------------
# Phase B: does any currency actually carry dollar signal?
# ---------------------------------------------------------------------------

def _candidate_currencies(sd: SlateData) -> dict:
    """Every ranking signal available pre-lock, as (C, M) matrices."""
    C, M = len(sd.cids), sd.M
    bc = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (C, M))  # noqa: E731
    out = {
        "p_win": sd.currency("p_win", "B"),
        "ev_dollars": sd.currency("ev_dollars", "B"),
        "ev_tail": sd.currency("ev_tail", "B"),
        "p_cash": sd.currency("p_cash", "B"),
        "p_beat99": sd.currency("p_beat99", "B"),
        "sim_mean": sd.currency("sim_mean", "B"),
        "sim_p99": sd.currency("sim_p99", "B"),
        "sim_std": sd.currency("sim_std", "B"),
        "proj_score": bc(sd.feats["proj_score"]),
        "neg_own": bc(-sd.feats["own_sum"]),
        "salary": bc(sd.feats["salary_sum"]),
        "max_stack": bc(sd.feats["max_stack"]),
    }
    # EV per dollar of entry fee, the quantity that actually sets ROI.
    out["ev_per_fee"] = sd.currency("ev_dollars", "B") / sd.fee[:, None]
    # Production's own prj_own currency (ev_type="prj_own", never used live):
    # projected score minus an ownership penalty scaled by field size.
    proj, own = sd.feats["proj_score"], sd.feats["own_sum"]
    out["prj_own"] = np.stack([proj - own * (nf / 30_000.0) for nf in sd.n_field])
    # Rank-space contrarian blends: equal-weighted so neither term's scale
    # decides the ranking, which the raw prj_own subtraction lets it do.
    rp, ro = _rank_norm(proj), _rank_norm(own)
    out["proj_minus_own"] = bc(rp - ro)
    out["ceiling_contrarian"] = bc(_rank_norm(sd.cur["B"]["sim_p99"]) - ro)
    # "Everything except the top decile of p_win" -- the cliff seen in the
    # decile-lift table, expressed as a currency: keep p_win ordering but
    # send its most confident 10% to the back.
    pw = sd.currency("p_win", "B")
    cut = np.percentile(pw, 90, axis=1, keepdims=True)
    out["p_win_no_top10"] = np.where(pw >= cut, -np.inf, pw)
    try:
        sab = load_saber(sd.slate)
        for tag in ("roi", "win_rate", "cash_rate", "dupes"):
            if tag in sab and np.isfinite(sab[tag]).any():
                out[f"saber_{tag}"] = bc(np.nan_to_num(sab[tag], nan=np.nanmin(sab[tag])))
    except Exception as exc:  # export without those blocks -- not fatal
        print(f"    note: {sd.slate} saber columns unavailable ({exc})")
    return out


def _decile_agg(df: pd.DataFrame, entry_weighted: bool = True) -> pd.DataFrame:
    """Pooled per-decile rates.

    Each row of `df` is one (slate, contest, currency, decile) cell holding n
    pool lineups. The quantity every table here wants is "what would an ENTRY
    drawn from this decile have returned", so a cell must contribute in
    proportion to the entries actually placed in that contest (k), not to how
    many pool lineups happen to sit in it (n). w = k/n converts a cell's
    lineup totals into that contest's entry-weighted share.

    entry_weighted=False restores the old cell-count pooling, kept only so the
    two can be printed side by side.
    """
    if entry_weighted:
        w = df.k / df.n
        return df.assign(_w=w).groupby(["currency", "decile"]).apply(
            lambda x: pd.Series({
                "$/entry": ((x.won * x._w).sum() - (x.fee * x.k).sum()) / x.k.sum(),
                "cash%": 100 * (x.cash * x._w).sum() / x.k.sum(),
                "top1%": 100 * (x.top1 * x._w).sum() / x.k.sum(),
                "top01%": 100 * (x.top01 * x._w).sum() / x.k.sum(),
            }), include_groups=False)
    return df.groupby(["currency", "decile"]).apply(
        lambda x: pd.Series({
            "$/entry": (x.won.sum() - (x.n * x.fee).sum()) / x.n.sum(),
            "cash%": 100 * x.cash.sum() / x.n.sum(),
            "top1%": 100 * x.top1.sum() / x.n.sum(),
            "top01%": 100 * x.top01.sum() / x.n.sum(),
        }), include_groups=False)


def _decile_consistency(df: pd.DataFrame, metric: str, n_deciles: int) -> pd.DataFrame:
    """Per-slate top-minus-bottom decile contrast, so a pooled gradient carried
    by one slate is visible in the same output that reports the gradient.

    This exists because the ownership fade passed every pooled screen and then
    failed here: its pooled top-1% gradient looked monotone while 98-100% of
    the contrast came from a single slate (07/25) and the sign was positive on
    only 5/8. Any pooled decile claim needs n_pos and a LOSO range next to it.
    """
    per = {}
    for (cur, s), g in df.groupby(["currency", "slate"]):
        a = _decile_agg(g.assign(slate=s))[metric]
        if len(a) < n_deciles:
            continue
        per.setdefault(cur, {})[s] = a.iloc[-1] - a.iloc[0]
    rows = []
    for cur, by_slate in per.items():
        d = np.array(list(by_slate.values()))
        if len(d) < 2:
            continue
        loso = np.array([(d.sum() - x) / (len(d) - 1) for x in d])
        rows.append({
            "currency": cur, "top-bot": d.mean(),
            "n_pos": f"{int((d > 0).sum())}/{len(d)}",
            "sign_p": _sign_p(int((d > 0).sum()), len(d)),
            "LOSO_lo": loso.min(), "LOSO_hi": loso.max(),
            "max_slate_share": (100 * d.max() / d[d > 0].sum()
                                if (d > 0).any() else np.nan),
        })
    return pd.DataFrame(rows).set_index("currency")


def cmd_currencies(seed: int = 42, calib: bool = False, n_deciles: int = 10) -> None:
    """Whole-pool decile lift: split every contest's pool into deciles by each
    candidate currency and measure the REALIZED $/entry, cash rate and top-1%
    rate in each. Pooled over 8 slates this is ~80k lineup-contest cells,
    versus the 24 portfolio draws the existing harness compares arms on.

    This is the honest test of whether a currency carries dollar signal at
    all, independent of any selection machinery wrapped around it -- but only
    once two things are right, both of which this originally got wrong:

    WEIGHTING. Pooling over cells gives every (contest, decile) cell weight
    proportional to its pool-lineup count, so a contest we place 1 entry in
    counts as much as one we place 95 in, and a slate's influence becomes its
    pool size x contest count -- which spans 40k-110k cells against 114-212
    real entries, a 2.8x spread unrelated to anything we care about. Entry
    weighting (w = k/n) is the decision-relevant one. Measured on ownership:
    it moves the top-1% d0/d9 gradient from 2.16x to 1.64x, i.e. a quarter of
    the apparent gradient was contest weighting alone.

    CONSISTENCY. A pooled gradient says nothing about whether it recurs. Every
    table is therefore printed with per-slate n_pos, a sign test and a LOSO
    range on the top-minus-bottom contrast. Ownership passed the pooled screen
    with a clean monotone gradient and then turned out to be 5/8 slates with
    98-100% of the contrast from one of them; salary did the same thing
    earlier. Read the consistency block first, the pooled table second.
    """
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    recs = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        curs = _candidate_currencies(sd)
        mask = proj_floor_mask(sd, 0.0)  # whole pool; the floor is measured separately
        for ci in range(len(sd.cids)):
            k = int(sd.k[ci])
            if k <= 0:
                continue  # no entries placed -> no weight under entry weighting
            idx = np.where(mask)[0]
            nf = int(sd.n_field[ci])
            g = sd.gross[ci][idx]
            r = sd.rank[ci][idx]
            top1 = ((r <= max(1, nf // 100)) & (r > 0)).astype(float)
            top01 = ((r <= max(1, nf // 1000)) & (r > 0)).astype(float)
            for name, v in curs.items():
                x = v[ci][idx]
                fin = np.isfinite(x)
                if fin.sum() < n_deciles * 10:
                    continue
                q = pd.qcut(pd.Series(x[fin]).rank(method="first"), n_deciles,
                            labels=False, duplicates="drop")
                for d in range(n_deciles):
                    m = q == d
                    if not m.any():
                        continue
                    recs.append({
                        "slate": s, "cid": sd.cids[ci], "currency": name, "decile": d,
                        "n": int(m.sum()), "k": k, "fee": float(sd.fee[ci]),
                        "won": float(g[fin][m].sum()), "top1": float(top1[fin][m].sum()),
                        "top01": float(top01[fin][m].sum()),
                        "cash": float((g[fin][m] > 0).sum()),
                    })
    df = pd.DataFrame(recs)
    out = ORACLE_DIR.parent / "lab_decile_lift.csv"
    df.to_csv(out, index=False)

    agg = _decile_agg(df, entry_weighted=True)
    old = _decile_agg(df, entry_weighted=False)

    print(f"\n===== WHOLE-POOL DECILE LIFT (seed {seed}, calib={calib}, "
          f"{len(slates)} slates) =====")
    print("Entry-weighted: each contest contributes in proportion to the entries")
    print("we actually place in it, not to how many pool lineups it happens to have.")
    print("Decile 9 = the currency's own top 10%. Signal should rise left-to-right.\n")

    for metric in ("$/entry", "top1%", "top01%"):
        piv = agg[metric].unstack()
        piv["top-bot"] = piv[piv.columns[-1]] - piv[piv.columns[0]]
        cons = _decile_consistency(df, metric, n_deciles)
        piv = piv.join(cons[["n_pos", "sign_p", "LOSO_lo", "LOSO_hi",
                             "max_slate_share"]])
        # how much of the contrast was the old cell weighting?
        o = old[metric].unstack()
        piv["cellwt_top-bot"] = o[o.columns[-1]] - o[o.columns[0]]
        print(f"-- {metric} by decile --")
        print(piv.round(3).sort_values("top-bot", ascending=False).to_string())
        print("   n_pos/LOSO are the per-slate top-minus-bottom contrast; a currency")
        print("   whose LOSO range straddles 0, or whose max_slate_share is near 100,")
        print("   is one slate, not a signal. cellwt_top-bot is the old weighting.\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Portfolio arms
# ---------------------------------------------------------------------------

def build_arms() -> dict:
    """{name: (currency, mode, kwargs)}.

    `currency` names a key of _candidate_currencies; selection always ranks on
    the B half and culls on the independent A half, exactly as production's
    two-stage winner's-curse guard does. `mode` is "greedy" (production's
    sequential per-contest fill) or "assign" (the global transportation
    solution). Random arms carry currency=None.
    """
    arms: dict = {}
    # Baselines: production, and the coin flip it currently loses to.
    arms["prod_p_win"] = ("p_win", "greedy", dict(floor_pct=30.0, admit_n=2000))
    arms["random"] = (None, "greedy", dict(floor_pct=30.0, admit_n=0))
    arms["random_nofloor"] = (None, "greedy", dict(floor_pct=0.0, admit_n=0))
    # Currency comparison at a fixed selection shape.
    for cur in ("ev_dollars", "ev_per_fee", "ev_tail", "p_cash", "p_beat99",
                "sim_mean", "sim_p99", "proj_score", "neg_own"):
        arms[f"{cur}"] = (cur, "greedy", dict(floor_pct=30.0, admit_n=2000))
    # Structure: global assignment vs sequential greedy, on the two currencies
    # most likely to matter.
    for cur in ("ev_dollars", "ev_per_fee", "p_win"):
        arms[f"{cur}@assign"] = (cur, "assign", dict(floor_pct=30.0, admit_n=2000))
    # Admit-window width on the new currency.
    for n in (100, 500, 2000, 0):
        arms[f"ev_dollars@admit{n}"] = ("ev_dollars", "greedy",
                                        dict(floor_pct=30.0, admit_n=n))
    # The projected-score floor, isolated. The funnel decomposition shows it
    # REMOVING realized value rather than concentrating it, so sweep it to
    # zero on production's own currency and on the coin flip.
    for f in (0.0, 10.0, 30.0, 50.0):
        arms[f"p_win@floor{f:g}"] = ("p_win", "greedy",
                                     dict(floor_pct=f, admit_n=2000))
        arms[f"random@floor{f:g}"] = (None, "greedy", dict(floor_pct=f, admit_n=0))
    # Contrarian currencies. The whole-pool decile lift shows ownership is
    # the one axis that orders top-1% rate monotonically (0.670 -> 1.450),
    # while every EV currency inverts at its own top decile.
    for cur in ("neg_own", "prj_own", "proj_minus_own", "ceiling_contrarian",
                "p_win_no_top10"):
        arms[cur] = (cur, "greedy", dict(floor_pct=30.0, admit_n=2000))
        arms[f"{cur}@nofloor"] = (cur, "greedy", dict(floor_pct=0.0, admit_n=0))
    return arms


def run_arms(seed: int = 42, calib: bool = False, slates=None) -> pd.DataFrame:
    slates = [s for s in (slates or BACKTEST_SLATES)
              if (ORACLE_DIR / f"{s}_real.npz").exists()]
    arms = build_arms()
    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        curs = _candidate_currencies(sd)
        curs_A = {"p_win": sd.currency("p_win", "A"),
                  "ev_dollars": sd.currency("ev_dollars", "A"),
                  "ev_tail": sd.currency("ev_tail", "A"),
                  "p_cash": sd.currency("p_cash", "A")}
        for name, (cur, mode, kw) in arms.items():
            if cur is None:
                # zlib.crc32, not hash() -- Python randomizes string hashing
                # per process, which would make the random arms irreproducible
                # across runs while still writing under the same label.
                import zlib
                rng = np.random.default_rng(
                    zlib.crc32(f"{s}|{seed}|{name}".encode()) & 0xFFFFFFFF)
                picks = select_greedy(sd, curs["p_win"], curs_A["p_win"],
                                      rng=rng, **kw)
            else:
                sel = curs[cur]
                # Cull on the independent A draw where we have one; currencies
                # with no A twin (static features) cull on themselves, which
                # is harmless -- they carry no sim noise to overfit to.
                cull = curs_A.get(cur, sel)
                fn = select_assign if mode == "assign" else select_greedy
                picks = fn(sd, sel, cull, **kw)
            frames.append(grade(sd, picks, name))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def cmd_arms(seed: int = 42, calib: bool = False) -> None:
    df = run_arms(seed, calib)
    out = ORACLE_DIR.parent / f"lab_arms_s{seed}_c{int(calib)}.csv"
    df.to_csv(out, index=False)
    res = report(df, baseline="prod_p_win")
    print_report(res, f"PORTFOLIO ARMS (seed {seed}, calib={calib}, "
                      f"{df.slate.nunique()} slates)")
    print(f"\nwrote {out}")


def cmd_stages(seed: int = 42, calib: bool = False) -> None:
    """Funnel decomposition: what does each stage do to realized value?

    Production's funnel is three successive narrowings -- the 30% projected-
    score floor, the top-2000 p_win admit window, then the ranked pick. Each
    is supposed to concentrate value. The oracle table lets us measure the
    realized $/entry, cash rate and top-1% rate of the surviving SET after
    each stage, against the pool average, so a stage that destroys value
    instead of concentrating it is visible on its own rather than only in the
    end-to-end result.
    """
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    recs = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        sel, cull = sd.currency("p_win", "B"), sd.currency("p_win", "A")
        for ci in range(len(sd.cids)):
            g, r, nf, fee = sd.gross[ci], sd.rank[ci], int(sd.n_field[ci]), float(sd.fee[ci])
            k = int(sd.k[ci])
            base = np.where(sd.ok)[0]
            floor = np.where(proj_floor_mask(sd, 30.0))[0]
            win = floor[np.argsort(-cull[ci][floor])[:2000]]
            top = win[np.argsort(-sel[ci][win])[:k]]
            for stage, idx in (("1 pool", base), ("2 +proj floor 30%", floor),
                               ("3 +p_win top2000", win), ("4 +ranked pick", top)):
                recs.append({
                    "slate": s, "cid": sd.cids[ci], "stage": stage, "n": len(idx),
                    "fee": fee, "won": float(g[idx].sum()),
                    "cash": float((g[idx] > 0).sum()),
                    "top1": float((r[idx] <= max(1, nf // 100)).sum()),
                    "entries": k,
                })
    df = pd.DataFrame(recs)
    # Weight every contest equally by its real entry count, so the summary is
    # "what would an entry drawn from this stage have returned", not an
    # average over wildly different set sizes.
    df["w"] = df.entries / df.n
    agg = df.groupby("stage").apply(lambda x: pd.Series({
        "mean $/entry": (x.won * x.w).sum() / x.entries.sum() - (x.fee * x.entries).sum() / x.entries.sum(),
        "gross $/entry": (x.won * x.w).sum() / x.entries.sum(),
        "cash%": 100 * (x.cash * x.w).sum() / x.entries.sum(),
        "top1%": 100 * (x.top1 * x.w).sum() / x.entries.sum(),
        "set size": x.n.mean(),
    }), include_groups=False)
    print(f"\n===== FUNNEL DECOMPOSITION (seed {seed}, calib={calib}, "
          f"{len(slates)} slates) =====")
    print("Value of the AVERAGE surviving lineup after each stage.")
    print("A stage that concentrates value raises these; one that destroys it lowers them.\n")
    print(agg.round(3).to_string())


# ---------------------------------------------------------------------------
# Phase E: fitted value model, leave-one-slate-out
# ---------------------------------------------------------------------------

def _rank_norm(x: np.ndarray) -> np.ndarray:
    """Within-slate rank in [0,1]. Every feature goes through this so a model
    fit across slates can't be thrown by a scale shift between them (pool
    sizes, ownership totals and sim scales all move slate to slate), and so
    the fit is driven by ordering, which is all a selection rule uses."""
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x)
    out = np.full(len(x), 0.5)
    if fin.sum() > 1:
        r = pd.Series(x[fin]).rank(method="average").to_numpy()
        out[fin] = (r - 0.5) / len(r)
    return out


def build_model_table(seed: int = 42, calib: bool = False, slates=None) -> pd.DataFrame:
    """One row per (slate, lineup): rank-normalized features + the realized
    outcome. A lineup's realized score is ONE number regardless of contest, so
    this is ~80k rows with a single target, not a lineup-by-contest cross
    product -- and the target is the lineup's percentile against the real
    field, which is exactly what the payout table is a function of.
    """
    slates = [s for s in (slates or BACKTEST_SLATES)
              if (ORACLE_DIR / f"{s}_real.npz").exists()]
    # Only features present on EVERY slate can be fit leave-one-slate-out:
    # SaberSim's ROI/Win Rate/Cash Rate/Sim Dupes blocks are absent from the
    # 07/28 onward exports, so a model including them would be fit on folds
    # where they exist and scored on folds where they don't.
    common = None
    per_slate = {}
    for s in slates:
        sd = load_slate(s, seed, calib)
        per_slate[s] = sd
        names = set(_candidate_currencies(sd))
        common = names if common is None else (common & names)
    dropped = set().union(*[set(_candidate_currencies(per_slate[s])) for s in slates]) - common
    if dropped:
        print(f"  dropped (not on every slate): {sorted(dropped)}")

    rows = []
    for s in slates:
        sd = per_slate[s]
        curs = {k: v for k, v in _candidate_currencies(sd).items() if k in common}
        ok = sd.ok
        big = int(np.argmax(sd.n_field))          # deepest field = finest percentile
        pct = 1.0 - (sd.rank[big] - 1) / sd.n_field[big]
        d = {"slate": s, "lineup": np.where(ok)[0], "realized": sd.actual[ok],
             "pct": pct[ok]}
        for name, v in curs.items():
            # collapse per-contest currencies to the deepest contest's column;
            # ranking within a slate is what the model learns
            d[name] = _rank_norm(v[big][ok])
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def cmd_model(seed: int = 42, calib: bool = False) -> None:
    """Fit realized field-percentile on pre-lock features, leave-one-slate-out.

    With 8 slates, in-sample fitting is meaningless -- so the model is fit on 7
    and scored only on the 8th, rotated. Both the honest (held-out) and the
    in-sample numbers are printed so the overfit gap is visible rather than
    implied. If held-out lift is flat that is itself the finding: it would say
    the pool is close to unpredictable at this horizon and effort belongs in
    coverage rather than ranking.
    """
    tab = build_model_table(seed, calib)
    feats = [c for c in tab.columns if c not in ("slate", "lineup", "realized", "pct")]
    slates = sorted(tab.slate.unique())
    print(f"\n===== LOSO VALUE MODEL ({len(slates)} slates, {len(tab):,} lineups, "
          f"{len(feats)} features) =====")

    X_all = tab[feats].to_numpy(dtype=np.float64)
    y_all = tab["pct"].to_numpy(dtype=np.float64)
    slate_col = tab["slate"].to_numpy()

    def ridge(X, y, alpha):
        """Closed-form ridge on centered features (numpy only -- sklearn is not
        in this venv and adding a dependency for a 5-line solve isn't worth
        it). The intercept is handled by centering, so alpha never penalizes
        it."""
        mx, my = X.mean(axis=0), y.mean()
        Xc = X - mx
        A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xc.T @ (y - my))
        return w, my - mx @ w

    def fit_with_inner_cv(X, y, groups):
        """alpha picked by an inner leave-one-slate-out over the TRAINING
        slates only, so the held-out slate never influences it."""
        alphas = np.logspace(-2, 5, 15)
        uniq = np.unique(groups)
        err = np.zeros(len(alphas))
        for g in uniq:
            tr, te = groups != g, groups == g
            for ai, a in enumerate(alphas):
                w, b = ridge(X[tr], y[tr], a)
                err[ai] += np.mean((X[te] @ w + b - y[te]) ** 2)
        best = alphas[int(np.argmin(err))]
        return (*ridge(X, y, best), best)

    preds = np.full(len(tab), np.nan)
    coefs, alphas_used = [], []
    for s in slates:
        tr, te = slate_col != s, slate_col == s
        w, b, a = fit_with_inner_cv(X_all[tr], y_all[tr], slate_col[tr])
        preds[te] = X_all[te] @ w + b
        coefs.append(pd.Series(w, index=feats))
        alphas_used.append(a)
    tab["pred"] = preds

    w_i, b_i, _ = fit_with_inner_cv(X_all, y_all, slate_col)
    tab["pred_ins"] = X_all @ w_i + b_i
    print(f"inner-CV alphas per fold: {[f'{a:.3g}' for a in alphas_used]}")

    print("\nper-slate Spearman(prediction, realized percentile):")
    print("  slate      held-out   in-sample   best single feature")
    for s in slates:
        g = tab[tab.slate == s]
        rho_o = g.pred.corr(g.pct, method="spearman")
        rho_i = g.pred_ins.corr(g.pct, method="spearman")
        singles = {f: g[f].corr(g.pct, method="spearman") for f in feats}
        bf = max(singles, key=lambda f: abs(singles[f]))
        print(f"  {s}   {rho_o:+8.3f}   {rho_i:+8.3f}   {bf} {singles[bf]:+.3f}")
    print(f"  {'POOLED':10s} {tab.pred.corr(tab.pct, method='spearman'):+8.3f}   "
          f"{tab.pred_ins.corr(tab.pct, method='spearman'):+8.3f}")

    cf = pd.concat(coefs, axis=1)
    print("\nmean LOSO coefficient (sign stability across folds in brackets):")
    for f, v in cf.mean(axis=1).sort_values(key=abs, ascending=False).items():
        agree = int((np.sign(cf.loc[f]) == np.sign(v)).sum())
        print(f"  {f:16s} {v:+8.4f}  [{agree}/{len(slates)}]")

    print("\nper-feature pooled Spearman vs realized percentile "
          "(single-signal view, no model):")
    for f in feats:
        per = [tab[tab.slate == s][f].corr(tab[tab.slate == s].pct, method="spearman")
               for s in slates]
        print(f"  {f:16s} mean {np.mean(per):+.3f}  range [{min(per):+.3f},{max(per):+.3f}]"
              f"  sign-consistent {int(sum(np.sign(p)==np.sign(np.mean(per)) for p in per))}"
              f"/{len(slates)}")

    out = ORACLE_DIR.parent / f"lab_model_s{seed}_c{int(calib)}.csv"
    tab.to_csv(out, index=False)
    print(f"\nwrote {out}")


def cmd_null(seed: int = 42, calib: bool = False, n_draws: int = 400) -> None:
    """The null distribution of each funnel configuration, precisely.

    Every dollar figure elsewhere is ONE draw of ~1,300 entries against
    payouts with a CV of 30-50, so two identical configurations can land
    $1.50/entry apart purely by luck. Rather than argue about single draws,
    this samples each configuration many times and reports the distribution
    -- which both estimates E[$/entry] far more precisely and says exactly
    how unusual production's realized result is against its own null.

    Cheap only because of the oracle tables: 400 draws x 8 slates x several
    configurations is a few million array lookups, not thousands of pipeline
    runs.
    """
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    configs = {
        "pool (no cull)": dict(floor_pct=0.0, admit_n=0),
        "floor30": dict(floor_pct=30.0, admit_n=0),
        "floor30 + p_win top2000": dict(floor_pct=30.0, admit_n=2000),
        "floor30 + p_win top100": dict(floor_pct=30.0, admit_n=100),
        "p_win top2000, no floor": dict(floor_pct=0.0, admit_n=2000),
    }
    rows = []
    for name, kw in configs.items():
        draws = []
        for t in range(n_draws):
            won = fees = n = cash = top1 = 0.0
            for s in slates:
                sd = load_slate(s, seed, calib)
                rng = np.random.default_rng(t * 7919 + zlib_hash(s))
                picks = select_greedy(sd, sd.currency("p_win", "B"),
                                      sd.currency("p_win", "A"), rng=rng, **kw)
                for ci, cid in enumerate(sd.cids):
                    idx = np.asarray(picks.get(cid, []), dtype=np.int64)
                    if not len(idx):
                        continue
                    g, r = sd.gross[ci][idx], sd.rank[ci][idx]
                    nf = int(sd.n_field[ci])
                    won += g.sum(); fees += float(sd.fee[ci]) * len(idx); n += len(idx)
                    cash += (g > 0).sum(); top1 += (r <= max(1, nf // 100)).sum()
            draws.append(((won - fees) / n, 100 * cash / n, 100 * top1 / n))
        a = np.array(draws)
        rows.append({
            "config": name, "draws": n_draws,
            "$/entry mean": a[:, 0].mean(), "sd": a[:, 0].std(),
            "p5": np.percentile(a[:, 0], 5), "p50": np.percentile(a[:, 0], 50),
            "p95": np.percentile(a[:, 0], 95),
            "cash% mean": a[:, 1].mean(), "top1% mean": a[:, 2].mean(),
        })
        print(f"  {name}: done", flush=True)
    res = pd.DataFrame(rows)
    print(f"\n===== NULL DISTRIBUTION ({n_draws} random portfolios per config, "
          f"{len(slates)} slates, seed {seed}) =====")
    print("What a RANDOM selection from each funnel stage actually returns.\n")
    print(res.round(3).to_string(index=False))
    out = ORACLE_DIR.parent / f"lab_null_s{seed}_c{int(calib)}.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")


def zlib_hash(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode()) & 0xFFFF


def cmd_slices(seed: int = 42, calib: bool = False, n_draws: int = 300) -> None:
    """Search for a slice of the pool with positive expected $/entry.

    "Rank by X and take the best k" is not the only way to use a signal, and
    the decile-lift table suggests it is the wrong one: several currencies
    have a profitable TOP DECILE while their top ~0.2% (which is what taking
    k of ~2000 amounts to) is not. So this measures the strategy "restrict to
    the top q by currency, then draw randomly inside it" -- the width q is
    the actual free parameter, and randomness inside the slice avoids the
    over-concentration that kills the ranked versions.

    Each configuration gets n_draws random portfolios for a precise mean, AND
    a leave-one-slate-out bootstrap so a slice that only works because of one
    slate is visible as such.
    """
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    sds = {s: load_slate(s, seed, calib) for s in slates}
    curs = {s: _candidate_currencies(sds[s]) for s in slates}

    cand = ["proj_score", "p_cash", "sim_mean", "sim_p99", "neg_own", "p_win",
            "ev_dollars", "prj_own", "salary"]
    fracs = [0.05, 0.10, 0.25, 0.50]
    rows = []
    for cur in cand:
        for q in fracs:
            per_slate = {s: [] for s in slates}
            for t in range(n_draws):
                for s in slates:
                    sd, cv = sds[s], curs[s][cur]
                    rng = np.random.default_rng(t * 7919 + zlib_hash(s))
                    base = proj_floor_mask(sd, 0.0)
                    won = fees = n = 0.0
                    mask = base.copy()
                    for ci, cid in enumerate(sd.cids):
                        k = int(sd.k[ci])
                        rem = np.where(mask & np.isfinite(cv[ci]))[0]
                        keep = max(k, int(round(q * len(rem))))
                        if len(rem) > keep:
                            rem = rem[np.argsort(-cv[ci][rem])[:keep]]
                        take = rng.choice(len(rem), size=min(k, len(rem)), replace=False)
                        idx = rem[take]
                        mask[idx] = False
                        won += sd.gross[ci][idx].sum()
                        fees += float(sd.fee[ci]) * len(idx)
                        n += len(idx)
                    per_slate[s].append((won - fees) / n)
            means = np.array([np.mean(per_slate[s]) for s in slates])
            rng = np.random.default_rng(0)
            bs = means[rng.integers(0, len(means), size=(20000, len(means)))].mean(axis=1)
            rows.append({
                "currency": cur, "keep": q, "$/entry": means.mean(),
                "lo95": np.percentile(bs, 2.5), "hi95": np.percentile(bs, 97.5),
                "pos_slates": f"{int((means > 0).sum())}/{len(means)}",
                "LOSO_min": min(means.sum() - m for m in means) / (len(means) - 1),
            })
        print(f"  {cur}: done", flush=True)
    res = pd.DataFrame(rows).sort_values("$/entry", ascending=False)
    print(f"\n===== SLICE SEARCH ({n_draws} draws/config, {len(slates)} slates) =====")
    print("Strategy: keep the top `keep` fraction by currency, then draw randomly.")
    print("Per-slate means averaged over draws, then bootstrapped over slates.\n")
    print(res.round(3).to_string(index=False))
    out = ORACLE_DIR.parent / f"lab_slices_s{seed}_c{int(calib)}.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")


def cmd_selector(seed: int = 42, calib: bool = False) -> None:
    """The diversity dial, against the REAL production baseline.

    Every arm in cmd_arms ranks by pure EV (evw=1.0), which is NOT what
    production does: it runs DeterminantPortfolioSelector at evw=0.25, so
    three quarters of the selection score is the diversity/hedge term. That
    distinction turns out to matter enormously -- pure p_win top-k is far
    worse than production -- so comparing anything to a pure-EV "production"
    arm would be comparing to a strawman.

    This builds the real correlation matrix per slate (the expensive part,
    ~5 min/slate) and sweeps currency x evw through production's own
    selector, so the baseline is genuinely production and the diversity
    weight is measured rather than inherited.
    """
    from tests.bt_core import build_slate_context, load_real_contests

    sweep_evw = [0.0, 0.1, 0.25, 0.5, 1.0]
    currencies = ["p_win", "p_cash", "sim_mean", "ev_dollars", "neg_own"]
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    frames = []
    for s in slates:
        d = PROJECT_ROOT / "archive" / s
        real = load_real_contests(d)
        ctx = build_slate_context(
            d, seed, calib, real, n_sims=25000, sharpness=0.05,
            sim_cache_dir=PROJECT_ROOT / "tests/backtest_output/sim_cache",
            want_corr=True, want_pwin=False,
        )
        corr = ctx["corr"]
        sd = load_slate(s, seed, calib)
        assert corr.shape[0] == sd.M, f"{s}: corr {corr.shape} vs oracle M={sd.M}"
        curs = _candidate_currencies(sd)
        curs_A = {"p_win": sd.currency("p_win", "A"),
                  "p_cash": sd.currency("p_cash", "A"),
                  "ev_dollars": sd.currency("ev_dollars", "A")}
        for cur in currencies:
            for evw in sweep_evw:
                sel = curs[cur]
                picks = select_greedy(
                    sd, sel, curs_A.get(cur, sel), floor_pct=30.0, admit_n=2000,
                    evw=evw, corr=corr,
                )
                frames.append(grade(sd, picks, f"{cur}@evw{evw:g}"))
        print(f"  {s}: {len(currencies) * len(sweep_evw)} selector arms graded",
              flush=True)
        del corr, ctx
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = ORACLE_DIR.parent / f"lab_selector_s{seed}_c{int(calib)}.csv"
    df.to_csv(out, index=False)
    res = report(df, baseline="p_win@evw0.25")
    print_report(res, f"DIVERSITY DIAL vs REAL PRODUCTION (p_win@evw0.25), "
                      f"seed {seed}, {df.slate.nunique()} slates")
    print(f"\nwrote {out}")


def cmd_seedcheck(calib: bool = False, seeds=(42, 137, 4242)) -> None:
    """Is production's own baseline stable across sim seeds?

    Everything production has been compared against -- the -51% ROI, the
    1.060% top-1%, the 0.076% top-0.1% -- was measured on seed 42 alone. A
    baseline that moves as much between seeds as the arms move against it
    cannot adjudicate anything, and every arm ranked against it inherits that
    seed's luck. The correlation matrix is seed-dependent too (it is built
    from the same sim), so each (slate, seed) gets its own context rather
    than reusing seed 42's corr with another seed's currencies.

    Two evw values because production's diversity weight is itself a live
    knob: evw=0.25 is what ships, evw=0.1 is the neighbouring setting.
    """
    from tests.bt_core import build_slate_context, load_real_contests

    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    frames = []
    for s in slates:
        d = PROJECT_ROOT / "archive" / s
        real = load_real_contests(d)
        for seed in seeds:
            ctx = build_slate_context(
                d, seed, calib, real, n_sims=25000, sharpness=0.05,
                sim_cache_dir=PROJECT_ROOT / "tests/backtest_output/sim_cache",
                want_corr=True, want_pwin=False,
            )
            corr = ctx["corr"]
            sd = load_slate(s, seed, calib)
            assert corr.shape[0] == sd.M, f"{s} s{seed}: corr {corr.shape} vs M={sd.M}"
            sel, cull = sd.currency("p_win", "B"), sd.currency("p_win", "A")
            for evw in (0.25, 0.1):
                picks = select_greedy(sd, sel, cull, floor_pct=30.0, admit_n=2000,
                                      evw=evw, corr=corr)
                frames.append(grade(sd, picks, f"p_win@evw{evw:g}|s{seed}"))
            print(f"  {s} s{seed}: done", flush=True)
            del corr, ctx
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = ORACLE_DIR.parent / f"lab_seedcheck_c{int(calib)}.csv"
    df.to_csv(out, index=False)

    print(f"\n===== PRODUCTION BASELINE ACROSS SEEDS (calib={calib}, "
          f"{df.slate.nunique()} slates) =====")
    agg = df.groupby("arm").apply(lambda g: pd.Series({
        "entries": len(g),
        "ROI%": 100 * (g.won.sum() - g.fee.sum()) / g.fee.sum(),
        "$/entry": (g.won.sum() - g.fee.sum()) / len(g),
        "cash%": 100 * g.cash.mean(),
        "top1%": 100 * g.top1.mean(),
        "top01%": 100 * g.top01.mean(),
        "best": int(g["rank"].min()),
    }), include_groups=False)
    print(agg.round(3).to_string())

    print("\nspread across the three seeds (this is the baseline's own noise):")
    for evw in (0.25, 0.1):
        sub = agg.loc[[i for i in agg.index if i.startswith(f"p_win@evw{evw:g}|")]]
        print(f"  evw={evw:g}: " + "  ".join(
            f"{c} {sub[c].min():.3f}-{sub[c].max():.3f} (range {sub[c].max()-sub[c].min():.3f})"
            for c in ("top1%", "top01%", "cash%", "$/entry")))

    print("\nper-slate top-1% rate (rows = slate, cols = arm):")
    ps = df.groupby(["slate", "arm"]).apply(
        lambda g: 100 * g.top1.mean(), include_groups=False).unstack()
    print(ps.round(3).to_string())
    print("\nper-slate top-0.1% rate:")
    ps01 = df.groupby(["slate", "arm"]).apply(
        lambda g: 100 * g.top01.mean(), include_groups=False).unstack()
    print(ps01.round(3).to_string())
    print(f"\nwrote {out}")


# ---------------------------------------------------------------------------
# Ownership: the one monotone signal in the decile-lift table
# ---------------------------------------------------------------------------

def own_set_rates(sd: SlateData, q: float, *, floor_pct: float = 0.0,
                  admit_n: int = 0, cull: np.ndarray = None,
                  key: np.ndarray = None) -> dict:
    """EXACT expected rates of "restrict to the slice, then draw uniformly".

    No Monte Carlo: drawing k entries uniformly without replacement from a set
    S has expected rate exactly mean(outcome over S), so the set mean IS the
    estimator and it carries zero simulation noise. Contests are weighted by
    their real entry count k_c, so the answer reads as "what an entry drawn
    from this slice would have returned", comparably to cmd_stages.

    The one thing this does NOT model is production's shared mask (a lineup is
    used at most once per slate), which depletes the slice for later contests.
    Slices here are 100-4,000 lineups against 114-212 total entries per slate,
    so depletion is second-order -- and `cmd_own` runs the real sequential
    arms too, which do carry it, as a check.
    """
    key = sd.feats["own_sum"] if key is None else key
    base = proj_floor_mask(sd, floor_pct)
    tot = cash = top1 = top01 = won = fees = 0.0
    sizes = []
    for ci in range(len(sd.cids)):
        k = int(sd.k[ci])
        if k <= 0:
            continue
        rem = np.where(base)[0]
        if admit_n > 0 and len(rem) > admit_n:
            rem = rem[np.argsort(-cull[ci][rem])[:admit_n]]
        if q < 1.0:
            keep = max(k, int(round(q * len(rem))))
            if len(rem) > keep:
                rem = rem[np.argsort(key[rem])[:keep]]   # lowest ownership first
        sizes.append(len(rem))
        nf = int(sd.n_field[ci])
        g, r = sd.gross[ci][rem], sd.rank[ci][rem]
        w = k / len(rem)
        won += float(g.sum()) * w
        fees += float(sd.fee[ci]) * k
        cash += float((g > 0).sum()) * w
        top1 += float((r <= max(1, nf // 100)).sum()) * w
        top01 += float((r <= max(1, nf // 1000)).sum()) * w
        tot += k
    return {"entries": tot, "slice": float(np.mean(sizes)),
            "cash%": 100 * cash / tot, "top1%": 100 * top1 / tot,
            "top01%": 100 * top01 / tot, "$/entry": (won - fees) / tot}


def _sign_p(n_pos: int, n: int) -> float:
    """Two-sided exact sign test -- the only test that needs no distributional
    assumption about per-slate lifts, which are heavy-tailed by construction."""
    from math import comb
    k = max(n_pos, n - n_pos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cmd_own(seed: int = 42, calib: bool = False, n_draws: int = 200) -> None:
    """Does the ownership gradient survive per slate, and how wide should the
    fade be if the objective is top-1% rate rather than dollars?

    Ownership is static -- it needs no sim, so this is seed-independent by
    construction and runs off the oracle tables in seconds. The seed argument
    only selects which currency file supplies p_win for the funnel variant.

    Discipline note: a competing static signal (salary) looked equally strong
    pooled and collapsed on exactly this per-slate check, so every table here
    is reported per slate with a sign test, and salary is carried alongside
    as the negative control rather than dropped.
    """
    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    sds = {s: load_slate(s, seed, calib) for s in slates}

    # -- 1. per-slate decile gradient -------------------------------------
    print(f"\n===== 1. OWNERSHIP DECILE GRADIENT, PER SLATE ({len(slates)} slates) =====")
    print("Whole pool split into deciles of lineup ownership sum; decile 0 = the")
    print("CONTRARIAN end. Cells are top-1% rate (%), k-weighted over contests.\n")
    for metric, lab in (("top1", "top-1% rate"), ("top01", "top-0.1% rate")):
        rows = {}
        for s in slates:
            sd = sds[s]
            own = sd.feats["own_sum"]
            idx_all = np.where(sd.ok)[0]
            order = idx_all[np.argsort(own[idx_all])]
            chunks = np.array_split(order, 10)
            num = np.zeros(10)
            den = 0.0
            for ci in range(len(sd.cids)):
                k = int(sd.k[ci])
                if k <= 0:
                    continue
                nf = int(sd.n_field[ci])
                cut = max(1, nf // (100 if metric == "top1" else 1000))
                hit = (sd.rank[ci] <= cut) & (sd.rank[ci] > 0)
                for d, ch in enumerate(chunks):
                    num[d] += 100 * hit[ch].mean() * k
                den += k
            rows[s] = num / den
        tab = pd.DataFrame(rows, index=[f"d{i}" for i in range(10)]).T
        tab["d0-d9"] = tab["d0"] - tab["d9"]
        pooled = tab.iloc[:, :10].mean(axis=0)
        n_pos = int((tab["d0-d9"] > 0).sum())
        print(f"-- {lab} by ownership decile --")
        print(tab.round(3).to_string())
        print(f"   pooled: {'  '.join(f'{v:.3f}' for v in pooled)}")
        print(f"   contrarian end higher on {n_pos}/{len(slates)} slates "
              f"(sign test p={_sign_p(n_pos, len(slates)):.3f})\n")

    # -- 2. fade-width sweep, standalone and in-funnel ---------------------
    fracs = [0.05, 0.10, 0.25, 0.50, 1.00]
    variants = {
        "standalone": dict(floor_pct=0.0, admit_n=0),
        "funnel(floor30+admit2000)": dict(floor_pct=30.0, admit_n=2000),
    }
    print(f"===== 2. FADE-WIDTH SWEEP ({len(slates)} slates, exact set means) =====")
    print("Keep the least-owned `keep` fraction, then draw uniformly. keep=1.00 is")
    print("the variant's own baseline (no ownership fade at all).\n")
    sweep_rows = []
    per_slate_store = {}
    for vname, vkw in variants.items():
        for q in fracs:
            per = {}
            for s in slates:
                sd = sds[s]
                per[s] = own_set_rates(sd, q, cull=sd.currency("p_win", "A"), **vkw)
            per_slate_store[(vname, q)] = per
            agg = {m: float(np.average([per[s][m] for s in slates],
                                       weights=[per[s]["entries"] for s in slates]))
                   for m in ("cash%", "top1%", "top01%", "$/entry")}
            sweep_rows.append({"variant": vname, "keep": q,
                               "slice": np.mean([per[s]["slice"] for s in slates]),
                               **agg})
    sw = pd.DataFrame(sweep_rows)
    print(sw.round(3).to_string(index=False))

    print("\n-- per-slate lift vs that variant's keep=1.00 baseline --")
    for vname in variants:
        base = per_slate_store[(vname, 1.00)]
        print(f"\n  [{vname}]")
        for metric in ("top1%", "top01%"):
            hdr = f"    {metric:7s} keep "
            print(hdr + "  ".join(f"{s[:4]:>7s}" for s in slates) + "   mean   pos  sign_p")
            for q in fracs[:-1]:
                per = per_slate_store[(vname, q)]
                d = np.array([per[s][metric] - base[s][metric] for s in slates])
                n_pos = int((d > 0).sum())
                print(f"    {'':7s} {q:.2f} " + "  ".join(f"{x:+7.3f}" for x in d)
                      + f"  {d.mean():+6.3f}  {n_pos}/{len(slates)}  "
                        f"{_sign_p(n_pos, len(slates)):.3f}")

    # -- 3. negative control: the same sweep on salary ---------------------
    print(f"\n===== 3. NEGATIVE CONTROL -- same sweep keyed on SALARY =====")
    print("Salary looked as good as ownership pooled and collapsed per slate.")
    print("If ownership's per-slate counts look like these, it is the same artifact.\n")
    for s_key, s_lab in ((lambda sd: -sd.feats["salary_sum"], "low salary"),
                         (lambda sd: sd.feats["salary_sum"], "high salary")):
        base = {s: own_set_rates(sds[s], 1.00, key=s_key(sds[s])) for s in slates}
        print(f"  [{s_lab} first]")
        for metric in ("top1%", "top01%"):
            for q in (0.10, 0.25):
                per = {s: own_set_rates(sds[s], q, key=s_key(sds[s])) for s in slates}
                d = np.array([per[s][metric] - base[s][metric] for s in slates])
                n_pos = int((d > 0).sum())
                print(f"    {metric:7s} keep {q:.2f}  mean {d.mean():+6.3f}  "
                      f"pos {n_pos}/{len(slates)}  sign_p {_sign_p(n_pos, len(slates)):.3f}")

    # -- 4. real sequential arms (carry the shared mask + dollars) ---------
    print(f"\n===== 4. REAL SEQUENTIAL ARMS ({n_draws} random draws inside each slice) =====")
    print("Same strategies run through the lab's own selection loop, so the")
    print("shared once-per-slate mask and the real per-contest fill are included.\n")
    frames = []
    for s in slates:
        sd = sds[s]
        cull = sd.currency("p_win", "A")
        arms = {}
        for vname, vkw in variants.items():
            for q in (0.05, 0.10, 0.25, 0.50, 1.00):
                arms[f"own{q:.2f}|{vname}"] = (q, vkw)
        for name, (q, vkw) in arms.items():
            for t in range(n_draws):
                rng = np.random.default_rng(t * 7919 + zlib_hash(s + name))
                picks = _slice_draw(sd, q, rng, cull=cull, **vkw)
                g = grade(sd, picks, name)
                if not g.empty:
                    g["draw"] = t
                    frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    out = ORACLE_DIR.parent / f"lab_own_s{seed}_c{int(calib)}.csv"
    df.to_csv(out, index=False)
    agg = df.groupby("arm").apply(lambda g: pd.Series({
        "entries/draw": len(g) / g.draw.nunique(),
        "$/entry": (g.won.sum() - g.fee.sum()) / len(g),
        "cash%": 100 * g.cash.mean(), "top1%": 100 * g.top1.mean(),
        "top01%": 100 * g.top01.mean(),
    }), include_groups=False).sort_values("top1%", ascending=False)
    print(agg.round(3).to_string())

    # -- 5. paired against production, per slate --------------------------
    prod_path = ORACLE_DIR.parent / f"lab_seedcheck_c{int(calib)}.csv"
    if prod_path.exists():
        prod = pd.read_csv(prod_path)
        print(f"\n===== 5. PAIRED vs PRODUCTION, PER SLATE =====")
        for arm in sorted(prod.arm.unique()):
            if not arm.startswith("p_win@evw0.25"):
                continue
            pr = prod[prod.arm == arm]
            p_top1 = pr.groupby("slate").top1.mean() * 100
            p_top01 = pr.groupby("slate").top01.mean() * 100
            for vname in variants:
                for q in (0.10, 0.25):
                    per = per_slate_store[(vname, q)]
                    d1 = np.array([per[int(s)]["top1%"] - p_top1[int(s)]
                                   if int(s) in p_top1.index else np.nan
                                   for s in p_top1.index])
                    d01 = np.array([per[int(s)]["top01%"] - p_top01[int(s)]
                                    for s in p_top01.index])
                    n = len(d1)
                    print(f"  {arm} vs own{q:.2f}|{vname}: "
                          f"top1 {d1.mean():+.3f} ({int((d1>0).sum())}/{n}, "
                          f"p={_sign_p(int((d1 > 0).sum()), n):.3f})  "
                          f"top01 {d01.mean():+.3f} ({int((d01>0).sum())}/{n}, "
                          f"p={_sign_p(int((d01 > 0).sum()), n):.3f})")
    else:
        print(f"\n(no {prod_path.name} yet -- run `seedcheck` for the "
              "production pairing in section 5)")
    print(f"\nwrote {out}")


def _slice_draw(sd: SlateData, q: float, rng, *, floor_pct: float = 0.0,
                admit_n: int = 0, cull: np.ndarray = None,
                key: np.ndarray = None) -> dict:
    """Sequential per-contest fill: cull to the funnel window, keep the
    least-owned q fraction, draw uniformly inside it, and remove picks from
    the shared mask exactly as select_greedy does."""
    key = sd.feats["own_sum"] if key is None else key
    mask = proj_floor_mask(sd, floor_pct)
    picks: dict = {}
    for ci in range(len(sd.cids)):
        k = int(sd.k[ci])
        if k <= 0:
            continue
        rem = np.where(mask)[0]
        if admit_n > 0 and len(rem) > admit_n:
            rem = rem[np.argsort(-cull[ci][rem])[:admit_n]]
        if q < 1.0:
            keep = max(k, int(round(q * len(rem))))
            if len(rem) > keep:
                rem = rem[np.argsort(key[rem])[:keep]]
        k = min(k, len(rem))
        chosen = rem[rng.choice(len(rem), size=k, replace=False)]
        picks[sd.cids[ci]] = list(map(int, chosen))
        mask[chosen] = False
    return picks


# ---------------------------------------------------------------------------
# Joint (whole-portfolio) vs single-insertion grading
# ---------------------------------------------------------------------------

def cmd_jointcheck(seeds=(42, 137, 4242)) -> None:
    """How much does joint insertion (grade_joint / bt_core.grade_portfolio)
    actually change realized $ versus the existing single-insertion grading
    (grade / grade_pick), on the arms this harness already uses elsewhere?

    Single insertion grades each of our entries as if it were the ONLY thing
    we added to the real field, so when we hold more than one entry in the
    same contest it can (a) let a worse entry of ours claim a rank a better
    entry of ours actually occupies (self-displacement), (b) let two of our
    own tied entries each claim the full tie-band prize instead of splitting
    it (self-tie-splitting), and (c) let literal duplicate lineups each
    collect a full prize (a special case of (b)). grade_joint fixes all
    three by inserting a contest's whole set of our picks at once. This
    quantifies the gap on real arms/slates/seeds rather than only on the
    synthetic property checks in cmd_verify.
    """
    arm_names = ["prod_p_win", "ev_dollars", "random"]
    arms = build_arms()
    have_real = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    slates = [s for s in have_real if (ORACLE_DIR / f"{s}_field.npz").exists()]
    missing_field = [s for s in have_real if s not in slates]
    if missing_field:
        print(f"  note: {missing_field} have no {{slate}}_field.npz sidecar yet "
              "-- run `python tests/backtest_oracle.py field` first; skipping them")

    tidy_rows = []
    contest_deltas = []  # (|delta|, delta, slate, seed, arm, contest, cid, k)
    for s in slates:
        for seed in seeds:
            sd = load_slate(s, seed, False)
            curs = _candidate_currencies(sd)
            curs_A = {"p_win": sd.currency("p_win", "A"),
                      "ev_dollars": sd.currency("ev_dollars", "A"),
                      "ev_tail": sd.currency("ev_tail", "A"),
                      "p_cash": sd.currency("p_cash", "A")}
            for name in arm_names:
                cur, mode, kw = arms[name]
                if cur is None:
                    import zlib
                    rng = np.random.default_rng(
                        zlib.crc32(f"{s}|{seed}|{name}".encode()) & 0xFFFFFFFF)
                    picks = select_greedy(sd, curs["p_win"], curs_A["p_win"], rng=rng, **kw)
                else:
                    sel = curs[cur]
                    cull = curs_A.get(cur, sel)
                    fn = select_assign if mode == "assign" else select_greedy
                    picks = fn(sd, sel, cull, **kw)

                g_single = grade(sd, picks, name)
                g_joint = grade_joint(sd, picks, name)
                if g_single.empty and g_joint.empty:
                    continue

                n_single, n_joint = len(g_single), len(g_joint)
                s_won, j_won = float(g_single.won.sum()), float(g_joint.won.sum())
                s_fee, j_fee = float(g_single.fee.sum()), float(g_joint.fee.sum())
                s_dpe = (s_won - s_fee) / max(n_single, 1)
                j_dpe = (j_won - j_fee) / max(n_joint, 1)
                tidy_rows.append({
                    "slate": s, "seed": seed, "arm": name, "entries": n_single,
                    "single_won": s_won, "joint_won": j_won,
                    "d_total_gross": j_won - s_won,
                    "single_dollar_per_entry": s_dpe, "joint_dollar_per_entry": j_dpe,
                    "d_dollar_per_entry": j_dpe - s_dpe,
                })

                for ci, cid in enumerate(sd.cids):
                    s_g = g_single.loc[g_single.cid == cid, "won"] if not g_single.empty else pd.Series(dtype=float)
                    j_g = g_joint.loc[g_joint.cid == cid, "won"] if not g_joint.empty else pd.Series(dtype=float)
                    if s_g.empty and j_g.empty:
                        continue
                    d = float(j_g.sum() - s_g.sum())
                    k_here = max(len(s_g), len(j_g))
                    contest_deltas.append((abs(d), d, s, seed, name, str(sd.contest[ci]), cid, k_here))
        print(f"  {s}: {len(seeds)} seeds x {len(arm_names)} arms graded both ways", flush=True)

    tidy = pd.DataFrame(tidy_rows)
    out_path = ORACLE_DIR.parent / "lab_jointcheck.csv"
    header = not out_path.exists()
    tidy.to_csv(out_path, mode="a", header=header, index=False)

    print(f"\n===== JOINT vs SINGLE INSERTION ({len(slates)} slates x {len(seeds)} "
          f"seeds x {len(arm_names)} arms) =====")
    print("delta = joint (grade_joint) minus single (grade), same picks either way.\n")

    print("-- per (slate, arm), pooled over seeds --")
    per_slate = tidy.groupby(["slate", "arm"]).apply(lambda g: pd.Series({
        "entries": g.entries.sum(),
        "d$/entry": g.d_dollar_per_entry.mean(),
        "d_total_gross": g.d_total_gross.sum(),
    }), include_groups=False)
    print(per_slate.round(4).to_string())

    print("\n-- pooled over everything, per arm --")
    pooled = tidy.groupby("arm").apply(lambda g: pd.Series({
        "entries": g.entries.sum(),
        "d$/entry": g.d_dollar_per_entry.mean(),
        "d_total_gross": g.d_total_gross.sum(),
    }), include_groups=False)
    print(pooled.round(4).to_string())

    print("\n-- 5 largest per-contest |delta| across everything --")
    top5 = sorted(contest_deltas, key=lambda t: -t[0])[:5]
    for absd, d, s, seed, name, contest, cid, k in top5:
        print(f"  delta ${d:+.2f}  {s} s{seed} arm={name} contest={contest!r} "
              f"(cid={cid}) k={k}")

    print(f"\nappended {len(tidy)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Amendment A1 (EVIDENCE_LOG.md, 2026-08-02): Phase 4' model-light adjudication
# ---------------------------------------------------------------------------

A1_BASELINE = "prod_faithful"
A1_NULL = "random@floor30"
A1_CHALLENGERS = ["proj_score", "p_cash", "p_cash@assign", "coverage_light"]
A1_ARMS = [A1_BASELINE, A1_NULL] + A1_CHALLENGERS
A1_SEEDS = (42, 137, 4242)

_CORR_CACHE_DIR = ORACLE_DIR.parent / "audit"
_CORR_MEM_CACHE: dict = {}


def _prod_corr(slate: str, seed: int) -> np.ndarray:
    """(M, M) float32 simulated-lineup-score correlation matrix -- exactly
    what production's DeterminantPortfolioSelector runs on, and what
    cmd_verify's check 2 builds via build_slate_context(want_corr=True). Not
    stored in the existing oracle/audit npz sidecars (checked: the
    {slate}_workunit_s{seed}_c{calib}.npz files hold PIT/crowding scalars,
    not a corr matrix), so this is its own cache -- a module-level dict for
    reuse within one process, and an on-disk copy under
    tests/backtest_output/audit/ so a rerun of cmd_adjudicate (or the
    prod_faithful self-check alone) doesn't pay the ~100s/slate context-build
    cost again. want_pwin=False: the oracle table's own p_win currency
    (already proven bit-identical to ep.compute_p_win by backtest_oracle.py)
    covers the EV side, so only the corr matrix needs to come from a fresh
    context build.
    """
    key = (slate, seed)
    if key in _CORR_MEM_CACHE:
        return _CORR_MEM_CACHE[key]
    path = _CORR_CACHE_DIR / f"{slate}_corr_s{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            corr = z["corr"]
    else:
        from tests.bt_core import build_slate_context, load_real_contests

        d = PROJECT_ROOT / "archive" / slate
        real = load_real_contests(d)
        ctx = build_slate_context(
            d, seed, False, real, n_sims=25000, sharpness=0.05,
            sim_cache_dir=PROJECT_ROOT / "tests/backtest_output/sim_cache",
            want_corr=True, want_pwin=False,
        )
        corr = ctx["corr"]
        _CORR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, corr=corr)
    _CORR_MEM_CACHE[key] = corr
    return corr


def _composition_overlap_fn(slate: str):
    """Lazy per-contest composition-overlap "correlation" for coverage_light:
    overlap(i, j) = |shared player_ids| / roster_size, built purely from
    {slate}_real.npz's player_ids -- no sim inputs anywhere in this arm.

    A dense (M, M) overlap matrix is 100M floats at M~10k (800MB) -- too big
    to hold per slate, and unnecessary: select_greedy's evw<1.0 path only
    ever needs the submatrix for one contest's admit window (`rem`, capped
    at admit_n=2000 rows). So a one-hot lineup x player incidence matrix is
    built once per slate (independent of seed -- composition doesn't change
    across seeds) and this returns a closure computing
    (H[rem] @ H[rem].T) / roster_size on demand: sparse x sparse, a few
    hundredths of a second even at admit_n=2000 (measured).
    """
    import scipy.sparse as sp

    with np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False) as z:
        pids = z["player_ids"]                       # (M, roster_size)
    roster_size = pids.shape[1]
    uniq, inv = np.unique(pids, return_inverse=True)
    inv = np.asarray(inv).reshape(pids.shape)
    rows = np.repeat(np.arange(pids.shape[0]), roster_size)
    H = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, inv.ravel())),
        shape=(pids.shape[0], len(uniq)),
    )

    def overlap(rem: np.ndarray) -> np.ndarray:
        hs = H[rem]
        return np.asarray((hs @ hs.T).toarray(), dtype=np.float64) / roster_size

    return overlap


def _a1_pick(sd: SlateData, arm: str, corr: np.ndarray, comp_fn) -> dict:
    """One arm's picks for the pre-registered A1 arm set (EVIDENCE_LOG.md,
    Amendment A1). Built directly here rather than through build_arms() so
    the pre-registered configuration is visible in one place and adding it
    can't perturb any existing command's arm registry.
    """
    curs = _candidate_currencies(sd)
    if arm == A1_BASELINE:
        # The faithful production baseline: p_win currency (B-half select,
        # A-half cull, floor 30, admit 2000) through evw=0.25 + the real
        # DeterminantPortfolioSelector + the sim corr matrix -- exactly the
        # configuration cmd_verify's check 2 validates against production's
        # own ep.allocate_contests, NOT the arm registry's evw=1.0
        # "prod_p_win" approximation.
        return select_greedy(
            sd, curs["p_win"], sd.currency("p_win", "A"),
            floor_pct=30.0, admit_n=2000, evw=0.25, corr=corr,
        )
    if arm == A1_NULL:
        import zlib
        rng = np.random.default_rng(
            zlib.crc32(f"{sd.slate}|{sd.seed}|{arm}".encode()) & 0xFFFFFFFF)
        return select_greedy(
            sd, curs["p_win"], sd.currency("p_win", "A"),
            floor_pct=30.0, admit_n=0, rng=rng,
        )
    if arm == "proj_score":
        sel = curs["proj_score"]
        return select_greedy(sd, sel, sel, floor_pct=30.0, admit_n=2000)
    if arm == "p_cash":
        return select_greedy(
            sd, curs["p_cash"], sd.currency("p_cash", "A"),
            floor_pct=30.0, admit_n=2000,
        )
    if arm == "p_cash@assign":
        return select_assign(
            sd, curs["p_cash"], sd.currency("p_cash", "A"),
            floor_pct=30.0, admit_n=2000,
        )
    if arm == "coverage_light":
        # proj_score as the EV term (a global currency, broadcast to every
        # contest) ranked through the real Det selector with COMPOSITION
        # correlation instead of a sim corr matrix -- no sim inputs in
        # either term.
        sel = curs["proj_score"]
        return select_greedy(
            sd, sel, sel, floor_pct=30.0, admit_n=2000, evw=0.25, corr_fn=comp_fn,
        )
    raise ValueError(f"unknown A1 arm {arm!r}")


def _selfcheck_prod_faithful(slate: str = "07222026", seed: int = 42) -> None:
    """Cheap startup self-check for cmd_adjudicate: prod_faithful's
    select_greedy(evw=0.25, corr=sim corr) must reproduce production's own
    ep.allocate_contests picks exactly (152/152 on 07222026 s42, the same
    number cmd_verify's check 2 prints) -- the identical equivalence, run
    standalone so a corr-cache bug or an accidental drift in prod_faithful's
    configuration can't silently invalidate everything cmd_adjudicate reports.
    Also seeds _CORR_MEM_CACHE / the on-disk corr cache for (slate, seed) so
    the main loop doesn't rebuild the same context a second time.
    """
    from src.api import external_pool as ep
    from tests.bt_core import build_slate_context, load_real_contests, _FakeGroup

    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    ctx = build_slate_context(
        d, seed, False, real, n_sims=25000, sharpness=0.05,
        sim_cache_dir=PROJECT_ROOT / "tests/backtest_output/sim_cache",
    )
    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ctx["contests"] if c["k"] > 0]
    alloc = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0, evw_base=0.25, evw_max=0.25,
        proj_scores=ctx["proj_scores"], proj_score_floor_percentile=30.0,
        ev_type="p_win", p_win_cull=ctx["p_win_cull"], p_win_select=ctx["p_win_select"],
        p_win_admit_n=2000, p_win_admit_multiplier=0.0,
    )
    idx_of = {id(lu): i for i, lu in enumerate(ctx["pool"].lineups)}
    prod_picks: dict = {}
    i = 0
    for g in groups:
        prod_picks[g.contest_id] = [idx_of[id(lu)] for lu, _ in alloc.portfolio[i:i + len(g.entries)]]
        i += len(g.entries)

    # Seed the corr cache with what we just built, before the main loop asks
    # _prod_corr for the same (slate, seed) again.
    _CORR_MEM_CACHE[(slate, seed)] = ctx["corr"]
    cache_path = _CORR_CACHE_DIR / f"{slate}_corr_s{seed}.npz"
    if not cache_path.exists():
        _CORR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, corr=ctx["corr"])

    sd = load_slate(slate, seed, False)
    lab_picks = select_greedy(
        sd, sd.currency("p_win", "B"), sd.currency("p_win", "A"),
        floor_pct=30.0, admit_n=2000, evw=0.25, corr=ctx["corr"],
    )
    same = sum(len(set(lab_picks.get(c, [])) & set(prod_picks.get(c, [])))
              for c in sd.cids)
    tot = sum(len(v) for v in prod_picks.values())
    print(f"  prod_faithful vs production allocate_contests ({slate} s{seed}): "
          f"{same}/{tot} identical picks")
    assert tot > 0 and same == tot, (
        f"prod_faithful self-check FAILED: {same}/{tot} identical picks "
        f"(expected {tot}/{tot}) -- prod_faithful no longer reproduces "
        "production's own allocate_contests; stop, nothing below this is "
        "trustworthy."
    )


def _a1_pooled(df: pd.DataFrame, arm: str, seed: int = None) -> dict:
    """Pooled (over slates, and over seeds unless `seed` narrows it) $/entry,
    cash% and top1% for one arm from a (slate, seed, arm)-tidy grade
    dataframe (grade()/grade_joint() output, possibly multi-seed)."""
    g = df[df.arm == arm]
    if seed is not None:
        g = g[g.seed == seed]
    if g.empty:
        return {"$/entry": np.nan, "cash%": np.nan, "top1%": np.nan}
    fees, won = g.fee.sum(), g.won.sum()
    return {
        "$/entry": (won - fees) / len(g),
        "cash%": 100 * g.cash.mean(),
        "top1%": 100 * g.top1.mean(),
    }


def cmd_adjudicate(calib: bool = False) -> None:
    """Amendment A1 (EVIDENCE_LOG.md, 2026-08-02): Phase 4' model-light
    adjudication, run against the FAITHFUL production baseline (evw=0.25 +
    DeterminantPortfolioSelector + sim corr -- see prod_faithful in
    _a1_pick), not the arm registry's evw=1.0 approximation. 9 slates x
    seeds 42/137/4242, calib=False. Every arm is graded BOTH single-insertion
    (grade, for comparability with the rest of this file) and jointly
    (grade_joint, the headline number here). Reports:

      * report() vs prod_faithful, single-insertion AND joint (two blocks).
      * realized per-slate log-growth Sum_s log(1 + net_s/(50*fees_s)) from
        JOINT grading, per (arm, seed), and how many seeds each challenger
        beats the baseline on (gate G5).
      * the mechanical G1-G5 gate table per challenger vs prod_faithful
        (joint grading) and the pre-registered verdict line.

    Tidy per-(slate, seed, arm) rows -> tests/backtest_output/lab_adjudicate.csv;
    the gate table -> tests/backtest_output/lab_adjudicate_gates.csv.
    """
    print("== self-check: prod_faithful reproduces production's allocate_contests ==")
    _selfcheck_prod_faithful()

    slates = [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]
    single_frames, joint_frames, tidy_rows = [], [], []
    for s in slates:
        comp_fn = _composition_overlap_fn(s)
        for seed in A1_SEEDS:
            sd = load_slate(s, seed, calib)
            corr = _prod_corr(s, seed)
            for arm in A1_ARMS:
                picks = _a1_pick(sd, arm, corr, comp_fn)
                g_single = grade(sd, picks, arm)
                g_joint = grade_joint(sd, picks, arm)
                if not g_single.empty:
                    single_frames.append(g_single)
                if not g_joint.empty:
                    joint_frames.append(g_joint)

                s_fees = float(g_single.fee.sum()) if not g_single.empty else 0.0
                s_won = float(g_single.won.sum()) if not g_single.empty else 0.0
                j_fees = float(g_joint.fee.sum()) if not g_joint.empty else 0.0
                j_won = float(g_joint.won.sum()) if not g_joint.empty else 0.0
                j_net = j_won - j_fees
                G_s = float(np.log(1.0 + j_net / (50.0 * j_fees))) if j_fees > 0 else np.nan
                tidy_rows.append({
                    "slate": s, "seed": seed, "calib": calib, "arm": arm,
                    "entries_single": len(g_single), "entries_joint": len(g_joint),
                    "single_fees": s_fees, "single_won": s_won,
                    "single_net": s_won - s_fees,
                    "single_dollar_per_entry": (s_won - s_fees) / len(g_single)
                                              if len(g_single) else np.nan,
                    "joint_fees": j_fees, "joint_won": j_won, "joint_net": j_net,
                    "joint_dollar_per_entry": j_net / len(g_joint) if len(g_joint) else np.nan,
                    "joint_cash_rate": 100 * g_joint.cash.mean() if not g_joint.empty else np.nan,
                    "joint_top1_rate": 100 * g_joint.top1.mean() if not g_joint.empty else np.nan,
                    "joint_top01_rate": 100 * g_joint.top01.mean() if not g_joint.empty else np.nan,
                    "G_s": G_s,
                })
            print(f"  {s} s{seed}: {len(A1_ARMS)} arms graded both ways", flush=True)

    single_df = pd.concat(single_frames, ignore_index=True)
    joint_df = pd.concat(joint_frames, ignore_index=True)
    tidy_df = pd.DataFrame(tidy_rows)

    out_tidy = ORACLE_DIR.parent / "lab_adjudicate.csv"
    header = not out_tidy.exists()
    tidy_df.to_csv(out_tidy, mode="a", header=header, index=False)

    res_single = report(single_df, baseline=A1_BASELINE)
    res_joint = report(joint_df, baseline=A1_BASELINE)
    print_report(res_single, "A1 ADJUDICATION -- SINGLE-INSERTION (comparability only)")
    print_report(res_joint, "A1 ADJUDICATION -- JOINT GRADING (headline)")

    # -- realized per-slate log-growth, from JOINT grading --------------
    sumG = tidy_df.groupby(["seed", "arm"])["G_s"].sum().unstack("arm")
    sumG = sumG[[a for a in A1_ARMS if a in sumG.columns]]
    print("\n===== REALIZED LOG-GROWTH  Sum_s log(1 + net_s/(50*fees_s))  "
          "(joint grading) =====")
    print(sumG.round(4).to_string())
    g5_beats = {
        arm: int(sum(sumG.loc[seed, arm] > sumG.loc[seed, A1_BASELINE]
                    for seed in A1_SEEDS))
        for arm in A1_CHALLENGERS
    }
    print("\nseeds where challenger's SumG beats prod_faithful's (gate G5, need >=2/3):")
    for arm in A1_CHALLENGERS:
        print(f"  {arm:16s} {g5_beats[arm]}/3")

    # -- mechanical G1-G5 gates, joint grading, vs prod_faithful ----------
    per_slate_seed = joint_df.groupby(["slate", "seed", "arm"]).apply(
        lambda g: (g.won.sum() - g.fee.sum()) / len(g), include_groups=False)

    base_by_seed = {seed: _a1_pooled(joint_df, A1_BASELINE, seed) for seed in A1_SEEDS}
    base_cash_range = (max(v["cash%"] for v in base_by_seed.values())
                       - min(v["cash%"] for v in base_by_seed.values()))
    base_top1_range = (max(v["top1%"] for v in base_by_seed.values())
                       - min(v["top1%"] for v in base_by_seed.values()))
    base_pooled = _a1_pooled(joint_df, A1_BASELINE)

    gate_rows = []
    for arm in A1_CHALLENGERS:
        # G1: pooled d$/entry > 0 on EVERY seed.
        g1_deltas = [_a1_pooled(joint_df, arm, seed)["$/entry"]
                    - base_by_seed[seed]["$/entry"] for seed in A1_SEEDS]
        g1 = all(d > 0 for d in g1_deltas)

        # G2: seed-averaged per-slate d$/entry, win_slates >= 6/9.
        slate_deltas = []
        for slate_ in slates:
            seed_deltas = []
            for seed in A1_SEEDS:
                key_c, key_b = (slate_, seed, arm), (slate_, seed, A1_BASELINE)
                if key_c in per_slate_seed.index and key_b in per_slate_seed.index:
                    seed_deltas.append(per_slate_seed[key_c] - per_slate_seed[key_b])
            if seed_deltas:
                slate_deltas.append(float(np.mean(seed_deltas)))
        g2_win = sum(d > 0 for d in slate_deltas)
        g2 = g2_win >= 6

        # G3: drop_max and LOSO_min (pooled joint report, all seeds) >= baseline's.
        base_row = res_joint.loc[res_joint.arm == A1_BASELINE].iloc[0]
        chall_row = res_joint.loc[res_joint.arm == arm].iloc[0]
        g3 = (chall_row["drop_max"] >= base_row["drop_max"]
             and chall_row["LOSO_min"] >= base_row["LOSO_min"])

        # G4: pooled top1%/cash% >= baseline's pooled rate minus baseline's
        # OWN 3-seed range (the seed noise floor).
        chall_pooled = _a1_pooled(joint_df, arm)
        g4 = (chall_pooled["top1%"] >= base_pooled["top1%"] - base_top1_range
             and chall_pooled["cash%"] >= base_pooled["cash%"] - base_cash_range)

        # G5: ΣG beats baseline's on >= 2/3 seeds.
        g5 = g5_beats[arm] >= 2

        if g1 and g2 and g3 and g4 and g5:
            verdict = "recommend prospective A/B"
        elif g1 and g3:
            verdict = "promising, extend prospectively"
        else:
            verdict = "no evidence -- production stands"

        gate_rows.append({
            "arm": arm, "G1_dpe_pos_all_seeds": g1, "G2_win_slates_ge6": g2,
            "G2_win_slates": f"{g2_win}/{len(slate_deltas)}", "G3_drop_loso_ge_base": g3,
            "G4_rate_ge_base_minus_range": g4, "G5_sumG_beats_ge2of3": g5,
            "G5_seeds": f"{g5_beats[arm]}/3", "verdict": verdict,
        })

    gate_df = pd.DataFrame(gate_rows)
    print("\n===== G1-G5 GATE TABLE vs prod_faithful (joint grading) =====")
    print(gate_df.to_string(index=False))

    out_gates = ORACLE_DIR.parent / "lab_adjudicate_gates.csv"
    header = not out_gates.exists()
    gate_df.to_csv(out_gates, mode="a", header=header, index=False)
    print(f"\nappended {len(tidy_df)} rows -> {out_tidy}")
    print(f"appended {len(gate_df)} rows -> {out_gates}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        cmd_verify()
    elif cmd == "currencies":
        cmd_currencies()
    elif cmd == "arms":
        cmd_arms()
    elif cmd == "model":
        cmd_model()
    elif cmd == "stages":
        cmd_stages()
    elif cmd == "selector":
        cmd_selector()
    elif cmd == "null":
        cmd_null()
    elif cmd == "slices":
        cmd_slices()
    elif cmd == "seedcheck":
        cmd_seedcheck()
    elif cmd == "own":
        cmd_own()
    elif cmd == "jointcheck":
        cmd_jointcheck()
    elif cmd == "adjudicate":
        cmd_adjudicate()
    else:
        raise SystemExit(
            f"unknown command {cmd!r} "
            "(verify|currencies|stages|arms|model|selector|null|slices|"
            "seedcheck|own|jointcheck|adjudicate)")


if __name__ == "__main__":
    main()
