"""Realized dollar ROI of any named entrant, from the archived standings.

`scripts/analyze_rival_portfolio.py` reconstructs a rival's portfolio STRUCTURE
(stack shapes, overlap, exposure) and `scripts/sim_evaluate_portfolios.py`
scores it against simulated worlds. Neither answers the plainest question:
what did they actually make? That needed the real per-contest payout tables to
be wired to the real standings, which the backtest work has now done
(tests/bt_core.load_real_contests).

Every entrant is already IN the field, so unlike our own backtested lineups
nothing is inserted -- their rank is their rank. Ties are split the way DK
splits them: the tie band's payouts averaged across everyone tied.

Why this matters beyond curiosity: the user reports ShaidyAdvice is a heavily
winning player over a horizon far longer than these 8 slates. If a known
winner does NOT show a clear profit here, that is a direct, empirical measure
of how much skill this sample can resolve at all -- which calibrates every
other conclusion drawn from it.

    source venv/bin/activate
    python scripts/analyze_rival_roi.py                       # default handles
    python scripts/analyze_rival_roi.py ShaidyAdvice shrek4224
    python scripts/analyze_rival_roi.py --top 15              # by entry count
"""
import csv
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import BACKTEST_SLATES, load_real_contests  # noqa: E402


def parse_standings(zip_path: Path) -> tuple[list, np.ndarray]:
    """([(entry_name, points)], sorted_scores) for one contest."""
    with zipfile.ZipFile(zip_path) as zf:
        fn = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(
            zf.read(fn).decode("utf-8-sig", errors="replace").splitlines()))
    entries = []
    for r in rows[1:]:
        if len(r) > 4 and r[0].strip().isdigit():
            try:
                # DK writes multi-entry names as "handle (12/150)". Without
                # stripping that, every single entry looks like its own handle
                # and a 150-max pro vanishes from any by-volume ranking.
                entries.append((r[2].split("(")[0].strip(), float(r[4])))
            except ValueError:
                continue
    scores = np.sort(np.array([p for _, p in entries], dtype=np.float64))
    return entries, scores


def payout_for(points: float, sorted_scores: np.ndarray,
               payout_arr: np.ndarray) -> tuple[float, int]:
    """(gross_$, rank) for an entry ALREADY in the field -- ties split evenly
    across the tie band, exactly as DK pays them."""
    n = len(sorted_scores)
    right = int(np.searchsorted(sorted_scores, points, side="right"))
    left = int(np.searchsorted(sorted_scores, points, side="left"))
    n_above = n - right
    n_tied = right - left
    lo, hi = n_above, min(n_above + n_tied, len(payout_arr))
    band = payout_arr[lo:hi] if lo < len(payout_arr) else np.array([])
    return (float(band.mean()) if len(band) else 0.0), n_above + 1


def collect(slates: list) -> pd.DataFrame:
    recs = []
    for slate in slates:
        d = PROJECT_ROOT / "archive" / slate
        if not d.exists():
            continue
        for c in load_real_contests(d):
            z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
            entries, scores = parse_standings(z)
            for name, pts in entries:
                gross, rank = payout_for(pts, scores, c["payout_arr"])
                recs.append({
                    "slate": slate, "contest": c["contest"], "entrant": name,
                    "fee": float(c["fee"]), "points": pts, "rank": rank,
                    "won": gross, "n_field": c["n_field"],
                    "top1": int(rank <= max(1, c["n_field"] // 100)),
                    "top01": int(rank <= max(1, c["n_field"] // 1000)),
                })
    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame, handles: list) -> None:
    field_fees = (df.fee).sum()
    field_won = df.won.sum()
    print(f"\n===== FIELD BASELINE ({df.slate.nunique()} slates, "
          f"{len(df):,} entries, {df.contest.nunique()} contest types) =====")
    print(f"  entries {len(df):,}   fees ${field_fees:,.0f}   won ${field_won:,.0f}   "
          f"ROI {100 * (field_won - field_fees) / field_fees:+.1f}%   "
          f"(this is the rake, and every entrant's average opponent)")

    rows = []
    rng = np.random.default_rng(0)
    for h in handles:
        g = df[df.entrant == h]
        if g.empty:
            print(f"  (no entries found for {h!r})")
            continue
        fees, won = g.fee.sum(), g.won.sum()
        per = g.groupby("slate").apply(
            lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
        bs = per.to_numpy()[rng.integers(0, len(per), size=(20000, len(per)))].mean(axis=1)
        rows.append({
            "entrant": h, "slates": g.slate.nunique(), "entries": len(g),
            "fees": fees, "won": won, "net": won - fees,
            "ROI%": 100 * (won - fees) / fees,
            "$/entry": (won - fees) / len(g),
            "lo95": np.percentile(bs, 2.5), "hi95": np.percentile(bs, 97.5),
            "pos_slates": f"{int((per > 0).sum())}/{len(per)}",
            "LOSO_min": min((per.sum() - x) / (len(per) - 1) for x in per),
            "cash%": 100 * (g.won > 0).mean(),
            "top1%": 100 * g.top1.mean(), "top01%": 100 * g.top01.mean(),
            "best": int(g["rank"].min()),
        })
    res = pd.DataFrame(rows).sort_values("$/entry", ascending=False)
    print("\n===== NAMED ENTRANTS =====")
    print(res.round(3).to_string(index=False))
    return res


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slates = BACKTEST_SLATES
    df = collect(slates)
    if "--top" in sys.argv:
        n = int(sys.argv[sys.argv.index("--top") + 1])
        handles = list(df.entrant.value_counts().head(n).index)
    else:
        handles = args or ["ShaidyAdvice"]
    res = summarize(df, handles)

    out = PROJECT_ROOT / "outputs" / "rival_roi.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(PROJECT_ROOT / "outputs" / "rival_entries.csv", index=False)
    if res is not None:
        res.to_csv(out, index=False)
    print(f"\nwrote {out} and outputs/rival_entries.csv")


if __name__ == "__main__":
    main()
