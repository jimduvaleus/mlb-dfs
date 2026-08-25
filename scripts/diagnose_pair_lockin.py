#!/usr/bin/env python
"""Is any pair of DIFFERENT-TEAM hitters riding along together more than chance?

The failure this looks for: a hitter appears in only a handful of entries, and
most of those entries carry the same hitter from another team -- so the few
lineups exposing him are structurally one bet rather than several. Same-team
pairs are excluded because co-occurrence there is stacking, which is the point.

WHY A RAW CO-OCCURRENCE RATE IS THE WRONG STATISTIC. It flags ubiquity, not
coupling. On the 08/24 build the top "offender" was a 75% pair whose partner
was simply in 22% of all lineups -- he rode along with everyone. Three
corrections, all needed:

  LIFT, not the raw rate. `P(B | A) / P(B)` discounts a partner who is
  everywhere. A 75% rate against a 22% base rate is 3.4x; the same 75% against
  a 10% base rate is 7.5x, and only the second is a coupling.

  AN EXACT TAIL, not a percentage. "3 of 4" is not evidence. The binomial tail
  `P(>= k | n appearances, base rate)` turns it into a probability that knows
  how few observations it rests on.

  MULTIPLE-TESTING CORRECTION. A portfolio generates ~1,000 cross-team hitter
  pairs. At raw p<0.05 you expect ~50 flags from noise alone; the 08/24 build
  produced 17, i.e. FEWER than chance, and none survived Benjamini-Hochberg.
  Reporting the raw p-values without this reliably manufactures a problem.

POWER, AND HOW IT VARIES. The shape that motivated this -- a hitter in 6
entries whose partner (10% base rate) appears in 5 of them -- is p ~ 5e-5. That
is q = 0.064 against a portfolio generating ~1,160 pairs (flagged) but q = 0.13
against one generating ~2,350 (not flagged), because the BH correction scales
with the number of tests. So the SAME coupling can cross the threshold or not
depending on how many distinct pairs the portfolio happens to produce.

Read the ranking as well as the q-value: a real coupling rises to the top of
the p-ordered list whether or not it clears an absolute threshold. And the test
is weak for hitters with 3-4 appearances, where almost nothing is
distinguishable from noise -- a clean report at low `--min-apps` means "not
detectable", not "not present".

POSITION CONTEXT is reported but deliberately NOT used to filter. A pairing can
be forced rather than chosen when a slot has few viable fillers. Note the share
is computed from `assigned_position` (the roster slot actually filled), not the
`position` string: that string is a multi-position eligibility label like
"2B/SS", and grouping on it compares a player against only those sharing his
exact label -- which reported a 5-entry player as holding 56% of his slot.

KNOWN LIMIT: pairs only. A recurring TRIPLE or a repeated 4-player block is the
same concern one level up and would not be flagged here.

    python scripts/diagnose_pair_lockin.py
    python scripts/diagnose_pair_lockin.py --portfolio archive/08242026/portfolio.csv
    python scripts/diagnose_pair_lockin.py --min-apps 5 --q 0.10 --top 15
"""
from __future__ import annotations

import argparse
import collections
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO = PROJECT_ROOT / "outputs" / "portfolio_draftkings.csv"


def load_lineups(path: Path):
    """(lineups, meta, slot_fillers) from a portfolio CSV."""
    df = pd.read_csv(path)
    need = {"lineup", "player_id", "name", "team", "assigned_position"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    lineups = {int(li): set(int(x) for x in g.player_id)
               for li, g in df.groupby("lineup")}
    meta = {int(r.player_id): {
        "name": r.name, "team": r.team, "slot": r.assigned_position,
        "salary": float(getattr(r, "salary", float("nan"))),
    } for r in df.itertuples()}
    hitters = df[df.assigned_position != "P"]
    slot_fillers = {s: set(g.player_id) for s, g in hitters.groupby("assigned_position")}
    slot_total = collections.Counter(hitters.assigned_position)
    return lineups, meta, slot_fillers, slot_total


def benjamini_hochberg(pvals):
    """BH q-values, with the monotonicity enforced (q is non-decreasing in p)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def analyse(lineups, meta, min_apps: int):
    n = len(lineups)
    hit = {p for p, m in meta.items() if m["slot"] != "P"}
    cnt = collections.Counter(p for s in lineups.values() for p in s if p in hit)

    pair = collections.Counter()
    for s in lineups.values():
        hs = sorted(p for p in s if p in hit)
        for a, b in itertools.combinations(hs, 2):
            if meta[a]["team"] != meta[b]["team"]:
                pair[(a, b)] += 1

    recs = []
    for (a, b), c in pair.items():
        rarer, other = (a, b) if cnt[a] <= cnt[b] else (b, a)
        n_rare = cnt[rarer]
        if n_rare < min_apps:
            continue
        base = cnt[other] / n
        if base <= 0:
            continue
        recs.append({
            "rarer": rarer, "other": other, "together": c, "n_rare": n_rare,
            "base": base, "lift": (c / n_rare) / base,
            "p": float(binom.sf(c - 1, n_rare, base)),
        })
    if recs:
        for r, q in zip(recs, benjamini_hochberg([r["p"] for r in recs])):
            r["q"] = q
    return recs, cnt, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    ap.add_argument("--min-apps", type=int, default=3,
                    help="ignore hitters with fewer appearances; below ~5 the "
                         "test has little power (default 3)")
    ap.add_argument("--q", type=float, default=0.10, help="BH threshold (default 0.10)")
    ap.add_argument("--top", type=int, default=10, help="rows to print (default 10)")
    args = ap.parse_args()

    lineups, meta, slot_fillers, slot_total = load_lineups(args.portfolio)
    recs, cnt, n = analyse(lineups, meta, args.min_apps)
    print(f"portfolio {args.portfolio}  ({n} lineups)")
    if not recs:
        print(f"no cross-team hitter pairs with the rarer hitter appearing "
              f">={args.min_apps} times.")
        return 0

    recs.sort(key=lambda r: r["p"])
    m = len(recs)
    n_raw = sum(1 for r in recs if r["p"] < 0.05)
    flagged = [r for r in recs if r["q"] < args.q]

    print(f"cross-team hitter pairs tested: {m:,}  (rarer hitter >= {args.min_apps} apps)\n")
    print(f"{'pair':52s} {'together':>9s} {'lift':>5s} {'p':>8s} {'BH q':>8s}")
    for r in recs[:args.top]:
        a, b = meta[r["rarer"]], meta[r["other"]]
        lbl = f"{str(a['name'])[:19]}({a['team']}) + {str(b['name'])[:19]}({b['team']})"
        mark = "  <<<" if r["q"] < args.q else ""
        print(f"{lbl:52s} {r['together']:3d}/{r['n_rare']:<5d} {r['lift']:5.1f} "
              f"{r['p']:8.4f} {r['q']:8.3f}{mark}")

    print(f"\npairs at raw p<0.05: {n_raw}   expected from noise alone: {0.05 * m:.0f}")
    print(f"pairs surviving BH at q<{args.q}: {len(flagged)}")
    if not flagged:
        print("\nVERDICT: no pair lock-in beyond chance. The extreme-looking rates "
              "above are what ~{:,} independent tests produce.".format(m))
    else:
        print("\nVERDICT: genuine pair lock-in. Position context for each partner "
              "(a thin slot can force a pairing rather than the selector choosing it):")
        for r in flagged:
            b = meta[r["other"]]
            s = b["slot"]
            share = cnt[r["other"]] / max(slot_total.get(s, 1), 1)
            print(f"  {str(b['name'])[:22]:22s} {b['team']:3s} slot {s:3s} "
                  f"${b['salary']:5.0f}  {cnt[r['other']]:3d} apps = {100*share:4.1f}% of "
                  f"{s} slots, filled by {len(slot_fillers.get(s, ())):3d} distinct players")
    return 0


if __name__ == "__main__":
    sys.exit(main())
