"""Does needlunchmoney's edge come from WHICH players he picks (stock-picking
skill) rather than HOW he arranges them (portfolio structure)? Structure has
been checked repeatedly and found statistically indistinguishable from ours
even under generous (LOSO) calibration -- see project-select-pool-loso-fix
memory. This is the natural remaining hypothesis.

Part A -- player outperformance: for every player-slot he actually rostered
(across all 10 archived slates), compute realized FPTS minus PRE-LOCK
projected mean ("My Proj" from the SaberSim per-player export). Compare the
pooled average against (a) our selected portfolio's rostered players, (b)
the real field's ownership-weighted average (a random-field-entrant
baseline). If his picks systematically beat their projections more than
ours/the field's do, that's real player-evaluation alpha.

Part B -- top-of-order bias: does he skew hitter selection toward
top-of-order batting slots (more PA/RBI opportunity) more than our
EV/ownership-driven selection, independent of raw projected mean?

Part C -- order clustering within a primary stack: "3-4-5-6" (tightly
clustered, low excess_gap) vs "1-2-5-7" (scattered, high excess_gap).
excess_gap = (max_order - min_order) - (k - 1) i.e. 0 for a perfectly
contiguous run of k slots, growing with dispersion. Compared against an
exact combinatorial null (mean excess_gap over every possible k-subset of
that team's ACTUAL available order slots that slate) rather than a generic
1-9 assumption -- cheaper and more rigorous than reconstructing the real
field's own stack-order behavior from standings zips.
"""
import ast
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.external_pool import discover_external_files, parse_player_projections  # noqa: E402
from scripts.emulate_needlunchmoney import find_target_contest, field_players_for_contest, TENTH_SLATE  # noqa: E402
from scripts.analyze_rival_portfolio import team_map, primary_teams, pitcher_names  # noqa: E402
from tests.bt_core import BACKTEST_SLATES  # noqa: E402

ACTUALS = pd.read_csv(PROJECT_ROOT / "outputs" / "profitable_entrants_lineups.csv")
OURS = pd.read_csv(PROJECT_ROOT / "outputs" / "select_needlunchmoney_pool_graded.csv", dtype={"slate": str})
TARGET_CONTESTS = ("Rally Cap", "Relay Throw")
ALL_SLATES = list(BACKTEST_SLATES) + [TENTH_SLATE]


def parse_names(s):
    return ast.literal_eval(s)


def null_excess_gap(orders_available: list, k: int) -> float:
    orders_available = sorted(orders_available)
    n = len(orders_available)
    if k < 2 or k > n:
        return np.nan
    total, count = 0.0, 0
    for combo in combinations(orders_available, k):
        total += (combo[-1] - combo[0]) - (k - 1)
        count += 1
    return total / count


pick_rows, order_rows, stack_rows = [], [], []
null_cache = {}

for slate in ALL_SLATES:
    adir = PROJECT_ROOT / "archive" / slate
    if not adir.exists():
        continue
    try:
        rc = find_target_contest(adir)
    except ValueError:
        continue

    his = ACTUALS[(ACTUALS.handle == "needlunchmoney") & (ACTUALS.slate == int(slate))
                  & (ACTUALS.contest.isin(TARGET_CONTESTS))]
    ours = OURS[(OURS.handle == "ours") & (OURS.slate == slate)]
    if his.empty or ours.empty:
        print(f"skip {slate}: his={len(his)} ours={len(ours)}")
        continue

    his_names = [parse_names(x) for x in his["names"]]
    our_names = [parse_names(x) for x in ours["names"]]

    found = discover_external_files(str(adir))
    proj_ext = parse_player_projections(Path(found["projections_path"]))
    proj_mean = proj_ext.set_index("name")["mean"].to_dict()
    proj_order = proj_ext.set_index("name")["order"].to_dict()

    field_df = field_players_for_contest(adir, rc).drop_duplicates("player")
    field_own = field_df.set_index("player")["pct_drafted"].to_dict()

    tmap = team_map(adir)
    pitchers = pitcher_names(adir)

    def diff_for(n):
        pm = proj_mean.get(n)
        fp = field_own_fpts.get(n)
        if pm is None or fp is None or np.isnan(pm) or np.isnan(fp):
            return None
        return fp - pm

    field_own_fpts = field_df.set_index("player")["fpts"].to_dict()

    def add_pick_rows(names_list, source):
        for lu in names_list:
            for n in lu:
                d = diff_for(n)
                if d is None:
                    continue
                pick_rows.append({"slate": slate, "source": source, "name": n,
                                   "diff": d, "is_pitcher": n in pitchers})

    add_pick_rows(his_names, "his")
    add_pick_rows(our_names, "ours")

    for _, row in field_df.iterrows():
        n = row["player"]
        d = diff_for(n)
        w = row["pct_drafted"]
        if d is None or np.isnan(w):
            continue
        pick_rows.append({"slate": slate, "source": "field", "name": n,
                           "diff": d, "weight": w, "is_pitcher": n in pitchers})

    def add_order_rows(names_list, source):
        for lu in names_list:
            for n in lu:
                if n in pitchers:
                    continue
                o = proj_order.get(n)
                if o is None or np.isnan(o):
                    continue
                order_rows.append({"slate": slate, "source": source, "order": o})

    add_order_rows(his_names, "his")
    add_order_rows(our_names, "ours")
    for _, row in field_df.iterrows():
        n = row["player"]
        if n in pitchers:
            continue
        o = proj_order.get(n)
        w = row["pct_drafted"]
        if o is None or (isinstance(o, float) and np.isnan(o)) or np.isnan(w):
            continue
        order_rows.append({"slate": slate, "source": "field", "order": o, "weight": w})

    team_orders = {}
    for n, t in tmap.items():
        o = proj_order.get(n)
        if n in pitchers or o is None or np.isnan(o):
            continue
        team_orders.setdefault(t, set()).add(o)

    def add_stack_rows(names_list, source):
        prim = primary_teams(names_list, tmap, pitchers)
        for lu, team in zip(names_list, prim):
            if not team:
                continue
            orders = sorted(
                proj_order[n] for n in lu
                if n not in pitchers and tmap.get(n) == team
                and proj_order.get(n) is not None and not np.isnan(proj_order.get(n))
            )
            k = len(orders)
            if k < 2:
                continue
            excess_gap = (orders[-1] - orders[0]) - (k - 1)
            key = (slate, team, k)
            if key not in null_cache:
                avail = team_orders.get(team, [])
                null_cache[key] = null_excess_gap(list(avail), k)
            stack_rows.append({"slate": slate, "source": source, "team": team, "size": k,
                                "excess_gap": excess_gap, "null_excess_gap": null_cache[key]})

    add_stack_rows(his_names, "his")
    add_stack_rows(our_names, "ours")

df_pick = pd.DataFrame(pick_rows)
df_order = pd.DataFrame(order_rows)
df_stack = pd.DataFrame(stack_rows)

df_pick.to_csv(PROJECT_ROOT / "outputs" / "player_pick_quality_picks.csv", index=False)
df_order.to_csv(PROJECT_ROOT / "outputs" / "player_pick_quality_order.csv", index=False)
df_stack.to_csv(PROJECT_ROOT / "outputs" / "player_pick_quality_stacks.csv", index=False)

print(f"\n{df_pick.slate.nunique()} slates used\n")

# ---------------------------------------------------------------------------
print("=" * 70)
print("PART A -- player outperformance (realized FPTS - pre-lock projection)")
print("=" * 70)


def weighted_mean(d, w):
    return float(np.average(d, weights=w))


for label, sub in [("all", df_pick), ("hitters", df_pick[~df_pick.is_pitcher]),
                    ("pitchers", df_pick[df_pick.is_pitcher])]:
    his_d = sub[sub.source == "his"]["diff"]
    our_d = sub[sub.source == "ours"]["diff"]
    fld = sub[sub.source == "field"]
    fld_mean = weighted_mean(fld["diff"], fld["weight"]) if len(fld) else np.nan
    print(f"\n-- {label} --  n_his={len(his_d)} n_ours={len(our_d)} n_field_rows={len(fld)}")
    print(f"  his_mean_diff   {his_d.mean():+.3f}  (median {his_d.median():+.3f})")
    print(f"  ours_mean_diff  {our_d.mean():+.3f}  (median {our_d.median():+.3f})")
    print(f"  field_own_wtd_mean_diff  {fld_mean:+.3f}")
    if len(his_d) > 5 and len(our_d) > 5:
        u, p = mannwhitneyu(his_d, our_d, alternative="two-sided")
        print(f"  Mann-Whitney his vs ours: p={p:.4f}")

    # per-slate: does his average beat ours, beat field, each slate?
    per_slate = sub[sub.source.isin(["his", "ours"])].groupby(["slate", "source"])["diff"].mean().unstack()
    if "his" in per_slate.columns and "ours" in per_slate.columns:
        per_slate = per_slate.dropna()
        beats = (per_slate["his"] > per_slate["ours"]).sum()
        print(f"  his > ours on {beats}/{len(per_slate)} slates")
        print(per_slate.round(2).to_string())

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART B -- top-of-order bias (hitters only)")
print("=" * 70)

for src in ("his", "ours"):
    sub = df_order[df_order.source == src]
    print(f"\n-- {src} --  n={len(sub)}  mean_order={sub['order'].mean():.2f}  "
          f"frac_order<=4  {(sub['order'] <= 4).mean():.3f}")

fld = df_order[df_order.source == "field"]
print(f"\n-- field (ownership-weighted) --  n={len(fld)}  "
      f"mean_order={weighted_mean(fld['order'], fld['weight']):.2f}  "
      f"frac_order<=4  {float(np.average(fld['order'] <= 4, weights=fld['weight'])):.3f}")

per_slate_o = df_order[df_order.source.isin(["his", "ours"])].groupby(["slate", "source"])["order"].mean().unstack().dropna()
print(f"\nper-slate mean order (lower = more top-of-order):")
print(per_slate_o.round(2).to_string())
print(f"his < ours (more top-heavy) on {(per_slate_o['his'] < per_slate_o['ours']).sum()}/{len(per_slate_o)} slates")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART C -- order clustering within primary stacks")
print("=" * 70)
print("excess_gap = (max_order - min_order) - (k-1): 0 = perfectly contiguous "
      "(e.g. 3-4-5-6), larger = more scattered (e.g. 1-2-5-7)")

for src in ("his", "ours"):
    sub = df_stack[df_stack.source == src].dropna(subset=["null_excess_gap"])
    print(f"\n-- {src} --  n_stacks={len(sub)}")
    print(f"  observed mean excess_gap  {sub['excess_gap'].mean():.3f}")
    print(f"  null (random-draw) mean excess_gap  {sub['null_excess_gap'].mean():.3f}")
    delta = sub["excess_gap"] - sub["null_excess_gap"]
    print(f"  mean(observed - null)  {delta.mean():+.3f}  "
          f"(negative = MORE clustered than random)")
    if len(sub) > 5:
        stat, p = mannwhitneyu(sub["excess_gap"], sub["null_excess_gap"], alternative="two-sided")
        print(f"  Mann-Whitney vs null: p={p:.4f}")

    by_size = sub.groupby("size").agg(n=("excess_gap", "size"),
                                       observed=("excess_gap", "mean"),
                                       null=("null_excess_gap", "mean"))
    print(by_size.round(3).to_string())

his_sub = df_stack[df_stack.source == "his"].dropna(subset=["null_excess_gap"])
our_sub = df_stack[df_stack.source == "ours"].dropna(subset=["null_excess_gap"])
if len(his_sub) > 5 and len(our_sub) > 5:
    stat, p = mannwhitneyu(his_sub["excess_gap"], our_sub["excess_gap"], alternative="two-sided")
    print(f"\nhis excess_gap vs ours excess_gap: p={p:.4f}  "
          f"(his mean {his_sub['excess_gap'].mean():.3f} vs ours {our_sub['excess_gap'].mean():.3f})")
