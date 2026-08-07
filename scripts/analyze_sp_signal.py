"""What pre-lock signal predicts a confirmed starting pitcher beating his
own projection? Follow-up to project-player-pick-quality: his pitcher picks
beat their pre-lock projection by +1.37 FPTS/slot (p<.0001) while ours and
the field don't -- that's real player-evaluation alpha, but doesn't say
WHICH signal he's reading. User-proposed candidate signals, all pulled
straight from the raw SaberSim per-player export (pre-lock, no lookahead):
SS Proj, Adj Own, SS Proj/Salary, opponent's implied team run total
("Saber Team" via the Opp column), dk_95/dk_99 percentile, dk_std, K.

Two-part test:
  Part 1 -- GENERAL predictive power: pooled across every CONFIRMED
  STARTER-slate (one row per starter per slate, not exposure-weighted),
  Spearman-correlate each signal against (realized FPTS - "My Proj"). This
  answers "is this signal a real market inefficiency" independent of who
  picks whom.
  Part 2 -- does he lean into the signal(s) that turn out to predict
  outperformance more than our selection / the field does? Exposure-weighted
  (his real lineup-slots, our selected lineup-slots, field ownership-
  weighted), same pattern as analyze_player_pick_quality.py Part A.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.emulate_needlunchmoney import find_target_contest, field_players_for_contest, TENTH_SLATE  # noqa: E402
from scripts.analyze_rival_portfolio import pitcher_names  # noqa: E402
from tests.bt_core import BACKTEST_SLATES  # noqa: E402

ACTUALS = pd.read_csv(PROJECT_ROOT / "outputs" / "profitable_entrants_lineups.csv")
OURS = pd.read_csv(PROJECT_ROOT / "outputs" / "select_needlunchmoney_pool_graded.csv", dtype={"slate": str})
TARGET_CONTESTS = ("Rally Cap", "Relay Throw")
ALL_SLATES = list(BACKTEST_SLATES) + [TENTH_SLATE]

SIGNALS = ["ss_proj", "adj_own", "ss_proj_per_sal", "opp_saber_team",
           "dk_95_percentile", "dk_99_percentile", "dk_std", "k_proj"]


def parse_names(s):
    return ast.literal_eval(s)


starter_rows = []   # Part 1: one row per confirmed starter per slate
pick_rows = []       # Part 2: one row per pitcher-slot rostered by his/ours/field

for slate in ALL_SLATES:
    adir = PROJECT_ROOT / "archive" / slate
    if not adir.exists():
        continue
    try:
        rc = find_target_contest(adir)
    except ValueError:
        continue

    files = list(adir.glob("MLB_*_DK_Main.csv"))
    if not files:
        continue
    raw = pd.read_csv(files[0])

    saber_team = raw.drop_duplicates("Team").set_index("Team")["Saber Team"].to_dict()
    is_sp = raw["Pos"].astype(str).isin(["P", "SP", "RP"])
    confirmed = raw[is_sp & (raw["Status"].astype(str) == "Confirmed")].copy()
    if confirmed.empty:
        continue
    confirmed["opp_saber_team"] = confirmed["Opp"].map(saber_team)
    confirmed["ss_proj_per_sal"] = confirmed["SS Proj"] / (confirmed["Salary"] / 1000.0)

    field_df = field_players_for_contest(adir, rc).drop_duplicates("player")
    field_fpts = field_df.set_index("player")["fpts"].to_dict()
    field_own_std = field_df.set_index("player")["pct_drafted"].to_dict()

    name_col = {
        "ss_proj": "SS Proj", "adj_own": "Adj Own", "ss_proj_per_sal": "ss_proj_per_sal",
        "opp_saber_team": "opp_saber_team", "dk_95_percentile": "dk_95_percentile",
        "dk_99_percentile": "dk_99_percentile", "dk_std": "dk_std", "k_proj": "K",
    }

    for _, row in confirmed.iterrows():
        n = row["Name"]
        fp = field_fpts.get(n)
        pm = row["My Proj"]
        if fp is None or np.isnan(fp) or np.isnan(pm):
            continue
        d = {"slate": slate, "name": n, "diff": fp - pm}
        for sig, col in name_col.items():
            d[sig] = row.get(col)
        starter_rows.append(d)

    proj_myproj = raw.set_index("Name")["My Proj"].to_dict()
    sig_lookup = {sig: raw.set_index("Name")[col].to_dict() if col in raw.columns
                  else confirmed.set_index("Name")[col].to_dict()
                  for sig, col in name_col.items()}
    # opp_saber_team / ss_proj_per_sal only computed on `confirmed`; merge those in
    opp_map = confirmed.set_index("Name")["opp_saber_team"].to_dict()
    persal_map = confirmed.set_index("Name")["ss_proj_per_sal"].to_dict()

    def sig_value(n, sig):
        if sig == "opp_saber_team":
            return opp_map.get(n, np.nan)
        if sig == "ss_proj_per_sal":
            return persal_map.get(n, np.nan)
        return sig_lookup[sig].get(n, np.nan)

    pitchers = pitcher_names(adir)
    his = ACTUALS[(ACTUALS.handle == "needlunchmoney") & (ACTUALS.slate == int(slate))
                  & (ACTUALS.contest.isin(TARGET_CONTESTS))]
    ours = OURS[(OURS.handle == "ours") & (OURS.slate == slate)]
    confirmed_names = set(confirmed["Name"])

    def add_pick_rows(names_list, source):
        for lu in names_list:
            for n in lu:
                if n not in confirmed_names:
                    continue
                row = {"slate": slate, "source": source, "name": n}
                for sig in SIGNALS:
                    row[sig] = sig_value(n, sig)
                pick_rows.append(row)

    if not his.empty:
        add_pick_rows([parse_names(x) for x in his["names"]], "his")
    if not ours.empty:
        add_pick_rows([parse_names(x) for x in ours["names"]], "ours")

    for _, row in field_df.iterrows():
        n = row["player"]
        if n not in confirmed_names:
            continue
        w = row["pct_drafted"]
        if np.isnan(w):
            continue
        d = {"slate": slate, "source": "field", "name": n, "weight": w}
        for sig in SIGNALS:
            d[sig] = sig_value(n, sig)
        pick_rows.append(d)

df_starter = pd.DataFrame(starter_rows)
df_pick = pd.DataFrame(pick_rows)
df_starter.to_csv(PROJECT_ROOT / "outputs" / "sp_signal_starters.csv", index=False)
df_pick.to_csv(PROJECT_ROOT / "outputs" / "sp_signal_picks.csv", index=False)

print(f"Part 1 pool: {len(df_starter)} confirmed-starter-slates across {df_starter.slate.nunique()} slates")
print("=" * 70)
print("PART 1 -- does this signal predict outperformance (realized - My Proj)?")
print("=" * 70)
results = []
for sig in SIGNALS:
    sub = df_starter[[sig, "diff"]].dropna()
    if len(sub) < 10:
        print(f"{sig:20s}  n={len(sub)} too few")
        continue
    rho, p = spearmanr(sub[sig], sub["diff"])
    results.append((sig, rho, p, len(sub)))
    print(f"{sig:20s}  n={len(sub):4d}  spearman_rho={rho:+.3f}  p={p:.4f}")

print("\nranked by |rho|:")
for sig, rho, p, n in sorted(results, key=lambda x: -abs(x[1])):
    flag = "  <-- significant" if p < 0.05 else ""
    print(f"  {sig:20s}  rho={rho:+.3f}  p={p:.4f}{flag}")

print("\n" + "=" * 70)
print("PART 2 -- does his pitcher selection lean into each signal more than ours/field?")
print("=" * 70)


def weighted_mean(d, w):
    return float(np.average(d, weights=w))


for sig in SIGNALS:
    his_v = df_pick[(df_pick.source == "his")][sig].dropna()
    our_v = df_pick[(df_pick.source == "ours")][sig].dropna()
    fld = df_pick[(df_pick.source == "field")][[sig, "weight"]].dropna()
    fld_mean = weighted_mean(fld[sig], fld["weight"]) if len(fld) else np.nan
    print(f"\n{sig}: his={his_v.mean():.3f}  ours={our_v.mean():.3f}  field={fld_mean:.3f}  "
          f"(n_his={len(his_v)} n_ours={len(our_v)} n_field={len(fld)})")
