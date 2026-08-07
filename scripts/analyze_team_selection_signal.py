"""Per-slate investigation: what team-level qualities correlate with needlunchmoney's
primary-stack team exposure, that our selector's exposure doesn't track as well?

Candidates checked per slate (all pre-lock / no lookahead bias):
  - saber_team:  SaberSim's team-specific implied run total ("Saber Team" column)
  - game_total:  SaberSim's game total ("Saber Total" column, shared by both teams)
  - team_own:    sum of hitters' projected ownership (My Own) on that team
  - team_val:    team's mean projected points per $1000 salary (hitters only)

For each slate/team we have his exposure fraction and our exposure fraction (from
primary_teams() on each portfolio's lineups). Spearman-correlate each signal against
exposure, per slate, separately for "his" and "ours" -- a signal that predicts HIS
picks much more than it predicts OURS is a candidate for a new selector term.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_rival_portfolio import team_map, primary_teams, pitcher_names

ARCHIVE = Path("archive")
ACTUALS = pd.read_csv("outputs/profitable_entrants_lineups.csv")
OURS = pd.read_csv("outputs/select_needlunchmoney_pool_graded.csv")

TARGET_CONTESTS = ("Rally Cap", "Relay Throw")

SLATES = sorted(OURS["slate"].unique())


def parse_names(s):
    if isinstance(s, str) and s.startswith("("):
        import ast
        return ast.literal_eval(s)
    return tuple(str(s).split("|"))


def load_raw_main(slate_dir: Path) -> pd.DataFrame:
    files = list(slate_dir.glob("MLB_*_DK_Main.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def team_features(raw: pd.DataFrame) -> pd.DataFrame:
    team_saber = raw[["Team", "Saber Team", "Saber Total"]].drop_duplicates("Team").set_index("Team")
    hitters = raw[raw["Pos"].astype(str).isin(["C", "1B", "2B", "3B", "SS", "OF"])].copy()
    hitters["val"] = hitters["dk_points"] / (hitters["Salary"] / 1000.0)
    agg = hitters.groupby("Team").agg(team_own=("My Own", "sum"), team_val=("val", "mean"),
                                       team_proj=("dk_points", "sum"))
    out = team_saber.join(agg, how="left")
    return out


def slate_exposure(names_list, tmap, pitchers):
    prim = primary_teams(names_list, tmap, pitchers)
    n = len(prim)
    s = pd.Series(prim).value_counts() / n
    return s


rows = []
for slate in SLATES:
    slate_str = str(slate).zfill(8)
    sdir = ARCHIVE / slate_str
    if not sdir.exists():
        print(f"skip {slate}: no archive dir")
        continue
    raw = load_raw_main(sdir)
    if raw.empty:
        print(f"skip {slate}: no raw main csv")
        continue
    tmap = team_map(sdir)
    pitchers = pitcher_names(sdir)
    feat = team_features(raw)

    his = ACTUALS[(ACTUALS["handle"] == "needlunchmoney") & (ACTUALS["slate"] == slate)
                  & (ACTUALS["contest"].isin(TARGET_CONTESTS))]
    ours = OURS[(OURS["handle"] == "ours") & (OURS["slate"] == slate)]
    if his.empty or ours.empty:
        print(f"skip {slate}: missing his={len(his)} ours={len(ours)}")
        continue

    his_names = [parse_names(x) for x in his["names"]]
    our_names = [parse_names(x) for x in ours["names"]]

    his_exp = slate_exposure(his_names, tmap, pitchers)
    our_exp = slate_exposure(our_names, tmap, pitchers)

    teams = feat.index.tolist()
    his_v = his_exp.reindex(teams).fillna(0.0)
    our_v = our_exp.reindex(teams).fillna(0.0)

    for sig in ["saber_team", "game_total", "team_own", "team_val", "team_proj"]:
        col = {"saber_team": "Saber Team", "game_total": "Saber Total"}.get(sig, sig)
        x = feat[col]
        valid = x.notna()
        if valid.sum() < 4:
            continue
        rh, ph = spearmanr(x[valid], his_v[valid])
        ro, po = spearmanr(x[valid], our_v[valid])
        rows.append({"slate": slate, "signal": sig, "n_teams": int(valid.sum()),
                      "corr_his": rh, "corr_ours": ro, "gap": rh - ro})

res = pd.DataFrame(rows)
res.to_csv("outputs/team_selection_signal.csv", index=False)

print(f"\n{len(SLATES)} slates checked, {res['slate'].nunique()} usable\n")
summary = res.groupby("signal").agg(
    mean_corr_his=("corr_his", "mean"),
    mean_corr_ours=("corr_ours", "mean"),
    mean_gap=("gap", "mean"),
    n_slates=("slate", "nunique"),
    n_his_pos=("corr_his", lambda s: (s > 0).sum()),
    n_ours_pos=("corr_ours", lambda s: (s > 0).sum()),
).round(3)
print(summary)

print("\nPer-slate detail:")
for sig in res["signal"].unique():
    sub = res[res["signal"] == sig].sort_values("slate")
    print(f"\n-- {sig} --")
    print(sub[["slate", "n_teams", "corr_his", "corr_ours", "gap"]].to_string(index=False))
