"""Quick weight sweep for the rescaled team_term: find a 'team' weight that
closes the entropy/max-exposure gap to needlunchmoney's near-uniform team
coverage (see project-team-selection-signal analysis) without wrecking the
other structural targets. Probes real select_portfolio calls, same pattern
as calibrate_pitcher_coverage_alpha/calibrate_hitter_own_weight.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.select_needlunchmoney_pool import (
    DEFAULT_WEIGHTS, PROJECT_ROOT, load_pool, build_feature_matrix,
    resolve_ev_column, load_needlunchmoney_actuals, TARGET, TARGET_HITTER_OWN,
    TARGET_MEAN_OVERLAP, TARGET_PITCHER_OWN, compute_pitcher_target_shares,
    calibrate_pitcher_coverage_alpha, calibrate_hitter_own_weight,
    select_portfolio,
)
from scripts.emulate_needlunchmoney import find_target_contest, field_players_for_contest, measure_structure
from scripts.analyze_rival_portfolio import team_map, primary_teams, pitcher_names

import ast

ACTUALS_DF = pd.read_csv(PROJECT_ROOT / "outputs" / "profitable_entrants_lineups.csv")
TARGET_CONTESTS = ("Rally Cap", "Relay Throw")


def his_entropy_and_max(slate: str, archive_dir: Path, n_teams_slate: int) -> tuple:
    his = ACTUALS_DF[(ACTUALS_DF["handle"] == "needlunchmoney") & (ACTUALS_DF["slate"] == int(slate))
                      & (ACTUALS_DF["contest"].isin(TARGET_CONTESTS))]
    if his.empty:
        return float("nan"), float("nan")
    names = [ast.literal_eval(x) for x in his["names"]]
    tmap = team_map(archive_dir)
    pitchers = pitcher_names(archive_dir)
    prim = primary_teams(names, tmap, pitchers)
    vc = pd.Series(prim).value_counts()
    p = (vc / vc.sum()).values
    h = -(p * np.log(p)).sum()
    return h / np.log(n_teams_slate), float(p.max())

SLATES = ["07252026", "07302026", "08032026"]
GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def entropy_and_max(portfolio_teams: list, n_teams_slate: int) -> tuple:
    vc = pd.Series(portfolio_teams).value_counts()
    p = (vc / vc.sum()).values
    h = -(p * np.log(p)).sum()
    return h / np.log(n_teams_slate), float(p.max()), len(vc)


for slate in SLATES:
    archive_dir = PROJECT_ROOT / "archive" / slate
    real_contest = find_target_contest(archive_dir)
    pool, proj_ext, features_df, confirmed_starter_ids, n_games, players_df = load_pool(archive_dir)
    ev_col = resolve_ev_column(n_games, real_contest["n_field"])
    feat = build_feature_matrix(pool.lineups, features_df, proj_ext, confirmed_starter_ids, ev_col)
    field_players_df = field_players_for_contest(archive_dir, real_contest)
    actuals = load_needlunchmoney_actuals(archive_dir, real_contest["contest"])

    target_chalk_index = actuals.get("chalk_index", TARGET["chalk_index"])
    target_hitter_own = actuals.get("hitter_own", TARGET_HITTER_OWN)
    target_mean_overlap = actuals.get("mean_overlap", TARGET_MEAN_OVERLAP)
    target_pitcher_own = actuals.get("pitcher_own", TARGET_PITCHER_OWN)
    target_pitcher_pair_rate = actuals.get("pitcher_pair_rate", TARGET["pitcher_pair_rate"])

    base_weights = dict(DEFAULT_WEIGHTS)
    alpha, _ = calibrate_pitcher_coverage_alpha(pool.lineups, feat, confirmed_starter_ids,
                                                 base_weights, target_pitcher_own)
    pitcher_target_share = compute_pitcher_target_shares(feat["conf_own"], alpha)
    w_hitter_own, _ = calibrate_hitter_own_weight(
        pool.lineups, feat, confirmed_starter_ids, players_df, archive_dir,
        field_players_df, base_weights, target_chalk_index, target_hitter_own, target_mean_overlap,
        target_pitcher_pair_rate, pitcher_target_share=pitcher_target_share)
    base_weights["hitter_own"] = w_hitter_own

    n_teams_slate = feat["n_teams_slate"]
    his_h, his_mx = his_entropy_and_max(slate, archive_dir, n_teams_slate)
    print(f"\n=== {slate}  n_teams_slate={n_teams_slate}  "
          f"his n_primary_teams={actuals.get('n_primary_teams', '?')}  "
          f"his entropy={his_h:.3f}  his max_exp={his_mx:.3f} ===")

    for tw in GRID:
        weights = dict(base_weights)
        weights["team"] = tw
        portfolio, selected_idx = select_portfolio(
            pool.lineups, feat, confirmed_starter_ids, weights,
            target_hitter_own=target_hitter_own, target_mean_overlap=target_mean_overlap,
            target_pitcher_pair_rate=target_pitcher_pair_rate, pitcher_target_share=pitcher_target_share)
        teams = feat["primary_team"][selected_idx]
        h, mx, n_uniq = entropy_and_max(list(teams), n_teams_slate)
        structure = measure_structure(portfolio, players_df, archive_dir, field_players_df)
        hitter_own_sel = feat["hitter_own"][selected_idx].mean()
        print(f"  team_w={tw:>5.1f}  entropy={h:.3f}  max_exp={mx:.3f}  n_teams_used={n_uniq}  "
              f"chalk_index={structure['chalk_index']:+.2f}  hitter_own={hitter_own_sel:.2f}  "
              f"mean_overlap={structure['mean_overlap']:.2f}  pitcher_pair_rate={structure['pitcher_pair_rate']:.3f}")
