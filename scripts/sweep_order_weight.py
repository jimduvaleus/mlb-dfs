"""Weight sweep for the new order/cluster terms, same pattern as
sweep_team_weight.py: probe real select_portfolio calls, LOSO targets."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.select_needlunchmoney_pool import (
    DEFAULT_WEIGHTS, PROJECT_ROOT, load_pool, build_feature_matrix,
    resolve_ev_column, load_needlunchmoney_actuals_loso, TARGET, TARGET_HITTER_OWN,
    TARGET_MEAN_OVERLAP, TARGET_PITCHER_OWN, TARGET_MEAN_ORDER, TARGET_EXCESS_GAP,
    compute_pitcher_target_shares, calibrate_pitcher_coverage_alpha,
    calibrate_hitter_own_weight, select_portfolio,
)
from scripts.emulate_needlunchmoney import find_target_contest, field_players_for_contest

SLATES = ["07252026", "07302026", "08032026"]
ORDER_GRID = [0.5, 2.0, 5.0, 10.0, 20.0]
CLUSTER_GRID = [0.5, 2.0, 5.0, 10.0]

for slate in SLATES:
    archive_dir = PROJECT_ROOT / "archive" / slate
    real_contest = find_target_contest(archive_dir)
    pool, proj_ext, features_df, confirmed_starter_ids, n_games, players_df = load_pool(archive_dir)
    ev_col = resolve_ev_column(n_games, real_contest["n_field"])
    feat = build_feature_matrix(pool.lineups, features_df, proj_ext, confirmed_starter_ids, ev_col)
    field_players_df = field_players_for_contest(archive_dir, real_contest)
    actuals = load_needlunchmoney_actuals_loso(slate)

    target_chalk_index = actuals.get("chalk_index", TARGET["chalk_index"])
    target_hitter_own = actuals.get("hitter_own", TARGET_HITTER_OWN)
    target_mean_overlap = actuals.get("mean_overlap", TARGET_MEAN_OVERLAP)
    target_pitcher_own = actuals.get("pitcher_own", TARGET_PITCHER_OWN)
    target_pitcher_pair_rate = actuals.get("pitcher_pair_rate", TARGET["pitcher_pair_rate"])
    target_mean_order = actuals.get("mean_order", TARGET_MEAN_ORDER)
    target_excess_gap = actuals.get("mean_excess_gap", TARGET_EXCESS_GAP)
    if np.isnan(target_excess_gap):
        target_excess_gap = TARGET_EXCESS_GAP

    base_weights = dict(DEFAULT_WEIGHTS)
    alpha, _ = calibrate_pitcher_coverage_alpha(pool.lineups, feat, confirmed_starter_ids,
                                                 base_weights, target_pitcher_own)
    pitcher_target_share = compute_pitcher_target_shares(feat["conf_own"], alpha)
    w_hitter_own, _ = calibrate_hitter_own_weight(
        pool.lineups, feat, confirmed_starter_ids, players_df, archive_dir,
        field_players_df, base_weights, target_chalk_index, target_hitter_own, target_mean_overlap,
        target_pitcher_pair_rate, pitcher_target_share=pitcher_target_share)
    base_weights["hitter_own"] = w_hitter_own

    print(f"\n=== {slate}  target_mean_order={target_mean_order:.2f}  "
          f"target_excess_gap={target_excess_gap:.2f} ===")

    print("  -- order weight sweep (cluster held at default 2.0) --")
    for ow in ORDER_GRID:
        weights = dict(base_weights)
        weights["order"] = ow
        portfolio, sel = select_portfolio(
            pool.lineups, feat, confirmed_starter_ids, weights,
            target_hitter_own=target_hitter_own, target_mean_overlap=target_mean_overlap,
            target_pitcher_pair_rate=target_pitcher_pair_rate, target_mean_order=target_mean_order,
            target_excess_gap=target_excess_gap, pitcher_target_share=pitcher_target_share)
        mo = feat["hitter_mean_order"][sel].mean()
        eg = feat["primary_excess_gap"][sel].mean()
        ho = feat["hitter_own"][sel].mean()
        print(f"    order_w={ow:>5.1f}  mean_order={mo:.2f}  excess_gap={eg:.2f}  hitter_own={ho:.2f}")

    print("  -- cluster weight sweep (order held at default 2.0) --")
    for cw in CLUSTER_GRID:
        weights = dict(base_weights)
        weights["cluster"] = cw
        portfolio, sel = select_portfolio(
            pool.lineups, feat, confirmed_starter_ids, weights,
            target_hitter_own=target_hitter_own, target_mean_overlap=target_mean_overlap,
            target_pitcher_pair_rate=target_pitcher_pair_rate, target_mean_order=target_mean_order,
            target_excess_gap=target_excess_gap, pitcher_target_share=pitcher_target_share)
        mo = feat["hitter_mean_order"][sel].mean()
        eg = feat["primary_excess_gap"][sel].mean()
        ho = feat["hitter_own"][sel].mean()
        print(f"    cluster_w={cw:>5.1f}  mean_order={mo:.2f}  excess_gap={eg:.2f}  hitter_own={ho:.2f}")
