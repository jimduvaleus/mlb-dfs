"""Sweep PITCHER_QUALITY_ALPHA (the independent quality exponent in the
multiplicative own_norm**alpha_own * quality_norm**alpha_quality design --
see compute_pitcher_target_shares' docstring for why this replaced the
linear-blend version, which was confirmed not to work).

For each candidate quality_alpha, alpha_own is recalibrated (real
select_portfolio probes, same as calibrate_pitcher_coverage_alpha) to hit
target_pitcher_own -- the two exponents are largely orthogonal by
construction (own_norm**a_own doesn't depend on quality, and vice versa),
so this should be a cleaner separation than the old shared-alpha blend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.select_needlunchmoney_pool import (
    DEFAULT_WEIGHTS, PROJECT_ROOT, load_pool, build_feature_matrix,
    resolve_ev_column, load_needlunchmoney_actuals_loso, TARGET_PITCHER_OWN,
    REFERENCE_HIS_PITCHER_SS_PROJ,
    calibrate_pitcher_coverage_alpha, compute_pitcher_target_shares, select_portfolio,
)
from scripts.emulate_needlunchmoney import find_target_contest, field_players_for_contest

SLATES = ["07252026", "07302026", "08032026"]
QUALITY_ALPHA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

for slate in SLATES:
    archive_dir = PROJECT_ROOT / "archive" / slate
    real_contest = find_target_contest(archive_dir)
    pool, proj_ext, features_df, confirmed_starter_ids, n_games, players_df = load_pool(archive_dir)
    ev_col = resolve_ev_column(n_games, real_contest["n_field"])
    feat = build_feature_matrix(pool.lineups, features_df, proj_ext, confirmed_starter_ids, ev_col)
    actuals = load_needlunchmoney_actuals_loso(slate)
    target_pitcher_own = actuals.get("pitcher_own", TARGET_PITCHER_OWN)

    print(f"\n=== {slate}  target_pitcher_own={target_pitcher_own:.2f}  "
          f"his_ss_proj_reference={REFERENCE_HIS_PITCHER_SS_PROJ:.2f} ===")

    for qa in QUALITY_ALPHA_GRID:
        alpha_own, _ = calibrate_pitcher_coverage_alpha(
            pool.lineups, feat, confirmed_starter_ids, DEFAULT_WEIGHTS, target_pitcher_own,
            quality_alpha=qa)
        share = compute_pitcher_target_shares(feat["conf_own"], alpha_own, feat.get("conf_quality"), qa)
        portfolio, sel = select_portfolio(pool.lineups, feat, confirmed_starter_ids, DEFAULT_WEIGHTS,
                                           pitcher_target_share=share)
        po = feat["pitcher_own"][sel].mean()
        ssp = feat["pitcher_ss_proj"][sel].mean()
        n_uniq_pitchers = len(set(int(p) for p in feat["pids_arr"][sel, 0:2].reshape(-1)))
        print(f"  quality_alpha={qa:.1f}  alpha_own_chosen={alpha_own:.2f}  pitcher_own={po:.2f}  "
              f"pitcher_ss_proj={ssp:.2f}  n_distinct_pitchers={n_uniq_pitchers}")
