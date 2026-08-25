"""Haugh & Singal line 2, solved with the quadratic term restored.

The load-bearing tests here are the two EQUIVALENCE ANCHORS, following the
pattern `test_mrp_marginal_reward.py` uses against `bt_core`:

  1. At lambda=0 the CP-SAT solver must return exactly what the CBC solver
     returns. That is what makes "same feasible set, richer objective" a claim
     rather than a hope -- the whole constraint block (C1-C9b) was ported by
     hand, and a silently different feasible region would be invisible in every
     other test in this file.
  2. `lineup_variance`'s pair form must equal the variance of the lineup's
     actual simulated scores. The solver optimises the pair form; if that is
     not the real w'Sigma w, the frontier is a frontier of something else.
"""
import numpy as np
import pandas as pd
import pytest

from src.optimization.mrp.frontier_qp import (
    _build_metadata,
    restrict_to_playable,
    _build_model,
    _OBJ_SCALE,
    calibrate_lambda_grid,
    frontier_lineups,
    solve_lambda,
)
from src.optimization.mrp.lineup_variance import (
    lineup_variance,
    unit_covariance_pairs,
    unit_player_groups,
)
from src.optimization.optimal_lineups import generate_optimal_lineups

TEAMS = [("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC")]
ROSTER = ("P", "P", "C", "C", "1B", "1B", "2B", "2B", "3B", "3B",
          "SS", "SS", "OF", "OF", "OF", "OF")
SOLVE_KW = dict(min_uniques=1, min_stack=3, timeout_s=10.0)


def _players_df(seed=7):
    """Non-degenerate means and salaries.

    Deliberately NOT the tied-mean fixture other MRP tests share: with ties,
    two solvers can both be optimal and still disagree, which would make the
    lambda=0 anchor test pass or fail on tie-breaking rather than on the
    feasible set.
    """
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for team, opp in TEAMS:
        for pos in ROSTER:
            pid += 1
            rows.append({
                "player_id": pid, "name": f"p{pid}", "position": pos,
                "eligible_positions": [pos], "team": team, "opponent": opp,
                "game": f"{team}@{opp}", "salary": int(rng.integers(2500, 4800)),
                "mean": float(6.0 + rng.normal(4.0, 2.0)), "std_dev": 5.0,
                "ownership": float(rng.uniform(2, 30)),
            })
    return pd.DataFrame(rows)


def _sigma_inputs(df, seed=11):
    rng = np.random.default_rng(seed)
    var = {int(p): float(rng.uniform(4, 60)) for p in df.player_id}
    cov = {}
    for _t, g in df[df.position != "P"].groupby("team"):
        ids = sorted(int(x) for x in g.player_id)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cov[(ids[i], ids[j])] = float(rng.uniform(-3, 15))
    return var, cov


def _zero_sigma(df):
    return {int(p): 0.0 for p in df.player_id}


def _assert_dk_legal(df, pids):
    meta = df.set_index("player_id")
    assert len(set(pids)) == 10
    pos = [meta.loc[p, "position"] for p in pids]
    assert sorted(pos) == sorted(["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"])
    assert sum(int(meta.loc[p, "salary"]) for p in pids) <= 50_000
    bat_teams = [meta.loc[p, "team"] for p in pids if meta.loc[p, "position"] != "P"]
    for t in set(bat_teams):
        assert bat_teams.count(t) <= 5, "C4: at most 5 batters per team"
    p_opps = {meta.loc[p, "opponent"] for p in pids if meta.loc[p, "position"] == "P"}
    assert not (set(bat_teams) & p_opps), "C5: no batter faces a rostered pitcher"



def _contest_env(df, n_sims=300, seed=5):
    """(contest_specs, sim_matrix, col_map) for the line-4 lambda* search.

    `thr` is the field score at each paying tier's last rank, per world -- what
    `field_order_statistics` returns in production. Two contests with different
    bars stand in for a small and a large field, so lambda* has something to
    discriminate on.
    """
    rng = np.random.default_rng(seed)
    pids = df.player_id.tolist()
    col_map = {int(p): i for i, p in enumerate(pids)}
    sim = rng.normal(df["mean"].to_numpy(), 5.0, size=(n_sims, len(pids))).astype(np.float32)
    specs = []
    for key, bar, steps in (("small", 92.0, np.array([50.0, 20.0, 10.0])),
                            ("large", 108.0, np.array([500.0, 100.0, 25.0]))):
        thr = np.stack([np.full(n_sims, bar + off) for off in (8.0, 3.0, 0.0)], axis=1)
        specs.append((key, thr, steps, 5))
    return specs, sim, col_map


# --------------------------------------------------------------- anchors ----

def test_lambda_zero_reproduces_the_cbc_optimum():
    """The ported constraint block defines the same feasible set as CBC's."""
    df = _players_df()
    var, cov = _sigma_inputs(df)
    md = _build_metadata(df)

    cbc = generate_optimal_lineups(df, n=1, min_uniques=1, min_stack=3)[0]
    sat = solve_lambda(md, 0.0, var, cov, _zero_sigma(df), n=1, **SOLVE_KW)[0]

    assert sorted(sat.player_ids) == sorted(cbc.player_ids)


def test_pair_form_variance_equals_the_simulated_lineup_variance():
    """`lineup_variance` is the real w'Sigma w, not a proxy for it.

    Built on a sim whose only dependence is WITHIN copula units, which is the
    structure `unit_covariance_pairs` assumes. If the estimator or the unit
    grouping were wrong, the pair form and the realised variance would diverge.
    """
    df = _players_df()
    rng = np.random.default_rng(5)
    n_sims = 40_000
    pids = df.player_id.tolist()
    col = {p: i for i, p in enumerate(pids)}

    mat = rng.normal(0.0, 1.0, size=(n_sims, len(pids)))
    # One shared factor per unit -- exactly the overlay's structure.
    for members in unit_player_groups(df).values():
        f = rng.normal(0.0, 1.0, size=n_sims)
        for p in members:
            mat[:, col[p]] += 0.8 * f
    mat = mat * 3.0 + df["mean"].to_numpy()

    var, cov = unit_covariance_pairs(mat, pids, df)
    lu = generate_optimal_lineups(df, n=1, min_uniques=1, min_stack=3)[0]

    realised = float(np.var(mat[:, [col[p] for p in lu.player_ids]].sum(axis=1)))
    assert lineup_variance(lu.player_ids, var, cov) == pytest.approx(realised, rel=0.02)


def test_mccormick_linearisation_is_exact():
    """Solver objective == independently recomputed mu + lambda(var - 2 sigma)."""
    from ortools.sat.python import cp_model

    df = _players_df()
    var, cov = _sigma_inputs(df)
    rng = np.random.default_rng(2)
    sig = {int(p): float(rng.uniform(0, 8)) for p in df.player_id}
    md = _build_metadata(df)

    for lam in (0.03, 0.1, 0.5):
        model, xp, _y, xp_list = _build_model(md, lam, var, cov, sig, 3, None, None, None)
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = 42
        solver.parameters.num_search_workers = 8
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        pids = sorted({xp_list[j][0] for j in range(len(xp_list)) if solver.Value(xp[j])})

        recomputed = (sum(md["mean_map"][p] for p in pids)
                      + lam * (lineup_variance(pids, var, cov)
                               - 2.0 * sum(sig[p] for p in pids)))
        # Tolerance is the integer-scaling rounding, nothing else.
        assert solver.ObjectiveValue() / _OBJ_SCALE == pytest.approx(recomputed, abs=2.0 / _OBJ_SCALE * 10)


# ------------------------------------------------------------- behaviour ----

def test_variance_rises_and_mean_falls_along_the_frontier():
    df = _players_df()
    var, cov = _sigma_inputs(df)
    md = _build_metadata(df)

    seen_var, seen_mean = [], []
    for lam in (0.0, 0.02, 0.05, 0.1):
        lu = solve_lambda(md, lam, var, cov, _zero_sigma(df), n=1, **SOLVE_KW)[0]
        seen_var.append(lineup_variance(lu.player_ids, var, cov))
        seen_mean.append(sum(md["mean_map"][p] for p in lu.player_ids))

    assert all(b >= a - 1e-6 for a, b in zip(seen_var, seen_var[1:])), seen_var
    assert all(b <= a + 1e-6 for a, b in zip(seen_mean, seen_mean[1:])), seen_mean
    assert seen_var[-1] > seen_var[0], "lambda must actually buy variance"


def test_sigma_dG_pushes_cutoff_movers_out():
    """The -2*lambda*w'sigma_dG term has the sign the paper gives it.

    Checked at lambda>0: the whole lambda-term is scaled by lambda, so at
    lambda=0 sigma_dG correctly does nothing at all.
    """
    df = _players_df()
    var, cov = _sigma_inputs(df)
    md = _build_metadata(df)

    base = solve_lambda(md, 0.05, var, cov, _zero_sigma(df), n=1, **SOLVE_KW)[0]
    penalised = sorted(base.player_ids)[:3]
    sig = {int(p): (500.0 if int(p) in penalised else 0.0) for p in df.player_id}

    after = solve_lambda(md, 0.05, var, cov, sig, n=1, **SOLVE_KW)[0]
    assert not (set(penalised) & set(after.player_ids))


def test_sigma_dG_is_inert_at_lambda_zero():
    df = _players_df()
    var, cov = _sigma_inputs(df)
    md = _build_metadata(df)
    huge = {int(p): 1e4 for p in df.player_id}

    a = solve_lambda(md, 0.0, var, cov, _zero_sigma(df), n=1, **SOLVE_KW)[0]
    b = solve_lambda(md, 0.0, var, cov, huge, n=1, **SOLVE_KW)[0]
    assert sorted(a.player_ids) == sorted(b.player_ids)


def test_calibrated_grid_starts_at_zero_and_increases():
    df = _players_df()
    var, cov = _sigma_inputs(df)
    md = _build_metadata(df)

    grid = calibrate_lambda_grid(md, var, cov, _zero_sigma(df),
                                 n_lambdas=8, target_uniques=4, **SOLVE_KW)
    assert grid[0] == 0.0, "the plain projection lineup is always on the frontier"
    assert len(grid) == 8
    assert all(b > a for a, b in zip(grid, grid[1:])), grid


def test_sweep_dedups_and_returns_legal_lineups():
    df = _players_df()
    var, cov = _sigma_inputs(df)

    specs, sim, cmap = _contest_env(df)
    lus, lams, diag = frontier_lineups(df, var, cov, _zero_sigma(df), specs, sim, cmap,
                                       n_lambdas=4, sample_n=2_000,
                                       # per_team is derived; the floor pins it
                                       target_lineups=1, min_per_team=3,
                                       n_anchors=1, n_generations=1,
                                       mutants_per_parent=2, **SOLVE_KW)
    assert diag["n_lambda_star"] >= 1
    assert lus, "sweep produced nothing"
    assert len(lams) == len(lus)
    keys = {frozenset(l.player_ids) for l in lus}
    assert len(keys) == len(lus), "sweep-wide dedup should leave no exact repeats"
    for lu in lus:
        _assert_dk_legal(df, lu.player_ids)


def test_missing_eligible_positions_is_rejected():
    """Production catches this in `_frontier_augment`, which fails soft and
    records why (see test_mrp_frontier_integration). This pins that the
    generator itself does not silently produce garbage instead."""
    df = _players_df().drop(columns=["eligible_positions"])
    var, cov = _sigma_inputs(_players_df())
    specs, sim, cmap = _contest_env(_players_df())
    with pytest.raises((ValueError, KeyError, AttributeError)):
        frontier_lineups(df, var, cov, _zero_sigma(df), specs, sim, cmap,
                         n_lambdas=2, sample_n=200,
                         target_lineups=1, min_per_team=1,
                         n_anchors=1, n_generations=0)


def test_generated_lineups_span_many_stack_teams():
    """THE REGRESSION THIS TEST EXISTS FOR.

    The first build solved one exact anchor per lambda and expanded it with
    `generate_shape_mutants`, which preserves the team-stack profile EXACTLY.
    Every lambda's argmax picks the same team -- lambda trades mean against
    variance and has no reason to switch -- so 100% of 733 generated lineups on
    the 08/18 slate were LAD stacks: 1 distinct team out of 18. The selector
    had to source all of its diversity from the external pool, and generation
    added depth only where the pool was already deep.

    Nothing else in this file would have caught it. Every lineup was legal,
    near-optimal and correctly scored -- it was a PORTFOLIO-level failure,
    invisible to per-lineup assertions.
    """
    df = _players_df()
    var, cov = _sigma_inputs(df)

    specs, sim, cmap = _contest_env(df)
    lus, _lams, _d = frontier_lineups(df, var, cov, _zero_sigma(df), specs, sim, cmap,
                                      n_lambdas=4, sample_n=4_000,
                                      target_lineups=1, min_per_team=3,
                                      n_anchors=0, n_generations=0, **SOLVE_KW)
    assert lus, "sweep produced nothing"

    meta = df.set_index("player_id")

    def primary(pids):
        counts: dict = {}
        for p in pids:
            if meta.loc[p, "position"] != "P":
                t = meta.loc[p, "team"]
                counts[t] = counts.get(t, 0) + 1
        return max(counts, key=counts.get) if counts else None

    teams = [primary(l.player_ids) for l in lus]
    distinct = set(teams)
    n_batter_teams = df[df.position != "P"].team.nunique()
    assert len(distinct) >= min(3, n_batter_teams), (
        f"generated lineups span only {len(distinct)} stack team(s) "
        f"({distinct}) of {n_batter_teams} -- the single-team collapse is back"
    )
    top_share = max(teams.count(t) for t in distinct) / len(teams)
    assert top_share <= 0.6, f"one team holds {top_share:.0%} of generated lineups"


# ------------------------------------------------------- player universe ----

def _staff_df():
    """A slate carrying whole pitching staffs, as players_df really does.

    `build_external_players_df` keeps every pitcher known to either source, so
    a real slate arrives with ~7 pitchers per team and only one of them
    starting (482 across 18 teams on 08/18).
    """
    rows, pid = [], 0
    for team, opp in TEAMS:
        for k in range(6):                      # 1 starter + 5 relievers
            pid += 1
            rows.append({
                "player_id": pid, "name": f"sp{pid}", "position": "P",
                "eligible_positions": ["P"], "team": team, "opponent": opp,
                "game": f"{team}@{opp}", "salary": 9000 if k == 0 else 4000,
                # The reliever that matters: low mean, HIGH variance, cheap --
                # exactly what a variance-maximising objective reaches for.
                "mean": 16.0 if k == 0 else 1.5, "std_dev": 4.0 if k == 0 else 9.0,
                "slot": 10, "ownership": 10.0,
            })
        for slot, pos in enumerate(("C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "C"), 1):
            pid += 1
            rows.append({
                "player_id": pid, "name": f"b{pid}", "position": pos,
                "eligible_positions": [pos], "team": team, "opponent": opp,
                "game": f"{team}@{opp}", "salary": 3500,
                "mean": 9.0, "std_dev": 5.0, "slot": slot, "ownership": 10.0,
            })
    return pd.DataFrame(rows)


def test_only_the_top_projected_pitcher_per_team_survives():
    df = _staff_df()
    kept, diag = restrict_to_playable(df, pool_pids=set())

    assert diag["n_pitchers_kept"] == len(TEAMS), "exactly one starter per team"
    assert diag["n_pitchers_before"] == 6 * len(TEAMS)
    per_team = kept[kept.position == "P"].groupby("team").size()
    assert set(per_team.unique()) == {1}
    for team, grp in df[df.position == "P"].groupby("team"):
        top = int(grp.loc[grp["mean"].idxmax(), "player_id"])
        assert top in set(kept.player_id), f"{team}'s starter must survive"


def test_high_variance_relievers_are_not_readmitted():
    """The regression that matters.

    A projection- or variance-threshold filter keeps these -- they sit at 1.5
    FPTS with a std_dev ABOVE every starter's and cost half as much, which is
    precisely what large lambda buys. A frontier lineup rostering one is
    unplayable and nothing downstream would catch it.
    """
    df = _staff_df()
    kept, _ = restrict_to_playable(df, pool_pids=set())
    relievers = df[(df.position == "P") & (df["mean"] < 5)]
    assert not (set(relievers.player_id) & set(kept.player_id))


def test_pool_membership_never_readmits_a_pitcher():
    """pool_pids widens the HITTER side only."""
    df = _staff_df()
    reliever = int(df[(df.position == "P") & (df["mean"] < 5)].iloc[0].player_id)
    kept, _ = restrict_to_playable(df, pool_pids={reliever})
    assert reliever not in set(kept.player_id)


def test_batters_outside_the_batting_order_need_pool_membership():
    df = _staff_df()
    benched = int(df[df.position == "C"].iloc[0].player_id)
    df.loc[df.player_id == benched, "slot"] = 0

    without, _ = restrict_to_playable(df, pool_pids=set())
    assert benched not in set(without.player_id)

    with_pool, _ = restrict_to_playable(df, pool_pids={benched})
    assert benched in set(with_pool.player_id), "a real SaberSim pick is kept"


def test_line4_picks_a_higher_lambda_for_the_harder_bar():
    """Algorithm 4 line 4, and the property that makes it worth restoring.

    A contest whose paying bar sits well above our reachable scores can only be
    won by variance, so line 4 should push lambda UP for it; one whose bar is
    reachable on projection alone should not. That contest-awareness is exactly
    what the invented "raise lambda until N players move" rule could not
    express -- it had no reference to the payout structure or the field at all.
    """
    from src.optimization.mrp.frontier_qp import lambda_search_grid, line4_lambda_star
    from src.optimization.mrp.lineup_variance import FrontierScorer
    from src.optimization.candidate_generator import CandidateGenerator

    df = _players_df()
    var, cov = _sigma_inputs(df)
    specs, sim, cmap = _contest_env(df)
    scorer = FrontierScorer(df, var, cov)
    cg = CandidateGenerator(df, df["ownership"].to_numpy(dtype=float), rng_seed=3)
    sampled = cg.generate(n_candidates=3_000)
    rows = np.array([l.player_ids for l in sampled], dtype=np.int64)

    grid = lambda_search_grid(scorer, rows, _zero_sigma(df), n_lambdas=8)
    assert len(grid) == 8 and all(b > a for a, b in zip(grid, grid[1:]))

    lam_star, curves = line4_lambda_star(scorer, rows, _zero_sigma(df), grid,
                                         specs, sim, cmap)
    assert set(lam_star) == {"small", "large"}
    assert all(len(c) == len(grid) for c in curves.values())
    assert lam_star["large"] >= lam_star["small"], (
        f"the harder bar should want at least as much variance: "
        f"small={lam_star['small']:.4g} large={lam_star['large']:.4g}"
    )


def test_per_team_slots_are_spent_on_distinct_shapes():
    """The per-team cap is a DIVERSITY BUDGET, not just an anti-monopoly cap.

    `generate_shape_mutants` differs from its parent by ~1 player, so without
    this the budget fills with 1-swap siblings: measured on the 08/24 slate,
    1,091 of 1,920 generated lineups shared a 9-player core with another, so a
    40-slot team was really offering ~25 distinct shapes. Skipping a sibling
    during selection does not shrink the pool -- the slot refills with the next
    best DISTINCT lineup -- which is what separates this from culling
    afterwards, which removes options without adding any.
    """
    df = _players_df()
    var, cov = _sigma_inputs(df)
    specs, sim, cmap = _contest_env(df)

    lus, _lams, _d = frontier_lineups(df, var, cov, _zero_sigma(df), specs, sim, cmap,
                                      n_lambdas=4, sample_n=6_000,
                                      target_lineups=1, min_per_team=8,
                                      n_anchors=0, n_generations=1,
                                      mutants_per_parent=3, **SOLVE_KW)
    assert lus, "sweep produced nothing"

    # No two lineups kept for the same lambda may share a 9-player core.
    from src.optimization.mrp.frontier_qp import _core_keys
    counts: dict = {}
    for l in lus:
        for c in _core_keys(l.player_ids):
            counts[c] = counts.get(c, 0) + 1
    sibling_cores = sum(1 for v in counts.values() if v > 1)
    # Mutation siblings can still cross lambda boundaries (the core set is
    # per-lambda so a later lambda is not starved by an earlier one's picks),
    # so this is a strong bound rather than zero.
    assert sibling_cores <= 0.15 * len(lus), (
        f"{sibling_cores} shared 9-player cores across {len(lus)} lineups -- "
        "the per-team budget is filling with 1-swap siblings again"
    )


def test_core_keys_detects_exactly_the_nine_of_ten_relation():
    from src.optimization.mrp.frontier_qp import _core_keys
    a = list(range(10))
    one_swap = list(range(9)) + [99]        # 9/10 overlap
    two_swap = list(range(8)) + [98, 99]    # 8/10 overlap
    ka = set(_core_keys(a))
    assert ka & set(_core_keys(one_swap))
    assert not (ka & set(_core_keys(two_swap)))
    assert len(_core_keys(a)) == 10
