"""The line-2 frontier where it meets the rest of the MRP path.

`test_mrp_frontier_qp.py` covers the solver. These cover the wiring, which is
where the two real hazards live:

  * ALIGNMENT. Frontier lineups are appended to the pool AFTER the caller has
    already computed `floor_scores`, and then the floor renumbers everything
    again. Three index spaces (pool, padded basis, candidate axis) have to stay
    consistent or a pick is attributed to the wrong lineup. This is the bug the
    topn precedent documents at pipeline.py:2962.
  * THE FLOOR EXEMPTION. Generated lineups have no SaberSim "99th" column, and
    the independence-assuming fallback proxy is biased against exactly the
    correlated stacks lambda buys. If the exemption regresses, the feature
    quietly cancels itself and every other test here still passes.
"""
import numpy as np
import pytest

from src.optimization.mrp.runner import MRPConfig, allocate_marginal_reward
from tests.test_mrp_runner import _fixture, _group

FRONTIER_CFG = MRPConfig(
    field_pool_size=400, max_sims_per_contest=300, seed=1,
    frontier_enabled=True, frontier_n_lambdas=3, frontier_per_team=2,
    frontier_sample_n=800, frontier_n_anchors=1, frontier_n_generations=1,
    frontier_mutants_per_parent=2, frontier_solver_timeout_s=5.0,
    # The toy roster's salaries top out around $40k for ten players, so the
    # real 47,500 default would make every lineup infeasible and the sampler
    # would return nothing.
    frontier_salary_floor=0.0,
)
OFF_CFG = MRPConfig(field_pool_size=400, max_sims_per_contest=300, seed=1)


def _groups():
    return [_group("c1", "Four-Seamer", 4), _group("c2", "Base Hit", 3)]


def test_disabled_frontier_is_a_no_op():
    df, sim, pool = _fixture()
    n_before = len(pool.lineups)
    alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), OFF_CFG)

    assert diag.frontier == {}, "no frontier diagnostics when disabled"
    assert len(pool.lineups) == n_before, "pool must not be mutated"
    assert len(alloc.portfolio) + len(alloc.unfilled) == 7


def test_frontier_adds_lineups_and_reports_them():
    df, sim, pool = _fixture()
    n_before = len(pool.lineups)
    alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)

    f = diag.frontier
    assert f and "skipped" not in f, f
    assert f["n_generated"] > 0
    assert f["n_kept"] == f["n_generated"] - f["n_dropped_duplicate"]
    assert f["n_real"] == n_before
    assert f["n_cov_pairs"] > 0, "within-unit pairs must be found"
    # lambda=0 is no longer pinned into the grid: line 4 chooses the operating
    # points, and a contest whose bar is reachable on projection simply gets a
    # small lambda* rather than exactly zero.
    assert f["lambda_min"] >= 0.0
    assert f["lambda_max"] >= f["lambda_min"]
    assert f["n_lambda_star"] >= 1
    assert set(f["lambda_star_by_contest"]), "every contest gets its own lambda*"
    assert 0 <= f["n_picked"] <= len(alloc.portfolio)
    assert "line-2 frontier" in diag.summary()
    # Every purchased slot is still accounted for.
    assert len(alloc.portfolio) + len(alloc.unfilled) == 7


def test_frontier_lineups_survive_an_aggressive_floor():
    """The exemption's whole job, isolated.

    The basis covers only the REAL pool, so every frontier lineup is padded
    with NaN -- which `_floor_keep_indices` would otherwise treat as a failure
    (`np.isfinite(basis) & ...`). If the exemption regresses, the frontier is
    culled in full and `n_frontier_exempt` goes to zero.
    """
    df, sim, pool = _fixture()
    basis = np.linspace(100.0, 200.0, len(pool.lineups))

    _alloc, diag = allocate_marginal_reward(
        pool, df, sim, _groups(), FRONTIER_CFG,
        floor_scores=basis, proj_score_floor_percentile=80.0,
    )
    assert diag.frontier["n_kept"] > 0
    assert diag.floor["n_frontier_exempt"] == diag.frontier["n_kept"], (
        "every frontier lineup carries a NaN basis, so all of them must be "
        "exempted rather than culled"
    )
    assert diag.frontier["n_surviving_floor"] == diag.frontier["n_kept"]


def test_only_exact_duplicates_are_dropped_from_the_frontier():
    """The 9/10 near-duplicate cull is deliberately NOT applied here.

    Shape mutants differ from their parent by a single player -- exactly what
    that cull targets, which cost 73% of them (520 -> 140) when tried. dR's
    demotion term already prices near-duplicates, so the cull is redundant as
    well as expensive. Exact duplicates are still errors and still go.
    """
    df, sim, pool = _fixture()
    n_real = len(pool.lineups)

    _alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)
    f = diag.frontier

    assert f["n_real"] == n_real, "no real lineup is ever removed"
    assert f["n_kept"] <= f["n_generated"]
    assert f["n_dropped_duplicate"] == f["n_generated"] - f["n_kept"]


def test_a_frontier_lineup_duplicating_a_pool_lineup_is_dropped():
    """Two identical entries in one contest is an error, not a diversity call."""
    from src.optimization.mrp.runner import _frontier_augment

    df, sim, pool = _fixture()
    # Stand in for the generator: hand the merge a lineup the pool already has.
    import src.optimization.mrp.frontier_qp as fq
    dupe = pool.lineups[0]
    real_fn = fq.frontier_lineups
    try:
        fq.frontier_lineups = lambda *a, **k: ([dupe], [0.0], {})
        rng = np.random.default_rng(0)
        sim_matrix = sim.results_matrix.astype(np.float32)
        from src.optimization.contest import ContestSimulator
        cs = ContestSimulator()
        own = df["ownership"].to_numpy(dtype=float)
        fp = cs.score_field(cs.generate_field(df, own, n_lineups=200, rng_seed=1),
                            sim_matrix, {int(p): i for i, p in enumerate(sim.player_ids)})
        newpool, n_frontier, fdiag, _fs = _frontier_augment(
            pool, df, sim, sim_matrix, fp, _groups(), FRONTIER_CFG, None, rng,
        )
    finally:
        fq.frontier_lineups = real_fn

    assert n_frontier == 0, "an exact duplicate of a pool lineup must not be added"
    assert fdiag["n_dropped_duplicate"] == 1
    assert len(newpool.lineups) == len(pool.lineups)


def test_picked_frontier_lineups_are_real_portfolio_entries():
    """`n_picked` must count actual portfolio members, not candidate indices."""
    df, sim, pool = _fixture()
    alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)

    n_picked = diag.frontier["n_picked"]
    assert 0 <= n_picked <= len(alloc.portfolio)
    for lu, delta in alloc.portfolio:
        assert len(set(lu.player_ids)) == 10
        assert np.isfinite(delta)


def test_frontier_fails_soft_without_eligible_positions():
    """A generator that cannot run must not take the run down with it."""
    df, sim, pool = _fixture()
    df = df.drop(columns=["eligible_positions"])

    alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)

    assert diag.frontier.get("skipped") == "players_df has no eligible_positions"
    assert "SKIPPED" in diag.summary()
    assert len(alloc.portfolio) + len(alloc.unfilled) == 7, "run still completes"


def test_frontier_does_not_change_the_number_of_purchased_slots():
    df, sim, pool = _fixture()
    groups = _groups()
    n_slots = sum(len(g.entries) for g in groups)

    off, _ = allocate_marginal_reward(pool, df, sim, groups, OFF_CFG)
    on, _ = allocate_marginal_reward(pool, df, sim, groups, FRONTIER_CFG)

    assert len(off.portfolio) + len(off.unfilled) == n_slots
    assert len(on.portfolio) + len(on.unfilled) == n_slots


# ----------------------------------------------------- progress + provenance --

def test_frontier_emits_the_events_the_ui_bar_needs():
    """The Run Progress bar reads these three stages; a rename breaks it
    silently (an SSE consumer just stops matching), so pin the contract."""
    df, sim, pool = _fixture()
    seen: list[dict] = []

    allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG,
                             progress_cb=seen.append)

    stages = [e.get("stage") for e in seen]
    assert "mrp_frontier_start" in stages
    assert "mrp_frontier_done" in stages
    assert "mrp_frontier" in stages, "per-lambda progress drives the live bar"

    start = next(e for e in seen if e["stage"] == "mrp_frontier_start")
    # `n_lambda_search` is the grid line 4 CHOOSES FROM. It is deliberately not
    # called n_lambdas: generation happens at the distinct lambda* only, and
    # labelling the search size as the sweep size told the user "16 λ" while
    # the bar counted to 5.
    assert {"n_lambda_search", "per_team", "n_sample", "n_pairs"} <= set(start)
    assert "n_lambdas" not in start, "the ambiguous name must not come back"

    prog = [e for e in seen if e["stage"] == "mrp_frontier"]
    assert all({"done", "total", "n_lineups"} <= set(e) for e in prog)
    # Monotonic on both axes, or the bar and the ETA both go backwards.
    assert [e["done"] for e in prog] == sorted(e["done"] for e in prog)
    assert [e["n_lineups"] for e in prog] == sorted(e["n_lineups"] for e in prog)
    assert prog[-1]["done"] <= prog[-1]["total"]

    # ONE denominator for the whole phase. The bar divides by `total`; when it
    # fell back to the search-grid size before the first progress event, the
    # readout jumped from "0 / 16" to "1 / 5" mid-run.
    assert len({e["total"] for e in prog}) == 1, "the bar's denominator must not change"
    # And the count is published before any generation is reported, so the bar
    # has a denominator for its whole life rather than only after the first
    # operating point has finished.
    assert prog[0]["done"] == 0


def test_from_generated_is_parallel_to_the_portfolio():
    """The Portfolio tab's GEN badge indexes this list positionally, so a
    length mismatch would mislabel real SaberSim lineups as generated."""
    df, sim, pool = _fixture()
    alloc, diag = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)

    assert len(alloc.from_generated) == len(alloc.portfolio)
    assert all(isinstance(f, bool) for f in alloc.from_generated)
    assert sum(alloc.from_generated) == diag.frontier["n_picked"]


def test_from_generated_is_empty_when_nothing_was_generated():
    """Every other allocator leaves it empty; the UI must then show no badge
    rather than a meaningless '0 of N' counter."""
    df, sim, pool = _fixture()
    alloc, _ = allocate_marginal_reward(pool, df, sim, _groups(), OFF_CFG)
    assert not any(alloc.from_generated)


def test_sigma_dG_is_blended_over_every_contest():
    """sigma_dG used to come from the LARGEST contest alone, which is arbitrary
    -- "biggest field" is not "where our money is", and on a slate whose
    largest contest holds two entries it tuned generation to a contest we
    barely play. dR allocates across all of them, so generation sees all of
    them, weighted by purchased entries."""
    df, sim, pool = _fixture()
    groups = _groups()

    _alloc, diag = allocate_marginal_reward(pool, df, sim, groups, FRONTIER_CFG)

    f = diag.frontier
    assert f["sigma_dG_contests"] == len(groups), (
        "every contest with a paying payout table must enter the blend"
    )
    # Worst per-contest agreement with the blend: a real correlation, and the
    # signal that a future slate's contests genuinely disagree.
    assert -1.0 <= f["sigma_dG_min_corr"] <= 1.0


def test_more_contests_widen_the_blend():
    """The count tracks the contests actually supplied, not a constant."""
    df, sim, pool = _fixture()
    one = [_group("c1", "Four-Seamer", 4)]
    _a, d1 = allocate_marginal_reward(pool, df, sim, one, FRONTIER_CFG)
    _b, d2 = allocate_marginal_reward(pool, df, sim, _groups(), FRONTIER_CFG)
    assert d1.frontier["sigma_dG_contests"] == 1
    assert d2.frontier["sigma_dG_contests"] == 2
