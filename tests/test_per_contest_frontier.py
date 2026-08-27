"""The per-contest external path augments the imported pool instead of only reading it.

`external_pool_ev_type: per_contest` used to take `pool.lineups` as the whole
candidate list, so `frontier_enabled: true` sat in config doing nothing and a
real run filled 110 entry slots out of a 4,219-lineup SaberSim export -- a 2.6%
cut, where the generated path cuts 0.4%. The menu was doing most of the
deciding, not the objective.

These tests cover the two seams that reintroduce that failure silently: the
ladder handed to the generator, and a generator that no-ops without saying so.
"""
import logging
from pathlib import Path

import numpy as np
import pytest

from src.api.pipeline import PipelineRunner
from src.optimization.contest import ContestSimulator
from src.optimization.mrp.runner import (
    MRPConfig, frontier_contests_from_groups, frontier_contests_from_slots,
)
from src.optimization.multi_contest import resolve_contest_slots
from tests.test_mrp_runner import _fixture, _group
from tests.test_per_contest_pipeline import _entries, _runner

# Mirrors tests/test_mrp_frontier_integration.py's FRONTIER_CFG: the toy roster
# tops out near $40k for ten players, so the real 47,500 floor would make every
# lineup infeasible and the sampler would return nothing.
FRONTIER_CFG_KWARGS = {
    "max_sims_per_contest": 300,
    "frontier_n_lambdas": 3,
    "frontier_target_lineups": 40,
    "frontier_min_per_team": 2,
    "frontier_sample_n": 800,
    "frontier_n_anchors": 1,
    "frontier_n_generations": 2,
    "frontier_mutants_per_parent": 2,
    "frontier_solver_timeout_s": 5.0,
    "frontier_salary_floor": 0.0,
    "frontier_mutant_workers": 1,
}


@pytest.fixture
def slots():
    """Two contests DK runs at several advertised sizes."""
    return resolve_contest_slots(_entries([
        ("MLB $175K Bat Flip [$50K to 1st]", "c-flip", 18.0, 175_000.0, 6),
        ("MLB $1.5K Hot Corner", "c-corner", 3.0, 1_500.0, 4),
    ]))


def _field(df, sim, n=400, seed=1):
    cs = ContestSimulator()
    own = df["ownership"].to_numpy(dtype=float)
    return cs, cs.generate_field(df, own, n_lineups=n, rng_seed=seed)


def test_slots_hand_the_generator_their_own_resolved_ladder(slots):
    """The size variant is matched on the advertised pool, not back-solved entries.

    `nearest_payout_structure`'s entry-count match is wrong by the rake and
    picked the wrong variant for 5 of 6 contests on a real entries file. A slot
    has already done that resolution correctly, so the frontier must not redo
    it by name -- this asserts the array is carried through by identity of
    value, not merely that some ladder arrived.
    """
    contests = frontier_contests_from_slots(slots)
    assert len(contests) == len(slots)
    for c, sl in zip(contests, slots):
        assert c.payout_arr is not None, "slot ladders must be passed through"
        np.testing.assert_array_equal(c.ladder(int(c.field_size)), sl.payout_arr)
        assert c.n_entries == sl.n_entries
        assert c.field_size == sl.field_size


def test_group_contests_still_resolve_by_name():
    """The MRP path has no pre-resolved ladder, so it must keep the name lookup."""
    contests = frontier_contests_from_groups([_group("c1", "Four-Seamer", 4)])
    assert contests[0].payout_arr is None
    ladder = contests[0].ladder(5_000)
    assert ladder.size and float(ladder.max()) > 0


def test_augmenting_grows_the_pool_and_keeps_every_imported_lineup():
    df, sim, pool = _fixture()
    before = [frozenset(lu.player_ids) for lu in pool.lineups]
    cs, field_raw = _field(df, sim)
    col_map = {int(p): i for i, p in enumerate(sim.player_ids)}
    slots = resolve_contest_slots(_entries([
        ("MLB $175K Bat Flip [$50K to 1st]", "c-flip", 18.0, 175_000.0, 6),
    ]))

    newpool, n_frontier, diag = _runner()._frontier_augment_pool(
        pool, df, sim, slots, dict(FRONTIER_CFG_KWARGS),
        field_raw, cs, col_map, seed=1,
    )

    assert n_frontier > 0, diag
    assert "skipped" not in diag, diag
    assert diag["n_pool_before"] == len(before)
    assert diag["n_pool_after"] == len(newpool.lineups) == len(before) + n_frontier
    # The user's own lineups are never dropped to make room, and they stay at
    # the FRONT: the floor exemption and the frontier-membership mask both key
    # off "the last n_frontier indices".
    assert [frozenset(lu.player_ids) for lu in newpool.lineups[:len(before)]] == before
    assert diag["elapsed_s"] >= 0.0


def test_a_generator_that_adds_nothing_says_so_out_loud(caplog):
    """A silent no-op here is the exact bug this wiring exists to close."""
    df, sim, pool = _fixture()
    df = df.drop(columns=["eligible_positions"], errors="ignore")
    cs, field_raw = _field(df, sim)
    col_map = {int(p): i for i, p in enumerate(sim.player_ids)}
    slots = resolve_contest_slots(_entries([
        ("MLB $1.5K Hot Corner", "c-corner", 3.0, 1_500.0, 4),
    ]))

    with caplog.at_level(logging.WARNING, logger="src.api.pipeline"):
        newpool, n_frontier, diag = _runner()._frontier_augment_pool(
            pool, df, sim, slots, dict(FRONTIER_CFG_KWARGS),
            field_raw, cs, col_map, seed=1,
        )

    assert n_frontier == 0
    assert diag.get("skipped")
    assert len(newpool.lineups) == len(pool.lineups)
    assert any("added nothing" in r.getMessage() for r in caplog.records)


def test_the_frontier_stage_is_announced_exactly_once():
    """One `mrp_frontier_start` per run, and it carries real numbers.

    `_frontier_augment` emits its own start once the covariance pairs are
    counted. A second one from the caller -- added to fill the gap while the
    field is being scored -- put two rows in the event log, the first claiming
    "0 covariance pairs" because they did not exist yet. The duplicate is
    invisible from Python: nothing raises, the run completes, and only the log
    is wrong.
    """
    df, sim, pool = _fixture()
    cs, field_raw = _field(df, sim)
    col_map = {int(p): i for i, p in enumerate(sim.player_ids)}
    slots = resolve_contest_slots(_entries([
        ("MLB $1.5K Hot Corner", "c-corner", 3.0, 1_500.0, 4),
    ]))

    seen = []
    r = _runner()
    r._cb = lambda stage, data: seen.append((stage, data))
    r._frontier_augment_pool(
        pool, df, sim, slots, dict(FRONTIER_CFG_KWARGS),
        field_raw, cs, col_map, seed=1,
    )

    starts = [d for st, d in seen if st == "mrp_frontier_start"]
    assert len(starts) == 1, f"expected one start row, got {len(starts)}"
    assert starts[0]["n_pairs"] > 0, (
        "the start row must be emitted after the covariance pairs are built, "
        "not before -- a 0 there means it fired too early"
    )
    assert len([1 for st, _ in seen if st == "mrp_frontier_done"]) == 1


def test_mrp_config_defaults_cover_every_frontier_knob():
    """`_frontier_augment_pool` reads the `marginal_reward:` block by design.

    `frontier_enabled: true` should mean the same thing whichever of the two
    sibling per-contest shapes is running, so both read one config block. An
    empty dict must therefore reproduce MRPConfig's own defaults rather than
    inventing a second set that drifts.
    """
    ref = MRPConfig()
    df, sim, pool = _fixture()
    cs, field_raw = _field(df, sim, n=50)
    col_map = {int(p): i for i, p in enumerate(sim.player_ids)}
    seen = {}

    import src.optimization.mrp.runner as rn
    real = rn._frontier_augment

    def spy(_pool, _df, _sim, _mat, _fp, _contests, cfg, *a, **k):
        seen.update(vars(cfg))
        return _pool, 0, {"skipped": "spy"}, None

    rn._frontier_augment = spy
    try:
        _runner()._frontier_augment_pool(
            pool, df, sim, [], {}, field_raw, cs, col_map, seed=7,
        )
    finally:
        rn._frontier_augment = real

    for k in (
        "frontier_n_lambdas", "frontier_target_lineups", "frontier_min_per_team",
        "frontier_sample_n", "frontier_n_anchors", "frontier_n_generations",
        "frontier_mutants_per_parent", "frontier_salary_floor",
        "frontier_mutant_workers", "max_sims_per_contest",
    ):
        assert seen[k] == getattr(ref, k), f"{k} drifted from MRPConfig's default"
    assert seen["frontier_enabled"] is True
    assert seen["seed"] == 7


# --------------------------------------------------------------------------
# Frontier attribution (`from_generated`)
# --------------------------------------------------------------------------

def _mini_players_df():
    import pandas as pd
    return pd.DataFrame({
        "player_id": list(range(1, 21)),
        "name": [f"P{i}" for i in range(1, 21)],
        "salary": [4000] * 20, "team": ["A"] * 20, "position": ["OF"] * 20,
        "slot": [1] * 20, "mean": [10.0] * 20,
    })


class _LU:
    def __init__(self, pids):
        self.player_ids = list(pids)


def test_attribution_is_per_lineup_so_every_arm_gets_its_own_answer():
    """The reason this is keyed by lineup and not by entry slot.

    Per-contest selection builds one portfolio per ARM off a shared pool, so
    slot 3 holds a different lineup in each of them. MRP's positional
    `from_generated` list describes one portfolio in one order and cannot
    describe six; keying on the lineup can.
    """
    gen = _LU(range(1, 11))
    imported = _LU(range(11, 21))
    keys = {frozenset(gen.player_ids)}
    df = _mini_players_df()

    arm_a = [(gen, 1.0), (imported, 1.0)]
    arm_b = [(imported, 1.0), (gen, 1.0)]      # same slots, swapped lineups
    rows_a = PipelineRunner._serialize_portfolio(arm_a, df, generated_keys=keys)
    rows_b = PipelineRunner._serialize_portfolio(arm_b, df, generated_keys=keys)

    assert [r["from_generated"] for r in rows_a] == [True, False]
    assert [r["from_generated"] for r in rows_b] == [False, True]


def test_unattributed_portfolios_omit_the_flag_rather_than_claiming_false():
    """Absent means "not tracked here"; False would mean "imported"."""
    df = _mini_players_df()
    rows = PipelineRunner._serialize_portfolio([(_LU(range(1, 11)), 1.0)], df)
    assert "from_generated" not in rows[0]


def test_the_key_set_is_dropped_between_runs():
    """A second run with the generator OFF must not inherit the first verdict.

    The server reuses one PipelineRunner across runs, so stale keys would not
    merely be untidy — they would label imported lineups as generated, which is
    a wrong provenance claim rather than a missing one.
    """
    r = _runner()
    r._external_generated_keys = {frozenset(range(1, 11))}
    r._external_from_generated = [True]
    # The clear at the top of _run_external, reproduced as the contract it is.
    import inspect
    src = inspect.getsource(PipelineRunner._run_external)
    head = src.split("gpp_cfg = cfg.get")[0]
    assert "self._external_generated_keys = None" in head, (
        "_run_external must clear frontier attribution before any branch runs"
    )
    assert "self._external_from_generated = []" in head


def test_the_entry_map_stops_overwriting_a_per_lineup_verdict():
    """Entry meta is merged ONTO serialized rows, so it must not clobber them."""
    from src.api.dk_entries import EntryRecord
    r = _runner()
    r._external_entry_plan = [
        (Path("DKEntries.csv"),
         EntryRecord(entry_id="1", contest_name="MLB $2K Pickoff [Single Entry]",
                     contest_id="c", entry_fee_cents=300, entry_fee_raw="$3"))
    ]
    r._external_from_generated = [True]

    r._external_generated_keys = None
    assert "from_generated" in r._build_external_entry_map()[1]

    r._external_generated_keys = {frozenset(range(1, 11))}
    assert "from_generated" not in r._build_external_entry_map()[1], (
        "with per-lineup attribution available the slot-indexed fallback must "
        "stand down, not overwrite it"
    )
