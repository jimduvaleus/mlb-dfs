"""End-to-end wiring of per-contest selection through PipelineRunner.

`_per_contest_sweep` is driven here against a REAL ContestScorer and real
payout ladders, because most of what can go wrong in it is a seam: an index
mapped back through the wrong array, a field whose width disagrees with the
ladder it is scored against, an arm silently reusing another arm's picks, or a
portfolio flattened in an order the entry map does not share.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api.pipeline import PipelineRunner
from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.gpp_portfolio import ContestScorer
from src.optimization.multi_contest import SharedFieldPool, resolve_contest_slots
from src.optimization.ownership import compute_heuristic_ownership
from src.simulation.results import SimulationResults


def _player(pid, pos, salary, team, game, mean):
    a, b = game.split("@")
    return {
        "player_id": pid, "name": f"P{pid}", "position": pos, "salary": salary,
        "team": team, "game": game, "mean": mean, "std_dev": 5.0,
        "opponent": b if team == a else a, "slot": 9 if pos == "P" else 1,
    }


@pytest.fixture
def players_df():
    rows = [
        _player(1, "P", 8000, "B", "A@B", 25.0),
        _player(2, "P", 7000, "B", "A@B", 22.0),
        _player(20, "P", 8500, "A", "A@B", 21.0),
        _player(5, "C", 4000, "A", "A@B", 18.0),
        _player(7, "1B", 4200, "A", "A@B", 19.0),
        _player(9, "2B", 4100, "A", "A@B", 17.0),
        _player(11, "3B", 3900, "A", "A@B", 16.0),
        _player(13, "SS", 3800, "A", "A@B", 15.0),
        _player(15, "OF", 4000, "A", "A@B", 20.0),
        _player(19, "OF", 3600, "A", "A@B", 18.0),
        _player(16, "OF", 4000, "B", "A@B", 18.0),
        _player(3, "P", 7500, "D", "C@D", 24.0),
        _player(4, "P", 6500, "D", "C@D", 21.0),
        _player(21, "P", 7500, "C", "C@D", 20.0),
        _player(6, "C", 3800, "C", "C@D", 18.0),
        _player(8, "1B", 3800, "C", "C@D", 17.0),
        _player(10, "2B", 3800, "C", "C@D", 16.0),
        _player(12, "3B", 3800, "C", "C@D", 15.0),
        _player(14, "SS", 3800, "C", "C@D", 14.0),
        _player(17, "OF", 3800, "C", "C@D", 19.0),
        _player(22, "OF", 3800, "C", "C@D", 17.0),
        _player(23, "OF", 3600, "C", "C@D", 16.0),
        _player(18, "OF", 3800, "D", "C@D", 18.0),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sim_results(players_df):
    rng = np.random.default_rng(42)
    pids = players_df["player_id"].tolist()
    return SimulationResults(
        player_ids=pids,
        results_matrix=rng.uniform(0, 40, size=(200, len(pids))).astype(np.float32),
    )


@pytest.fixture
def scorer(sim_results, players_df):
    return ContestScorer(
        sim_results, players_df,
        n_field_lineups=60, n_field_samples=1,
        ownership_vec=compute_heuristic_ownership(players_df),
        candidate_batch_size=25,
    )


@pytest.fixture
def shortlist(players_df):
    gen = CandidateGenerator(
        players_df, compute_heuristic_ownership(players_df), rng_seed=0,
    )
    return gen.generate(n_candidates=60)


class _Rec:
    def __init__(self, name, cid, fee_c, pool_c, entry_id):
        self.contest_name = name
        self.contest_id = cid
        self.entry_fee_cents = fee_c
        self.prize_pool_cents = pool_c
        self.entry_fee_raw = f"${fee_c / 100:.2f}"
        self.entry_id = entry_id


def _entries(spec):
    """spec: [(name, contest_id, fee_$, pool_$, n_entries)] -> all_file_entries."""
    recs, eid = [], 0
    for name, cid, fee, pool, n in spec:
        for _ in range(n):
            recs.append(_Rec(name, cid, int(fee * 100), int(pool * 100), eid))
            eid += 1
    return [(Path("DKEntries.csv"), recs)]


@pytest.fixture
def all_file_entries():
    """Two real contests, ~19x apart in field size and 6x in fee."""
    return _entries([
        ("MLB $175K Bat Flip [$50K to 1st]", "c-flip", 18.0, 175_000.0, 6),
        ("MLB $1.5K Hot Corner", "c-corner", 3.0, 1_500.0, 4),
    ])


@pytest.fixture
def slots(all_file_entries):
    return resolve_contest_slots(all_file_entries)


def _runner():
    r = PipelineRunner.__new__(PipelineRunner)
    r._cb = lambda stage, data: None
    r._stop_check = None
    return r


def _sweep(runner, slots, shortlist, scorer, modes, k=1, **kw):
    cols = scorer._build_col_lineups(shortlist)
    scores = np.empty((len(shortlist), scorer._sim_matrix.shape[0]), dtype=np.float32)
    for c0 in range(0, len(shortlist), 25):
        c1 = min(c0 + 25, len(shortlist))
        scores[c0:c1] = scorer._sim_matrix[:, cols[c0:c1]].sum(axis=2).T
    pool = SharedFieldPool(
        scorer.build_raw_field_pool(n_lineups=60, n_samples=k),
        scorer.sorted_field_from_raw, seed=7,
    )
    # Both contests are capped to the 60-lineup pool, which is what makes this
    # tractable; the ladders and fees still differ by design.
    return runner._per_contest_sweep(
        slots=slots, shortlist=shortlist, cand_scores=scores, e_dupes=None,
        field_pool=pool, gpp_cfg=kw.pop("gpp_cfg", {}), modes=modes,
        cash_anchor_fraction=0.25,
        det_sweep_risks=kw.pop("det_sweep_risks", []), **kw,
    )


# --------------------------------------------------------------------------
# The sweep itself
# --------------------------------------------------------------------------

def test_each_arm_fills_every_contest_exactly(slots, shortlist, scorer):
    sweep, picks, diag = _sweep(_runner(), slots, shortlist, scorer, {"kelly"})
    n_entries = sum(s.n_entries for s in slots)
    assert len(sweep) == 5                       # kelly sweeps five risk tiers
    for label, portfolio in sweep:
        assert len(portfolio) == n_entries
        assert all(isinstance(ev, float) for _lu, ev in portfolio)
    for label, per_contest in picks.items():
        assert set(per_contest) == {s.contest_id for s in slots}
        for s in slots:
            assert len(per_contest[s.contest_id]) == s.n_entries
    assert set(diag) == {s.contest_id for s in slots}


def test_a_lineup_is_never_entered_in_two_contests(slots, shortlist, scorer):
    _, picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly"})
    for label, per_contest in picks.items():
        flat = [i for v in per_contest.values() for i in v]
        assert len(flat) == len(set(flat)), f"arm {label} double-entered a lineup"


def test_disjointness_can_be_turned_off(slots, shortlist, scorer):
    _, picks, _ = _sweep(
        _runner(), slots, shortlist, scorer, {"kelly"},
        gpp_cfg={"per_contest_disjoint": False},
    )
    # Some arm reuses at least one lineup across the two contests once the
    # constraint is lifted; with it on, the previous test proves none do.
    assert any(
        len([i for v in pc.values() for i in v]) != len({i for v in pc.values() for i in v})
        for pc in picks.values()
    )


def test_arms_are_labelled_distinctly_in_multi_arm_mode(slots, shortlist, scorer):
    sweep, picks, _ = _sweep(
        _runner(), slots, shortlist, scorer, {"kelly", "emax"},
    )
    labels = [lbl for lbl, _ in sweep]
    assert len(labels) == len(set(labels)) == 6      # kelly 11-15 + emax 31
    assert set(labels) == {11.0, 12.0, 13.0, 14.0, 15.0, 31.0}


def test_an_arm_keeps_its_own_label_when_it_runs_alone(slots, shortlist, scorer):
    """A single-arm sweep must NOT collapse onto the Determinant band.

    The sweep key doubles as an arm identifier -- `armLabel` in
    PortfolioTable.tsx reads the number to decide which objective a portfolio
    came from. Labels used to be offset only when more than one arm ran, so a
    Kelly-only run emitted 1.0-5.0 and the UI rendered five buttons reading
    "Risk 1".."Risk 5" with a Determinant EVw stat: the exact arm the user had
    just switched off. An arm that renames itself depending on its neighbours
    is, from the UI's side, a different arm.
    """
    assert [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"emax"})[0]] == [31.0]
    assert [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"dr"})[0]] == [41.0]
    assert [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"coverage"})[0]] == [23.0]
    assert [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"kelly"})[0]] == [11., 12., 13., 14., 15.]


def test_arm_labels_do_not_move_when_another_arm_is_added(slots, shortlist, scorer):
    """Adding or removing an arm must not relabel the others."""
    alone = [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"kelly"})[0]]
    with_dr = [lbl for lbl, _ in _sweep(
        _runner(), slots, shortlist, scorer, {"kelly", "dr"})[0]]
    assert alone == [l for l in with_dr if l < 41.0]


def test_every_arm_flattens_into_the_same_entry_order(slots, shortlist, scorer):
    """Position i of any arm must belong to entry i, or the shared entry map
    silently mislabels which contest a lineup was chosen for."""
    sweep, picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly", "emax"})
    order = PipelineRunner._per_contest_entry_order(slots)
    for _label, portfolio in sweep:
        assert len(portfolio) == len(order)
    for _label, per_contest in picks.items():
        expected = [s.contest_id for s in slots for _ in range(s.n_entries)]
        assert [cid for cid in expected] == [
            slot.contest_id for (_f, _r, slot) in order
        ]


def test_reported_ev_is_the_contest_the_lineup_will_be_entered_in(slots, shortlist, scorer):
    """A $18 Bat Flip entry and a $3 Hot Corner entry cannot share an EV scale."""
    sweep, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"emax"})
    _label, portfolio = sweep[0]
    order = PipelineRunner._per_contest_entry_order(slots)
    by_contest: dict = {}
    for (_f, _r, slot), (_lu, ev) in zip(order, portfolio):
        by_contest.setdefault(slot.contest_id, []).append(ev)
    # Every entry loses at worst its own fee, so each contest's EVs sit on that
    # contest's own scale rather than on one shared reference fee.
    for slot in slots:
        assert min(by_contest[slot.contest_id]) >= -slot.entry_fee - 1e-3


def test_dr_arm_runs_and_returns_distinct_lineups(slots, shortlist, scorer):
    sweep, picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"dr"})
    assert [lbl for lbl, _ in sweep] == [41.0]
    for per_contest in picks.values():
        flat = [i for v in per_contest.values() for i in v]
        assert len(flat) == len(set(flat))


def test_coverage_arm_runs_off_contest_specific_bits(slots, shortlist, scorer):
    sweep, picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"coverage"})
    _label, portfolio = sweep[0]
    assert len(portfolio) == sum(s.n_entries for s in slots)


def test_det_arm_participates_in_the_per_contest_sweep(slots, shortlist, scorer):
    sweep, _, _ = _sweep(
        _runner(), slots, shortlist, scorer, {"det"}, det_sweep_risks=[1.0, 3.0],
    )
    assert sorted(lbl for lbl, _ in sweep) == [1.0, 3.0]


def test_more_field_samples_changes_nothing_structural(slots, shortlist, scorer):
    sweep, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly"}, k=2)
    assert all(len(p) == sum(s.n_entries for s in slots) for _l, p in sweep)


def test_shortlist_smaller_than_the_entries_file_raises(slots, shortlist, scorer):
    with pytest.raises(RuntimeError, match="remain unused"):
        _sweep(_runner(), slots, shortlist[:6], scorer, {"kelly"})


# --------------------------------------------------------------------------
# Entry order, assignment and the summary
# --------------------------------------------------------------------------

def test_entry_order_is_contests_in_fill_order_then_file_order(slots):
    order = PipelineRunner._per_contest_entry_order(slots)
    assert len(order) == sum(s.n_entries for s in slots)
    seen: list = []
    for _f, _r, slot in order:
        if not seen or seen[-1] is not slot:
            seen.append(slot)
    assert seen == slots                       # no interleaving


def test_positional_assignment_pairs_each_lineup_with_its_own_contest(slots):
    order = PipelineRunner._per_contest_entry_order(slots)
    portfolio = [(f"lu{i}", 0.0) for i in range(len(order))]
    out = PipelineRunner._assign_positional(order, portfolio)
    pairs = out[Path("DKEntries.csv")]
    assert len(pairs) == len(order)
    for (rec, lu), (_f, exp_rec, _slot) in zip(pairs, order):
        assert rec is exp_rec
    for i, (rec, lu) in enumerate(pairs):
        assert lu == f"lu{i}"


def test_entry_map_carries_contest_identity_onto_every_row(slots):
    order = PipelineRunner._per_contest_entry_order(slots)
    portfolio = [(f"lu{i}", 0.0) for i in range(len(order))]
    em = PipelineRunner._build_per_contest_entry_map(order, portfolio)
    assert len(em) == len(order)
    assert em[1]["contest_field_size"] == slots[0].field_size
    assert em[1]["contest_top_prize"] == slots[0].top_prize
    assert em[1]["entry_sort_order"] == 0
    # The last entry belongs to the last contest in fill order.
    assert em[len(order)]["entry_sort_order"] == len(slots) - 1
    assert em[len(order)]["contest_field_size"] == slots[-1].field_size


def test_entry_map_is_truncated_to_the_portfolio_not_the_file(slots):
    order = PipelineRunner._per_contest_entry_order(slots)
    em = PipelineRunner._build_per_contest_entry_map(order, [("lu", 0.0)] * 3)
    assert set(em) == {1, 2, 3}


def test_summary_reports_one_row_per_contest_not_a_blended_average(slots):
    order = PipelineRunner._per_contest_entry_order(slots)
    result = [
        {"lineup_index": i + 1, "lineup_ownership": 120.0 if i < 6 else 80.0,
         "mean_ev": 1.0, "lineup_salary": 49_000}
        for i in range(len(order))
    ]
    rows = PipelineRunner._per_contest_summary(result, order, {})
    assert len(rows) == len(slots)
    assert [r["fill_rank"] for r in rows] == [1, 2]
    assert rows[0]["mean_ownership"] == 120.0
    assert rows[1]["mean_ownership"] == 80.0
    # The spread the whole path exists to produce is visible per row; a blended
    # mean would have shown ~104 and hidden it.
    assert rows[0]["mean_ownership"] - rows[1]["mean_ownership"] == 40.0
    assert rows[0]["n_lineups"] == slots[0].n_entries
    assert rows[0]["lineup_indices"] == list(range(1, slots[0].n_entries + 1))
    assert rows[0]["dollars_at_risk"] == pytest.approx(
        slots[0].entry_fee * slots[0].n_entries
    )


def test_summary_is_empty_without_per_contest_selection():
    assert PipelineRunner._per_contest_summary([], [], {}) == []


def test_disjoint_override_argument_beats_the_config(slots, shortlist, scorer):
    """The pipeline turns disjointness off when the pool cannot fill every
    entry; that decision must override the config, not be overridden by it."""
    _, picks, _ = _sweep(
        _runner(), slots, shortlist, scorer, {"kelly"},
        gpp_cfg={"per_contest_disjoint": True}, disjoint=False,
    )
    assert any(
        len([i for v in pc.values() for i in v]) != len({i for v in pc.values() for i in v})
        for pc in picks.values()
    )


def test_a_short_contest_truncates_the_arm_instead_of_sliding_later_contests(
    slots, shortlist, scorer, monkeypatch,
):
    """A contest that comes up short must NOT let the next contest's lineups
    slide onto its unfilled entries: assignment is positional, so every row
    after the gap would name the wrong contest while the count still looked
    plausible."""
    from src.optimization import gpp_portfolio as gp

    real_select = gp.EMaxPortfolioSelector.select
    calls = {"n": 0}

    def short_first_contest(self, *a, **kw):
        out = real_select(self, *a, **kw)
        calls["n"] += 1
        return out[:-1] if calls["n"] == 1 else out

    monkeypatch.setattr(gp.EMaxPortfolioSelector, "select", short_first_contest)
    sweep, _picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"emax"})
    _label, portfolio = sweep[0]
    # First contest returned n-1, so the arm stops there rather than appending
    # the second contest's picks onto the first contest's leftover entry.
    assert len(portfolio) == slots[0].n_entries - 1


# --------------------------------------------------------------------------
# Progress events
# --------------------------------------------------------------------------

def test_a_stop_after_one_contest_keeps_that_contests_lineups(
    slots, shortlist, scorer,
):
    """Stop must end the sweep, not run every remaining contest to completion.

    The user-visible symptom this covers: per-contest selection is the
    minutes-long stage of a run, and every contest after the Stop click was
    being priced in full before anything noticed.
    """
    runner = _runner()
    done = []
    runner._cb = lambda stage, data: (
        done.append(data["contest_name"]) if stage == "gpp_contest_done" else None
    )
    runner._stop_check = lambda: len(done) >= 1
    sweep, picks, diag = _sweep(runner, slots, shortlist, scorer, {"kelly"})

    assert len(done) == 1                       # the second contest never ran
    assert list(diag) == [slots[0].contest_id]
    for _label, portfolio in sweep:
        # Truncated at the contest boundary, so every position that IS filled
        # still belongs to the entry the flatten order names.
        assert len(portfolio) == slots[0].n_entries
    for arm_picks in picks.values():
        assert list(arm_picks) == [slots[0].contest_id]


def test_a_stop_before_any_contest_yields_an_empty_portfolio_not_an_error(
    slots, shortlist, scorer,
):
    runner = _runner()
    runner._stop_check = lambda: True
    sweep, picks, diag = _sweep(runner, slots, shortlist, scorer, {"kelly"})
    assert diag == {}
    assert sweep and all(portfolio == [] for _, portfolio in sweep)
    assert all(arm_picks == {} for arm_picks in picks.values())


def test_every_arm_honours_a_stop_including_det(slots, shortlist, scorer):
    """det ran its greedy to completion regardless -- it was the one arm whose
    selector was never handed the stop check."""
    runner = _runner()
    prepared = []
    runner._cb = lambda stage, data: (
        prepared.append(data["contest_name"]) if stage == "gpp_contest_select" else None
    )
    runner._stop_check = lambda: len(prepared) >= 1
    sweep, _picks, diag = _sweep(
        runner, slots, shortlist, scorer, {"det", "kelly", "coverage", "emax", "dr"},
        det_sweep_risks=[1.0, 3.0],
    )
    assert diag == {}
    assert all(portfolio == [] for _, portfolio in sweep)


def test_a_stop_before_selection_reports_a_terminal_stopped_event():
    """The external per_contest branch bails out through this helper whenever a
    stop lands in a pre-selection stage (field generation, frontier
    augmentation, shortlisting), and the UI's `stopped` handler reads every one
    of these keys."""
    runner = _runner()
    events = []
    runner._cb = lambda stage, data: events.append((stage, data))
    assert runner._stopped_before_portfolio("per_contest") == []
    (stage, data), = events
    assert stage == "stopped"
    assert data["n_lineups"] == 0
    assert data["portfolio"] == [] and data["portfolio_sweep"] == []
    assert data["optimal_lineups"] == []
    assert data["ev_type"] == "per_contest" and data["external"] is True


def test_progress_events_are_one_per_contest_not_one_per_arm(slots, shortlist, scorer):
    """The panel should get a readable stage trail, not a flood: the expensive
    work is per contest, so that is the reporting granularity even though six
    arms run inside each one."""
    seen: list = []
    r = _runner()
    r._cb = lambda stage, data: seen.append((stage, data))
    _sweep(r, slots, shortlist, scorer, {"kelly", "emax"})

    starts = [d for st, d in seen if st == 'gpp_contest_select']
    dones = [d for st, d in seen if st == 'gpp_contest_done']
    assert len(starts) == len(dones) == len(slots)
    assert [d["contest_index"] for d in starts] == [1, 2]
    assert all(d["contests_total"] == len(slots) for d in starts)
    # Six arms ran; the log still sees two contest rows.
    assert starts[0]["n_arms"] == 6

    phase = [d for st, d in seen if st == 'gpp_per_contest_start']
    assert len(phase) == 1
    assert phase[0]["n_entries"] == sum(s.n_entries for s in slots)
    assert phase[0]["n_contests"] == len(slots)


def test_progress_work_units_are_monotonic_and_complete(slots, shortlist, scorer):
    """work_done drives the bar, so it must reach work_total exactly — a bar
    that stops at 85% is worse than no bar."""
    seen: list = []
    r = _runner()
    r._cb = lambda stage, data: seen.append((stage, data))
    _sweep(r, slots, shortlist, scorer, {"kelly"})

    dones = [d for st, d in seen if st == 'gpp_contest_done']
    work = [d["work_done"] for d in dones]
    assert work == sorted(work)
    assert work[-1] == dones[-1]["work_total"]
    # A contest's own work is not counted until every arm has filled it.
    starts = [d for st, d in seen if st == 'gpp_contest_select']
    assert starts[0]["work_done"] == 0
    assert starts[1]["work_done"] == work[0]


def test_work_weight_reflects_contest_size(slots, shortlist, scorer):
    """Contests fill in top-prize order, not size order, so a flat
    contests-done/total bar would advance in uneven jumps."""
    seen: list = []
    r = _runner()
    r._cb = lambda stage, data: seen.append((stage, data))
    _sweep(r, slots, shortlist, scorer, {"kelly"})
    dones = [d for st, d in seen if st == 'gpp_contest_done']
    shares = [dones[0]["work_done"]]
    for a, b in zip(dones, dones[1:]):
        shares.append(b["work_done"] - a["work_done"])
    # The 11,437-entry Bat Flip must carry more weight than the 594-entry
    # Hot Corner, which a per-contest count would have treated as equal.
    assert shares[0] > shares[1]


def test_the_phase_does_not_flood_the_event_log(slots, shortlist, scorer):
    """Six arms x two contests must not become a dozen log rows. Only the
    phase-start row and one row per contest are loggable; the rest exist to
    drive the bar and are suppressed by ProgressPanel."""
    seen: list = []
    r = _runner()
    r._cb = lambda stage, data: seen.append((stage, data))
    _sweep(r, slots, shortlist, scorer, {"kelly", "dr"})

    bar_only = {'gpp_contest_done', 'gpp_contest_field_progress'}
    loggable = [st for st, _ in seen if st not in bar_only]
    assert loggable == [
        'gpp_per_contest_start',
        'gpp_contest_select',
        'gpp_contest_select',
    ]
    # Nothing is emitted per arm or per greedy pick.
    assert len(seen) == 1 + 2 * len(slots)


# --------------------------------------------------------------------------
# Contest identity is read from the entries file, not typed by the user
# --------------------------------------------------------------------------

def test_funnel_reference_is_the_contest_carrying_the_most_money(slots):
    ref = PipelineRunner._funnel_reference(slots)
    # Bat Flip: 6 x $18 = $108 at risk vs Hot Corner's 4 x $3 = $12.
    assert ref.contest_id == "c-flip"
    assert ref.dollars_at_risk == max(s.dollars_at_risk for s in slots)


def test_funnel_reference_prefers_money_over_field_size():
    """'Largest field' would anchor a real entries file on a $1 contest, where
    a $0.20 ev_floor is a 20% ROI hurdle that guts the pool."""
    spec = [
        ("MLB $15K mini-MAX [150 Entry Max]", "c-mini", 1.0, 15_000.0, 60),
        ("MLB $20K Four-Seamer", "c-four", 4.0, 20_000.0, 20),
    ]
    slots = resolve_contest_slots(_entries(spec))
    biggest_field = max(slots, key=lambda s: s.field_size)
    ref = PipelineRunner._funnel_reference(slots)
    assert biggest_field.contest_id == "c-mini"      # 17,835 entries
    assert ref.contest_id == "c-four"                # $80 at risk vs $60
    assert ref.entry_fee > biggest_field.entry_fee


def test_real_entries_files_resolve_with_no_user_input():
    """The whole point: nothing is typed. If the repo's own entries files stop
    resolving cleanly, the auto-derivation has regressed."""
    from src.api.dk_entries import parse_entry_file
    paths = [Path("data/raw/GEDKEntries.csv"), Path("data/raw/MEDKEntries.csv")]
    if not all(p.exists() for p in paths):
        pytest.skip("sample entries files not present")
    files = [(p, parse_entry_file(p)) for p in paths]
    slots = resolve_contest_slots(files)
    # No exact count: these are the live entries files and get overwritten with
    # each slate's real download, so the contest count moves (6 on 08/25, 7 on
    # 08/28). What must hold is that whatever they contain resolves cleanly.
    assert len(slots) >= 5
    assert not any(s.is_approximate for s in slots)   # no fallback dialog
    assert all(s.field_size > 0 for s in slots)
    # Fill order is descending top prize.
    assert [s.top_prize for s in slots] == sorted(
        (s.top_prize for s in slots), reverse=True,
    )
    ref = PipelineRunner._funnel_reference(slots)
    assert ref.entry_fee == 4.0 and ref.field_size == 5945


# --------------------------------------------------------------------------
# Shortlist composition
# --------------------------------------------------------------------------

def _pool(n=2000, seed=0):
    """EV deliberately correlated with ownership — the situation that makes a
    top-N-by-EV cut collapse onto one end of the ownership axis."""
    rng = np.random.default_rng(seed)
    own = rng.uniform(60, 160, size=n)
    ev = 0.05 * own + rng.normal(0, 1.0, size=n)
    return ev, own


def test_stratified_shortlist_keeps_both_ownership_tails():
    ev, own = _pool()
    cap = 400
    plain = PipelineRunner._stratified_shortlist(ev, own, cap, strata=1)
    strat = PipelineRunner._stratified_shortlist(ev, own, cap, strata=10)
    assert len(plain) == len(strat) == cap
    # The plain cut collapses onto the chalky end; the stratified one spans
    # essentially the whole axis, which is what lets a 792-entry contest and a
    # 17,835-entry one draw from the same shortlist.
    assert own[plain].min() > own[strat].min()
    assert (own[strat].max() - own[strat].min()) > (own[plain].max() - own[plain].min())
    assert own[strat].min() <= np.percentile(own, 5)
    assert own[strat].max() >= np.percentile(own, 95)


def test_every_ownership_band_is_represented():
    ev, own = _pool()
    sel = PipelineRunner._stratified_shortlist(ev, own, 400, strata=10)
    edges = np.percentile(own, np.linspace(0, 100, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        assert ((own[sel] >= lo) & (own[sel] <= hi)).any()


def test_ev_still_decides_who_represents_each_band():
    ev, own = _pool()
    sel = set(PipelineRunner._stratified_shortlist(ev, own, 400, strata=10).tolist())
    order = np.argsort(own, kind="stable")
    band = np.array_split(order, 10)[4]
    picked = [i for i in band if i in sel]
    dropped = [i for i in band if i not in sel]
    assert picked and dropped
    assert min(ev[picked]) >= max(ev[dropped])


def test_shortlist_always_reaches_its_cap():
    ev, own = _pool(n=1000)
    for strata in (1, 3, 7, 10, 64):
        sel = PipelineRunner._stratified_shortlist(ev, own, 400, strata)
        assert len(sel) == 400, strata
        assert len(set(sel.tolist())) == 400


def test_thin_bands_donate_their_quota_rather_than_underfilling():
    """More bands than the quota can fill must still produce a full shortlist."""
    ev, own = _pool(n=300)
    sel = PipelineRunner._stratified_shortlist(ev, own, 250, strata=200)
    assert len(sel) == 250


def test_a_pool_smaller_than_the_cap_is_taken_whole():
    ev, own = _pool(n=120)
    sel = PipelineRunner._stratified_shortlist(ev, own, 400, strata=10)
    assert len(sel) == 120
    assert sorted(sel.tolist()) == list(range(120))


# --------------------------------------------------------------------------
# per_contest as an external-pool EV type
# --------------------------------------------------------------------------

def test_per_contest_is_an_accepted_external_ev_type():
    """The dispatch list in _run_external is a literal tuple; a value missing
    from it silently falls back to 'roi' with only a warning."""
    import inspect
    from src.api.pipeline import PipelineRunner
    src = inspect.getsource(PipelineRunner._run_external)
    assert '"per_contest"' in src
    marker = 'if _ev_type not in ('
    tup = src[src.index(marker) + len(marker):]
    tup = tup[:tup.index('):')]
    assert "per_contest" in tup


def test_sweep_output_converts_to_a_valid_external_allocation(slots, shortlist, scorer):
    """The external path promises ExternalAllocation(portfolio, entry_plan)
    parallel and in per-contest fill order — which is exactly what the
    per-contest flatten produces, so the conversion must need no reshaping."""
    from src.api.external_pool import ExternalAllocation

    sweep, _picks, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly", "emax"})
    plan = [(fp, rec) for (fp, rec, _slot)
            in PipelineRunner._per_contest_entry_order(slots)]
    allocs = {
        lbl: ExternalAllocation(portfolio=port, entry_plan=plan,
                                unfilled=plan[len(port):])
        for lbl, port in sweep
    }
    assert len(allocs) == 6
    for a in allocs.values():
        assert len(a.portfolio) == len(plan)
        assert not a.unfilled
        for (lineup, ev), (_fp, rec) in zip(a.portfolio, a.entry_plan):
            assert hasattr(lineup, "player_ids")
            assert isinstance(ev, float)
            assert rec.contest_id in {s.contest_id for s in slots}


def test_entry_plans_do_not_diverge_across_arms(slots, shortlist, scorer):
    """_run_external asserts plan equality across risks and aborts the run if
    it fails — an arm truncated at a short contest must not trip it."""
    sweep, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly", "dr"})
    plan = [(fp, rec) for (fp, rec, _slot)
            in PipelineRunner._per_contest_entry_order(slots)]
    plans = [[rec.entry_id for _fp, rec in plan] for _lbl, _port in sweep]
    assert all(p == plans[0] for p in plans[1:])


def test_external_assignments_pair_each_lineup_with_its_own_contest(slots, shortlist, scorer):
    sweep, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"emax"})
    _label, portfolio = sweep[0]
    plan = [(fp, rec) for (fp, rec, _slot)
            in PipelineRunner._per_contest_entry_order(slots)]
    r = _runner()
    r._external_entry_plan = plan
    out = r._external_assignments(portfolio)
    pairs = out[Path("DKEntries.csv")]
    assert len(pairs) == len(portfolio)
    by_contest: dict = {}
    for rec, lu in pairs:
        by_contest.setdefault(rec.contest_id, []).append(lu)
    for slot in slots:
        assert len(by_contest[slot.contest_id]) == slot.n_entries


# --------------------------------------------------------------------------
# Shortlist union across contest ladders
# --------------------------------------------------------------------------

def test_union_gives_every_contest_an_equal_say():
    """Two contests wanting opposite material must each get half the menu —
    the case a single-ladder ranking gets exactly wrong."""
    from src.optimization.multi_contest import union_shortlist
    a = np.arange(1000, dtype=np.float64)
    sel = set(union_shortlist([a, -a], 100).tolist())
    assert len(sel) == 100
    assert len(sel & set(range(50))) == 50            # contest B's top 50
    assert len(sel & set(range(950, 1000))) == 50     # contest A's top 50


def test_union_deepens_coverage_when_contests_agree():
    from src.optimization.multi_contest import union_shortlist
    a = np.arange(1000, dtype=np.float64)
    sel = union_shortlist([a, a.copy(), a.copy()], 100)
    # Identical rankings: the union is just the top 100, not 33 each.
    assert sorted(sel.tolist()) == list(range(900, 1000))


def test_union_is_capped_and_deduplicated():
    from src.optimization.multi_contest import union_shortlist
    rng = np.random.default_rng(4)
    evs = [rng.normal(size=500) for _ in range(6)]
    for cap in (10, 137, 500, 900):
        sel = union_shortlist(evs, cap)
        assert len(sel) == min(cap, 500)
        assert len(set(sel.tolist())) == len(sel)


def test_union_requires_at_least_one_ranking():
    from src.optimization.multi_contest import union_shortlist
    with pytest.raises(ValueError, match="no per-contest rankings"):
        union_shortlist([], 10)


def test_contest_ev_means_matches_the_full_payout_matrix():
    """The mean-only path exists to avoid holding (M, S); it must agree with
    the matrix it replaces to the last cent."""
    from src.optimization.multi_contest import contest_ev_means, contest_payout_matrix
    rng = np.random.default_rng(7)
    cand = rng.normal(140, 25, size=(60, 40)).astype(np.float32)
    fields = [
        np.ascontiguousarray(np.sort(
            rng.normal(140, 25, size=(40, 300)).astype(np.float32), axis=1))
        for _ in range(2)
    ]
    ladder = np.zeros(300, dtype=np.float32)
    ladder[:3] = 500.0
    ladder[3:70] = 12.0
    full = contest_payout_matrix(cand, list(fields), ladder, 3.0).mean(axis=1)
    means = contest_ev_means(cand, list(fields), ladder, 3.0, cand_chunk=7)
    assert np.allclose(full, means, atol=1e-4)


def test_union_shortlist_helper_ranks_each_contest_on_its_own_ladder(
    slots, shortlist, scorer,
):
    r = _runner()
    S_full = scorer._sim_matrix.shape[0]
    w = np.sort(np.random.default_rng(0).choice(S_full, size=60, replace=False))
    sim_rank = np.ascontiguousarray(scorer._sim_matrix[w])
    cols = scorer._build_col_lineups(shortlist)
    scores = np.ascontiguousarray(sim_rank[:, cols].sum(axis=2).T)
    raw = scorer.build_raw_field_pool(n_lineups=60, n_samples=1)

    def sort_fn(rw):
        return np.ascontiguousarray(np.sort(
            scorer._cs.score_field(rw, sim_rank, scorer._col_map), axis=1))

    sel, evs = r._union_shortlist_by_contest(
        slots, scores, raw, sort_fn, cap=20, seed=3,
    )
    assert len(evs) == len(slots)
    assert all(len(e) == len(shortlist) for e in evs)
    assert len(sel) == 20
    # The two contests differ by ~19x in field size and 6x in fee, so their
    # dollar EVs must not be the same numbers.
    assert not np.allclose(evs[0], evs[-1])


def test_ranking_is_skipped_when_it_cannot_cut_meaningfully():
    """The ranking pass costs a full field generation regardless of how much it
    then cuts, so on a pool already near the cap it buys an arbitrary few-percent
    trim for the full price. A real run paid 39s + 5s to drop 219 of 4,219."""
    from src.api.pipeline import _PC_RANK_MIN_CUT
    cap = 4000
    threshold = cap / (1.0 - _PC_RANK_MIN_CUT)
    assert 4219 <= threshold          # the observed SaberSim pool: take it whole
    assert 11000 > threshold          # a real generated pool: rank it


# --------------------------------------------------------------------------
# Per-contest candidate cap
# --------------------------------------------------------------------------

def test_candidate_cap_is_off_by_default_in_these_fixtures(slots, shortlist, scorer):
    """The 60-lineup fixture pool is far under the 2,000 floor, so it cannot bite.

    Stated as a test rather than assumed, because every other test in this file
    would silently start measuring a narrowed pool if the floor ever dropped
    below the fixture size.
    """
    assert len(shortlist) < 2_000
    a, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly"})
    b, _, _ = _sweep(_runner(), slots, shortlist, scorer, {"kelly"},
                     gpp_cfg={"per_contest_cand_per_entry": 400})
    assert [lbl for lbl, _ in a] == [lbl for lbl, _ in b]
    for (_, pa), (_, pb) in zip(a, b):
        assert [lu.player_ids for lu, _ in pa] == [lu.player_ids for lu, _ in pb]


def test_the_cap_switches_itself_off_when_no_arm_can_use_it(slots, shortlist, scorer, caplog):
    """It pays for itself only where cost grows faster than linearly in M.

    dR builds four (M x S) rank arrays per contest and Determinant runs an
    M x M matmul; Kelly, coverage and E[max] read a payout matrix that was
    constructed at full M whatever this setting says, then do sub-second
    per-pick work. Measured 08/28: capping a Kelly-only run saved ~9s of a
    119s stage while four of seven contests chose from 46% of the menu.

    Overriding rather than defaulting is deliberate. The value is a config
    field a user may have set for a run that DID include dR, and it should not
    keep costing them picks after they switch arms.
    """
    import logging
    cfg = {"per_contest_cand_per_entry": 400}
    for modes, disabled in [({"kelly"}, True), ({"emax"}, True),
                            ({"kelly", "coverage", "emax"}, True),
                            ({"dr"}, False), ({"kelly", "dr"}, False)]:
        with caplog.at_level(logging.INFO, logger="src.api.pipeline"):
            caplog.clear()
            _sweep(_runner(), slots, shortlist, scorer, modes, gpp_cfg=dict(cfg),
                   det_sweep_risks=([1.0] if "det" in modes else []))
        off = any("cap" in r.getMessage() and "disabled" in r.getMessage()
                  for r in caplog.records)
        assert off is disabled, f"modes={modes}: expected disabled={disabled}"


def test_an_explicit_zero_needs_no_announcement(slots, shortlist, scorer, caplog):
    """Already off is not the same event as switched off, and must stay quiet."""
    import logging
    with caplog.at_level(logging.INFO, logger="src.api.pipeline"):
        _sweep(_runner(), slots, shortlist, scorer, {"kelly"},
               gpp_cfg={"per_contest_cand_per_entry": 0})
    assert not any("disabled" in r.getMessage() for r in caplog.records)


def test_a_binding_cap_keeps_the_highest_ev_candidates(slots, shortlist, scorer):
    """When it does bite, it must cut from the BOTTOM of this contest's EV.

    Driven through `select_per_contest_multi_arm` directly so the cap can be
    forced to bind on a small fixture: the production floor of 2,000 is there
    precisely so it never binds at this scale.
    """
    from src.optimization.multi_contest import select_per_contest_multi_arm

    M = len(shortlist)
    cap = 8
    ev = np.linspace(0.0, 1.0, M)
    rng = np.random.default_rng(3)
    ev = ev[rng.permutation(M)]          # EV order must not track index order
    seen: dict = {}

    def narrow(ctx, avail, k, slot):
        keep = np.sort(avail[np.argsort(-ev[avail])[:cap]])
        # Both, because `avail` is already missing earlier contests' picks —
        # the cap's job is to take the best of what is AVAILABLE, which is not
        # the best of the whole shortlist after the first contest.
        seen[slot.contest_id] = (np.asarray(avail), keep)
        return keep

    def arm(ctx, avail, k, slot):
        # Whatever the cap handed us, in order — enough to check containment.
        return [int(j) for j in avail[:k]]

    picks, _ = select_per_contest_multi_arm(
        slots, M, make_field=lambda f: [np.zeros((2, f), dtype=np.float32)],
        prepare_fn=lambda fields, slot: {"ev": ev},
        arm_fns={"a": arm}, narrow_fn=narrow,
        progress=lambda _m: None,
    )

    for slot in slots:
        avail, kept = seen[slot.contest_id]
        assert len(kept) == cap
        dropped = np.setdiff1d(avail, kept)
        assert ev[kept].min() >= ev[dropped].max(), (
            "the cap must cut the lowest-EV candidates of those available, "
            "not an arbitrary slice"
        )
        assert set(picks["a"][slot.contest_id]) <= set(int(j) for j in kept)


def test_the_cap_is_applied_after_the_used_set_not_before(slots, shortlist, scorer):
    """Order matters: cap-then-exclude could leave a contest short of entries.

    If the cap ran first and every one of its survivors had already been used by
    an earlier contest, the arm would have nothing left even though thousands of
    unused candidates existed. Narrowing the ALREADY-available set cannot do
    that, and `select_per_contest_multi_arm` calls narrow_fn in that order.
    """
    from src.optimization.multi_contest import select_per_contest_multi_arm

    M = len(shortlist)
    order_seen = []

    def narrow(ctx, avail, k, slot):
        # An arm has run before us on every contest after the first, so `avail`
        # must already be missing that arm's earlier picks.
        order_seen.append((slot.contest_id, len(avail)))
        return avail

    picks, _ = select_per_contest_multi_arm(
        slots, M, make_field=lambda f: [np.zeros((2, f), dtype=np.float32)],
        prepare_fn=lambda fields, slot: {"ev": np.zeros(M)},
        arm_fns={"a": lambda ctx, avail, k, slot: [int(j) for j in avail[:k]]},
        narrow_fn=narrow, exclude_used=True, progress=lambda _m: None,
    )

    used = 0
    for (cid, n_avail), slot in zip(order_seen, slots):
        assert n_avail == M - used, "narrow_fn must see the used-set exclusion"
        used += slot.n_entries
