"""The payout-coverage gate.

`nearest_payout_structure` never returns None: an unregistered contest silently
borrows the closest-size table of ANY registered type. For a per-contest
allocator that misranks candidates inside one contest. For MRP it is worse --
dR is denominated in dollars and the greedy compares marginal dollars ACROSS
contests, so a borrowed curve misallocates entries BETWEEN contests, slate-wide
and invisibly.

The backtest path already treats this as a hard stop (`load_real_contests`
raises SystemExit; PROSPECTIVE_PROTOCOL says not to silence it). These tests
pin the same detector for the live path.
"""
import numpy as np
import pytest

from src.api.external_pool import ContestGroup, _DK_RAKE
from src.optimization.mrp.runner import check_payout_coverage, describe_payout_fallbacks


def _group(name, k=5, fee_cents=400, n_field=5000):
    prize = int(round(n_field * fee_cents * (1.0 - _DK_RAKE)))
    return ContestGroup(
        contest_id=f"c:{name}", contest_name=name, entry_fee_cents=fee_cents,
        prize_pool_cents=prize, single_entry_tag=False,
        entries=[(f"/tmp/{name}.csv", f"e{j}") for j in range(k)],
    )


def test_registered_contests_resolve_exactly():
    groups = [_group("MLB $20K Four-Seamer [20 Entry Max]", n_field=5945, fee_cents=400),
              _group("MLB $15K mini-MAX [150 Entry Max]", n_field=17835, fee_cents=100)]
    rows = check_payout_coverage(groups)
    assert all(r["exact"] for r in rows), [r for r in rows if not r["exact"]]
    assert describe_payout_fallbacks(rows) == "", "no prompt when everything resolves"


def test_unregistered_contest_is_flagged_not_silently_substituted():
    rows = check_payout_coverage([_group("MLB $50K Totally Fictional Slugfest")])
    assert len(rows) == 1
    r = rows[0]
    assert r["exact"] is False, "an unknown contest must not pass as exact"
    # It still resolves to SOMETHING -- that is the danger the gate exists for.
    assert r["table_name"], "nearest_payout_structure always returns a table"
    assert r["table_entries"] > 0


def test_description_names_the_contest_and_what_it_will_borrow():
    rows = check_payout_coverage([
        _group("MLB $20K Four-Seamer [20 Entry Max]", n_field=5945),
        _group("MLB $50K Totally Fictional Slugfest", k=3),
    ])
    desc = describe_payout_fallbacks(rows)
    assert "Totally Fictional Slugfest" in desc
    assert "Four-Seamer" not in desc.split("will use")[0].split("\n")[0], \
        "the summary line should count only the unresolved ones"
    assert "will use" in desc
    assert "1 of 2 contests" in desc
    # The reason must travel with the warning, not live only in a docstring.
    assert "between" in desc.lower() and "dollar" in desc.lower()


def test_reported_field_size_and_fee_come_from_the_contest_not_the_table():
    """The dialog has to show what we ARE entering next to what it will BORROW,
    or the user cannot judge whether the substitution is acceptable."""
    g = _group("MLB $50K Totally Fictional Slugfest", k=7, fee_cents=1500, n_field=3000)
    r = check_payout_coverage([g])[0]
    assert r["k"] == 7
    assert r["entry_fee"] == 15.0
    assert r["implied_field_size"] == pytest.approx(3000, rel=0.02)
    assert r["table_prize_pool"] > 0


def test_coverage_check_has_no_side_effects_on_the_groups():
    g = _group("MLB $50K Totally Fictional Slugfest")
    before = (g.contest_name, len(g.entries), g.prize_pool_cents)
    check_payout_coverage([g])
    assert (g.contest_name, len(g.entries), g.prize_pool_cents) == before


def test_pipeline_aborts_at_the_gate_when_it_cannot_ask():
    """A non-interactive caller must not silently take the risky branch."""
    import inspect

    from src.api import pipeline

    src = inspect.getsource(pipeline.PipelineRunner._run_external)
    branch = src.split('elif _ev_type == "marginal_reward":', 1)[1]
    assert "check_payout_coverage" in branch
    assert "if self._await_confirmation is None:" in branch
    assert "raise RuntimeError" in branch, "must abort, not proceed, with no way to ask"


# ---------------------------------------------------------------------------
# The server-side gate: the pipeline thread blocks on this
# ---------------------------------------------------------------------------

def _reset_gate():
    from src.api import server
    server._confirm_event.clear()
    server._confirm_state["pending"] = None
    server._confirm_answer["proceed"] = False
    server._stop_event.clear()


def test_gate_returns_true_when_the_user_proceeds():
    import threading

    from src.api import server
    _reset_gate()

    result = {}
    t = threading.Thread(
        target=lambda: result.update(
            ok=server._await_confirmation("k", {"a": 1})), daemon=True)
    t.start()
    # Wait for the gate to actually register before answering.
    for _ in range(200):
        if server._confirm_state["pending"] is not None:
            break
        __import__("time").sleep(0.01)
    assert server._confirm_state["pending"] == {"kind": "k", "payload": {"a": 1}}

    server.answer_confirmation({"proceed": True})
    t.join(timeout=5)
    assert not t.is_alive(), "gate did not release -- the run would hang"
    assert result["ok"] is True
    assert server._confirm_state["pending"] is None, "pending must be cleared"


def test_gate_returns_false_when_the_user_declines():
    import threading
    import time

    from src.api import server
    _reset_gate()

    result = {}
    t = threading.Thread(
        target=lambda: result.update(ok=server._await_confirmation("k", {})), daemon=True)
    t.start()
    for _ in range(200):
        if server._confirm_state["pending"] is not None:
            break
        time.sleep(0.01)

    server.answer_confirmation({"proceed": False})
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["ok"] is False


def test_stopping_the_run_releases_a_waiting_gate():
    """Otherwise Stop at a dialog parks the executor thread forever."""
    import threading
    import time

    from src.api import server
    _reset_gate()

    result = {}
    t = threading.Thread(
        target=lambda: result.update(ok=server._await_confirmation("k", {})), daemon=True)
    t.start()
    for _ in range(200):
        if server._confirm_state["pending"] is not None:
            break
        time.sleep(0.01)

    server._stop_event.set()          # what POST /api/run/stop does
    t.join(timeout=5)
    assert not t.is_alive(), "stop did not release the gate"
    assert result["ok"] is False, "a stop must read as a decline, not a proceed"
    _reset_gate()


def test_answering_with_nothing_pending_is_rejected():
    from fastapi import HTTPException

    from src.api import server
    _reset_gate()
    with pytest.raises(HTTPException):
        server.answer_confirmation({"proceed": True})
