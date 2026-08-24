"""Progress-event key sync across the three layers an MRP event crosses.

An event is emitted in src/optimization/mrp/runner.py, rebuilt field by field
by the `_mrp_progress` relay in src/api/pipeline.py, and read by
ui/src/components/ProgressPanel.tsx through the interfaces in ui/src/types.ts.
The relay does not forward `info` -- it constructs a NEW dict, naming every
key -- so renaming an emitter key leaves `info.get("old_name")` returning None
and the field reaches the browser as null.

That is silent on the Python side and fatal on the JS side: it is exactly how
`mrp_frontier_start` came to send n_lambdas/n_per_lambda while the emitter had
moved to n_lambda_search/per_team/n_sample, and the panel's
`n_sample.toLocaleString()` threw during render and blanked the whole app
mid-run. Same class of silent-drop failure as the config sync in
test_mrp_config_sync.py, and it gets a test for the same reason.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "optimization" / "mrp" / "runner.py"
PIPELINE = ROOT / "src" / "api" / "pipeline.py"
TYPES_TS = ROOT / "ui" / "src" / "types.ts"

# Stages whose emitted payload is a dict literal, so the key set is knowable
# from the source. The rest are emitted as `{"stage": ..., **some_dict}` and
# their keys only exist at runtime -- see test_spread_emitters_are_accounted_for.
LITERAL_STAGES = {"mrp_frontier_start", "mrp_frontier", "mrp_build", "mrp_pick"}


def _emitted_payloads() -> tuple[dict[str, set[str]], set[str]]:
    """stage -> explicitly-named keys, plus the stages emitted via `**spread`."""
    tree = ast.parse(RUNNER.read_text())
    literal: dict[str, set[str]] = {}
    spread: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "progress_cb"
                and node.args
                and isinstance(node.args[0], ast.Dict)):
            continue
        d = node.args[0]
        keys = {k.value for k in d.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        stage = next((v.value for k, v in zip(d.keys, d.values)
                      if isinstance(k, ast.Constant) and k.value == "stage"
                      and isinstance(v, ast.Constant)), None)
        if stage is None:
            continue
        if any(k is None for k in d.keys):      # `**diag` -- not statically knowable
            spread.add(stage)
        else:
            literal[stage] = keys - {"stage"}
    return literal, spread


def _relay_branches() -> dict[str, dict[str, set[str] | str | None]]:
    """Incoming stage -> {reads: keys pulled off `info`, writes: keys sent to
    the UI, out_stage: the stage name the UI receives}."""
    tree = ast.parse(PIPELINE.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_mrp_progress"), None)
    assert fn is not None, "_mrp_progress relay not found in src/api/pipeline.py"

    branches: dict[str, dict] = {}

    def visit_chain(stmts):
        for stmt in stmts:
            if not isinstance(stmt, ast.If):
                continue
            test = stmt.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name) and test.left.id == "stage"
                    and isinstance(test.comparators[0], ast.Constant)):
                branches[test.comparators[0].value] = _branch_keys(stmt.body)
            visit_chain(stmt.orelse)

    visit_chain(fn.body)
    return branches


def _branch_keys(body) -> dict:
    reads: set[str] = set()
    writes: set[str] = set()
    out_stage = None
    for node in body:
        for sub in ast.walk(node):
            # info["k"]
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name) and sub.value.id == "info"
                    and isinstance(sub.slice, ast.Constant)):
                reads.add(sub.slice.value)
            # info.get("k") / info.get("k", default)
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
                    and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "info"
                    and sub.args and isinstance(sub.args[0], ast.Constant)):
                reads.add(sub.args[0].value)
            # {k: info.get(k) for k in ("a", "b", ...)}
            if isinstance(sub, ast.DictComp):
                for gen in sub.generators:
                    if isinstance(gen.iter, ast.Tuple):
                        names = {e.value for e in gen.iter.elts
                                 if isinstance(e, ast.Constant)}
                        reads |= names
                        writes |= names
            # self._cb("out_stage", {...})
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute) and sub.func.attr == "_cb"
                    and sub.args and isinstance(sub.args[0], ast.Constant)):
                if out_stage is None:
                    out_stage = sub.args[0].value
                if len(sub.args) > 1 and isinstance(sub.args[1], ast.Dict):
                    writes |= {k.value for k in sub.args[1].keys
                               if isinstance(k, ast.Constant)}
    return {"reads": reads, "writes": writes, "out_stage": out_stage}


def _ts_interface_fields(name: str) -> set[str]:
    ts = TYPES_TS.read_text()
    import re
    block = re.search(rf"export interface {name} extends SSEEvent \{{(.*?)\n\}}", ts, re.S)
    assert block, f"{name} missing from ui/src/types.ts"
    body = re.sub(r"/\*.*?\*/", "", block.group(1), flags=re.S)   # strip doc comments
    return set(re.findall(r"^\s*(\w+)\??:", body, re.M)) - {"stage"}


EMITTED, SPREAD = _emitted_payloads()
RELAY = _relay_branches()


def test_the_expected_stages_are_actually_reachable_statically():
    """Guards the guard: if an emit site changes shape, the stage silently
    drops out of LITERAL_STAGES and the sync below would check nothing."""
    assert LITERAL_STAGES <= set(EMITTED), (
        f"expected dict-literal emitters not found in runner.py: "
        f"{LITERAL_STAGES - set(EMITTED)}"
    )
    assert LITERAL_STAGES <= set(RELAY), (
        f"stages with no relay branch in pipeline.py: {LITERAL_STAGES - set(RELAY)}"
    )


@pytest.mark.parametrize("stage", sorted(LITERAL_STAGES))
def test_relay_only_reads_keys_the_emitter_actually_sends(stage):
    """The regression: relay asking for a key the emitter no longer sends."""
    reads = RELAY[stage]["reads"]
    stale = reads - EMITTED[stage]
    assert not stale, (
        f"src/api/pipeline.py's `{stage}` relay reads {sorted(stale)}, which "
        f"runner.py does not emit (it sends {sorted(EMITTED[stage])}). These "
        f"reach the UI as null."
    )


@pytest.mark.parametrize("stage", sorted(LITERAL_STAGES))
def test_relay_forwards_everything_the_emitter_sends(stage):
    """The other direction: a new emitter field nobody relayed is dead weight,
    and usually means the UI change that motivated it was left half-wired."""
    dropped = EMITTED[stage] - RELAY[stage]["reads"]
    assert not dropped, (
        f"runner.py emits {sorted(dropped)} on `{stage}` but the relay in "
        f"src/api/pipeline.py never reads them, so the UI cannot see them."
    )


@pytest.mark.parametrize("stage", sorted(LITERAL_STAGES | {"mrp_frontier_done"}))
def test_relay_output_matches_the_typescript_interface(stage):
    """Third layer: what the relay sends vs what ProgressPanel expects."""
    out_stage = RELAY[stage]["out_stage"]
    assert out_stage and out_stage.startswith("mrp_")
    iface = "".join(p.capitalize() for p in out_stage.split("_")) + "Event"
    ts_fields = _ts_interface_fields(iface)
    writes = RELAY[stage]["writes"]
    assert writes == ts_fields, (
        f"`{out_stage}`: relay sends {sorted(writes)}, {iface} in types.ts "
        f"declares {sorted(ts_fields)}; only in relay: "
        f"{sorted(writes - ts_fields)}, only in TS: {sorted(ts_fields - writes)}"
    )


def test_the_frontier_start_regression_stays_closed():
    """The exact rename that blanked the UI: emitter moved to
    n_lambda_search/per_team/n_sample, relay was left on the old names."""
    reads = RELAY["mrp_frontier_start"]["reads"]
    assert {"n_lambda_search", "per_team", "n_sample", "n_pairs"} <= reads
    assert not ({"n_lambdas", "n_per_lambda"} & reads), "old field names are back"


def test_spread_emitters_are_accounted_for():
    """Documents what this test file deliberately cannot cover: stages emitted
    as `{"stage": ..., **some_dict}` have no statically knowable key set."""
    assert SPREAD == {"mrp_frontier_done", "mrp_floor", "mrp_preflight"}, (
        f"the set of spread-emitted MRP stages changed: {sorted(SPREAD)}. If a "
        f"stage became a dict literal, add it to LITERAL_STAGES for full "
        f"coverage; if a new one appeared, confirm the UI guards its fields."
    )
