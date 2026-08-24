"""Every ConfigForm number input must accept its own default value.

An `<input type="number">` with `step` and `min` set only accepts values on the
lattice `min + n*step`. A default off that lattice is rejected by the browser as
a `stepMismatch`, and because the whole form validates before submit, ONE bad
field blocks Save Config for every other field too -- with no error message
pointing at the culprit.

That is exactly what shipped with `frontier_n_per_lambda` (default 200 against
`step=25 min=1`, whose lattice is 1, 26, 51, ...): the user enabled frontier
generation and simply could not save. Nothing errored; the button did nothing.

The check is a text scan rather than a DOM test because the repo has no
frontend test runner, and the failure lives entirely in the JSX attributes.
"""
import re
from pathlib import Path

import pytest

FORM = Path(__file__).resolve().parents[1] / "ui" / "src" / "components" / "ConfigForm.tsx"

# `.*?/>` rather than `[^>]*?/>`: the onChange arrow function contains a `>`,
# so an exclusion class stops at `e =>` and matches nothing useful.
_INPUT_RE = re.compile(r'<input type="number".*?/>', re.S)


def _number_inputs():
    """(name, min, step, default) for each number input that declares a step
    and a `?? default` fallback. Inputs with `step="any"` are unconstrained."""
    out = []
    for block in _INPUT_RE.findall(FORM.read_text()):
        step = re.search(r'step=(?:\{([^}]*)\}|"([^"]*)")', block)
        default = re.search(r'\?\?\s*([0-9.]+)\s*\}', block)
        if not step or not default:
            continue
        raw = (step.group(1) or step.group(2) or "").strip().strip('"')
        if raw in ("any", ""):
            continue
        mn = re.search(r"min=\{([^}]*)\}", block)
        path = re.search(r"draft\.([\w.?]+)", block)
        try:
            out.append((
                path.group(1) if path else block[:60],
                float(mn.group(1)) if mn else 0.0,
                float(raw),
                float(default.group(1)),
            ))
        except (ValueError, AttributeError):
            continue
    return out


def test_the_scan_finds_the_forms_inputs():
    """Guards the regex itself -- a silently-zero scan would pass everything."""
    found = _number_inputs()
    assert len(found) >= 20, f"only matched {len(found)} inputs; regex is stale"


@pytest.mark.parametrize("name,minimum,step,default", _number_inputs(),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_default_is_on_the_step_lattice(name, minimum, step, default):
    steps = (default - minimum) / step
    assert abs(steps - round(steps)) < 1e-9, (
        f"{name}: default {default:g} is not reachable from min {minimum:g} in "
        f"steps of {step:g} — the browser rejects it as a step mismatch and "
        f"Save Config silently fails for the ENTIRE form. Nearest valid value: "
        f"{minimum + round(steps) * step:g}"
    )
