#!/usr/bin/env python
"""Weekly walk-forward evaluation runner (see PROSPECTIVE_PROTOCOL.md).

    python scripts/weekly_walkforward.py 08052026 08062026 ...

For the given NEW slates (plus everything already in bt_core.BACKTEST_SLATES),
in order:

  1. build oracle tables for the new slates (all seeds, both calib flavors,
     field sidecars) -- skip-if-exists, so re-runs are cheap;
  2. `backtest_lab.py verify` -- hard stop if not green;
  3. refresh the model-error audit (`backtest_audit.py all`) -- the error
     model / signal / crowding numbers are walk-forward quantities;
  4. run the pre-registered adjudication (`backtest_lab.py adjudicate`);
  5. tee everything into tests/backtest_output/evidence/YYYYMMDD/.

The combined slate list is passed to every child through the BT_SLATES env
override; bt_core's committed constant is only updated at milestones.

Mining control (PROSPECTIVE_PROTOCOL.md): this runner only executes the
standing, pre-registered arm set. A new arm/hypothesis needs an EVIDENCE_LOG
entry BEFORE it runs here.
"""
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import BACKTEST_SLATES  # noqa: E402  (respects BT_SLATES itself)


def run(step: str, cmd: list[str], env: dict, log_dir: Path) -> None:
    log = log_dir / f"{step}.log"
    print(f"== {step}: {' '.join(cmd)} -> {log}", flush=True)
    with open(log, "w") as f:
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(
            f"{step} failed (exit {p.returncode}) -- see {log}. "
            "Nothing downstream was run; fix and re-run."
        )


def main() -> None:
    new = [s for s in sys.argv[1:] if s.isdigit()]
    slates = list(dict.fromkeys(BACKTEST_SLATES + new))  # ordered de-dupe
    stamp = dt.date.today().strftime("%Y%m%d")
    log_dir = PROJECT_ROOT / "tests" / "backtest_output" / "evidence" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, BT_SLATES=",".join(slates))
    py = sys.executable
    print(f"walk-forward over {len(slates)} slates ({len(new)} new) -> {log_dir}")

    if new:
        run("oracle", [py, "tests/backtest_oracle.py", *new], env, log_dir)
        run("field", [py, "tests/backtest_oracle.py", "field", *new], env, log_dir)
    run("verify", [py, "tests/backtest_lab.py", "verify"], env, log_dir)
    run("audit", [py, "tests/backtest_audit.py", "all"], env, log_dir)
    run("adjudicate", [py, "tests/backtest_lab.py", "adjudicate"], env, log_dir)
    print(f"done -- review {log_dir}/adjudicate.log and append the gate table "
          "to the open EVIDENCE_LOG.md entry")


if __name__ == "__main__":
    main()
