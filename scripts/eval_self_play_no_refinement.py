"""Refinement-ablation companion to scripts/eval_self_play_selector.py: same
methodology, same slates, same round_n_sims/pool size defaults, but with
precision refinement fully disabled (precise_n_sims=None) -- answers
"how much is the precision-refinement pass actually buying us" directly,
since refinement is both the dominant memory cost (its on-demand precise
scoring is what drives multi-GB peaks, see self_play._MMAP_THRESHOLD_BYTES's
comment) and a meaningful compute cost (~34% of construction time on the one
slate measured with granular per-contest timing) for, on that same slate,
only 4 swaps across 10 contests.

Reuses scripts/eval_self_play_selector.py's run_slate/main wholesale (module-
level override, not a copy) so this can never silently drift from the
with-refinement run's methodology -- only PRECISE_N_SIMS and the output CSV
paths differ, so results are directly comparable row-for-row.

Usage
-----
    source venv/bin/activate
    python scripts/eval_self_play_no_refinement.py <slate MMDDYYYY> [<slate> ...]

Env vars: same as eval_self_play_selector.py (BT_NSIMS, SELF_PLAY_POOL_SIZE,
SELF_PLAY_ROUND_NSIMS, SELF_PLAY_REFRESH_EVERY, SELF_PLAY_SEED,
SELF_PLAY_FORCE) EXCEPT SELF_PLAY_PRECISE_NSIMS, which this script pins to
"disabled" regardless of the environment.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_self_play_selector as base  # noqa: E402

base.PRECISE_N_SIMS = None
base.RESULTS_CSV = base.OUT_DIR / "results_no_refinement.csv"
base.ROUND_LOG_CSV = base.OUT_DIR / "round_log_no_refinement.csv"
base.REFINEMENT_LOG_CSV = base.OUT_DIR / "refinement_log_no_refinement.csv"

if __name__ == "__main__":
    base.main()
