"""Per-round statistics for EV-guided candidate pool refinement.

Pure functions so the mutant-vs-parent accounting (including holdout
regression checks) is unit-testable outside the pipeline.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def mutant_round_stats(
    parents: list,
    parent_evs: list[float],
    mutants: list,
    mutant_evs: np.ndarray,
    label_fn: Callable[[int], str],
    parent_evs_holdout: Optional[list[float]] = None,
    mutant_evs_holdout: Optional[np.ndarray] = None,
) -> dict:
    """Compare each mutant to its source parent (max player overlap).

    parent_evs / mutant_evs are the *selection* metric (train-half means when
    a holdout split is active, full-matrix means otherwise). When the holdout
    arrays are provided, the best swap's EV delta is also reported on the
    held-out columns — noise-mined swaps regress toward zero there, real
    improvements don't.

    Returns a dict with: n_beat_parent, best_swap_out, best_swap_in,
    best_swap_ev_delta, best_mutant_ev, and (when holdout arrays are given)
    best_swap_ev_delta_holdout.
    """
    parent_sets = [set(int(p) for p in lu.player_ids) for lu in parents]

    n_beat_parent = 0
    best_delta = -np.inf
    best_mi = -1
    best_pi = -1
    for mi, mu in enumerate(mutants):
        mset = set(int(p) for p in mu.player_ids)
        pi = max(range(len(parent_sets)), key=lambda k: len(mset & parent_sets[k]))
        delta = float(mutant_evs[mi]) - parent_evs[pi]
        if delta > 0:
            n_beat_parent += 1
        if delta > best_delta:
            best_delta = delta
            best_mi = mi
            best_pi = pi

    stats: dict = {
        "n_beat_parent": n_beat_parent,
        "best_swap_out": [],
        "best_swap_in": [],
        "best_swap_ev_delta": 0.0,
        "best_mutant_ev": 0.0,
    }
    if best_mi < 0:
        return stats

    best_mset = set(int(p) for p in mutants[best_mi].player_ids)
    stats["best_swap_out"] = sorted(
        label_fn(p) for p in (parent_sets[best_pi] - best_mset)
    )
    stats["best_swap_in"] = sorted(
        label_fn(p) for p in (best_mset - parent_sets[best_pi])
    )
    stats["best_swap_ev_delta"] = float(best_delta)
    stats["best_mutant_ev"] = float(mutant_evs[best_mi])

    if parent_evs_holdout is not None and mutant_evs_holdout is not None:
        stats["best_swap_ev_delta_holdout"] = float(
            mutant_evs_holdout[best_mi] - parent_evs_holdout[best_pi]
        )
    return stats


def split_sim_columns(
    n_sims: int, holdout_fraction: float, rng_seed: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Deterministically split sim columns into (train, holdout) index arrays.

    Returns (None, None) when the fraction yields no holdout columns or no
    train columns, signalling that the split is disabled.
    """
    n_holdout = int(n_sims * holdout_fraction)
    if n_holdout < 1 or n_holdout >= n_sims:
        return None, None
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_sims)
    holdout = np.sort(perm[:n_holdout]).astype(np.int64)
    train = np.sort(perm[n_holdout:]).astype(np.int64)
    return train, holdout
