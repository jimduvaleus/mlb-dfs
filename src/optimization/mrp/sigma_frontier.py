"""LEGACY linear frontier: sigma_dG only, no quadratic term.

SUPERSEDED FOR GENERATION by `frontier_qp.py`, which solves the paper's line 2
whole. Kept because `scripts/eval_field_covariance.py` measures on this linear
form and remains a working diagnostic, and because `pool_sigma_coverage` below
is the pool-spanning gate regardless of which solver produced the frontier.

WHY IT WAS SUPERSEDED. Equation (14) makes line 2's lambda-term
`Var(w'delta - G) - Var(G)` -- margin variance. This module keeps only the
cross-term, which is half that object, and the half that measured as ~84%
ownership (commit 5128a7f). The other half, `w'Sigma w`, was dropped for a
solver limitation rather than a modelling reason and was never measured.

WHY THIS IS GATED, NOT ASSUMED. This repo has a measured negative on generated
supplements: `diagnose_ilp_supplement_pwin.py` found ILP supplements scoring
0.57-0.68x the external pool's p_win, with ZERO supplement lineups reaching the
combined pool's top 1% across 7 slates (106.2 expected under random placement),
and `compare_candidate_pools.py` found augmentation "never helps -- and
sometimes hurts -- the SELECTED portfolio's hit99 rate."

That negative is real but its diagnosis is specific: "a per-world argmax lineup
is optimal for ONE specific simulated world, not necessarily a lineup that
performs well across the distribution of worlds." It condemns SIM-OPTIMAL
generation. A sigma_dG frontier is a different object -- a DISTRIBUTIONAL
linear functional, one solve per lambda over slate-level expectations, aimed at
the low-cutoff-covariance region a commercial ROI-maximising optimiser has no
reason to populate.

The counter-argument for generating anything at all is that the SaberSim pool
is a shared commodity, not proprietary edge (memory
project-pipeline-is-a-random-draw: a 3-entry rival played our pool's #3 lineup
with all 10 players matching). dR cannot select what is not in the pool.

So: run `pool_sigma_coverage` FIRST. If the external pool already spans the
low-sigma region densely, generation buys nothing and should be skipped.

LINEAR, hence a plain CBC solve rather than a MIQP. That was the whole reason
the quadratic term was dropped here -- an implementation constraint, not a
judgement that the term was worthless. `frontier_qp.py` lifts it by moving to
CP-SAT with McCormick product variables. `df["mean"]` is
`generate_optimal_lineups`' objective coefficient vector, so swapping it is the
same one-line substitution `generate_sim_optimal_lineups` already makes.
"""
from __future__ import annotations

import numpy as np

from src.optimization.optimal_lineups import generate_optimal_lineups


def sigma_objective(
    mu: np.ndarray,
    sigma_dG: np.ndarray,
    lam: float,
) -> np.ndarray:
    """The paper's objective MINUS its quadratic term, on a rescaled lambda.

    Haugh & Singal maximise `w'mu + lambda(w'Sigma w - 2 w'sigma_dG)`; dropping
    the quadratic term (CBC cannot express it) leaves `w'mu - 2 lambda
    w'sigma_dG`. Use `frontier_qp.frontier_lineups` for the full objective --
    this is the diagnostic form only.

    sigma_dG is a covariance in FPTS^2 while mu is in FPTS, so a raw lambda
    grid would be neither interpretable nor comparable across slates. sigma is
    therefore rescaled to mu's own cross-player spread, making lambda a pure
    mixing weight: 0 is the plain projection ILP, 1 weights the leverage term
    equally with projection.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sig = np.asarray(sigma_dG, dtype=np.float64)
    sd = sig.std()
    if sd <= 0:
        return mu.copy()          # sigma carries no cross-player information
    # Scale sigma onto mu's own spread so lambda is a pure mixing weight. If mu
    # is degenerate (no cross-player spread) fall back to unit scale rather
    # than collapsing the term to zero -- otherwise lambda would silently stop
    # doing anything on exactly the slate where projections are uninformative
    # and leverage is all there is to go on.
    scale = mu.std()
    if scale <= 0:
        scale = 1.0
    return mu - float(lam) * (sig / sd) * scale


def sigma_frontier(
    df,
    sigma_dG: np.ndarray,
    lambdas=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00),
    n_per_lambda: int = 50,
    min_uniques: int = 3,
    min_stack: int = 4,
    salary_floor=None,
    min_secondary=None,
    progress_cb=None,
) -> list:
    """Solve the ILP across a lambda grid; return unique lineups.

    `df` follows `generate_optimal_lineups`' contract; its "mean" column is
    replaced per lambda, exactly as the sim-optimal generator does per world.
    Deduplication is shared across the whole sweep via the `seen` set, so
    neighbouring lambdas that resolve to the same lineup cost one entry, not
    two.
    """
    mu = df["mean"].to_numpy(dtype=np.float64)
    seen: set = set()
    out: list = []
    for i, lam in enumerate(lambdas):
        d = df.copy()
        d["mean"] = sigma_objective(mu, sigma_dG, lam)
        got = generate_optimal_lineups(
            d, n=n_per_lambda, min_uniques=min_uniques, min_stack=min_stack,
            salary_floor=salary_floor, seen=seen, min_secondary=min_secondary,
        )
        out.extend(got)
        if progress_cb is not None:
            progress_cb(i + 1, len(lambdas))
    return out


def pool_sigma_coverage(pool_scores: np.ndarray, frontier_scores: np.ndarray) -> dict:
    """THE GATE. Does the external pool already reach the low-sigma_dG region?

    `pool_scores` / `frontier_scores` are per-lineup `w'sigma_dG`
    (`field_covariance.lineup_sigma_scores`). Lower is better: a low-sigma
    lineup's fortunes move less with the payout cutoff.

    Returns the frontier's position within the pool's own distribution. If the
    frontier's best lineups sit inside the pool's existing left tail
    (`frontier_pct_in_pool` not small, `n_pool_below_frontier_min` not tiny),
    the pool already spans that region and generation should be SKIPPED -- the
    repo's prior negative on augmentation then stands unchallenged.
    """
    pool = np.asarray(pool_scores, dtype=np.float64)
    fr = np.asarray(frontier_scores, dtype=np.float64)
    if pool.size == 0 or fr.size == 0:
        return {"decision": "insufficient-data", "n_pool": int(pool.size), "n_frontier": int(fr.size)}

    fr_min = float(fr.min())
    below = int((pool < fr_min).sum())
    pct = float((pool < fr_min).mean() * 100.0)
    return {
        "n_pool": int(pool.size),
        "n_frontier": int(fr.size),
        "pool_min": float(pool.min()),
        "pool_p01": float(np.percentile(pool, 1)),
        "pool_median": float(np.median(pool)),
        "frontier_min": fr_min,
        "frontier_median": float(np.median(fr)),
        # Where the frontier's most extreme lineup sits in the pool's own
        # distribution, as a percentile.
        "frontier_pct_in_pool": pct,
        "n_pool_below_frontier_min": below,
        "decision": "generate" if below == 0 else "pool-already-spans",
    }
