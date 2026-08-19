"""Stochastic ownership: Dir(alpha) draws instead of one point estimate.

Haugh & Singal footnote 7, on their own first attempt: "we assumed p was fixed
and known but this led to over-certainty and poor performance of the resulting
portfolios." We have exactly that over-certainty -- every simulated field draw
in this repo reuses ONE ownership vector and varies only the RNG seed, so
field-model uncertainty is represented purely as lineup-sampling noise and
never as uncertainty about where the chalk actually is.

PER-POSITION-GROUP, and that is forced by the consumer rather than chosen:
`ContestSimulator._build_pos_pools` renormalises ownership within each position
(`w / w.sum()`), so field generation is SCALE-INVARIANT and only within-position
relative weights can matter. A slate-wide Dirichlet would waste most of its
variance on a degree of freedom the field generator discards.

-----------------------------------------------------------------------------
THE TENSION THIS MODULE MUST NOT PAPER OVER
-----------------------------------------------------------------------------
EVIDENCE_LOG H3(b) came back INVERTED: the real top band is LESS crowded than
our simulated field on 6/9 slates -- the field model already over-concentrates,
and the crowding payout correction was dropped because its premise reversed.

A plain Dirichlet-multinomial makes within-draw concentration WORSE, not
better: every opponent in a draw shares the same perturbed p, so they pile onto
the same randomly-boosted players. Fitting alpha_0 from ownership prediction
error and stopping there would therefore push an already-wrong quantity further
wrong.

So these are TWO knobs fitted to TWO measurements, and they must be moved
together:

  alpha_0        location uncertainty -- how wrong our ownership POINT ESTIMATE
                 is across slates. Fitted here, from archived realized
                 %Drafted vs prediction (`fit_concentration`).
  concentration  within-draw crowding -- how much opponents agree with each
                 other. Lives in contest.py's stack/secondary-stack
                 probabilities, and is measured against real top-band crowding,
                 not against ownership error.

Target defect for the pair: the simulated field's p99 exceeds the max realized
score by 47-90 FPTS on every archived slate.
"""
from __future__ import annotations

import numpy as np

# Dirichlet requires strictly positive concentration parameters. A player the
# model gives exactly 0 would otherwise be undrawable AND make alpha invalid;
# this floor keeps him rare rather than impossible, matching the reasoning in
# `build_external_players_df` (a hard 0 makes a player mathematically
# impossible to draw into a simulated opponent field).
_ALPHA_FLOOR = 1e-6


def dirichlet_ownership(
    ownership_vec: np.ndarray,
    positions: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One draw of ownership, resampled within each position group.

    Mean-preserving by construction: E[Dir(alpha)] = alpha / alpha.sum(), and
    alpha is set proportional to the point estimate, so averaging many draws
    returns the input. Each group's TOTAL is preserved exactly, so the result
    is on the caller's original scale (fraction summing to slot count, or
    percentage points) and no consumer needs to know this happened.

    `concentration` is alpha_0 per group: large means tight around the point
    estimate, small means diffuse. `np.inf` returns the input unchanged, which
    is the "today's behaviour" arm every comparison needs.
    """
    own = np.asarray(ownership_vec, dtype=np.float64)
    pos = np.asarray(positions)
    if own.shape[0] != pos.shape[0]:
        raise ValueError(f"ownership {own.shape} does not match positions {pos.shape}")
    if not np.isfinite(concentration):
        return own.copy()
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0, got {concentration}")

    out = np.empty_like(own)
    for group in np.unique(pos):
        m = pos == group
        w = own[m]
        total = w.sum()
        if total <= 0:
            out[m] = w
            continue
        alpha = np.maximum(w / total * concentration, _ALPHA_FLOOR)
        out[m] = rng.dirichlet(alpha) * total
    return out


def fit_concentration(pred: np.ndarray, actual: np.ndarray,
                      positions: np.ndarray) -> dict:
    """Fit alpha_0 per position group from archived predicted/realized ownership.

    Under p ~ Dir(alpha) with mean m and alpha_0 = sum(alpha),

        Var(p_i) = m_i (1 - m_i) / (alpha_0 + 1)

    so matching the observed squared error of our point estimate to that
    variance gives a method-of-moments estimate

        alpha_0 = mean( m (1 - m) ) / mean( err^2 ) - 1

    Note this is a RATIO OF MEANS, not a mean of ratios. The per-player form
    `mean(m(1-m)/err^2)` is the obvious-looking version and is badly biased:
    any player whose realized ownership happens to land near its prediction
    contributes a near-zero denominator and dominates the average (measured:
    it recovered 33,496 for a true alpha_0 of 250). Averaging numerator and
    denominator separately is the stable estimator of the same identity.

    Computed on the WITHIN-GROUP SIMPLEX (each group's values divided by the
    group total), because that is the only thing the field generator reads.

    This measures how wrong our ownership predictions are, which is exactly the
    location uncertainty the paper's footnote is about. It says nothing about
    within-draw crowding -- see the module docstring; do not let one stand in
    for the other.
    """
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    pos = np.asarray(positions)
    out: dict = {}
    for group in np.unique(pos):
        m = pos == group
        p, a = pred[m], actual[m]
        ps, as_ = p.sum(), a.sum()
        if ps <= 0 or as_ <= 0 or m.sum() < 2:
            continue
        p, a = p / ps, a / as_
        err2 = (a - p) ** 2
        mean_err2 = float(np.mean(err2))
        if mean_err2 <= 0:
            continue
        alpha0 = float(np.mean(p * (1.0 - p)) / mean_err2 - 1.0)
        out[str(group)] = {
            "alpha_0": max(alpha0, 1.0),
            "n": int(m.sum()),
            "mean_abs_err": float(np.mean(np.abs(a - p))),
        }
    return out


def field_ownership_draws(
    ownership_vec: np.ndarray,
    positions: np.ndarray,
    n_draws: int,
    concentration: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """`n_draws` ownership vectors, one per field sample.

    The intended call site is the K-field-draw loop
    (`ContestScorer._build_field_sorted`'s callers, or
    `allocate_contests_topn_coverage._draw_thresholds`), where today all K
    draws share one vector and differ only by seed.
    """
    return [dirichlet_ownership(ownership_vec, positions, concentration, rng)
            for _ in range(int(n_draws))]
