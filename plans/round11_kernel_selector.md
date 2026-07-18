# Round-11 pre-registration — kernel selector (composition-only diversification)

Registered 2026-07-17, BEFORE implementation. Branch shape-preserving-seed-mutation.

## Question

Does the Det selector's diversification term need the contest-sim payout
matrix, or is payout covariance between lineups reconstructible from
composition alone? Motivated by the shaidy architecture (his tool consumes
the lineup × sim return matrix from a partner, blind to players; the
kernel variant is a further reduction to scalar ROI + composition) and by
round-9's currency null (sim-derived tail metrics added nothing over
scalar mined EV for ranking — but ranking ≠ diversification, so this is
undetermined for the covariance term).

## Design

Implement `KernelPortfolioSelector`: identical greedy mean/diversity
machinery to `DeterminantPortfolioSelector`, but the covariance between
lineups i,j is a structured kernel computed from composition only:

    K(i,j) = Σ_{p ∈ i∩j} w_p               (shared-player term, dominant)
           + γ_team · |team-overlap of hitter stacks|
           + γ_own  · f(own_i, own_j)       (field-correlation proxy:
                                             chalk lineups co-move with the
                                             field → compressed payouts;
                                             leverage anti-correlates)

with w_p ∝ player projected variance share; γ knobs fit once on a
calibration slate by regressing realized robust_payout covariances on the
kernel features (calibration slates excluded from judgment).

Mean vector: unchanged — per-lineup EV from the existing ContestScorer
(this half of the payout matrix is retained by design; EV is
payout-structure-specific and cannot be composition-derived).

## Head-to-head

Same pools (existing blend_full_p150 run dirs where cached), same slates,
same 150-lineup portfolios per risk tier: sim-Det (incumbent) vs
kernel-Det. Judged on the real fields.

## Judgment, pre-committed

- PRIMARY (equivalence): kernel-Det portfolios match sim-Det on mean_pct
  and cash rate within ±1.5pp pooled over ≥30 slates, and prefix-20
  blocks match within ±2pp. Match ⇒ selector needs composition+EV only;
  the portfolio stage becomes applicable to external lineup/ROI feeds
  (shaidy-as-a-service architecture) and drops the K×M matrix cost.
- SECONDARY (mechanism): selection overlap (Jaccard of chosen sets),
  anchor-concentration and ownership-gradient profiles of kernel vs sim
  portfolios — do the emergent conviction trees survive?
- FAILURE MODE WORTH LEARNING: if kernel-Det degrades specifically on
  top-heavy tail metrics but not cash/percentile, the payout-curve
  nonlinearity (points-space vs dollar-space correlation) is the
  irreducible content of the matrix — quantifies exactly what the sims
  buy at selection.
- Meta-rule (round-9 lesson): no conclusions below ~30 events on any
  tail metric; equivalence judged on the fast-converging currencies.

## Non-goals

No changes to generation, scoring, funnel, or live config. Pure selector
experiment, offline only.
