# Model-error audit memo — 2026-08-02

Phase 2 of the season program (pre-registered 2026-08-02 in `EVIDENCE_LOG.md`,
commit `bb74d5f`, before any result existed). Instrument: 9-slate oracle
archive, seed 42 / uncalibrated grids unless stated; produced by
`tests/backtest_audit.py` (`pit | varcomp | signal | crowding`), tables in
`tests/backtest_output/audit/`.

## Headline verdicts

1. **H1 — sim ranking signal beyond model-free controls: FAILED, decisively.**
   p_win, ev_dollars, and ev_tail lose the per-slate head-to-head against a
   plain projections rank (proj_score) on 7 of 9 slates at all three seeds
   (mean deficit $1.1–$4.5/entry). The pre-registered pivot rule fires:
   selection development moves to model-light currencies; the robust-worlds /
   Kelly-selector build is descoped.
2. **The predictable margin lives in the cash zone, not the tail.**
   Top-minus-bottom decile lift on realized $/entry: proj_score +$2.17,
   sim p_cash +$2.38 (6/9 slates positive; sign_p ≈ 0.5, so suggestive, not
   proven). Every upside-weighted currency is ~zero or negative
   ($/entry: p_win +0.23, ev_dollars −0.44, ev_tail −1.24, SaberSim ROI
   −3.46). On top-1% rate, *nothing* clears the ~±0.5pp resolution floor
   (best: neg_own +0.22pp, p_cash +0.21pp; all EV currencies negative).
   The tournament ceiling is unpredictable in this sample; the min-cash zone
   is where measurable skill exists.
3. **The sim's tails are inflated, and the batter side is the worst-measured
   part.** Mean lineup PIT is below 0.5 on 8/9 slates (0.31–0.56): realized
   scores sit in the lower half of their own sim distributions. The
   simulated field's p99 ceiling exceeds the best lineup that actually
   happened by 47–90 FPTS on every slate. Variance decomposition of
   z = (realized − sim_mean)/sim_std: σ_team(batter stacks) = 0.49 z
   (~13.9 FPTS, CI [0.42, 0.53]) > σ_pitcher = 0.35 (~10.0, CI [0.29, 0.38])
   > σ_slate = 0.27 (~7.6, CI [0.13, 0.36]). Confirms the prior: batter-side
   error dominates, pitcher signal is meaningfully better, and a slate-wide
   run-environment miss of ~7.6 FPTS per lineup is on top of both.
4. **H3(b) — real-field crowding: direction INVERTED.** The real top-1% band
   is *less* duplicated and *less* stack-concentrated than our
   ownership-sampled simulated field on 6/9 slates (e.g. 07/30: real dupe
   rate 17.6% vs simulated 37.1%; 07/29 modal-stack share 50.1% real vs
   93.3% simulated). The field model over-concentrates the top, which biases
   every field-relative currency (p_win, ev_dollars) — but the planned
   "crowding correction" (penalizing consensus-proximate lineups harder) is
   built on the opposite premise and is therefore dropped.
5. **H3(a) — "dialed-in slates are hardest to separate on": directionally
   sympathetic, unproven.** Spearman(model accuracy, p_win realized lift) =
   −0.50 across slates, p = 0.17 at n=9, and the median-split cut is
   inconsistent. Stays a live hypothesis for the prospective sample; not a
   design driver today.

## Secondary observations

- PIT is seed-stable (07/29 mean PIT varies by 0.0005 across seeds
  42/137/4242): these are model properties, not sampling noise.
- Sim p_cash ≈ proj_score in both lift and head-to-head (4–5/9): the sim's
  *cash-probability* estimate is as good as projections, just not better.
  Its upside estimates are the broken part.
- Joint grading (Phase 1) showed single-insertion grading flattered
  EV-concentrated selection (−$350 on one contest where a naive EV arm
  self-duplicated 12 entries) while diversified arms moved <$0.01/entry —
  independent evidence that concentration is mispriced by naive sim-EV.

## What this means for the program (per the pre-registered pivot rule)

- **Descoped**: per-world payout matrices, `select_kelly`, robust-world
  augmentation as a selection engine, the crowding payout correction.
  Rationale: with no sim ranking signal beyond projections and tails
  inflated by ~50–90 FPTS, a Kelly maximizer over sim worlds optimizes
  noise; the pre-registered rule anticipated exactly this.
- **Kept, reframed**: the fitted error components (σ_team 0.49z,
  σ_pitcher 0.35z, σ_slate 0.27z) become the *validation argument* for
  diversification levels: under Kelly with B = 50× fees, the model-error
  covariance between same-stack lineups is large enough that concentrated
  portfolios are indefensible even though the sampling-covariance penalty
  alone would be negligible. Production's diversity behavior is
  approximately what a robust objective would prescribe.
- **Phase 4′ (adjudication, pre-registered as amendment A1)**: model-light
  arms — proj_score, p_cash, `coverage_light` (proj_score ranked through
  production's Det selector with a *composition-based* correlation:
  shared-player/stack overlap, no sim), p_cash@assign — against a faithful
  production baseline (evw 0.25 + Det selector) and the random+floor null,
  gates G1–G5, joint grading, 3 seeds.
- The field-model concentration bias (finding 4) is logged as a prospective
  work item (fix the ownership-field sampler), not a selection lever now.
