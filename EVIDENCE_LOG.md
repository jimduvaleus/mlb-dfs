# Evidence log (pre-registration required before running)

Every archive-facing comparison gets a dated entry BEFORE it runs: date,
hypothesis, exact config, applicable gates, and what failure looks like.
Results are appended to the same entry after the run. See
`PROSPECTIVE_PROTOCOL.md` for the gate definitions (G1–G5).

---

## 2026-08-02 — Season program kickoff (plan: our-goal-is-to-wobbly-lampson)

Pre-registered hypotheses for the Phase 2 audit + Phase 4/5 adjudication,
all against baseline `prod_p_win` (= live flat2000_uc), 9 slates × seeds
42/137/4242, calib=False, entry-weighted, joint grading for headlines:

- **H1**: sim currencies (p_win, ev_dollars, ev_tail, p_cash) carry realized
  top-minus-bottom decile lift beyond model-free controls (proj_score,
  neg_own, Saber ROI). Failure: per-slate head-to-head difference vs the
  best model-free control has n_pos ≤ 5/9 or sign_p > 0.5. On failure the
  program pivots to model-light selection (pivot rule).
- **H2**: robust-worlds selection (`kelly_B50_robust`) ≥ non-robust EV
  control (`worldEV_topk`) on gates G1/G3/G4.
- **H3**: (a) sim-top-decile realized lift shrinks as slate-level model
  accuracy rises (Spearman < 0 across 9 slates); (b) real top-band crowding
  (dupes, modal-stack share) exceeds the ownership-sampled simulated
  field's. Directional at n=9.
- **H4**: best challenger vs production on G1–G5; verdict vocabulary fixed
  (all gates → "recommend prospective A/B"; G1+G3 → "promising, extend
  prospectively"; else → "no evidence — production stands").

Arms registered for Phase 4: `prod_p_win`, `worldEV_topk`, `kelly_B50`,
`kelly_B50_robust`, `kelly_B10_robust`, `kelly_meanvar_mu`,
`coverage_light`, `kelly_crowd_robust` (only if H3(b) holds),
`ev_dollars_dupe` (4 dupe-bearing slates only, capped at "promising"),
plus the existing contrarian registry (`neg_own`, `prj_own`,
`p_win_no_top10`, `ceiling_contrarian`) as references.

Results (2026-08-02, full memo in `docs/audit_2026-08-02_model_error.md`):

- **H1 FAILED**: p_win/ev_dollars/ev_tail lose to proj_score 7/9 slates at
  all 3 seeds (n_pos 2/9, well under the ≤5/9 failure line). p_cash ties
  proj_score (4–5/9). → **Pivot rule invoked**: model-light selection;
  robust-worlds/Kelly build descoped; H2 mooted.
- **H3(a)** directionally supported, unproven (Spearman −0.50, p=0.17, n=9).
- **H3(b) INVERTED**: the real top band is LESS crowded than the simulated
  field on 6/9 slates — the field model over-concentrates; the crowding
  payout correction is dropped (premise reversed); `kelly_crowd_robust`
  not built.
- Calibration: mean PIT < 0.5 on 8/9 slates; sim field p99 exceeds max
  realized by 47–90 FPTS every slate; σ_team(batter) 0.49z > σ_pitcher
  0.35z > σ_slate 0.27z.

## 2026-08-02 — Amendment A1: Phase 4′ model-light adjudication (post-pivot)

Pre-registered BEFORE running. Baseline: faithful production
(`prod_p_win` with evw=0.25 + DeterminantPortfolioSelector + sim corr, the
cmd_verify-validated configuration — NOT the arm registry's evw=1.0
approximation). Null: `random@floor30`. Challengers:

- `proj_score` (greedy, floor 30, admit 2000) — the model-free control as
  an arm.
- `p_cash` (same shape) — the one sim currency that matched projections.
- `p_cash@assign` — global assignment routing on p_cash.
- `coverage_light` — proj_score ranked through the real Det selector
  (evw 0.25) with **composition** correlation (shared-player overlap),
  no sim inputs in either term.

9 slates × seeds 42/137/4242, calib=False, entry-weighted, gates G1–G5 vs
the faithful baseline, **joint grading for headline numbers**, single-
insertion reported for comparability. Verdict vocabulary as above.
Failure looks like: no challenger passes G1+G3 → "no evidence — production
stands" is the recorded outcome.

Results (2026-08-02, joint grading, 9 slates × 3 seeds, baseline
`prod_faithful` self-checked 152/152 vs live allocate_contests):

| arm | $/entry | cash% | top1% | G1 | G2 | G3 | G4 | G5 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| prod_faithful | +3.48 | 17.4 | 1.02 | — | — | — | — | — | baseline |
| p_cash | +4.81 | 31.9 | 1.46 | ✗ | ✗ 4/9 | ✓ | ✓ | ✓ 3/3 | **no evidence — production stands** |
| p_cash@assign | +1.02 | 32.0 | 1.51 | ✗ | ✗ 4/9 | ✓ | ✓ | ✓ 2/3 | no evidence — production stands |
| proj_score | +0.66 | 31.7 | 1.74 | ✗ | ✗ 4/9 | ✓ | ✓ | ✓ 2/3 | no evidence — production stands |
| coverage_light | −1.22 | 26.8 | 0.91 | ✗ | ✗ 5/9 | ✗ | ✓ | ✗ | no evidence — production stands |
| random@floor30 | −0.41 | 21.4 | 0.95 | (null) | | | | | |

- G1 fails for every challenger: each has ≥1 seed with negative pooled
  d$/entry vs baseline — no seed-robust dollar edge exists in this sample.
- p_cash is the strongest challenger (G3+G4+G5, 3/3 seeds on log-growth,
  drop_max +2.49 vs baseline −1.17 — its edge survives removing its own
  largest payout) but wins only 4/9 slates. Logged as the one arm to track
  prospectively as new slates accrue; NOT promoted.
- coverage_light (composition-only diversity) underperforms the random
  null on $/entry and the whole rate ladder — composition overlap through
  the Det selector is anti-signal as implemented, a genuine negative result.
- Dollar/ROI columns remain outlier-dominated at n=9 (baseline +99% ROI
  rests on outright contest wins); the gates, not the ROI column, are the
  decision instrument.

## Backlog — open questions (not yet pre-registered)

### Bit-packing density waste in topn_coverage (raised 2026-08-11)

`allocate_contests_topn_coverage` stores every candidate's crossing set as a
dense bit-plane, `(n_cand, R, K x n_sims_g / 8)` uint8. That is the right
representation for the loose outer rung, where a large fraction of slots are
set. It is badly wrong for tight ranks: on the 08/10 mini-MAX contest a
rank-1 crossing set holds **1.9 set bits out of 53,775 slots — density
3.5e-5**, i.e. ~6.7 KB spent to store ~2 bits of information (~900x waste).

That waste is what makes "just raise the sim budget" look infeasible for the
payout ladder: at the ~10x sims rank-1 would need to settle, the dense bit
array alone is ~26 GB, against a ~11 GB ceiling.

Questions to answer:

1. What is the actual density-vs-rank curve across contest sizes, and where
   does the crossover sit at which a sparse index list beats a dense bitset
   (both in bytes and in gain-evaluation speed)?
2. What sim headroom does a hybrid buy — sparse index lists for tight rungs,
   dense bit-planes for loose ones — holding peak RSS at ~11 GB?
3. `_draw_thresholds` materializes an `(n_sims_g x field_size_g)` float32
   array (~768 MB for mini-MAX, plus an `np.partition` copy) purely to
   extract a few per-world order statistics. Threshold extraction is
   per-world independent, so this is chunkable over WORLDS with identical
   results. How much does that alone decouple n_sims from peak memory?
4. Does the greedy's per-pick popcount stay the fastest option under a
   sparse representation, or does a lazy/priority-queue greedy (exact for
   monotone submodular coverage) become the better lever?

Not a pre-registered experiment — this is a capability question (what can we
afford?), not a currency question (does it make money?). Its value is
entirely instrumental: it only matters if more sim worlds turn out to buy
real stability, which the 1x/2x/3x sweep is testing.
