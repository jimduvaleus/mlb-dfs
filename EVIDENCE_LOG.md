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

Results: (pending)
