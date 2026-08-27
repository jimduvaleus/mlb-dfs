# Handoff — contest-aware portfolio selection (2026-08-26)

Delete this file once the work is merged. It exists because 27 files are
uncommitted and because several *negative* results are worth more than the code.

## State: nothing is committed

13 modified core files + 13 new scripts + 1 new module, on branch
`parallel-shape-mutants`. **1,202 tests pass**, `tsc` clean. Commit before doing
anything else, or the first `git checkout` loses a day's work.

## What was built

**New objectives, wired into the pipeline sweep.** `gpp.selector_mode` is now a
comma list (`kelly,dr,emax,coverage`, default `kelly,dr`), rendered as
checkboxes in ConfigForm. Sweep keys double as arm ids: det 1-5, kelly 11-15,
coverage 23, E[max] 31, dR 41 — `PortfolioTable.armLabel()` maps them to names,
keep it in sync with `pipeline.py`.

**Real contest ladders.** The GPP path hardcoded `dk_classic_gpp_5001` — one
5,001-entry curve for every contest. `gpp.contest_structure` /
`contest_field_size` now resolve the real ladder, and an unregistered contest
gates the run through the existing `PayoutFallbackDialog` (payload shape matches
`mrp_payout_fallback`, one dialog serves both).

**Per-contest selection** — `src/optimization/multi_contest.py`. Groups an
entries file by contest, resolves each ladder, orders them, and selects each
contest's own disjoint slice. **Built and verified standalone; NOT yet wired
into `pipeline._run`** — that is the next task.

**Fixes found along the way** (each was silent, each found by checking an
invariant rather than by anything failing):
- `CandidateGenerator.generate` produced 1.97% duplicate lineups — now
  `dedupe=True` by default. Any pre-08/26 replay baseline used a pool with
  duplicates; `dedupe=False` reproduces it.
- `generate_shape_mutants` returned **0** mutants when `eligible_positions` was
  a string rather than a list — strictly worse than omitting the column (104).
  Fixed by `normalize_eligible_positions` in `lineup.py`.
- `generate_sim_optimal_lineups` opened 20 CBC threads on 8 physical cores.
  Measured 743/218/142/139/139 ms per solve at 1/4/8/16/20 workers — flat past
  the physical count. Now capped via `physical_cores()`.
- Payout structure resolution matched size variants by back-solved entries,
  which ignores rake — wrong for 5 of 6 real contests (a $15K mini-MAX resolved
  to a $12,000 table). Now matches on the **advertised pool**, which the contest
  name states exactly.
- Ladder bands and DK tie-splitting were hardcoded to a 670-payer contest.
- `n_bytes = W // 8` under-allocated the coverage bitset for any world count not
  divisible by 8.

## Settled findings

Four contests, two slates: Main Event Warm Up (3,335 @ $333), Bat Flip
(10,170 @ $17), Hot Corner (594 @ $3, 5-max), Skipper (234 @ $25, 1-max).

- **Self-competition objectives beat ownership-ranked and correlation selection
  by 12-24 points of portfolio ROI**, consistently across all four.
- **Kelly / dR / coverage are statistically tied** (median per-arm sd 0.45 ME /
  1.13 BF over 5 grading seeds; pairwise z < 2). Do not claim a winner.
  E[max] sits a rung below (z 2.4-3.0).
- **Contest-awareness falls out of the payout math**: ~19 points of ownership
  swing from a 235-entry to an 11,437-entry contest, unprompted. Per-contest
  selection reproduced a 45-point spread across six real contests in one run.
- **Coverage is structurally contest-blind** (slope 0.00 across a 63x field-size
  range). It ties on large contests and was right on Hot Corner *by accident*.
  Treat with care on small fields.
- **Marginal grading is blind to self-competition** — 50 identical lineups grade
  **+7.6% marginal, -60.1% portfolio**. Always grade in portfolio mode
  (`scripts/portfolio_grading.py`).

## Negative results — do not redo these

- **Generation is not the constraint.** 2.5x the ILP anchors: **+0.18 pts**.
  Dropping all 7,850 mutants: **~0**. Ownership headroom was never binding.
- **Rank-denominated gate currency: -0.03 pts**, despite ranking lineups far
  more like the payout does (Spearman 0.94-0.97 vs 0.79-0.90, +13pt recall).
  Available as `--gate-currency rank`; it *does* help det-style selectors (+3.6
  to +9.1) which cannot price rank themselves.
- **Perfect ownership projection is worth ~+1.4 to +1.9 pts** (oracle field),
  against 16 points for the objective choice. Feeding realized ownership to the
  pool and gate as well is usually *worse*.
- **Anchor world deciles do not matter.** corr(decile, ceiling) = -0.097. A
  quiet world is quiet for everyone including the field, so a quiet-world
  optimum is not a low-ceiling lineup.

Pattern worth internalising: **two strong proxy improvements both converted to
nothing.** A 150-lineup portfolio draws from ~750 candidates above its own mean
ceiling — 5:1 coverage. Improving *what reaches* the objectives does not bind.

## Open work, in order

1. **Wire `select_per_contest` into `pipeline._run`.** Shortlist once, compute
   `cand_scores` once (contest-independent), call it per arm, route through
   `assign_per_contest` instead of `assign_lineups_to_entries`. Two contract
   questions to settle first:
   - The sweep is keyed by one number per portfolio, but selection is now
     per-arm AND per-contest (2 arms x 6 contests = 12 portfolios). Either add a
     dimension or let an arm's portfolio be the concatenation across contests
     (which `assign_per_contest` produces naturally).
   - `portfolio_size` becomes per-contest, not global.
2. **First real UI run.** `replay_slate.py` cannot exercise any of this — no
   archive has `market_odds_projections.csv`. Only a smoke test has run
   (`_field_sorted_list` / `_sim_matrix` / `_build_col_lineups` verified against
   a real `ContestScorer`; world axes agree).
3. **Multi-slate validation.** Everything rests on two slates. `replay_slate.py
   --all --variants` is what would turn any of this into a design principle.
4. **Cross-contest risk** — measured, not built. Rebuilding contest B under an
   exact-duplicate constraint raised P(at least one contest profits) **+4.1 pts**
   and P(at least one doubles) **+2.9**, costing 5.7% of mean EV. `gamma_out`
   below 9 buys nothing beyond the duplicate rule. `select_per_contest`'s
   `exclude_used=True` already delivers this within a run.

## Gotchas

- Coverage silently yields nothing if `gpp.compute_tail_metrics` is off.
- dR is the only arm with real memory cost — it retains the sorted field
  (~0.6-1.2 GB) the pipeline otherwise frees; `gpp.dr_shortlist` caps it.
- Any new `gpp.*` config key must land in **three** places (`config.yaml`,
  `models.py`, `types.ts`) or Pydantic drops it silently on save.
- Grading noise: gaps under ~1.3 pts (ME) / ~3.2 pts (BF) are unresolved at 40k
  sims. Use `scripts/regrade_portfolios.py` to size it before believing a gap.
